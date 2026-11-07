# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import socket
import struct

import numpy as np
import torch

from .transport import PipelineTransport
from .socket_pool import ConnectionHealth


class TcpTransport(PipelineTransport):
    """Pipeline transport using TCP sockets for cross-platform communication.

    Supports two modes:

    - **connect-per-send**: Opens a new TCP connection for each tensor,
      transmits, then closes. ~180 μs per 16KB round-trip.
    - **persistent** (with SocketPool): Uses a connection pool to reuse
      TCP connections across multiple ``send()`` calls. ~58 μs per 16KB
      round-trip (3.1x faster). Connections are health-checked on every
      acquire/release and idle connections are evicted after ``idle_timeout``.

    Args:
        send_port: TCP port for sending to next stage. Default 20000.
        recv_port: TCP port for receiving from previous stage. Default 20001.
        host: Host to bind/connect to. Default ``"127.0.0.1"``.
        persistent: If True, use SocketPool for connection reuse.
            Default False (backward compatible).
        pool_size: Max connections per stage-pair in the pool. Default 4.
        idle_timeout: Seconds before an unused connection is evicted. Default 60.
    """

    _TORCH_DTYPE_CODES = {
        torch.float32: 0,
        torch.float64: 1,
        torch.int32: 2,
        torch.int64: 3,
        torch.float16: 4,
        torch.bfloat16: 5,
        torch.uint8: 6,
        torch.int8: 7,
        torch.int16: 8,
        torch.bool: 9,
    }
    _CODE_TO_NP = {
        0: np.float32,
        1: np.float64,
        2: np.int32,
        3: np.int64,
        4: np.float16,
        5: np.float16,
        6: np.uint8,
        7: np.int8,
        8: np.int16,
        9: np.bool_,
    }

    def __init__(self,
                 send_port=20000,
                 recv_port=20001,
                 host="127.0.0.1",
                 persistent=False,
                 pool_size=4,
                 idle_timeout=60.0):
        if send_port < 1 or send_port > 65535:
            raise ValueError(f"Invalid send_port: {send_port}")
        if recv_port < 1 or recv_port > 65535:
            raise ValueError(f"Invalid recv_port: {recv_port}")
        self._send_port = send_port
        self._recv_port = recv_port
        self._host = host
        self._persistent = persistent
        if persistent:
            from .socket_pool import SocketPoolManager
            self._pool_manager = SocketPoolManager(
                host=host,
                recv_port=send_port,  # connect to remote's recv port, not local
                max_connections_per_pair=pool_size,
                idle_timeout=idle_timeout,
            )
        else:
            self._pool_manager = None
        self._recv_sock = None
        self._conn = None
        self._initialized = False

    def send(self, tensor, dest_stage):
        """Send a tensor over TCP.

        In persistent mode, acquires a connection from the pool,
        sends the payload, and releases the connection back. DEAD
        connections are discarded by the pool on release.

        Args:
            tensor: The tensor to send.
            dest_stage: Destination stage ID.
        """
        if not self._initialized:
            raise RuntimeError("TcpTransport not initialized. Call initialize() first.")

        t = tensor.cpu().detach().to(torch.float32).contiguous()
        arr = t.numpy()
        numel = arr.size
        dtype_code = self._TORCH_DTYPE_CODES.get(tensor.dtype, 0)
        header = struct.pack("!II", numel, dtype_code)
        data = arr.tobytes()
        payload = header + data

        if self._pool_manager:
            conn = self._pool_manager.get_connection(dest_stage)
            try:
                conn.sock.sendall(payload)
            except (ConnectionError, BrokenPipeError, OSError):
                conn.health = ConnectionHealth.DEAD
                raise
            finally:
                self._pool_manager.return_connection(dest_stage, conn)
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            try:
                sock.connect((self._host, self._send_port))
                sock.sendall(payload)
            finally:
                sock.close()

    def recv(self, tensor, src_stage):
        """Receive a tensor over TCP.

        In persistent mode, reuses the same accepted connection.
        In connect-per-send mode, accepts a new connection per call.

        Args:
            tensor: Pre-allocated buffer (unused).
            src_stage: Source stage ID.

        Returns:
            torch.Tensor: The received tensor.
        """
        if not self._initialized:
            raise RuntimeError("TcpTransport not initialized. Call initialize() first.")

        if self._persistent:
            if self._conn is None:
                self._conn, _ = self._recv_sock.accept()
            conn = self._conn
            header = conn.recv(8)
            if len(header) < 8:
                raise ConnectionError("Failed to receive header")
            numel, dtype_code = struct.unpack("!II", header)
            data = b""
            while len(data) < numel * 4:
                chunk = conn.recv(numel * 4 - len(data))
                if not chunk:
                    raise ConnectionError("Connection closed")
                data += chunk
            np_dtype = self._CODE_TO_NP.get(dtype_code, np.float32)
            itemsize = np.dtype(np_dtype).itemsize
            arr = np.frombuffer(data[:numel * itemsize], dtype=np_dtype)
            return torch.from_numpy(arr.copy())
        else:
            conn, _ = self._recv_sock.accept()
            try:
                header = conn.recv(8)
                if len(header) < 8:
                    raise ConnectionError("Failed to receive header")
                numel, dtype_code = struct.unpack("!II", header)
                data = b""
                while len(data) < numel * 4:
                    chunk = conn.recv(numel * 4 - len(data))
                    if not chunk:
                        raise ConnectionError("Connection closed")
                    data += chunk
            finally:
                conn.close()
            np_dtype = self._CODE_TO_NP.get(dtype_code, np.float32)
            itemsize = np.dtype(np_dtype).itemsize
            arr = np.frombuffer(data[:numel * itemsize], dtype=np_dtype)
            return torch.from_numpy(arr.copy())

    def initialize(self, topology):
        """Bind the receive socket and start the connection pool eviction thread.

        Args:
            topology: Pipeline topology (unused for TCP transport).
        """
        if self._initialized:
            return
        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._recv_sock.bind((self._host, self._recv_port))
        self._recv_sock.listen(1)
        if self._pool_manager:
            self._pool_manager.start_eviction()
        self._initialized = True

    def shutdown(self):
        """Close all connections and release resources.

        Drains the connection pool in persistent mode, closes the
        persistent receive connection if any, and shuts down the
        listening socket.
        """
        if self._pool_manager:
            self._pool_manager.drain()
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._recv_sock is not None:
            self._recv_sock.close()
            self._recv_sock = None
        self._initialized = False
