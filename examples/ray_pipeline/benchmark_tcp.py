# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
Benchmark: Connect-per-send TCP vs Persistent TCP vs Unix domain sockets.

Uses the actual TcpTransport class for TCP modes and raw Unix sockets
for the local-only comparison. Measures full round-trip latency for
tensor sizes from 4 bytes to 4 MB.
"""

import socket
import struct
import threading
import time
import statistics
import os
import tempfile
import torch
import numpy as np

# Direct import of TcpTransport to avoid deepspeed.__init__ chain
import importlib.util
import sys


def _import_tcp_transport():
    spec = importlib.util.spec_from_file_location(
        'tcp_transport',
        os.path.join(os.getcwd(), 'deepspeed', 'runtime', 'pipe', 'tcp_transport.py'))
    mod = importlib.util.module_from_spec(spec)

    # Set up transport ABC stub
    class _PipelineTransport:
        pass

    # Set up parent modules
    import types
    mod2 = types.ModuleType('transport')
    mod2.PipelineTransport = _PipelineTransport
    sys.modules['transport'] = mod2
    mod.__package__ = 'deepspeed.runtime.pipe'
    spec.loader.exec_module(mod)
    return mod.TcpTransport


TcpTransport = _import_tcp_transport()

TENSOR_SIZES = [1, 16, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
ITERATIONS = 100


def benchmark_tcp_connect():
    """Benchmark TcpTransport with persistent=False (default)."""
    send_transport = TcpTransport(send_port=22001, recv_port=22002, persistent=False)
    recv_transport = TcpTransport(send_port=22000, recv_port=22001, persistent=False)
    _benchmark_roundtrip(send_transport, recv_transport, "TCP\nconnect-per-send")


def benchmark_tcp_persistent():
    """Benchmark TcpTransport with persistent=True."""
    send_transport = TcpTransport(send_port=22003, recv_port=22004, persistent=True)
    recv_transport = TcpTransport(send_port=22002, recv_port=22003, persistent=True)
    _benchmark_roundtrip(send_transport, recv_transport, "TCP\npersistent")


def benchmark_unix_connect():
    """Benchmark raw Unix domain sockets (connect per send)."""
    sock_path = os.path.join(tempfile.gettempdir(), "ds-bench-unix.sock")
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    listen_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listen_sock.bind(sock_path)
    listen_sock.listen(1)

    latencies = {s: [] for s in TENSOR_SIZES}
    ready = threading.Event()

    def server():
        ready.set()
        for _ in range(ITERATIONS * len(TENSOR_SIZES) + 20):
            conn, _ = listen_sock.accept()
            header = conn.recv(8)
            n, dc = struct.unpack("!II", header)
            itemsize = 4  # float32
            body = b""
            while len(body) < n * itemsize:
                body += conn.recv(n * itemsize - len(body))
            conn.close()

    srv = threading.Thread(target=server, daemon=True)
    srv.start()
    ready.wait()
    time.sleep(0.05)

    # Warmup
    for _ in range(10):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(sock_path)
        t = torch.randn(1024)
        arr = t.numpy()
        payload = struct.pack("!II", arr.size, 0) + arr.tobytes()
        s.sendall(payload)
        s.close()

    for size in TENSOR_SIZES:
        tensor = torch.randn(size)
        arr = tensor.numpy()
        payload = struct.pack("!II", arr.size, 0) + arr.tobytes()
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(sock_path)
            s.sendall(payload)
            s.close()
            latencies[size].append((time.perf_counter() - start) * 1e6)

    listen_sock.close()
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    return _stats("Unix\nper-send", latencies)


def _benchmark_roundtrip(send_transport, recv_transport, label):
    """Benchmark a transport pair using the TcpTransport API."""
    send_transport.initialize(None)
    recv_transport.initialize(None)

    latencies = {s: [] for s in TENSOR_SIZES}
    received = [None]
    ready = threading.Event()

    def receiver():
        ready.set()
        for _ in range(ITERATIONS * len(TENSOR_SIZES) + 20):
            received[0] = recv_transport.recv(torch.zeros(1), src_stage=0)

    recv_thread = threading.Thread(target=receiver, daemon=True)
    recv_thread.start()
    ready.wait()
    time.sleep(0.1)

    # Warmup
    for _ in range(10):
        send_transport.send(torch.randn(1024), dest_stage=0)
        _ = received[0]

    for size in TENSOR_SIZES:
        tensor = torch.randn(size)
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            send_transport.send(tensor, dest_stage=0)
            _ = received[0]
            latencies[size].append((time.perf_counter() - start) * 1e6)

    send_transport.shutdown()
    recv_transport.shutdown()

    return _stats(label, latencies)


def _stats(label, latencies):
    return {
        'label': label,
        'results': {s: {
            'mean': statistics.mean(lats),
            'p50': statistics.median(lats),
        } for s, lats in latencies.items()},
    }


def main():
    print("=" * 72)
    print("Transport Benchmark: TCP vs Persistent TCP vs Unix Sockets")
    print(f"  Iterations: {ITERATIONS}")
    print(f"  Tensor sizes: {', '.join(str(s) for s in TENSOR_SIZES)}")
    print("=" * 72)

    all_results = [
        benchmark_tcp_connect(),
        benchmark_tcp_persistent(),
        benchmark_unix_connect(),
    ]

    # Header
    print()
    header = f"{'Size':>8} {'Bytes':>8}"
    for r in all_results:
        header += f" {r['label']:>18}"
    header += f" {'Persist vs Connect':>20} {'Unix vs Connect':>17}"
    print(header)
    print("-" * 72)

    tcp_connect = all_results[0]['results']
    tcp_persist = all_results[1]['results']
    unix_data = all_results[2]['results']

    for size in TENSOR_SIZES:
        bytes_sz = size * 4
        c_us = tcp_connect[size]['mean']
        p_us = tcp_persist[size]['mean']
        u_us = unix_data[size]['mean']

        persist_improvement = (1 - p_us / c_us) * 100 if c_us > 0 else 0
        unix_improvement = (1 - u_us / c_us) * 100 if c_us > 0 else 0

        row = f"{size:>8} {bytes_sz:>8}"
        row += f" {c_us:>17.1f}μs"
        row += f" {p_us:>17.1f}μs"
        row += f" {u_us:>17.1f}μs"
        row += f" {persist_improvement:>19.0f}%"
        row += f" {unix_improvement:>16.0f}%"
        print(row)

    print("-" * 72)

    # Summary
    total_connect = sum(tcp_connect[s]['mean'] for s in TENSOR_SIZES)
    total_persist = sum(tcp_persist[s]['mean'] for s in TENSOR_SIZES)
    total_unix = sum(unix_data[s]['mean'] for s in TENSOR_SIZES)

    print(f"\nTotal latency (sum of means across sizes):")
    print(f"  Connect-per-send TCP:  {total_connect:>8.0f} μs")
    print(f"  Persistent TCP:        {total_persist:>8.0f} μs")
    print(f"  Unix per-send:         {total_unix:>8.0f} μs")
    print(f"\n  Persistent vs Connect:  {total_connect/total_persist:.1f}x faster "
          f"({(1-total_persist/total_connect)*100:.0f}% reduction)")
    print(f"  Unix vs Connect:        {total_connect/total_unix:.1f}x faster "
          f"({(1-total_unix/total_connect)*100:.0f}% reduction)")
    print(f"  Persistent vs Unix:     {total_unix/total_persist:.1f}x faster "
          f"({(1-total_persist/total_unix)*100:.0f}% reduction)")

    # Detailed stats for 16KB
    size_16k = 4096
    print(f"\nDetailed latency for {size_16k} elements ({size_16k*4} bytes):")
    for r in all_results:
        d = r['results'][size_16k]
        print(f"  {r['label']:>20}: mean={d['mean']:6.1f} μs  p50={d['p50']:6.1f} μs")


if __name__ == "__main__":
    main()
