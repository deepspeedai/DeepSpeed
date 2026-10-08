# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tests for the persistent TCP socket pool.

Covers PooledConnection, SocketPool, SocketPoolManager, _is_healthy
edge cases, and performance benchmarks under load.
"""

import socket
import statistics
import threading
import time

import pytest


class TestPooledConnection:
    """Unit tests for PooledConnection dataclass."""

    def test_create_tracks_metadata(self):
        from deepspeed.runtime.pipe.socket_pool import PooledConnection, ConnectionHealth
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        t0 = time.monotonic()
        conn = PooledConnection(sock=sock, stage_pair="0->1")
        t1 = time.monotonic()
        assert conn.created_at >= t0
        assert conn.created_at <= t1
        assert conn.last_used_at >= t0
        assert conn.health == ConnectionHealth.HEALTHY
        sock.close()

    def test_health_starts_healthy(self):
        from deepspeed.runtime.pipe.socket_pool import PooledConnection, ConnectionHealth
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn = PooledConnection(sock=sock, stage_pair="2->3")
        assert conn.health == ConnectionHealth.HEALTHY
        sock.close()

    def test_health_transitions(self):
        from deepspeed.runtime.pipe.socket_pool import PooledConnection, ConnectionHealth
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn = PooledConnection(sock=sock, stage_pair="0->1")
        conn.health = ConnectionHealth.STALE
        assert conn.health == ConnectionHealth.STALE
        conn.health = ConnectionHealth.DEAD
        assert conn.health == ConnectionHealth.DEAD
        sock.close()


class TestSocketPool:
    """Unit tests for SocketPool lifecycle."""

    def _create_pool(self, max_size=4, idle_timeout=60.0):
        from deepspeed.runtime.pipe.socket_pool import SocketPool
        return SocketPool(stage_pair="0->1", host="127.0.0.1", port=0, max_size=max_size, idle_timeout=idle_timeout)

    def test_pool_created_empty(self):
        pool = self._create_pool()
        assert pool.total_connections() == 0

    def test_acquire_creates_new_connection(self):
        pool = self._create_pool(max_size=2)
        conn = pool.acquire()
        assert conn is not None
        assert conn.stage_pair == "0->1"
        assert pool.total_connections() == 1

    def test_release_returns_to_pool(self):
        pool = self._create_pool()
        conn1 = pool.acquire()
        pool.release(conn1)
        conn2 = pool.acquire()
        assert conn1 is conn2

    def test_max_size_limit(self):
        pool = self._create_pool(max_size=2)
        c1 = pool.acquire()
        c2 = pool.acquire()
        assert pool.total_connections() == 2
        pool.release(c1)
        c3 = pool.acquire()
        assert c3 is c1
        assert pool.total_connections() == 2

    def test_dead_connection_not_reused(self):
        from deepspeed.runtime.pipe.socket_pool import ConnectionHealth
        pool = self._create_pool()
        conn = pool.acquire()
        conn.health = ConnectionHealth.DEAD
        pool.release(conn)
        assert pool.total_connections() == 0

    def test_drain_closes_all(self):
        pool = self._create_pool(max_size=3)
        c1 = pool.acquire()
        c2 = pool.acquire()
        pool.release(c1)
        pool.release(c2)
        assert pool.total_connections() == 2
        pool.drain(timeout=1)
        assert pool.total_connections() == 0


class TestSocketPoolIdleEviction:
    """Tests for idle connection eviction."""

    def _create_pool(self, max_size=4, idle_timeout=1.0):
        from deepspeed.runtime.pipe.socket_pool import SocketPool
        return SocketPool(stage_pair="test", host="127.0.0.1", port=0, max_size=max_size, idle_timeout=idle_timeout)

    def _fill(self, pool, n):
        """Add n healthy socketpair connections to pool."""
        import socket
        from deepspeed.runtime.pipe.socket_pool import PooledConnection
        pairs = [socket.socketpair() for _ in range(n)]
        for a, b in pairs:
            pool._connections.append(PooledConnection(sock=a, stage_pair="test"))
        return pairs

    def test_evicts_connections_beyond_timeout(self):
        """Connections idle longer than timeout are evicted."""
        pool = self._create_pool(idle_timeout=60.0)
        pairs = self._fill(pool, 3)

        # Simulate: last used 120 seconds ago — past 60s timeout
        now = time.monotonic()
        for conn in pool._connections:
            conn.last_used_at = now - 120.0

        assert pool.total_connections() == 3
        pool.evict_idle(now)
        assert pool.total_connections() == 0

        for a, b in pairs:
            a.close()
            b.close()

    def test_keeps_recently_used_connections(self):
        """Connections used within timeout are kept."""
        pool = self._create_pool(idle_timeout=60.0)
        pairs = self._fill(pool, 3)

        now = time.monotonic()
        # Two idle 120s ago, one idle 30s ago
        pool._connections[0].last_used_at = now - 120.0
        pool._connections[1].last_used_at = now - 120.0
        pool._connections[2].last_used_at = now - 30.0

        assert pool.total_connections() == 3
        pool.evict_idle(now)
        assert pool.total_connections() == 1  # only recent kept

        for a, b in pairs:
            a.close()
            b.close()

    def test_boundary_at_timeout_kept(self):
        """Connection idle exactly at timeout boundary is kept."""
        pool = self._create_pool(idle_timeout=60.0)
        pairs = self._fill(pool, 1)

        now = time.monotonic()
        pool._connections[0].last_used_at = now - 60.0  # exactly at boundary

        pool.evict_idle(now)
        assert pool.total_connections() == 1  # kept — boundary is not past

        for a, b in pairs:
            a.close()
            b.close()

    def test_boundary_just_past_timeout_evicted(self):
        """Connection idle fractionally past timeout is evicted."""
        pool = self._create_pool(idle_timeout=60.0)
        pairs = self._fill(pool, 1)

        now = time.monotonic()
        pool._connections[0].last_used_at = now - 60.001  # just past

        pool.evict_idle(now)
        assert pool.total_connections() == 0

        for a, b in pairs:
            a.close()
            b.close()

    def test_eviction_on_empty_pool_noop(self):
        """evict_idle on empty pool is a no-op."""
        pool = self._create_pool()
        assert pool.total_connections() == 0
        pool.evict_idle(time.monotonic())  # should not raise
        assert pool.total_connections() == 0

    def test_eviction_with_now_none(self):
        """evict_idle with now=None uses current time."""
        pool = self._create_pool(idle_timeout=0.001)  # 1ms timeout
        pairs = self._fill(pool, 1)

        time.sleep(0.01)  # wait past 1ms timeout
        pool.evict_idle(now=None)
        assert pool.total_connections() == 0

        for a, b in pairs:
            a.close()
            b.close()

    def test_acquire_refreshes_last_used(self):
        """acquire() updates last_used_at, preventing eviction."""
        pool = self._create_pool(idle_timeout=60.0)
        pairs = self._fill(pool, 1)

        now = time.monotonic()
        pool._connections[0].last_used_at = now - 120.0  # past timeout

        # acquire refreshes last_used_at to now
        conn = pool.acquire()
        assert conn.last_used_at >= now

        # Now it shouldn't be evicted
        pool.evict_idle(time.monotonic())
        assert pool.total_connections() == 1  # recent, kept

        pool.release(conn)
        for a, b in pairs:
            a.close()
            b.close()
        pool.drain()


class TestSocketPoolManager:
    """Unit tests for SocketPoolManager multi-pool orchestration."""

    def _create_manager(self, max_per_pair=2, idle_timeout=60.0):
        from deepspeed.runtime.pipe.socket_pool import SocketPoolManager
        return SocketPoolManager(host="127.0.0.1",
                                 recv_port=0,
                                 max_connections_per_pair=max_per_pair,
                                 idle_timeout=idle_timeout)

    def test_manager_creates_pools_on_demand(self):
        mgr = self._create_manager()
        assert len(mgr._pools) == 0
        conn = mgr.get_connection("stage_1")
        assert conn is not None
        assert "stage_1" in mgr._pools

    def test_separate_pools_per_dest(self):
        mgr = self._create_manager()
        c1 = mgr.get_connection("stage_1")
        c2 = mgr.get_connection("stage_2")
        assert "stage_1" in mgr._pools
        assert "stage_2" in mgr._pools
        assert mgr._pools["stage_1"] is not mgr._pools["stage_2"]

    def test_return_connection(self):
        mgr = self._create_manager()
        conn = mgr.get_connection("stage_1")
        pool = mgr._pools["stage_1"]
        assert pool.total_connections() == 1
        mgr.return_connection("stage_1", conn)
        assert pool.total_connections() == 1

    def test_return_to_unknown_pool_no_error(self):
        mgr = self._create_manager()
        conn = mgr.get_connection("stage_1")
        mgr.return_connection("nonexistent", conn)

    def test_drain_removes_all_pools(self):
        mgr = self._create_manager()
        mgr.get_connection("stage_1")
        mgr.get_connection("stage_2")
        assert len(mgr._pools) == 2
        mgr.drain(timeout=1)
        assert len(mgr._pools) == 0


class TestSocketPoolPerformance:
    """Benchmark tests for SocketPool under load.

    Each test method gets a fresh echo server via the _start_server fixture.
    Cleanup is handled by request.addfinalizer.
    """

    @pytest.fixture(autouse=True)
    def _start_server(self, request):
        """Start an echo server on a random port."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(('127.0.0.1', 0))
        self._server_sock.listen(8)
        self._server_port = self._server_sock.getsockname()[1]
        self._server_running = True

        def serve():
            while self._server_running:
                try:
                    conn, _ = self._server_sock.accept()
                    t = threading.Thread(target=self._echo, args=(conn, ), daemon=True)
                    t.start()
                except OSError:
                    break

        self._server_thread = threading.Thread(target=serve, daemon=True)
        self._server_thread.start()

        def cleanup():
            self._server_running = False
            try:
                self._server_sock.close()
            except Exception:
                pass

        request.addfinalizer(cleanup)

    def _echo(self, conn):
        """Read 4-byte ping and echo back."""
        try:
            data = conn.recv(4)
            if data:
                conn.sendall(data)
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def test_acquire_release_throughput(self):
        """Measure acquire/release cycles per second."""
        from deepspeed.runtime.pipe.socket_pool import SocketPool

        pool = SocketPool(stage_pair="bench", host="127.0.0.1", port=self._server_port, max_size=4)
        N = 1000

        for _ in range(20):
            c = pool.acquire()
            pool.release(c)

        start = time.perf_counter()
        for _ in range(N):
            c = pool.acquire()
            pool.release(c)
        elapsed = time.perf_counter() - start

        rate = N / elapsed
        assert rate > 0
        pool.drain()

    def test_pool_vs_raw_socket_latency(self):
        """Compare pool acquire latency vs raw socket creation."""
        from deepspeed.runtime.pipe.socket_pool import SocketPool

        N = 200
        pool = SocketPool(stage_pair="bench", host="127.0.0.1", port=self._server_port, max_size=4)

        conns = [pool.acquire() for _ in range(4)]
        for c in conns:
            pool.release(c)

        pool_lats = []
        for _ in range(N):
            start = time.perf_counter()
            c = pool.acquire()
            pool_lats.append((time.perf_counter() - start) * 1e6)
            pool.release(c)

        pool_mean = statistics.mean(pool_lats)

        raw_lats = []
        for _ in range(N):
            start = time.perf_counter()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect(('127.0.0.1', self._server_port))
            raw_lats.append((time.perf_counter() - start) * 1e6)
            s.close()

        raw_mean = statistics.mean(raw_lats)
        assert pool_mean < raw_mean
        pool.drain()

    def test_health_check_overhead(self):
        """Measure _is_healthy overhead on a pooled connection."""
        from deepspeed.runtime.pipe.socket_pool import SocketPool

        pool = SocketPool(stage_pair="bench", host="127.0.0.1", port=self._server_port, max_size=1)
        conn = pool.acquire()

        N = 1000
        lats = []
        for _ in range(N):
            start = time.perf_counter()
            pool._is_healthy(conn)
            lats.append((time.perf_counter() - start) * 1e6)

        pool.release(conn)
        pool.drain()

        mean = statistics.mean(lats)
        assert mean < 100

    def test_concurrent_acquire_no_deadlock(self):
        """Multiple threads should not deadlock on acquire/release."""
        from deepspeed.runtime.pipe.socket_pool import SocketPool

        pool = SocketPool(stage_pair="bench", host="127.0.0.1", port=self._server_port, max_size=4)
        errors = []
        results = []

        def worker(worker_id):
            try:
                for _ in range(50):
                    conn = pool.acquire()
                    time.sleep(0.001)
                    pool.release(conn)
                results.append(worker_id)
            except Exception as e:
                errors.append((worker_id, e))

        threads = [threading.Thread(target=worker, args=(i, )) for i in range(4)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        elapsed = time.perf_counter() - start

        pool.drain()
        assert len(errors) == 0
        assert len(results) == 4
        assert elapsed < 10

    def test_pool_reuse_ratio(self):
        """Verify pool reuses connections after release."""
        from deepspeed.runtime.pipe.socket_pool import SocketPool

        pool = SocketPool(stage_pair="bench", host="127.0.0.1", port=self._server_port, max_size=4)

        conns = [pool.acquire() for _ in range(4)]
        assert pool.total_connections() == 4
        for c in conns:
            pool.release(c)

        reused = pool.acquire()
        assert pool.total_connections() == 4
        pool.release(reused)
        pool.drain()


class TestSocketPoolIsHealthy:
    """Edge case tests for SocketPool._is_healthy()."""

    def _create_pool(self):
        from deepspeed.runtime.pipe.socket_pool import SocketPool
        return SocketPool(stage_pair="test", host="127.0.0.1", port=0, max_size=1)

    def test_healthy_socket_no_data(self):
        from deepspeed.runtime.pipe.socket_pool import PooledConnection

        pool = self._create_pool()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(True)
        conn = PooledConnection(sock=sock, stage_pair="test")
        result = pool._is_healthy(conn)
        assert result is True
        sock.close()

    def test_closed_socket_eof(self):
        from deepspeed.runtime.pipe.socket_pool import PooledConnection

        pool = self._create_pool()
        a, b = socket.socketpair()
        b.close()
        conn = PooledConnection(sock=a, stage_pair="test")
        result = pool._is_healthy(conn)
        assert result is False
        a.close()


class TestSocketPoolRecovery:
    """Tests for health check failure recovery paths.

    Covers acquire skipping dead connections, pool creating new
    connections when all are dead, release not returning dead
    connections, sequential failure recovery, and mixed live/dead batches.
    """

    def _create_pool(self, max_size=4):
        from deepspeed.runtime.pipe.socket_pool import SocketPool
        return SocketPool(stage_pair="test", host="127.0.0.1", port=0, max_size=max_size)

    def _dead_conn(self):
        from deepspeed.runtime.pipe.socket_pool import PooledConnection
        a, b = socket.socketpair()
        b.close()
        return PooledConnection(sock=a, stage_pair="test")

    def _live_conn(self):
        from deepspeed.runtime.pipe.socket_pool import PooledConnection
        a, b = socket.socketpair()
        return PooledConnection(sock=a, stage_pair="test"), b

    def test_acquire_skips_dead_returns_healthy(self):
        pool = self._create_pool()
        dead = self._dead_conn()
        live, peer = self._live_conn()
        pool._connections.append(dead)
        pool._connections.append(live)
        conn = pool.acquire()
        assert conn is live
        assert pool.total_connections() == 1
        peer.close()
        pool.drain()

    def test_acquire_all_dead_creates_new(self):
        pool = self._create_pool(max_size=2)
        pool._connections.append(self._dead_conn())
        pool._connections.append(self._dead_conn())
        conn = pool.acquire()
        assert conn is not None
        assert pool.total_connections() == 3
        pool.drain()

    def test_release_dead_not_returned(self):
        from deepspeed.runtime.pipe.socket_pool import ConnectionHealth
        pool = self._create_pool()
        conn, peer = self._live_conn()
        pool._connections.append(conn)
        c = pool.acquire()
        c.health = ConnectionHealth.DEAD
        pool.release(c)
        assert pool.total_connections() == 0
        peer.close()

    def test_sequential_failure_recovery(self):
        from deepspeed.runtime.pipe.socket_pool import ConnectionHealth
        pool = self._create_pool(max_size=3)
        for _ in range(3):
            c, p = self._live_conn()
            pool._connections.append(c)
        for i in range(3):
            c = pool.acquire()
            c.health = ConnectionHealth.DEAD
            pool.release(c)
            assert pool.total_connections() == 2 - i
        c = pool.acquire()
        assert c is not None
        pool.drain()

    def test_health_after_recovery(self):
        from deepspeed.runtime.pipe.socket_pool import ConnectionHealth
        pool = self._create_pool()
        pool._connections.append(self._dead_conn())
        recovered = pool.acquire()
        recovered.health = ConnectionHealth.HEALTHY
        pool.release(recovered)
        c = pool.acquire()
        assert pool._is_healthy(c) is True
        pool.release(c)
        pool.drain()

    def test_mixed_dead_and_live_batch(self):
        pool = self._create_pool(max_size=4)
        d1, d2 = self._dead_conn(), self._dead_conn()
        l1, p1 = self._live_conn()
        l2, p2 = self._live_conn()
        pool._connections.append(d2)
        pool._connections.append(l2)
        pool._connections.append(d1)
        pool._connections.append(l1)
        r1 = pool.acquire()
        r2 = pool.acquire()
        assert {r1, r2} == {l1, l2}
        assert pool.total_connections() == 2
        p1.close()
        p2.close()
        pool.drain()

    def test_connection_reset(self):
        from deepspeed.runtime.pipe.socket_pool import PooledConnection

        pool = self._create_pool()
        a, b = socket.socketpair()
        b.shutdown(socket.SHUT_RDWR)
        b.close()
        conn = PooledConnection(sock=a, stage_pair="test")
        result = pool._is_healthy(conn)
        assert result is False
        a.close()


class TestSocketPoolAcquireTimeout:
    """Tests for blocking acquire with timeout when pool is exhausted."""

    def _create_pool(self, max_size=2):
        from deepspeed.runtime.pipe.socket_pool import SocketPool
        import socket
        # Create listener so _create() works
        self._ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._ls.bind(('127.0.0.1', 0))
        self._ls.listen(4)
        self._port = self._ls.getsockname()[1]
        self._running = True

        def srv():
            while self._running:
                try:
                    c, _ = self._ls.accept()
                    c.close()
                except OSError:
                    break

        threading.Thread(target=srv, daemon=True).start()
        pool = SocketPool(stage_pair="t", host="127.0.0.1", port=self._port, max_size=max_size)
        # Override drain to close listener
        pool._cleanup_listener = lambda: [setattr(self, '_running', False) or self._ls.close()]
        return pool

    def _live(self):
        from deepspeed.runtime.pipe.socket_pool import PooledConnection
        a, b = socket.socketpair()
        return PooledConnection(sock=a, stage_pair="t"), b

    def test_timeout_zero_raises_when_exhausted(self):
        """timeout=0 raises PoolExhaustedError instead of returning None."""
        from deepspeed.runtime.pipe.socket_pool import PoolExhaustedError
        pool = self._create_pool(max_size=1)
        l, peer = self._live()
        pool._connections.append(l)
        c1 = pool.acquire()
        assert c1 is not None
        with pytest.raises(PoolExhaustedError, match="exhausted"):
            pool.acquire(timeout=0)
        pool.release(c1)
        peer.close()
        pool.drain()

    def test_blocks_until_release(self):
        pool = self._create_pool(max_size=1)
        l, peer = self._live()
        pool._connections.append(l)
        c1 = pool.acquire()
        errors = []

        def delayed_release():
            time.sleep(0.05)
            try:
                pool.release(c1)
            except Exception as e:
                errors.append(e)

        threading.Thread(target=delayed_release, daemon=True).start()
        start = time.perf_counter()
        c2 = pool.acquire(timeout=5)
        elapsed = time.perf_counter() - start
        assert c2 is not None
        assert elapsed > 0.03
        assert not errors
        pool.release(c2)
        peer.close()
        pool.drain()

    def test_timeout_expires_returns_none(self):
        pool = self._create_pool(max_size=1)
        l, peer = self._live()
        pool._connections.append(l)
        pool.acquire()
        start = time.perf_counter()
        result = pool.acquire(timeout=0.1)
        elapsed = time.perf_counter() - start
        assert result is None
        assert 0.08 < elapsed < 0.3
        peer.close()
        pool.drain()

    def test_condition_notify_wakes_blocked_acquirer(self):
        pool = self._create_pool(max_size=1)
        l, peer = self._live()
        pool._connections.append(l)
        c = pool.acquire()
        result_holder = [None]

        def blocked():
            result_holder[0] = pool.acquire(timeout=5)

        t = threading.Thread(target=blocked, daemon=True)
        t.start()
        time.sleep(0.05)
        pool.release(c)
        t.join(timeout=3)
        assert result_holder[0] is not None
        assert not t.is_alive()
        peer.close()
        pool.drain()

    def test_default_no_timeout_creates_below_max(self):
        """acquire() without timeout creates new connections below max_size."""
        pool = self._create_pool(max_size=3)
        peers = []

        # Create 3 connections — all should succeed without blocking
        for _ in range(3):
            l, peer = self._live()
            pool._connections.append(l)
            peers.append(peer)

        # All 3 acquired without timeout
        conns = [pool.acquire() for _ in range(3)]
        assert len(conns) == 3
        assert pool.total_in_use() == 3

        for c in conns:
            pool.release(c)
        for p in peers:
            p.close()
        pool.drain()

    def test_default_no_timeout_blocks_and_recovers(self):
        """acquire() without timeout blocks on exhausted pool, recovers on release."""
        pool = self._create_pool(max_size=1)
        l, peer = self._live()
        pool._connections.append(l)
        c = pool.acquire()
        assert c is not None
        assert pool.total_in_use() == 1

        result = [None]

        def delayed():
            time.sleep(0.05)
            pool.release(c)

        threading.Thread(target=delayed, daemon=True).start()

        start = time.perf_counter()
        result[0] = pool.acquire()  # blocks indefinitely, wakes on notify
        elapsed = time.perf_counter() - start

        assert result[0] is not None, "Default acquire should get connection after release"
        assert 0.03 < elapsed < 1.0, f"Should block briefly, got {elapsed*1000:.0f}ms"

        pool.release(result[0])
        peer.close()
        pool.drain()

    def test_timeout_expires_returns_none_on_exhausted(self):
        """acquire(timeout=N) returns None when timeout expires."""
        pool = self._create_pool(max_size=1)
        l, peer = self._live()
        pool._connections.append(l)
        pool.acquire()  # exhaust

        start = time.perf_counter()
        result = pool.acquire(timeout=0.1)
        elapsed = time.perf_counter() - start

        assert result is None, "Timed-out acquire should return None"
        assert 0.08 < elapsed < 0.3, f"Expected ~100ms timeout, got {elapsed*1000:.0f}ms"
        peer.close()
        pool.drain()
