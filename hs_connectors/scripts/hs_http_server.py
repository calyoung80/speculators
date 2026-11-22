"""Lightweight HTTP key-value server for hidden-states transfer.

Run on the node reachable from both the vLLM producers and the trainer::

    python3 scripts/hs_http_server.py --port 50088

The vLLM-side ``HttpHiddenStatesConnector`` PUTs packed tensor bytes to
``/hs/{handle}``; the trainer-side ``HttpTransfer`` GETs them back and
DELETEs them when done.  Data lives in memory only — nothing is written
to disk.

Endpoints:
    PUT    /hs/{handle}   -> store bytes (body = payload)
    GET    /hs/{handle}   -> fetch bytes (200) or 404 if absent  [read-once:
                             the entry is removed after a successful GET]
    DELETE /hs/{handle}   -> drop the entry

Memory is bounded by ``--max-buffer-gib``: when the total payload
bytes exceed this budget the oldest entries are evicted (FIFO).

Chunked-transfer protocol (used by HttpStore for large payloads):
    PUT  /hs/{handle}/{i}   -> store chunk i
    PUT  /hs/{handle}/nchunks -> completion marker (body = chunk count)
    GET  /hs/{handle}/nchunks -> poll for completeness
    GET  /hs/{handle}/{i}   -> fetch chunk i
"""

from __future__ import annotations

import argparse
import http.server
import threading
from collections import OrderedDict


class _Store:
    """Thread-safe in-memory key-value store with byte-budget eviction.

    GET uses pop() semantics (read-once): the entry is removed after a
    successful read, mirroring the training pipeline's generate→fetch→
    delete lifecycle and preventing unbounded accumulation.
    """

    def __init__(self, max_bytes: int = 64 * 1024**3):
        self._data: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()
        self._max_bytes = max_bytes
        self._cur_bytes = 0

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            # Evict oldest entries until we fit the byte budget.
            while self._cur_bytes + len(value) > self._max_bytes and self._data:
                _, old = self._data.popitem(last=False)
                self._cur_bytes -= len(old)
            old = self._data.pop(key, None)
            if old is not None:
                self._cur_bytes -= len(old)
            self._data[key] = value
            self._cur_bytes += len(value)
            self._data.move_to_end(key)

    def get(self, key: str) -> bytes | None:
        with self._lock:
            return self._data.get(key)

    def pop(self, key: str) -> bytes | None:
        """Read-once: fetch and remove in one atomic step."""
        with self._lock:
            value = self._data.pop(key, None)
            if value is not None:
                self._cur_bytes -= len(value)
            return value

    def delete(self, key: str) -> None:
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self._cur_bytes -= len(old)

    def stats(self) -> tuple[int, int]:
        """(num_entries, total_bytes) snapshot."""
        with self._lock:
            return len(self._data), self._cur_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def make_handler(store: _Store) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive

        def _key(self) -> str:
            return self.path.lstrip("/").removeprefix("hs/")

        def _reply(self, status: int, body: bytes = b"", content_type: str = "text/plain") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_PUT(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            store.put(self._key(), data)
            self._reply(200, b"ok")

        def do_GET(self) -> None:  # noqa: N802
            # Read-once semantics: pop removes the entry after reading.
            data = store.pop(self._key())
            if data is None:
                self._reply(404)
                return
            self._reply(200, data, content_type="application/octet-stream")

        def do_DELETE(self) -> None:  # noqa: N802
            store.delete(self._key())
            self._reply(200)

        def log_message(self, fmt: str, *args) -> None:  # quiet
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9020)
    parser.add_argument(
        "--max-buffer-gib",
        type=float,
        default=64.0,
        help="Max total payload bytes in GiB (default 64); oldest evicted beyond this.",
    )
    args = parser.parse_args()

    store = _Store(max_bytes=int(args.max_buffer_gib * 1024**3))
    server = http.server.ThreadingHTTPServer(
        (args.host, args.port), make_handler(store)
    )
    print(
        f"[hs-http-server] listening on {args.host}:{args.port} "
        f"(max_buffer={args.max_buffer_gib:.0f} GiB, read-once GET)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
