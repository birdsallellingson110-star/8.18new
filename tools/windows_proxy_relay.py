#!/usr/bin/env python3
"""Tiny HTTP CONNECT relay: listen 127.0.0.1:19090 -> forward to 127.0.0.1:7890.

Why: SSH RemoteForward traffic is from ssh.exe; some Clash Fake-IP / process
rules answer curl.exe but stall ssh.exe. Relaying via python.exe is more stable.

Run on Windows:
  C:\\Python314\\python.exe windows_proxy_relay.py

Then SSH config:
  RemoteForward 7890 127.0.0.1:19090
"""

from __future__ import annotations

import select
import socket
import socketserver
import threading

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 19090
UPSTREAM = ("127.0.0.1", 7890)
BUF = 65536


def pipe(a: socket.socket, b: socket.socket) -> None:
    try:
        while True:
            r, _, _ = select.select([a, b], [], [], 300)
            if not r:
                break
            for src in r:
                dst = b if src is a else a
                data = src.recv(BUF)
                if not data:
                    return
                dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        upstream = socket.create_connection(UPSTREAM, timeout=10)
        t = threading.Thread(target=pipe, args=(self.request, upstream), daemon=True)
        t.start()
        t.join()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    print(f"relay http://{LISTEN_HOST}:{LISTEN_PORT} -> http://{UPSTREAM[0]}:{UPSTREAM[1]}")
    print("Keep this window open. Test: curl.exe -x http://127.0.0.1:19090 -I https://www.google.com")
    with Server((LISTEN_HOST, LISTEN_PORT), Handler) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    main()
