"""Serving the UI and opening the native shell.

Porthole always serves the UI over **HTTP**, even for the native window, and the
native window is just a browser pointed at that URL. Two consequences, both
deliberate:

1. The page origin is `http://127.0.0.1:<port>`, never `file://`. Opaque
   `file://` origins are the documented cause of localhost WebSocket failures
   inside embedded webviews, and serving over HTTP avoids the class entirely.
2. "Open it in Chrome on another machine" costs nothing, because it is the same
   URL the native window uses. That is what makes the browser a real target
   rather than an accident.

pywebview is an optional dependency. Everything except `open_window` works
without it.
"""

from __future__ import annotations

import logging
import threading
import urllib.parse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger("porthole.window")


class _SPAHandler(SimpleHTTPRequestHandler):
    """Static file handler with SPA fallback and extra mounted directories.

    Asset mounts exist because some payloads do not belong on the bus. A
    robot's visual meshes are megabytes that never change; the browser already
    has a caching, streaming, off-thread loader for them, and the bus does not.
    See `docs/scene-protocol.md` §7.
    """

    def __init__(self, *args, assets: dict[str, Path] | None = None, **kwargs):
        self._assets = assets or {}
        super().__init__(*args, **kwargs)

    def send_response(self, code, message=None):  # noqa: D102
        super().send_response(code, message)
        # A dev shell must never serve a stale bundle.
        self.send_header("Cache-Control", "no-store")

    def _asset_root(self, url_path: str) -> tuple[str, Path] | None:
        clean = urllib.parse.urlsplit(url_path).path
        for prefix, root in self._assets.items():
            if clean == prefix or clean.startswith(prefix + "/"):
                return clean[len(prefix):].lstrip("/"), root
        return None

    def translate_path(self, path):  # noqa: D102
        mount = self._asset_root(path)
        if mount is None:
            return super().translate_path(path)

        relative, root = mount
        target = (root / relative).resolve()
        # Containment check, not string prefixing: `..` segments and symlinks
        # both resolve away here, and a mount is a directory a user chose to
        # expose, not permission to walk out of it.
        if target != root and root not in target.parents:
            log.warning("blocked asset path escape: %s", path)
            return str(root / "__blocked__")
        return str(target)

    def do_GET(self):  # noqa: N802, D102
        resolved = self.translate_path(self.path)
        if not Path(resolved).exists():
            if self._asset_root(self.path) is not None:
                # Never answer a missing mesh with index.html: a 3D loader
                # handed HTML reports a parse error, which reads like a
                # corrupt asset rather than a wrong path.
                self.send_error(404, "asset not found")
                return
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, fmt, *args):  # noqa: D102
        log.debug(fmt, *args)


def serve_ui(
    dist: str | Path,
    host: str = "127.0.0.1",
    port: int = 8766,
    assets: dict[str, str | Path] | None = None,
) -> str:
    """Serve a built frontend directory on a background thread. Returns the URL.

    ``assets`` mounts extra directories by URL prefix::

        serve_ui(dist, assets={"/meshes/ur5e": "~/.cache/.../visual"})

    which makes `/meshes/ur5e/wrist3.dae` fetchable by the UI. Use it for large
    static geometry rather than sending it over the bus.
    """
    dist = Path(dist).resolve()
    index = dist / "index.html"
    if not index.exists():
        raise FileNotFoundError(
            f"no index.html in {dist} — build the frontend first (npm run build)"
        )

    mounts: dict[str, Path] = {}
    for prefix, directory in (assets or {}).items():
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"asset mount {prefix!r} -> {root} is not a directory")
        mounts["/" + prefix.strip("/")] = root

    handler = partial(_SPAHandler, directory=str(dist), assets=mounts)
    server = ThreadingHTTPServer((host, port), handler)
    threading.Thread(
        target=server.serve_forever, name="porthole-http", daemon=True
    ).start()

    url = f"http://{host}:{port}"
    log.info("serving %s at %s", dist, url)
    for prefix, root in mounts.items():
        log.info("  mount %s -> %s", prefix, root)
    return url


def open_window(
    url: str,
    title: str = "Porthole",
    width: int = 1600,
    height: int = 1000,
    debug: bool = False,
    gui: str | None = None,
) -> None:
    """Open the native shell at ``url`` and block until it closes.

    Requires the `native` extra: ``pip install 'porthole[native]'``.

    ``gui`` picks the pywebview backend (``"qt"``, ``"gtk"``). Leave it None to
    let pywebview choose. On a conda Python the GTK backend is usually
    unavailable — system PyGObject is built for the distro's interpreter and
    will not import under a different minor version — so Qt is the portable
    choice there.
    """
    try:
        import webview
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "the native window needs pywebview: pip install 'porthole[native]'\n"
            f"Meanwhile the UI is already usable in any browser at {url}"
        ) from exc

    webview.create_window(title, url, width=width, height=height)
    webview.start(debug=debug, **({"gui": gui} if gui else {}))


def run_native_app(
    dist: str | Path,
    title: str = "Porthole",
    host: str = "127.0.0.1",
    port: int = 8766,
    width: int = 1600,
    height: int = 1000,
    debug: bool = False,
    gui: str | None = None,
    assets: dict[str, str | Path] | None = None,
) -> None:
    """Serve a built frontend and open it in the native shell. Blocks.

    Start your bus before calling this; the window connects to it over the
    socket like any other client.
    """
    url = serve_ui(dist, host=host, port=port, assets=assets)
    log.info("opening native window at %s", url)
    open_window(url, title=title, width=width, height=height, debug=debug, gui=gui)
