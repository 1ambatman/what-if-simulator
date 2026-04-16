"""Launch the app and open the system browser."""

from __future__ import annotations

import threading
import time
import webbrowser

import uvicorn

from what_if_app.config import settings


def _open_browser() -> None:
    time.sleep(1.0)
    webbrowser.open(f"http://{settings.app_host}:{settings.app_port}/")


def main() -> None:
    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(
        "what_if_app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
