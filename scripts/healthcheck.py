#!/usr/bin/env python3
"""容器健康检查：GET /health，成功（HTTP 200）退出码 0，否则 1。"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    port = os.environ.get("PORT", "8000")
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status != 200:
                return 1
            return 0
    except (urllib.error.URLError, OSError):
        return 1


if __name__ == "__main__":
    sys.exit(main())
