"""HTTP 服务：POST /api/current-traces/align。

仅依赖 Python 标准库，便于离线容器构建。

* ``GET  /health``                 健康检查；
* ``POST /api/current-traces/align`` 联合对齐；
* ``GET  /``                       服务信息。
"""

from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Tuple

from .alignment import AlignmentError, solve_alignment

LOGGER = logging.getLogger("nanopore_align")

_ALLOWED_FIELDS = {
    "reference_levels",
    "observations",
    "drift_min",
    "drift_max",
    "residual_limit",
    "dwell_min",
    "dwell_max",
    "max_skips",
}

_REQUIRED_FIELDS = (
    "reference_levels",
    "observations",
    "drift_min",
    "drift_max",
    "residual_limit",
)


def align_from_payload(payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """校验请求体并求解，返回 (HTTP 状态码, 响应字典)。"""
    if not isinstance(payload, dict):
        return _bad_request("invalid_body", "请求体必须为 JSON 对象")

    unknown = set(payload) - _ALLOWED_FIELDS
    if unknown:
        return _bad_request(
            "unknown_fields", f"存在未知字段: {sorted(unknown)}"
        )

    missing = [f for f in _REQUIRED_FIELDS if f not in payload]
    if missing:
        return _bad_request(
            "missing_fields", f"缺少必填字段: {missing}"
        )

    kwargs = {
        "reference": payload["reference_levels"],
        "observations": payload["observations"],
        "drift_min": payload["drift_min"],
        "drift_max": payload["drift_max"],
        "residual_limit": payload["residual_limit"],
    }
    if "dwell_min" in payload:
        kwargs["dwell_min"] = payload["dwell_min"]
    if "dwell_max" in payload:
        kwargs["dwell_max"] = payload["dwell_max"]
    if "max_skips" in payload:
        kwargs["max_skips"] = payload["max_skips"]

    try:
        result = solve_alignment(**kwargs)
    except AlignmentError as exc:
        status, body = _bad_request("invalid_request", str(exc))
        if exc.field:
            # 明确字段错误（如区间倒置、字段缺失、区间项数量超限）。
            body["field"] = exc.field
        return status, body
    except TypeError as exc:
        return _bad_request("invalid_request", f"字段类型错误: {exc}")

    return HTTPStatus.OK, result


def _bad_request(code: str, message: str) -> Tuple[int, Dict[str, Any]]:
    return HTTPStatus.BAD_REQUEST, {
        "feasible": False,
        "error": code,
        "message": message,
    }


class AlignHandler(BaseHTTPRequestHandler):
    server_version = "NanoporeAlign/1.0"

    def _write_json(self, status: int, body: Dict[str, Any]) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 (stdlib 命名)
        if self.path.split("?", 1)[0] == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path.split("?", 1)[0] == "/":
            self._write_json(
                HTTPStatus.OK,
                {
                    "service": "nanopore-current-trace-alignment",
                    "endpoint": "/api/current-traces/align",
                    "method": "POST",
                },
            )
            return
        self._write_json(
            HTTPStatus.NOT_FOUND,
            {"error": "not_found", "message": f"未知路径: {self.path}"},
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/api/current-traces/align":
            self._write_json(
                HTTPStatus.NOT_FOUND,
                {"error": "not_found", "message": f"未知路径: {self.path}"},
            )
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._write_json(
                *_bad_request("invalid_body", "Content-Length 非法")
            )
            return
        if length < 0:
            self._write_json(
                *_bad_request("invalid_body", "Content-Length 非法")
            )
            return
        if length == 0:
            self._write_json(
                *_bad_request("invalid_body", "缺少 JSON 请求体")
            )
            return
        if length > 1_000_000:
            self._write_json(
                *_bad_request("payload_too_large", "请求体超过 1MB 上限")
            )
            return

        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._write_json(
                *_bad_request("invalid_body", f"JSON 解析失败: {exc}")
            )
            return

        status, body = align_from_payload(payload)
        self._write_json(status, body)

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def build_server(host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), AlignHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="纳米孔电流对齐 API 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--port", type=int, default=8000, help="监听端口，0 表示随机端口"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    server = build_server(args.host, args.port)
    actual_port = server.server_address[1]
    # verify 脚本在以随机端口自启服务时读取此行。
    print(f"LISTENING {actual_port}", flush=True)
    LOGGER.info("API 监听 http://%s:%d", args.host, actual_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
