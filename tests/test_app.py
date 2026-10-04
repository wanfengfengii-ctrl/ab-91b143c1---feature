"""HTTP 层测试：直接调用处理函数与真实 HTTP 端到端冒烟。"""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Any, Dict, Tuple

from nanopore_align.app import align_from_payload, build_server


FEASIBLE_PAYLOAD: Dict[str, Any] = {
    "reference_levels": [10, 20, 30, 40, 50, 60, 70, 80],
    "observations": [15, 25, 35, 45, 55, 65, 75, 85],
    "drift_min": -10,
    "drift_max": 10,
    "residual_limit": 2,
    "dwell_min": 1,
    "dwell_max": 1,
}

INFEASIBLE_PAYLOAD: Dict[str, Any] = {
    "reference_levels": [0, 10, 20, 30, 40, 50, 60, 70],
    "observations": [900] * 8,
    "drift_min": -5,
    "drift_max": 5,
    "residual_limit": 1,
}


class AlignFromPayloadTests(unittest.TestCase):
    def test_feasible(self) -> None:
        status, body = align_from_payload(FEASIBLE_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertTrue(body["feasible"])
        self.assertEqual(body["drift"], 5)
        self.assertEqual(body["residual_sum"], 0)
        self.assertEqual(len(body["levels"]), 8)

    def test_infeasible_is_explicit_conclusion(self) -> None:
        status, body = align_from_payload(INFEASIBLE_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertFalse(body["feasible"])
        self.assertEqual(body["reason"], "no_alignment_exists")
        self.assertIn("message", body)

    def test_missing_field_rejected(self) -> None:
        bad = dict(FEASIBLE_PAYLOAD)
        del bad["drift_max"]
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        self.assertFalse(body["feasible"])
        self.assertEqual(body["error"], "missing_fields")

    def test_unknown_field_rejected(self) -> None:
        bad = dict(FEASIBLE_PAYLOAD)
        bad["extra"] = 1
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "unknown_fields")

    def test_bad_size_rejected(self) -> None:
        bad = dict(FEASIBLE_PAYLOAD)
        bad["reference_levels"] = list(range(7))
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_request")

    def test_too_many_observations_rejected(self) -> None:
        bad = dict(FEASIBLE_PAYLOAD)
        bad["observations"] = list(range(61))
        status, body = align_from_payload(bad)
        self.assertEqual(status, 400)

    def test_wrong_type_rejected(self) -> None:
        status, body = align_from_payload("not-a-dict")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_body")


class HttpEndToEndTests(unittest.TestCase):
    server: ThreadingHTTPServer
    thread: threading.Thread
    base: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = build_server("127.0.0.1", 0)
        port = cls.server.server_address[1]
        cls.base = f"http://127.0.0.1:{port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.daemon = True
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _post(self, payload: Any) -> Tuple[int, Dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}/api/current-traces/align",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_health(self) -> None:
        with urllib.request.urlopen(f"{self.base}/health", timeout=5) as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(json.loads(r.read())["status"], "ok")

    def test_feasible_roundtrip(self) -> None:
        status, body = self._post(FEASIBLE_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertTrue(body["feasible"])
        covered = [
            s["index"]
            for lv in body["levels"]
            for s in lv["samples"]
        ]
        self.assertEqual(covered, list(range(8)))

    def test_infeasible_roundtrip(self) -> None:
        status, body = self._post(INFEASIBLE_PAYLOAD)
        self.assertEqual(status, 200)
        self.assertFalse(body["feasible"])

    def test_invalid_roundtrip(self) -> None:
        bad = dict(FEASIBLE_PAYLOAD)
        bad["drift_min"] = 99
        bad["drift_max"] = 0
        status, body = self._post(bad)
        self.assertEqual(status, 400)
        self.assertFalse(body["feasible"])

    def test_malformed_json(self) -> None:
        req = urllib.request.Request(
            f"{self.base}/api/current-traces/align",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            self.fail("应当返回 400")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)

    def test_unknown_route_404(self) -> None:
        try:
            urllib.request.urlopen(f"{self.base}/nope", timeout=5)
            self.fail("应当返回 404")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 404)


if __name__ == "__main__":
    unittest.main()
