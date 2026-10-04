#!/usr/bin/env python3
"""一次性验证服务（compose 服务名: verify）。

按顺序执行三个阶段，并以退出码汇报总体结果：

1. 包构建：用 setuptools 构建 wheel 到临时目录；
2. 代码测试：运行仓库内全部 unittest 测试；
3. HTTP 冒烟：对运行中的 API 提交
   - 混合观测（整数 + min/max 区间）可行轨迹：期望 feasible=true，
     漂移/逐级证据/区间见证值齐全，并按区间距离规则复算最终裁决；
   - 混合观测无解轨迹：期望 feasible=false 且给出明确结论；
   - 非法区间请求（区间倒置 / 区间字段缺失 / 区间项数量超限）：
     期望 HTTP 400 与明确字段错误；
   - 纯整数兼容轨迹：期望响应与原语义完全一致（无区间字段），
     非法纯整数请求仍返回 HTTP 400。

API 地址取环境变量 ``API_BASE_URL``（compose 中为 http://api:8000）；
若该地址不可达且未显式要求使用远端服务，则在本地以随机端口临时启动
一个服务实例进行冒烟，方便脱离 compose 直接运行本脚本。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "/api/current-traces/align"

# 混合观测可行用例：整数与 min/max 区间混合。唯一解为 drift=0，
# 残差和 10、最大残差 2；区间见证 (index=1 -> 20/+0, index=4 -> 51/+1)。
MIXED_FEASIBLE_CASE: Dict[str, Any] = {
    "reference_levels": [10, 20, 30, 40, 50, 60, 70, 80],
    "observations": [
        12,
        {"min": 18, "max": 21},
        31,
        38,
        {"min": 51, "max": 54},
        58,
        71,
        79,
    ],
    "drift_min": -5,
    "drift_max": 5,
    "residual_limit": 2,
    "dwell_min": 1,
    "dwell_max": 1,
}
MIXED_FEASIBLE_EXPECT: Dict[str, Any] = {
    "drift": 0,
    "residual_sum": 10,
    "max_abs_residual": 2,
    "interval_witnesses": [
        {"index": 1, "min": 18, "max": 21, "witness": 20, "residual": 0},
        {"index": 4, "min": 51, "max": 54, "witness": 51, "residual": 1},
    ],
}

# 混合观测无解用例：区间项与整数项都远超残差上限可及范围。
MIXED_INFEASIBLE_CASE: Dict[str, Any] = {
    "reference_levels": [0, 10, 20, 30, 40, 50, 60, 70],
    "observations": [
        {"min": 900, "max": 910},
        901,
        902,
        903,
        904,
        905,
        906,
        907,
    ],
    "drift_min": -5,
    "drift_max": 5,
    "residual_limit": 1,
}

# 纯整数兼容用例：唯一解 drift=0，残差和 12、最大残差 2、边界 [1..7]。
PURE_INTEGER_CASE: Dict[str, Any] = {
    "reference_levels": [10, 20, 30, 40, 50, 60, 70, 80],
    "observations": [12, 19, 31, 38, 52, 58, 71, 79],
    "drift_min": -5,
    "drift_max": 5,
    "residual_limit": 2,
    "dwell_min": 1,
    "dwell_max": 1,
}
PURE_INTEGER_EXPECT: Dict[str, Any] = {
    "drift": 0,
    "residual_sum": 12,
    "max_abs_residual": 2,
    "boundaries": [1, 2, 3, 4, 5, 6, 7],
    "num_levels_used": 8,
}


def _interval_payload(observations: List[Any]) -> Dict[str, Any]:
    return {
        "reference_levels": [10, 20, 30, 40, 50, 60, 70, 80],
        "observations": observations,
        "drift_min": -5,
        "drift_max": 5,
        "residual_limit": 2,
        "dwell_min": 1,
        "dwell_max": 1,
    }


# 非法区间用例：区间倒置 / 区间字段缺失 / 区间项数量超限（> 6）。
INVALID_INTERVAL_CASES: List[Tuple[str, Dict[str, Any]]] = [
    (
        "区间倒置",
        _interval_payload(
            [12, {"min": 22, "max": 18}, 31, 38, 52, 58, 71, 79]
        ),
    ),
    (
        "区间字段缺失",
        _interval_payload([12, {"min": 19}, 31, 38, 52, 58, 71, 79]),
    ),
    (
        "区间项数量超限",
        _interval_payload(
            [
                {"min": 12, "max": 12},
                {"min": 19, "max": 19},
                {"min": 31, "max": 31},
                {"min": 38, "max": 38},
                {"min": 52, "max": 52},
                {"min": 58, "max": 58},
                {"min": 71, "max": 71},
                79,
            ]
        ),
    ),
]

# 纯整数非法请求（结构非法），用于确认失败语义保持兼容。
INVALID_CASE: Dict[str, Any] = {
    "reference_levels": [1, 2, 3],  # 少于 8 个
    "observations": [1, 2, 3, 4, 5, 6, 7, 8],
    "drift_min": 0,
    "drift_max": 0,
    "residual_limit": 0,
}


class StageFailure(Exception):
    pass


def _log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def stage_build_package() -> Path:
    _log("build", "开始构建 wheel ...")
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            tmp,
            str(REPO_ROOT),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stdout.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            raise StageFailure("wheel 构建失败")
        wheels = list(Path(tmp).glob("*.whl"))
        if not wheels:
            raise StageFailure("未找到构建产物 wheel")
        # wheel 位于临时目录，复制到持久临时路径以便汇报。
        out_dir = Path(tempfile.mkdtemp(prefix="nanopore-wheel-"))
        wheel = out_dir / wheels[0].name
        wheel.write_bytes(wheels[0].read_bytes())
    _log("build", f"包构建成功: {wheel.name}")
    return wheel


def stage_run_tests() -> None:
    _log("tests", "运行 unittest 测试套件 ...")
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        raise StageFailure("代码测试失败")
    _log("tests", "全部测试通过")


def _http_request(
    base_url: str, payload: Any
) -> Tuple[int, Dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + ENDPOINT,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _wait_healthy(base_url: str, attempts: int = 30) -> None:
    url = base_url.rstrip("/") + "/health"
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise StageFailure(f"服务健康检查未通过: {url}")


def _maybe_start_local_server() -> Tuple[Optional[subprocess.Popen], str]:
    """优先使用 API_BASE_URL；不可达时本地起随机端口服务。"""
    base = os.environ.get("API_BASE_URL")
    if base:
        _wait_healthy(base)
        return None, base

    _log("smoke", "API_BASE_URL 未设置，本地临时启动 API 实例 ...")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "nanopore_align.app",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    port: Optional[int] = None
    deadline = time.time() + 10
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise StageFailure("本地 API 进程提前退出")
            time.sleep(0.05)
            continue
        if line.startswith("LISTENING "):
            port = int(line.split()[1])
            break
    if port is None:
        proc.kill()
        raise StageFailure("未能读取本地 API 端口")
    base = f"http://127.0.0.1:{port}"
    _wait_healthy(base)
    return proc, base


def _obs_pairs(observations: List[Any]) -> List[Tuple[int, int]]:
    """混合观测归一化为 (lo, hi) 闭区间对（整数 o 视为 [o, o]）。"""
    pairs: List[Tuple[int, int]] = []
    for o in observations:
        if isinstance(o, dict):
            pairs.append((o["min"], o["max"]))
        else:
            pairs.append((o, o))
    return pairs


def _check_evidence(
    body: Dict[str, Any], observations: List[Any], residual_limit: int
) -> None:
    """逐级证据必须覆盖全部采样，并能按区间距离规则复算最终裁决。"""
    pairs = _obs_pairs(observations)
    levels = body.get("levels")
    if not isinstance(levels, list) or not levels:
        raise StageFailure("可行轨迹缺少逐级采样区间")
    covered: List[int] = []
    total = 0
    worst = 0
    for lv in levels:
        adopted = lv["adopted_level"]
        if adopted != lv["reference_level"] + body["drift"]:
            raise StageFailure("采用电平与参考电平/漂移不一致")
        for s in lv["samples"]:
            t = s["index"]
            lo, hi = pairs[t]
            witness = min(max(adopted, lo), hi)
            residual = witness - adopted
            if isinstance(observations[t], dict):
                if s.get("observed") != {"min": lo, "max": hi}:
                    raise StageFailure(f"样本 {t} 区间回显错误: {s}")
                if s.get("witness") != witness:
                    raise StageFailure(f"样本 {t} 见证值错误: {s}")
            else:
                if "witness" in s:
                    raise StageFailure(f"整数样本 {t} 不应含见证值字段")
                if s.get("observed") != lo:
                    raise StageFailure(f"样本 {t} 观测回显错误: {s}")
            if s.get("residual") != residual:
                raise StageFailure(f"样本 {t} 残差错误: {s}")
            if abs(residual) > residual_limit:
                raise StageFailure(f"样本 {t} 残差越限: {s}")
            total += abs(residual)
            if abs(residual) > worst:
                worst = abs(residual)
            covered.append(t)
    if covered != list(range(len(observations))):
        raise StageFailure("残差证据未恰好覆盖每个观测一次")
    if body.get("residual_sum") != total:
        raise StageFailure(
            f"按区间距离规则复算的残差和 {total} 与响应 "
            f"{body.get('residual_sum')} 不一致"
        )
    if body.get("max_abs_residual") != worst:
        raise StageFailure(
            f"按区间距离规则复算的最大残差 {worst} 与响应 "
            f"{body.get('max_abs_residual')} 不一致"
        )


def stage_http_smoke(base_url: str) -> None:
    _log("smoke", f"对 {base_url} 发起 HTTP 冒烟 ...")

    # 1) 混合观测可行：裁决值、区间见证值与逐级证据复算。
    status, body = _http_request(base_url, MIXED_FEASIBLE_CASE)
    if status != 200 or not body.get("feasible"):
        raise StageFailure(
            f"混合观测可行用例失败: status={status} body={body}"
        )
    for key, want in MIXED_FEASIBLE_EXPECT.items():
        if body.get(key) != want:
            raise StageFailure(
                f"混合观测可行用例字段 {key} 不符: "
                f"期望 {want} 实际 {body.get(key)}"
            )
    _check_evidence(
        body,
        MIXED_FEASIBLE_CASE["observations"],
        MIXED_FEASIBLE_CASE["residual_limit"],
    )
    _log(
        "smoke",
        f"混合观测可行通过: drift={body['drift']} "
        f"sum={body['residual_sum']} max={body['max_abs_residual']} "
        f"见证值={body['interval_witnesses']}",
    )

    # 2) 混合观测无解：明确结论。
    status, body = _http_request(base_url, MIXED_INFEASIBLE_CASE)
    if status != 200 or body.get("feasible") is not False:
        raise StageFailure(
            f"混合观测无解用例失败: status={status} body={body}"
        )
    if body.get("reason") != "no_alignment_exists":
        raise StageFailure("无解轨迹缺少明确结论 reason")
    _log("smoke", "混合观测无解通过: 服务返回 feasible=false 及明确结论")

    # 3) 非法区间：倒置 / 字段缺失 / 数量超限均返回 400 与明确字段错误。
    for name, payload in INVALID_INTERVAL_CASES:
        status, body = _http_request(base_url, payload)
        if status != 400:
            raise StageFailure(
                f"{name} 应返回 400，实际 status={status} body={body}"
            )
        if not body.get("message"):
            raise StageFailure(f"{name} 缺少错误信息: {body}")
        if not str(body.get("field", "")).startswith("observations"):
            raise StageFailure(f"{name} 缺少明确字段定位: {body}")
    _log("smoke", "非法区间通过: 倒置/字段缺失/数量超限均返回 400 与字段错误")

    # 4) 纯整数兼容：裁决值、边界与响应形状与原语义一致。
    status, body = _http_request(base_url, PURE_INTEGER_CASE)
    if status != 200 or not body.get("feasible"):
        raise StageFailure(
            f"纯整数兼容用例失败: status={status} body={body}"
        )
    for key, want in PURE_INTEGER_EXPECT.items():
        if body.get(key) != want:
            raise StageFailure(
                f"纯整数兼容用例字段 {key} 不符: "
                f"期望 {want} 实际 {body.get(key)}"
            )
    if "interval_witnesses" in body:
        raise StageFailure("纯整数响应不应包含 interval_witnesses")
    _check_evidence(
        body,
        PURE_INTEGER_CASE["observations"],
        PURE_INTEGER_CASE["residual_limit"],
    )
    _log(
        "smoke",
        f"纯整数兼容通过: drift={body['drift']} "
        f"sum={body['residual_sum']} max={body['max_abs_residual']}",
    )

    # 4b) 纯整数非法请求仍返回 400（失败语义兼容）。
    status, body = _http_request(base_url, INVALID_CASE)
    if status != 400:
        raise StageFailure(f"非法请求应返回 400，实际 status={status}")
    _log("smoke", "纯整数非法请求通过: 返回 HTTP 400 拒绝")


def main() -> int:
    failures = []
    wheel: Optional[Path] = None
    proc: Optional[subprocess.Popen] = None
    try:
        wheel = stage_build_package()
    except StageFailure as exc:
        failures.append(f"包构建: {exc}")

    try:
        stage_run_tests()
    except StageFailure as exc:
        failures.append(f"代码测试: {exc}")

    try:
        proc, base_url = _maybe_start_local_server()
        stage_http_smoke(base_url)
    except StageFailure as exc:
        failures.append(f"HTTP 冒烟: {exc}")

    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("-" * 60)
    if failures:
        print("VERIFY 结果: 失败")
        for item in failures:
            print(f"  - {item}")
        if wheel is not None:
            print(f"  wheel 产物: {wheel}")
        return 1

    print("VERIFY 结果: 全部通过（包构建 / 代码测试 / HTTP 冒烟）")
    if wheel is not None:
        print(f"  wheel 产物: {wheel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
