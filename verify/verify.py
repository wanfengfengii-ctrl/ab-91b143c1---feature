#!/usr/bin/env python3
"""一次性验证服务（compose 服务名: verify）。

按顺序执行三个阶段，并以退出码汇报总体结果：

1. 包构建：用 setuptools 构建 wheel 到临时目录；
2. 代码测试：运行仓库内全部 unittest 测试；
3. HTTP 冒烟：对运行中的 API 提交
   - 一条**混合观测**可行轨迹（点观测与 min/max 区间混用；
     期望 feasible=true、区间项带 witness/有符号残差，且逐级证据
     可按区间最短距离规则复算残差和、最大残差与覆盖），
   - 一条纯整数可行轨迹（兼容旧请求/响应语义），
   - 一条无解轨迹（期望 feasible=false 且给出明确结论），
   - 两条非法请求（区间倒置、字段缺失/数量超限，期望 HTTP 400）。

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
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
ENDPOINT = "/api/current-traces/align"

# 纯整数可行轨迹（验证旧请求/响应语义完全兼容）。
INTEGER_FEASIBLE_CASE: Dict[str, Any] = {
    "reference_levels": [10, 20, 30, 40, 50, 60, 70, 80],
    "observations": [12, 19, 31, 38, 52, 58, 71, 79],
    "drift_min": -5,
    "drift_max": 5,
    "residual_limit": 2,
    "dwell_min": 1,
    "dwell_max": 1,
}

# 混合观测可行轨迹：前 4 项为整数（恰为 R+5），后 4 项为整数闭区间。
# 点观测把可行漂移限制在 d∈[3,5]；区间 [57,60] 仅在 d=5 时与采用
# 电平 55 的距离（2）不越限，从而唯一钉住漂移 +5。此时：
#   - observations[4]=[57,60]：最近见证 57，有符号残差 +2；
#   - 其余区间分别含 65/75/85，见证即采用电平、残差 0。
MIXED_FEASIBLE_CASE: Dict[str, Any] = {
    "reference_levels": [10, 20, 30, 40, 50, 60, 70, 80],
    "observations": [
        15,
        25,
        35,
        45,
        {"min": 57, "max": 60},
        {"min": 65, "max": 70},
        {"min": 75, "max": 80},
        {"min": 85, "max": 90},
    ],
    "drift_min": -5,
    "drift_max": 5,
    "residual_limit": 2,
    "dwell_min": 1,
    "dwell_max": 1,
}
MIXED_FEASIBLE_INTERVAL_INDICES = {4, 5, 6, 7}
MIXED_FEASIBLE_EXPECTED_DRIFT = 5

INFEASIBLE_CASE: Dict[str, Any] = {
    "reference_levels": [0, 10, 20, 30, 40, 50, 60, 70],
    "observations": [900, 901, 902, 903, 904, 905, 906, 907],
    "drift_min": -5,
    "drift_max": 5,
    "residual_limit": 1,
}

# 非法区间：min > max（区间倒置）。
INVALID_INVERTED_CASE: Dict[str, Any] = {
    "reference_levels": [0, 10, 20, 30, 40, 50, 60, 70],
    "observations": [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        {"min": 80, "max": 70},
    ],
    "drift_min": 0,
    "drift_max": 0,
    "residual_limit": 1,
}

# 非法区间：对象缺少 min 字段。
INVALID_MISSING_FIELD_CASE: Dict[str, Any] = {
    "reference_levels": [0, 10, 20, 30, 40, 50, 60, 70],
    "observations": [
        0,
        10,
        20,
        30,
        40,
        50,
        60,
        {"max": 70},
    ],
    "drift_min": 0,
    "drift_max": 0,
    "residual_limit": 1,
}

# 非法区间：区间项数量超过 6 个上限。
INVALID_TOO_MANY_INTERVALS_CASE: Dict[str, Any] = {
    "reference_levels": [0, 10, 20, 30, 40, 50, 60, 70],
    "observations": [
        0,
        {"min": 9, "max": 11},
        {"min": 19, "max": 21},
        {"min": 29, "max": 31},
        {"min": 39, "max": 41},
        {"min": 49, "max": 51},
        {"min": 59, "max": 61},
        {"min": 69, "max": 71},
    ],
    "drift_min": 0,
    "drift_max": 0,
    "residual_limit": 1,
}

# 纯整数但序列规模非法（保留原有非法请求语义检查）。
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


def _signed_interval_residual(
    level: int, lo: int, hi: int
) -> Tuple[int, int]:
    """采用电平到闭区间 [lo, hi] 的 (最近见证值, 有符号残差)。"""
    witness = min(max(level, lo), hi)
    return witness, witness - level


def _check_levels_evidence(
    case: Dict[str, Any],
    body: Dict[str, Any],
    interval_indices: set,
    residual_limit: int,
) -> None:
    """逐级证据必须覆盖全部采样，且残差可按区间最短距离规则独立复算。"""
    observations = case["observations"]
    levels = body.get("levels")
    if not isinstance(levels, list) or not levels:
        raise StageFailure("可行结果缺少逐级采样区间 levels")

    covered = []
    sum_dist = 0
    max_dist = 0
    for lv in levels:
        adopted = lv["adopted_level"]
        for s in lv["samples"]:
            t = s["index"]
            covered.append(t)
            residual = s["residual"]
            if t in interval_indices:
                interval = observations[t]
                lo_v = interval["min"]
                hi_v = interval["max"]
                if not (
                    isinstance(s.get("observed_min"), int)
                    and isinstance(s.get("observed_max"), int)
                    and isinstance(s.get("witness"), int)
                ):
                    raise StageFailure(
                        f"区间项 observations[{t}] 缺少 witness/边界字段: {s}"
                    )
                if s["observed_min"] != lo_v or s["observed_max"] != hi_v:
                    raise StageFailure(
                        f"区间项 observations[{t}] 回显边界与请求不一致"
                    )
                witness, want_residual = _signed_interval_residual(
                    adopted, lo_v, hi_v
                )
                if s["witness"] != witness:
                    raise StageFailure(
                        f"区间项 observations[{t}] witness={s['witness']} "
                        f"与最近整数见证 {witness} 不符"
                    )
                if residual != want_residual:
                    raise StageFailure(
                        f"区间项 observations[{t}] 有符号残差 {residual} "
                        f"与复算值 {want_residual} 不符"
                    )
                if "observed" in s:
                    raise StageFailure(
                        f"区间项 observations[{t}] 不应携带点观测字段 observed"
                    )
            else:
                if residual != observations[t] - adopted:
                    raise StageFailure(
                        f"点观测 observations[{t}] 残差复算不一致"
                    )
                if "witness" in s or "observed_min" in s:
                    raise StageFailure(
                        f"点观测 observations[{t}] 不应携带区间字段"
                    )
            dist = abs(residual)
            if dist > residual_limit:
                raise StageFailure(
                    f"observations[{t}] 距离 {dist} 超过残差上限 "
                    f"{residual_limit}"
                )
            sum_dist += dist
            max_dist = max(max_dist, dist)

    if covered != list(range(len(observations))):
        raise StageFailure("残差证据未恰好覆盖每个观测一次")
    if sum_dist != body.get("residual_sum"):
        raise StageFailure(
            f"复算残差和 {sum_dist} 与响应 {body.get('residual_sum')} 不符"
        )
    if max_dist != body.get("max_abs_residual"):
        raise StageFailure(
            f"复算最大残差 {max_dist} 与响应 "
            f"{body.get('max_abs_residual')} 不符"
        )


def stage_http_smoke(base_url: str) -> None:
    _log("smoke", f"对 {base_url} 发起 HTTP 冒烟 ...")

    # 1) 混合观测可行轨迹。
    status, body = _http_request(base_url, MIXED_FEASIBLE_CASE)
    if status != 200 or not body.get("feasible"):
        raise StageFailure(f"混合观测可行用例失败: status={status} body={body}")
    if not isinstance(body.get("drift"), int):
        raise StageFailure("混合观测可行结果缺少整数漂移字段")
    if body["drift"] != MIXED_FEASIBLE_EXPECTED_DRIFT:
        raise StageFailure(
            f"混合观测漂移 {body['drift']} 与真值 "
            f"{MIXED_FEASIBLE_EXPECTED_DRIFT} 不符"
        )
    _check_levels_evidence(
        MIXED_FEASIBLE_CASE,
        body,
        MIXED_FEASIBLE_INTERVAL_INDICES,
        MIXED_FEASIBLE_CASE["residual_limit"],
    )
    # 区间项必须逐个回传见证值。
    interval_samples = {
        s["index"]: s
        for lv in body["levels"]
        for s in lv["samples"]
        if s["index"] in MIXED_FEASIBLE_INTERVAL_INDICES
    }
    if set(interval_samples) != MIXED_FEASIBLE_INTERVAL_INDICES:
        raise StageFailure("并非每个区间项都返回了见证证据")
    for t, s in interval_samples.items():
        if not (s["observed_min"] <= s["witness"] <= s["observed_max"]):
            raise StageFailure(f"区间项 observations[{t}] 见证值越界")
    _log(
        "smoke",
        f"混合观测可行通过: drift={body['drift']} "
        f"levels={len(body['levels'])} sum={body['residual_sum']} "
        f"max={body['max_abs_residual']}（区间距离已独立复算）",
    )

    # 2) 纯整数可行轨迹（旧请求/响应语义兼容）。
    status, body = _http_request(base_url, INTEGER_FEASIBLE_CASE)
    if status != 200 or not body.get("feasible"):
        raise StageFailure(
            f"纯整数可行用例失败: status={status} body={body}"
        )
    _check_levels_evidence(INTEGER_FEASIBLE_CASE, body, set(), 2)
    for lv in body["levels"]:
        for s in lv["samples"]:
            if set(s) != {"index", "observed", "residual"}:
                raise StageFailure(
                    f"纯整数响应字段被污染: {s}"
                )
    _log(
        "smoke",
        f"纯整数兼容通过: drift={body['drift']} "
        f"levels={len(body['levels'])} sum={body['residual_sum']}",
    )

    # 3) 无解轨迹。
    status, body = _http_request(base_url, INFEASIBLE_CASE)
    if status != 200 or body.get("feasible") is not False:
        raise StageFailure(f"无解轨迹用例失败: status={status} body={body}")
    if body.get("reason") != "no_alignment_exists":
        raise StageFailure("无解轨迹缺少明确结论 reason")
    _log("smoke", "无解轨迹通过: 服务返回 feasible=false 及明确结论")

    # 4) 非法区间与非法请求。
    for label, case in (
        ("区间倒置", INVALID_INVERTED_CASE),
        ("区间字段缺失", INVALID_MISSING_FIELD_CASE),
        ("区间项超限", INVALID_TOO_MANY_INTERVALS_CASE),
        ("序列规模非法", INVALID_CASE),
    ):
        status, body = _http_request(base_url, case)
        if status != 400:
            raise StageFailure(
                f"{label}应返回 400，实际 status={status} body={body}"
            )
        if not body.get("message"):
            raise StageFailure(f"{label}的 400 响应缺少明确字段错误信息")
        _log("smoke", f"{label}通过: 返回 HTTP 400（{body['message']}）")


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
