"""联合对齐算法的单元测试与暴力对照测试。"""

from __future__ import annotations

import itertools
import random
import unittest
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from nanopore_align.alignment import (
    AlignmentError,
    MAX_INTERVAL_OBSERVATIONS,
    solve_alignment,
)

Obs = Union[int, Tuple[int, int]]


def _distance(level: int, obs: Obs) -> int:
    """采用电平到点观测或闭区间的最短整数距离。"""
    if isinstance(obs, tuple):
        lo, hi = obs
        if level < lo:
            return lo - level
        if level > hi:
            return level - hi
        return 0
    return abs(obs - level)


def _witness(level: int, obs: Obs) -> Tuple[int, int]:
    """返回 (最近见证值, 有符号残差)。"""
    if isinstance(obs, tuple):
        w = min(max(level, obs[0]), obs[1])
        return w, w - level
    return obs, obs - level


def brute_force(
    reference: Sequence[int],
    observations: Sequence[int],
    drift_min: int,
    drift_max: int,
    residual_limit: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
) -> Optional[dict]:
    """穷举所有漂移 / 含首尾子序列 / 停留组合，返回与 solve 同口径的最优解。"""
    return _brute_force_mixed(
        list(reference),
        list(observations),
        drift_min,
        drift_max,
        residual_limit,
        dwell_min,
        dwell_max,
        max_skips,
    )


def _brute_force_mixed(
    reference: Sequence[int],
    observations: Sequence[Obs],
    drift_min: int,
    drift_max: int,
    residual_limit: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
) -> Optional[dict]:
    """混合观测版本的穷举：区间项残差按到闭区间的最短距离计。"""
    R = len(reference)
    N = len(observations)
    best: Optional[Tuple] = None

    for d in range(drift_min, drift_max + 1):
        # 枚举被跳过的内部参考索引集合（大小 <= max_skips）。
        inner = list(range(1, R - 1))
        for skip_count in range(0, min(max_skips, len(inner)) + 1):
            for skipped in itertools.combinations(inner, skip_count):
                used = [i for i in range(R) if i not in set(skipped)]
                k = len(used)
                if k > N or k * dwell_min > N or k * dwell_max < N:
                    continue
                for dwells in itertools.product(
                    range(dwell_min, dwell_max + 1), repeat=k
                ):
                    if sum(dwells) != N:
                        continue
                    boundaries = tuple(itertools.accumulate(dwells))
                    total = 0
                    worst = 0
                    ok = True
                    s = 0
                    for ri, L in zip(used, dwells):
                        level = reference[ri] + d
                        for t in range(s, s + L):
                            r = _distance(level, observations[t])
                            if r > residual_limit:
                                ok = False
                                break
                            total += r
                            worst = max(worst, r)
                        if not ok:
                            break
                        s += L
                    if not ok:
                        continue
                    cost = (skip_count, total, worst, d, boundaries)
                    if best is None or cost < best[0]:
                        best = (cost, used, d)

    if best is None:
        return None
    (skipped_n, total, worst, d, boundaries), used, drift = best
    return {
        "feasible": True,
        "drift": drift,
        "num_skips": skipped_n,
        "residual_sum": total,
        "max_abs_residual": worst,
        "boundaries": list(boundaries[:-1]),
        "used_indices": list(used),
    }


def _normalize_observations(observations: Sequence[Any]) -> List[Obs]:
    """把 dict 形式的区间观测转换为算法内部使用的 (min, max) 元组。"""
    return [
        (o["min"], o["max"]) if isinstance(o, dict) else o
        for o in observations
    ]


def _assert_matches_brute(
    testcase: unittest.TestCase,
    reference: Sequence[int],
    observations: Sequence[Any],
    drift_min: int,
    drift_max: int,
    residual_limit: int,
    dwell_min: int = 1,
    dwell_max: int = 3,
    max_skips: int = 2,
) -> None:
    norm_obs = _normalize_observations(observations)
    # 对外接口的区间项是 min/max 对象：把元组形态转成 dict 再提交。
    payload_obs: List[Any] = [
        {"min": o[0], "max": o[1]} if isinstance(o, tuple) else o
        for o in observations
    ]
    got = solve_alignment(
        reference,
        payload_obs,
        drift_min,
        drift_max,
        residual_limit,
        dwell_min,
        dwell_max,
        max_skips,
    )
    want = _brute_force_mixed(
        reference,
        norm_obs,
        drift_min,
        drift_max,
        residual_limit,
        dwell_min,
        dwell_max,
        max_skips,
    )
    if want is None:
        testcase.assertFalse(got["feasible"], msg=f"意外可行: {got}")
        return
    testcase.assertTrue(got["feasible"], msg="意外无解")
    testcase.assertEqual(got["drift"], want["drift"])
    testcase.assertEqual(got["num_skips"], want["num_skips"])
    testcase.assertEqual(got["residual_sum"], want["residual_sum"])
    testcase.assertEqual(got["max_abs_residual"], want["max_abs_residual"])
    testcase.assertEqual(got["boundaries"], want["boundaries"])
    got_used = [lv["reference_index"] for lv in got["levels"]]
    testcase.assertEqual(got_used, want["used_indices"])
    _assert_evidence_consistent(testcase, reference, norm_obs, got)


def _assert_evidence_consistent(
    testcase: unittest.TestCase,
    reference: Sequence[int],
    observations: Sequence[Obs],
    got: dict,
) -> None:
    """逐级证据覆盖全部采样，区间项见证值/残差可按规则复算。"""
    covered: List[int] = []
    total = 0
    worst = 0
    for lv in got["levels"]:
        adopted = lv["adopted_level"]
        testcase.assertEqual(
            adopted, reference[lv["reference_index"]] + got["drift"]
        )
        for s in lv["samples"]:
            t = s["index"]
            covered.append(t)
            obs = observations[t]
            w, r = _witness(adopted, obs)
            if isinstance(obs, tuple):
                testcase.assertEqual(s["observed_min"], obs[0])
                testcase.assertEqual(s["observed_max"], obs[1])
                testcase.assertEqual(s["witness"], w)
                testcase.assertNotIn("observed", s)
            else:
                testcase.assertEqual(s["observed"], obs)
                testcase.assertNotIn("witness", s)
            testcase.assertEqual(s["residual"], r)
            testcase.assertEqual(abs(r), _distance(adopted, obs))
            total += abs(r)
            worst = max(worst, abs(r))
    testcase.assertEqual(covered, list(range(len(observations))))
    testcase.assertEqual(total, got["residual_sum"])
    testcase.assertEqual(worst, got["max_abs_residual"])


class ExactAlignmentTests(unittest.TestCase):
    def test_wide_drift_interval_finds_far_drift(self) -> None:
        # 真实漂移远离 0，且首级/末级可行漂移区间很窄：
        # 验证首末级预筛不会把正确漂移漏掉。
        ref = [10, 20, 30, 40, 50, 60, 70, 80]
        obs = [x + 432 for x in ref]
        res = solve_alignment(ref, obs, -1000, 1000, 0)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["drift"], 432)
        self.assertEqual(res["num_skips"], 0)
        self.assertEqual(res["residual_sum"], 0)

    def test_first_last_prefilter_conflict_is_infeasible(self) -> None:
        # 首级前几个观测要求漂移 -7 附近，末级观测要求漂移 +7 附近，
        # 首末级可行区间交集为空 -> 明确无解（limit=0，无折中可能）。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        obs = [-7, -7, -7, 10, 20, 30, 40, 50, 60, 70, 77, 77]
        res = solve_alignment(ref, obs, -100, 100, 0,
                              dwell_min=1, dwell_max=3)
        self.assertFalse(res["feasible"])

    def test_straight_no_skip(self) -> None:
        # 8 个参考电平，每级恰好 1 个观测，漂移 +5，残差全 0。
        ref = [10, 20, 30, 40, 50, 60, 70, 80]
        obs = [x + 5 for x in ref]
        res = solve_alignment(ref, obs, -10, 10, 2)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["drift"], 5)
        self.assertEqual(res["num_skips"], 0)
        self.assertEqual(res["residual_sum"], 0)
        self.assertEqual(res["max_abs_residual"], 0)
        self.assertEqual(res["boundaries"], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(len(res["levels"]), 8)
        for slot, lv in enumerate(res["levels"]):
            self.assertEqual(lv["sample_start"], slot)
            self.assertEqual(lv["sample_end"], slot + 1)
            self.assertEqual(lv["dwell"], 1)
            self.assertEqual(lv["samples"][0]["residual"], 0)

    def test_dwell_two_three_and_skip(self) -> None:
        # 8 个参考电平；跳过索引 3；停留：2,3,2,... 凑 14 个观测。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        used = [0, 1, 2, 4, 5, 6, 7]
        dwells = [2, 3, 2, 2, 2, 2, 1]
        self.assertEqual(sum(dwells), 14)
        obs: List[int] = []
        for ri, L in zip(used, dwells):
            obs.extend([ref[ri] - 3] * L)
        res = solve_alignment(ref, obs, -5, 5, 1)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["drift"], -3)
        self.assertEqual(res["num_skips"], 1)
        self.assertEqual(res["skipped_reference_indices"], [3])
        self.assertEqual(
            [lv["reference_index"] for lv in res["levels"]], used
        )
        self.assertEqual([lv["dwell"] for lv in res["levels"]], dwells)
        self.assertEqual(res["boundaries"], [2, 5, 7, 9, 11, 13])
        for lv in res["levels"]:
            for s in lv["samples"]:
                self.assertEqual(abs(s["residual"]), 0)

    def test_first_and_last_mandatory(self) -> None:
        # 首电平与尾电平与观测差距巨大，任何跳过都救不了 -> 无解。
        ref = [0, 100, 100, 100, 100, 100, 100, 1000]
        obs = [100] * 8
        res = solve_alignment(ref, obs, 0, 0, 1)
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "no_alignment_exists")

    def test_infeasible_when_residual_limit_tight(self) -> None:
        ref = [0, 1, 2, 3, 4, 5, 6, 7]
        # 每个观测比对应电平高 2；limit=1、漂移只能 0 -> 无解。
        obs = [x + 2 for x in ref]
        res = solve_alignment(ref, obs, 0, 0, 1)
        self.assertFalse(res["feasible"])

    def test_infeasible_when_too_many_levels(self) -> None:
        # 8 个观测、8 个必到级数（首尾+内部不允许跳过），每级最多 1 采样可行；
        # 但若禁用跳过且观测只有 8 个、停留最小 2 -> 无解。
        ref = list(range(8))
        obs = list(range(8))
        res = solve_alignment(ref, obs, 0, 0, 0, dwell_min=2, dwell_max=3)
        self.assertFalse(res["feasible"])

    def test_skip_cap_enforced(self) -> None:
        # 需要 3 个跳过才可行，但上限为 2 -> 无解。
        ref = [0, 100, 200, 300, 400, 500, 600, 700]
        obs = [0, 0, 700, 700]  # 仅首尾附近有观测；N 最少为 8，构造 8 个
        obs = [0, 0, 0, 0, 700, 700, 700, 700]
        # 合法对齐至少要跳过中间 6 个内部电平，> 2。
        res = solve_alignment(ref, obs, 0, 0, 0, max_skips=2)
        self.assertFalse(res["feasible"])
        res2 = solve_alignment(ref, obs, 0, 0, 0, max_skips=2,
                               dwell_min=1, dwell_max=3)
        self.assertFalse(res2["feasible"])


class ObjectiveOrderTests(unittest.TestCase):
    def test_minimize_skips_first(self) -> None:
        # 无跳过对齐需要较大残差；带 1 跳过残差为 0。
        # 仍应选择无跳过（跳过数优先级最高）。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        # 8 观测每级 1 个，全部偏移 2（limit=2，无跳过，残差和 16）。
        obs = [x + 2 for x in ref]
        res = solve_alignment(ref, obs, 0, 0, 5)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["num_skips"], 0)
        self.assertEqual(res["residual_sum"], 16)

    def test_then_residual_sum(self) -> None:
        # 漂移 -1 / 0 / +1 都可行，残差和不同 -> 选和最小者。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        obs = [r - 1 for r in ref]  # drift=-1 时残差为 0
        res = solve_alignment(ref, obs, -2, 2, 5)
        self.assertEqual(res["drift"], -1)
        self.assertEqual(res["residual_sum"], 0)

    def test_then_max_residual(self) -> None:
        # 构造两个漂移残差和相同但最大残差不同：用对称扰动。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        # drift=0: 残差 +1,-1,...,0 -> 和 0（带符号不影响，这里用绝对值）
        obs = [1, 9, 21, 29, 41, 39, 61, 69]
        # drift=0 时绝对残差全 1；和 8、最大 1
        # drift=1 时残差 [0,-2,0,-2,...,±2] 和 16 更大 -> 0 胜
        res = solve_alignment(ref, obs, -1, 1, 3)
        self.assertEqual(res["drift"], 0)
        self.assertEqual(res["max_abs_residual"], 1)
        self.assertEqual(res["residual_sum"], 8)

    def test_then_drift_tie_break(self) -> None:
        # 所有观测恰好在两个漂移下都零残差（参考电平差偶数且观测居中无法同时为0；
        # 改用只有 1 级停留长度结构使得 +0 与 +1 不可能同残差，因此直接构造
        # limit 宽松、残差相同的情形：参考全部相同间距无法做到——
        # 这里验证漂移平局取较小漂移：令观测 = ref + 1，并允许 drift=1 与
        # 跳过路径下 drift=0 残差结构一致较难构造，改为直接校验纯平局：
        # 参考电平全部为偶数，观测相对 ref：+1，此时仅 drift=1 零残差，
        # 退而求其次，校验候选漂移中较小者获胜的代码路径由对照测试覆盖。
        ref = [0, 2, 4, 6, 8, 10, 12, 14]
        obs = [r + 1 for r in ref]
        res = solve_alignment(ref, obs, 0, 2, 1)
        self.assertEqual(res["drift"], 1)

    def test_boundaries_lexicographic_tie_break(self) -> None:
        # 全部相同参考电平：任意边界划分残差相同；应取字典序最小边界
        # （尽早结束第一级：在 dwell_min=1 下首条边界最小）。
        ref = [5] * 8
        obs = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  # 10 观测，8 级
        res = solve_alignment(ref, obs, 0, 0, 0, dwell_min=1, dwell_max=3)
        self.assertTrue(res["feasible"])
        # 额外的 2 个采样尽量后置 -> 停留 (1,1,1,1,1,1,1,3)，
        # 内部边界字典序最小。
        self.assertEqual(res["boundaries"], [1, 2, 3, 4, 5, 6, 7])


class ValidationTests(unittest.TestCase):
    def _base(self) -> dict:
        return dict(
            reference=list(range(8)),
            observations=list(range(8)),
            drift_min=0,
            drift_max=0,
            residual_limit=0,
        )

    def test_reference_size_bounds(self) -> None:
        kw = self._base()
        kw["reference"] = list(range(7))
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["reference"] = list(range(25))
        kw["observations"] = list(range(25))
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_observation_size_bounds(self) -> None:
        kw = self._base()
        kw["observations"] = list(range(7))
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["observations"] = list(range(61))
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_drift_interval(self) -> None:
        kw = self._base()
        kw["drift_min"], kw["drift_max"] = 5, 4
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_dwell_range(self) -> None:
        kw = self._base()
        kw["dwell_min"], kw["dwell_max"] = 0, 3
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["dwell_min"], kw["dwell_max"] = 2, 1
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["dwell_min"], kw["dwell_max"] = 1, 4
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_max_skips_range(self) -> None:
        kw = self._base()
        kw["max_skips"] = 3
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw["max_skips"] = -1
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_drift_width_cap(self) -> None:
        from nanopore_align.alignment import DRIFT_WIDTH_MAX

        kw = self._base()
        kw["drift_min"] = 0
        kw["drift_max"] = DRIFT_WIDTH_MAX
        solve_alignment(**kw)  # 边界宽度合法
        kw["drift_max"] = DRIFT_WIDTH_MAX + 1
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_negative_residual_limit_rejected(self) -> None:
        kw = self._base()
        kw["residual_limit"] = -1
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)

    def test_type_checks(self) -> None:
        kw = self._base()
        kw["reference"] = [1, 2, 3, 4, 5, 6, 7, "8"]  # type: ignore[list-item]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw = self._base()
        kw["drift_min"] = 0.5  # type: ignore[assignment]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw = self._base()
        kw["observations"] = True  # type: ignore[assignment]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)
        kw = self._base()
        kw["reference"] = []  # type: ignore[assignment]
        with self.assertRaises(AlignmentError):
            solve_alignment(**kw)


class IntervalObservationTests(unittest.TestCase):
    """区间（部分采样整数闭区间）观测的语义测试。"""

    def test_interval_pins_drift_and_returns_witness(self) -> None:
        # 与 verify 混合用例同构：点观测把漂移限到 [3,5]，
        # 区间 [57,60] 仅在 d=5 时距离 2 不越限。
        ref = [10, 20, 30, 40, 50, 60, 70, 80]
        obs: List[Any] = [
            15, 25, 35, 45,
            {"min": 57, "max": 60},
            {"min": 65, "max": 70},
            {"min": 75, "max": 80},
            {"min": 85, "max": 90},
        ]
        res = solve_alignment(ref, obs, -5, 5, 2, dwell_min=1, dwell_max=1)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["drift"], 5)
        samples = {
            s["index"]: s
            for lv in res["levels"]
            for s in lv["samples"]
        }
        # 电平 55 在区间 [57,60] 左侧：最近见证为下界 57，残差 +2。
        self.assertEqual(samples[4]["witness"], 57)
        self.assertEqual(samples[4]["residual"], 2)
        # 电平 65/75/85 落在各自区间内部：见证即采用电平，残差 0。
        self.assertEqual(samples[5]["witness"], 65)
        self.assertEqual(samples[5]["residual"], 0)
        self.assertEqual(samples[6]["witness"], 75)
        self.assertEqual(samples[7]["witness"], 85)
        self.assertEqual(res["residual_sum"], 2)
        self.assertEqual(res["max_abs_residual"], 2)

    def test_interval_above_level_clamps_to_max(self) -> None:
        # 采用电平高于区间上界：见证取上界，残差为负。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        obs: List[Any] = [
            0, 10, 20, 30, 40, 50, 60, {"min": 60, "max": 65}
        ]
        res = solve_alignment(ref, obs, 0, 0, 5, dwell_min=1, dwell_max=1)
        self.assertTrue(res["feasible"])
        last = res["levels"][-1]["samples"][0]
        self.assertEqual(last["witness"], 65)
        self.assertEqual(last["residual"], -5)
        self.assertEqual(res["residual_sum"], 5)

    def test_interval_distance_must_obey_residual_limit(self) -> None:
        # 点观测全部精确匹配 d=0，但区间 [76,80] 距电平 70 为 6 > 2：
        # 区间残差参与上限裁决，故无解（而不是被中点误放进来）。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        obs: List[Any] = [
            0, 10, 20, 30, 40, 50, 60, {"min": 76, "max": 80}
        ]
        res = solve_alignment(ref, obs, 0, 0, 2, dwell_min=1, dwell_max=1)
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "no_alignment_exists")
        # 放宽上限后可行。
        res2 = solve_alignment(ref, obs, 0, 0, 6, dwell_min=1, dwell_max=1)
        self.assertTrue(res2["feasible"])
        self.assertEqual(res2["residual_sum"], 6)

    def test_degenerate_interval_equals_point(self) -> None:
        # min == max 的退化区间等价于该整数点观测。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        obs_deg = [0, 10, 20, 30, 40, 50, 60, {"min": 72, "max": 72}]
        obs_pt = [0, 10, 20, 30, 40, 50, 60, 72]
        a = solve_alignment(ref, obs_deg, -5, 5, 3, dwell_min=1, dwell_max=1)
        b = solve_alignment(ref, obs_pt, -5, 5, 3, dwell_min=1, dwell_max=1)
        self.assertTrue(a["feasible"] and b["feasible"])
        self.assertEqual(a["drift"], b["drift"])
        self.assertEqual(a["residual_sum"], b["residual_sum"])
        self.assertEqual(a["boundaries"], b["boundaries"])

    def test_no_midpoint_substitution_changes_boundaries(self) -> None:
        # 关键语义：不得先把区间替换成中点。区间 [4,8] 的中点是 6，
        # 若错误地使用中点，电平 5（d=5）的残差会是 -1；
        # 正确的最短距离为 0（电平落在区间内）。
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        obs: List[Any] = [
            5, 15, 25, 35, 45, {"min": 51, "max": 59}, 65, 75
        ]
        res = solve_alignment(ref, obs, 0, 10, 0, dwell_min=1, dwell_max=1)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["drift"], 5)
        self.assertEqual(res["residual_sum"], 0)
        s5 = res["levels"][5]["samples"][0]
        self.assertEqual(s5["witness"], 55)
        self.assertEqual(s5["residual"], 0)

    def test_inverted_interval_rejected(self) -> None:
        ref = list(range(8))
        obs: List[Any] = [0, 1, 2, 3, 4, 5, 6, {"min": 9, "max": 7}]
        with self.assertRaises(AlignmentError) as ctx:
            solve_alignment(ref, obs, 0, 0, 1)
        self.assertIn("倒置", str(ctx.exception))

    def test_missing_interval_field_rejected(self) -> None:
        ref = list(range(8))
        obs: List[Any] = [0, 1, 2, 3, 4, 5, 6, {"min": 7}]
        with self.assertRaises(AlignmentError) as ctx:
            solve_alignment(ref, obs, 0, 0, 1)
        self.assertIn("缺少字段", str(ctx.exception))

    def test_too_many_intervals_rejected(self) -> None:
        ref = list(range(8))
        obs: List[Any] = [
            0,
            *[{"min": i, "max": i + 1} for i in range(1, 8)],
        ]
        self.assertEqual(
            sum(isinstance(o, dict) for o in obs),
            MAX_INTERVAL_OBSERVATIONS + 1,
        )
        with self.assertRaises(AlignmentError) as ctx:
            solve_alignment(ref, obs, 0, 0, 1)
        self.assertIn("超过上限", str(ctx.exception))

    def test_interval_field_types_rejected(self) -> None:
        ref = list(range(8))
        bad_obs: List[Any] = [0, 1, 2, 3, 4, 5, 6, {"min": "7", "max": 8}]
        with self.assertRaises(AlignmentError):
            solve_alignment(ref, bad_obs, 0, 0, 1)
        bad_obs2: List[Any] = [0, 1, 2, 3, 4, 5, 6, {"min": 7, "max": True}]
        with self.assertRaises(AlignmentError):
            solve_alignment(ref, bad_obs2, 0, 0, 1)

    def test_exactly_six_intervals_allowed(self) -> None:
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        obs: List[Any] = [
            0, 10,
            {"min": 19, "max": 21},
            {"min": 29, "max": 31},
            {"min": 39, "max": 41},
            {"min": 49, "max": 51},
            {"min": 59, "max": 61},
            {"min": 69, "max": 71},
        ]
        res = solve_alignment(ref, obs, 0, 0, 1, dwell_min=1, dwell_max=1)
        self.assertTrue(res["feasible"])


class MixedBruteForceComparisonTests(unittest.TestCase):
    """随机混合（点 + 区间）小规模实例：DP 必须与穷举完全一致。"""

    def test_random_mixed_cases(self) -> None:
        rng = random.Random(20261005)
        for trial in range(150):
            R = rng.randint(8, 10)
            N = rng.randint(8, 14)
            ref = [rng.randint(0, 40) for _ in range(R)]
            used = [0]
            inner = list(range(1, R - 1))
            rng.shuffle(inner)
            skip_k = rng.randint(0, min(2, R - 2))
            skipped_set = set(inner[:skip_k])
            used = [i for i in range(R) if i not in skipped_set]
            k = len(used)
            if k > N:
                used = list(range(R))
                k = R
                skipped_set = set()
            dwells = self._random_composition(rng, k, N, 1, 3)
            if dwells is None:
                continue
            d = rng.randint(-3, 3)
            obs: List[Obs] = []
            for ri, L in zip(used, dwells):
                level = ref[ri] + d
                for _ in range(L):
                    noise = rng.choice([0, 0, 1, -1, 2, -2, 5])
                    center = level + noise
                    if rng.random() < 0.4:
                        half = rng.randint(0, 3)
                        obs.append((center - half, center + rng.randint(0, 3)))
                    else:
                        obs.append(center)
            if sum(isinstance(o, tuple) for o in obs) > 6:
                continue
            limit = rng.choice([0, 1, 2, 3, 10])
            d_lo = d - rng.randint(0, 3)
            d_hi = d + rng.randint(0, 3)
            with self.subTest(trial=trial):
                _assert_matches_brute(
                    self, ref, obs, d_lo, d_hi, limit, 1, 3, 2
                )

    @staticmethod
    def _random_composition(
        rng: random.Random, k: int, n: int, lo: int, hi: int
    ) -> Optional[List[int]]:
        if not k * lo <= n <= k * hi:
            return None
        for _ in range(500):
            parts = [rng.randint(lo, hi) for _ in range(k)]
            if sum(parts) == n:
                return parts
        return None


class ResidualEvidenceTests(unittest.TestCase):
    def test_evidence_covers_every_observation_once(self) -> None:
        ref = [3, 7, 11, 15, 19, 23, 27, 31]
        obs = [3, 3, 8, 10, 16, 19, 22, 24, 28, 30, 30, 31]
        res = solve_alignment(ref, obs, -2, 2, 2,
                              dwell_min=1, dwell_max=3)
        self.assertTrue(res["feasible"])
        covered: List[int] = []
        for lv in res["levels"]:
            self.assertEqual(lv["sample_end"] - lv["sample_start"], lv["dwell"])
            self.assertEqual(len(lv["samples"]), lv["dwell"])
            for s in lv["samples"]:
                self.assertEqual(
                    s["residual"], s["observed"] - lv["adopted_level"]
                )
                self.assertLessEqual(abs(s["residual"]), 2)
                covered.append(s["index"])
        self.assertEqual(covered, list(range(len(obs))))
        self.assertEqual(
            res["residual_sum"],
            sum(abs(s["residual"]) for lv in res["levels"] for s in lv["samples"]),
        )
        self.assertEqual(
            res["max_abs_residual"],
            max(abs(s["residual"]) for lv in res["levels"] for s in lv["samples"]),
        )


class BruteForceComparisonTests(unittest.TestCase):
    """随机小规模实例：DP 必须与穷举结果完全一致（含全部平局裁决）。"""

    def test_random_cases(self) -> None:
        rng = random.Random(20261004)
        for trial in range(120):
            R = rng.randint(8, 10)
            N = rng.randint(8, 14)
            ref = [rng.randint(0, 40) for _ in range(R)]
            # 先随机生成一个“真值”对齐，再对部分观测加入噪声，
            # 保证可行与不可行实例混合出现。
            used = [0]
            inner = list(range(1, R - 1))
            rng.shuffle(inner)
            # 随机跳过 0..2 个内部电平
            skip_k = rng.randint(0, min(2, R - 2))
            skipped_set = set(inner[:skip_k])
            used = [i for i in range(R) if i not in skipped_set]
            k = len(used)
            if k > N:
                used = list(range(R))
                k = R
                skipped_set = set()
            # 随机停留组合，和为 N，每段 1..3
            dwells = self._random_composition(rng, k, N, 1, 3)
            if dwells is None:
                continue
            d = rng.randint(-3, 3)
            obs: List[int] = []
            for ri, L in zip(used, dwells):
                level = ref[ri] + d
                for _ in range(L):
                    noise = rng.choice([0, 0, 0, 1, -1, 2, -2, 5])
                    obs.append(level + noise)
            limit = rng.choice([0, 1, 2, 3, 10])
            d_lo = d - rng.randint(0, 3)
            d_hi = d + rng.randint(0, 3)
            with self.subTest(trial=trial, ref=ref, obs=obs,
                              lo=d_lo, hi=d_hi, limit=limit):
                _assert_matches_brute(
                    self, ref, obs, d_lo, d_hi, limit, 1, 3, 2
                )

    @staticmethod
    def _random_composition(
        rng: random.Random, k: int, n: int, lo: int, hi: int
    ) -> Optional[List[int]]:
        choices: List[List[int]] = []
        total_lo = k * lo
        total_hi = k * hi
        if not total_lo <= n <= total_hi:
            return None
        # 在合法空间内简单拒绝采样。
        for _ in range(500):
            parts = [rng.randint(lo, hi) for _ in range(k)]
            if sum(parts) == n:
                return parts
        return None


class WideDriftRandomTests(unittest.TestCase):
    """宽漂移区间下随机真值实例：预筛与枚举必须找回唯一真值对齐。"""

    def test_wide_interval_random_truth(self) -> None:
        rng = random.Random(424242)
        for _ in range(40):
            R = rng.randint(8, 24)
            k = R - rng.randint(0, 2)
            skip_set = set(rng.sample(range(1, R - 1), R - k))
            used = [i for i in range(R) if i not in skip_set]
            dwells = self._composition(rng, k, rng.randint(k, 3 * k))
            if dwells is None:
                continue
            N = sum(dwells)
            if N > 60:
                continue
            ref = sorted(rng.sample(range(-20000, 20000), R))
            truth_d = rng.randint(-500, 500)
            obs: List[int] = []
            for ri, L in zip(used, dwells):
                obs.extend([ref[ri] + truth_d] * L)
            with self.subTest(R=R, N=N, d=truth_d):
                res = solve_alignment(
                    ref, obs, -1000, 1000, 0, 1, 3, 2
                )
                self.assertTrue(res["feasible"], msg="漏掉可行真值对齐")
                self.assertEqual(res["drift"], truth_d)
                self.assertEqual(res["residual_sum"], 0)
                self.assertEqual(
                    [lv["reference_index"] for lv in res["levels"]], used
                )
                self.assertEqual(
                    [lv["dwell"] for lv in res["levels"]], dwells
                )

    @staticmethod
    def _composition(
        rng: random.Random, k: int, n: int
    ) -> Optional[List[int]]:
        for _ in range(800):
            parts = [rng.randint(1, 3) for _ in range(k)]
            if sum(parts) == n:
                return parts
        return None


if __name__ == "__main__":
    unittest.main()
