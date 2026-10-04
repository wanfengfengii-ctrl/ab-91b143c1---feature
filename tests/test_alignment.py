"""联合对齐算法的单元测试与暴力对照测试。"""

from __future__ import annotations

import itertools
import random
import unittest
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nanopore_align.alignment import (
    MAX_INTERVAL_OBSERVATIONS,
    AlignmentError,
    solve_alignment,
)


def obs_pairs(observations: Sequence[Any]) -> List[Tuple[int, int]]:
    """混合观测归一化为 (lo, hi) 闭区间对（整数 o 视为 [o, o]）。"""
    pairs: List[Tuple[int, int]] = []
    for o in observations:
        if isinstance(o, dict):
            pairs.append((o["min"], o["max"]))
        else:
            pairs.append((o, o))
    return pairs


def interval_distance(level: int, lo: int, hi: int) -> int:
    """电平到闭区间 [lo, hi] 的最短整数距离。"""
    if level < lo:
        return lo - level
    if level > hi:
        return level - hi
    return 0


def brute_force(
    reference: Sequence[int],
    observations: Sequence[Any],
    drift_min: int,
    drift_max: int,
    residual_limit: int,
    dwell_min: int,
    dwell_max: int,
    max_skips: int,
) -> Optional[dict]:
    """穷举所有漂移 / 含首尾子序列 / 停留组合，返回与 solve 同口径的最优解。"""
    pairs = obs_pairs(observations)
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
                            lo, hi = pairs[t]
                            r = interval_distance(level, lo, hi)
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


def assert_evidence_consistent(
    testcase: unittest.TestCase,
    result: dict,
    observations: Sequence[Any],
    residual_limit: int,
) -> None:
    """逐级证据必须覆盖全部采样，并能按区间距离规则复算最终裁决。"""
    pairs = obs_pairs(observations)
    covered: List[int] = []
    total = 0
    worst = 0
    sample_map: Dict[int, Tuple[dict, int]] = {}
    for lv in result["levels"]:
        adopted = lv["adopted_level"]
        testcase.assertEqual(adopted, lv["reference_level"] + result["drift"])
        testcase.assertEqual(
            lv["sample_end"] - lv["sample_start"], lv["dwell"]
        )
        testcase.assertEqual(len(lv["samples"]), lv["dwell"])
        for s in lv["samples"]:
            t = s["index"]
            lo, hi = pairs[t]
            witness = min(max(adopted, lo), hi)
            residual = witness - adopted
            if isinstance(observations[t], dict):
                testcase.assertEqual(s["observed"], {"min": lo, "max": hi})
                testcase.assertEqual(s["witness"], witness)
            else:
                testcase.assertNotIn("witness", s)
                testcase.assertEqual(s["observed"], lo)
            testcase.assertEqual(s["residual"], residual)
            testcase.assertLessEqual(abs(residual), residual_limit)
            total += abs(residual)
            worst = max(worst, abs(residual))
            covered.append(t)
            sample_map[t] = (s, adopted)
    testcase.assertEqual(covered, list(range(len(observations))))
    testcase.assertEqual(result["residual_sum"], total)
    testcase.assertEqual(result["max_abs_residual"], worst)

    n_interval = sum(1 for o in observations if isinstance(o, dict))
    if n_interval == 0:
        testcase.assertNotIn("interval_witnesses", result)
        return
    witnesses = result.get("interval_witnesses")
    testcase.assertIsNotNone(witnesses)
    testcase.assertEqual(len(witnesses), n_interval)
    for rec in witnesses:
        t = rec["index"]
        lo, hi = pairs[t]
        testcase.assertEqual(rec["min"], lo)
        testcase.assertEqual(rec["max"], hi)
        s, adopted = sample_map[t]
        testcase.assertEqual(rec["witness"], min(max(adopted, lo), hi))
        testcase.assertEqual(rec["residual"], rec["witness"] - adopted)
        testcase.assertEqual(rec["witness"], s["witness"])
        testcase.assertEqual(rec["residual"], s["residual"])


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
    got = solve_alignment(
        reference,
        observations,
        drift_min,
        drift_max,
        residual_limit,
        dwell_min,
        dwell_max,
        max_skips,
    )
    want = brute_force(
        reference,
        observations,
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
    assert_evidence_consistent(testcase, got, observations, residual_limit)


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


class IntervalObservationTests(unittest.TestCase):
    """区间（min/max）观测的残差语义、见证值回报与输入校验。"""

    REF = [10, 20, 30, 40, 50, 60, 70, 80]

    def _solve(self, observations: List[Any], limit: int = 3, **kw: Any) -> dict:
        args: Dict[str, Any] = dict(
            reference=self.REF,
            observations=observations,
            drift_min=0,
            drift_max=0,
            residual_limit=limit,
            dwell_min=1,
            dwell_max=1,
        )
        args.update(kw)
        return solve_alignment(**args)

    def test_level_inside_interval_zero_residual(self) -> None:
        obs: List[Any] = [10, 20, {"min": 28, "max": 33}, 40, 50, 60, 70, 80]
        res = self._solve(obs)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["residual_sum"], 0)
        s = res["levels"][2]["samples"][0]
        self.assertEqual(s["observed"], {"min": 28, "max": 33})
        self.assertEqual(s["witness"], 30)
        self.assertEqual(s["residual"], 0)
        self.assertEqual(
            res["interval_witnesses"],
            [{"index": 2, "min": 28, "max": 33, "witness": 30, "residual": 0}],
        )

    def test_level_below_interval_witness_is_min(self) -> None:
        obs: List[Any] = [10, 20, {"min": 33, "max": 36}, 40, 50, 60, 70, 80]
        res = self._solve(obs)
        self.assertTrue(res["feasible"])
        s = res["levels"][2]["samples"][0]
        self.assertEqual(s["witness"], 33)
        self.assertEqual(s["residual"], 3)
        self.assertEqual(res["residual_sum"], 3)
        self.assertEqual(res["max_abs_residual"], 3)

    def test_level_above_interval_witness_is_max(self) -> None:
        obs: List[Any] = [10, 20, {"min": 24, "max": 27}, 40, 50, 60, 70, 80]
        res = self._solve(obs)
        self.assertTrue(res["feasible"])
        s = res["levels"][2]["samples"][0]
        self.assertEqual(s["witness"], 27)
        self.assertEqual(s["residual"], -3)
        self.assertEqual(res["residual_sum"], 3)

    def test_interval_distance_must_not_exceed_limit(self) -> None:
        # 电平 30 到区间 [33, 36] 的距离为 3：limit=3 可行，limit=2 无解。
        obs: List[Any] = [10, 20, {"min": 33, "max": 36}, 40, 50, 60, 70, 80]
        self.assertTrue(self._solve(obs, limit=3)["feasible"])
        res = self._solve(obs, limit=2)
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "no_alignment_exists")

    def test_interval_rescues_otherwise_impossible_exact_match(self) -> None:
        # limit=0 下任何整数观测都无法对齐（观测与电平错位），
        # 区间项因电平落在区间内（距离 0）使对齐可行。
        obs: List[Any] = [10, 20, {"min": 29, "max": 31}, 40, 50, 60, 70, 80]
        res = self._solve(obs, limit=0)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["residual_sum"], 0)

    def test_mixed_feasible_deterministic(self) -> None:
        obs: List[Any] = [
            12,
            {"min": 18, "max": 21},
            31,
            38,
            {"min": 51, "max": 54},
            58,
            71,
            79,
        ]
        res = solve_alignment(self.REF, obs, -5, 5, 2, 1, 1)
        self.assertTrue(res["feasible"])
        self.assertEqual(res["drift"], 0)
        self.assertEqual(res["residual_sum"], 10)
        self.assertEqual(res["max_abs_residual"], 2)
        self.assertEqual(
            res["interval_witnesses"],
            [
                {"index": 1, "min": 18, "max": 21, "witness": 20, "residual": 0},
                {"index": 4, "min": 51, "max": 54, "witness": 51, "residual": 1},
            ],
        )
        assert_evidence_consistent(self, res, obs, 2)

    def test_mixed_infeasible(self) -> None:
        ref = [0, 10, 20, 30, 40, 50, 60, 70]
        obs: List[Any] = [
            {"min": 900, "max": 910},
            901, 902, 903, 904, 905, 906, 907,
        ]
        res = solve_alignment(ref, obs, -5, 5, 1)
        self.assertFalse(res["feasible"])
        self.assertEqual(res["reason"], "no_alignment_exists")

    def test_max_interval_count_allowed(self) -> None:
        obs: List[Any] = [
            {"min": 10, "max": 10},
            {"min": 20, "max": 20},
            {"min": 30, "max": 30},
            {"min": 40, "max": 40},
            {"min": 50, "max": 50},
            {"min": 60, "max": 60},
            70,
            80,
        ]
        res = self._solve(obs, limit=0)
        self.assertTrue(res["feasible"])
        self.assertEqual(len(res["interval_witnesses"]), 6)

    def test_too_many_intervals_rejected(self) -> None:
        obs: List[Any] = [
            {"min": v, "max": v}
            for v in (10, 20, 30, 40, 50, 60, 70)
        ] + [80]
        self.assertEqual(
            sum(1 for o in obs if isinstance(o, dict)),
            MAX_INTERVAL_OBSERVATIONS + 1,
        )
        with self.assertRaises(AlignmentError) as ctx:
            self._solve(obs)
        self.assertIn(str(MAX_INTERVAL_OBSERVATIONS), str(ctx.exception))
        self.assertEqual(ctx.exception.field, "observations")

    def test_inverted_interval_rejected(self) -> None:
        obs: List[Any] = [10, 20, {"min": 35, "max": 32}, 40, 50, 60, 70, 80]
        with self.assertRaises(AlignmentError) as ctx:
            self._solve(obs)
        self.assertEqual(ctx.exception.field, "observations[2]")

    def test_missing_bound_rejected(self) -> None:
        for bad in ({"min": 30}, {"max": 30}, {}):
            obs: List[Any] = [10, 20, bad, 40, 50, 60, 70, 80]
            with self.assertRaises(AlignmentError) as ctx:
                self._solve(obs)
            self.assertEqual(ctx.exception.field, "observations[2]")

    def test_unknown_interval_key_rejected(self) -> None:
        obs: List[Any] = [
            10, 20, {"min": 30, "max": 30, "mean": 30}, 40, 50, 60, 70, 80,
        ]
        with self.assertRaises(AlignmentError) as ctx:
            self._solve(obs)
        self.assertEqual(ctx.exception.field, "observations[2]")

    def test_non_integer_bound_rejected(self) -> None:
        for bad in ({"min": 30.5, "max": 31}, {"min": 30, "max": True}):
            obs: List[Any] = [10, 20, bad, 40, 50, 60, 70, 80]
            with self.assertRaises(AlignmentError):
                self._solve(obs)

    def test_bad_observation_type_rejected(self) -> None:
        for bad in ("30", None, [30], 30.5):
            obs: List[Any] = [10, 20, bad, 40, 50, 60, 70, 80]
            with self.assertRaises(AlignmentError):
                self._solve(obs)

    def test_interval_bound_out_of_range_rejected(self) -> None:
        obs: List[Any] = [
            10, 20, {"min": -2_000_000_000, "max": 30}, 40, 50, 60, 70, 80,
        ]
        with self.assertRaises(AlignmentError):
            self._solve(obs)

    def test_pure_integer_response_has_no_interval_fields(self) -> None:
        obs = [15, 25, 35, 45, 55, 65, 75, 85]
        res = solve_alignment(self.REF, obs, -10, 10, 2, 1, 1)
        self.assertTrue(res["feasible"])
        self.assertNotIn("interval_witnesses", res)
        for lv in res["levels"]:
            for s in lv["samples"]:
                self.assertEqual(set(s), {"index", "observed", "residual"})


class MixedBruteForceComparisonTests(unittest.TestCase):
    """随机混合（整数 + 区间）实例：DP 必须与穷举结果完全一致。"""

    def test_random_mixed_cases(self) -> None:
        rng = random.Random(20261005)
        for trial in range(120):
            R = rng.randint(8, 10)
            N = rng.randint(8, 14)
            ref = [rng.randint(0, 40) for _ in range(R)]
            inner = list(range(1, R - 1))
            rng.shuffle(inner)
            skip_k = rng.randint(0, min(2, R - 2))
            skipped_set = set(inner[:skip_k])
            used = [i for i in range(R) if i not in skipped_set]
            k = len(used)
            if k > N:
                used = list(range(R))
                k = R
            dwells = BruteForceComparisonTests._random_composition(
                rng, k, N, 1, 3
            )
            if dwells is None:
                continue
            d = rng.randint(-3, 3)
            obs: List[Any] = []
            interval_left = MAX_INTERVAL_OBSERVATIONS
            for ri, L in zip(used, dwells):
                level = ref[ri] + d
                for _ in range(L):
                    noise = rng.choice([0, 0, 0, 1, -1, 2, -2, 5])
                    center = level + noise
                    if interval_left > 0 and rng.random() < 0.3:
                        # 区间项：以带噪真值为中心的随机宽度闭区间。
                        a = rng.randint(0, 2)
                        b = rng.randint(0, 2)
                        obs.append({"min": center - a, "max": center + b})
                        interval_left -= 1
                    else:
                        obs.append(center)
            limit = rng.choice([0, 1, 2, 3, 10])
            d_lo = d - rng.randint(0, 3)
            d_hi = d + rng.randint(0, 3)
            with self.subTest(trial=trial, ref=ref, obs=obs,
                              lo=d_lo, hi=d_hi, limit=limit):
                _assert_matches_brute(
                    self, ref, obs, d_lo, d_hi, limit, 1, 3, 2
                )


if __name__ == "__main__":
    unittest.main()
