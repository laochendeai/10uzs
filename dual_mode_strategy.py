# -*- coding: utf-8 -*-
"""
双模式自适应交易核心（箱体网格 + 突破趋势）

仅生成决策意图，不直接下单，调用方负责执行/风控。
依赖：pattern_detection.detect_box_range / detect_support_resistance_levels / detect_swings
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import math

from config import DUAL_MODE_CONFIG
from pattern_detection import (
    detect_box_range,
    detect_support_resistance_levels,
    detect_swings,
)


@dataclass
class ModeState:
    mode: str = "grid"  # grid | trend_long | trend_short
    box: Optional[Dict[str, float]] = None
    levels: List[float] = field(default_factory=list)
    last_break_dir: Optional[str] = None
    last_break_bar: Optional[int] = None
    trail_stop: Optional[float] = None


class DualModeStrategy:
    def __init__(self, config: Optional[Dict] = None):
        self.cfg = dict(DUAL_MODE_CONFIG)
        if config:
            self.cfg.update(config)
        self.state = ModeState()

    def _compute_atr(self, highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> float:
        n = len(closes)
        if n < self.cfg["atr_window"] + 1:
            return 0.0
        window = self.cfg["atr_window"]
        trs: List[float] = []
        for i in range(n - window, n):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            trs.append(tr)
        return sum(trs) / window if trs else 0.0

    def _build_grid_levels(self, box: Dict[str, float], atr: float) -> Tuple[List[float], List[float]]:
        s = box["support"]
        r = box["resistance"]
        h = r - s
        # 动态高度：使用 ATR 放大，避免过窄
        target_h = max(h, atr * self.cfg["atr_mult_grid"])
        mid = (s + r) / 2

        # 基于手续费+滑点+最小间距过滤网格层，避免过密
        min_spacing_pct = max(
            self.cfg.get("grid_min_spacing_pct", 0.0),
            self.cfg.get("grid_fee_buffer_pct", 0.0) + self.cfg.get("grid_slippage_pct", 0.0)
        )
        min_spacing_abs = mid * min_spacing_pct if mid else 0.0
        if target_h < 2 * min_spacing_abs:
            # 箱体太窄不足以覆盖费用/滑点，放弃网格
            return [], []

        levels: List[float] = []
        weights: List[float] = []
        for rel, w in zip(self.cfg["grid_levels"], self.cfg["grid_weights"]):
            levels.append(mid - target_h * rel)  # 下侧做多
            levels.append(mid + target_h * rel)  # 上侧做空
            weights.extend([w, w])
        # 边界权重再加强
        levels.append(s)
        levels.append(r)
        weights.extend([self.cfg["grid_weights"][0], self.cfg["grid_weights"][0]])
        # 去重、排序、最小间距过滤
        combined = []
        weights_sorted = []
        for lvl, w in sorted(zip(levels, weights), key=lambda x: x[0]):
            if combined and abs(lvl - combined[-1]) < min_spacing_abs:
                continue
            combined.append(lvl)
            weights_sorted.append(w)
        return combined, weights_sorted

    def _detect_box(self, highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], atr: float = 0.0) -> Optional[Dict[str, float]]:
        n = len(closes)
        # 0) 视觉式判定：固定窗口 + 裁剪均值，更接近人工目测
        visual_box = self._detect_visual_box(highs, lows, closes)
        if visual_box:
            return visual_box

        min_lb = max(int(self.cfg.get("box_dynamic_min_lookback", self.cfg["box_lookback"])), self.cfg["box_min_bars"])
        max_lb = max(int(self.cfg.get("box_dynamic_max_lookback", min_lb)), min_lb)
        step = max(int(self.cfg.get("box_dynamic_step", 30)), 1)
        if n < min_lb:
            return None

        # 1) 结构化判定：分型/波段构造箱体，免参数
        swings = detect_swings(
            closes,
            change_threshold=self.cfg.get("swing_change_threshold", 0.004),
            min_bars_between=self.cfg.get("swing_min_bars_between", 3),
            max_pivots=self.cfg.get("swing_max_pivots", 10),
            atr_series=None,
            atr_mult=0.0,
        )
        tops = [s for s in swings if s["type"] == "high"]
        bottoms = [s for s in swings if s["type"] == "low"]
        if len(tops) >= 2 and len(bottoms) >= 2:
            last_highs = tops[-2:]
            last_lows = bottoms[-2:]
            res = sum(h["price"] for h in last_highs) / len(last_highs)
            sup = sum(l["price"] for l in last_lows) / len(last_lows)
            if res > sup > 0:
                mid = (res + sup) / 2
                height_pct = (res - sup) / max(abs(mid), 1e-12)
                tol = max(
                    self.cfg["box_tol_pct"],
                    (self.cfg.get("box_atr_mult", 0.0) * atr / max(abs(mid), 1e-12)) if atr > 0 else 0.0
                )
                if height_pct <= tol:
                    return {
                        "support": sup,
                        "resistance": res,
                        "mid": mid,
                        "height_pct": height_pct,
                        "slope_pct": 0.0,
                        "bars": n,
                        "tolerance_used": tol,
                        "lookback_used": n,
                        "source": "pivots"
                    }

        # 2) 动态窗口搜索（高/低参与），从最小窗口开始放大
        atr_series = [atr] * n if atr > 0 and self.cfg.get("box_atr_mult", 0) > 0 else None
        atr_series = None
        if atr > 0 and self.cfg.get("box_atr_mult", 0) > 0:
            atr_series = [atr] * len(closes)

        for lb in range(min_lb, min(max_lb, n) + 1, step):
            tolerance = self.cfg["box_tol_pct"]
            box = detect_box_range(
                closes,
                lookback=lb,
                tolerance_pct=tolerance,
                min_bars=lb,
                use_quantile=True,
                q=self.cfg["box_quantile"],
                max_slope_pct=self.cfg["box_max_slope_pct"],
                atr_series=atr_series,
                atr_mult=self.cfg.get("box_atr_mult", 1.0),
                highs=highs,
                lows=lows
            )
            if box:
                box["lookback_used"] = lb
                box["tolerance_used"] = box.get("tolerance_used", tolerance)
                return box

        # 3) 最后兜底：用 min_lb 窗口的高/低分位数直接构造箱体，放宽容差
        lb = min(min_lb, n)
        def _quantile(vals: Sequence[float], prob: float) -> float:
            if not vals:
                return 0.0
            vals_sorted = sorted(vals)
            k = max(0, min(len(vals_sorted) - 1, int(prob * (len(vals_sorted) - 1))))
            return vals_sorted[k]

        highs_window = highs[-lb:]
        lows_window = lows[-lb:]
        closes_window = closes[-lb:]
        if highs_window and lows_window:
            high = _quantile(highs_window, 1 - self.cfg["box_quantile"])
            low = _quantile(lows_window, self.cfg["box_quantile"])
            mid = (high + low) / 2 if (high + low) else 0.0
            height_pct = (high - low) / max(abs(mid), 1e-12)
            slope_pct = 0.0
            # 粗略坡度用收盘回归
            n_window = len(closes_window)
            if n_window >= 2:
                x_mean = (n_window - 1) / 2
                y_mean = sum(closes_window) / n_window
                num = 0.0
                den = 0.0
                for i, p in enumerate(closes_window):
                    num += (i - x_mean) * (p - y_mean)
                    den += (i - x_mean) ** 2
                slope = num / den if den != 0 else 0.0
                slope_pct = slope / max(abs(mid), 1e-12)
            tol = max(
                self.cfg["box_tol_pct"] * 2,
                (self.cfg.get("box_atr_mult", 0.0) * atr / max(abs(mid), 1e-12)) if atr > 0 else 0.0
            )
            if height_pct <= tol:
                return {
                    "support": low,
                    "resistance": high,
                    "mid": mid,
                    "height_pct": height_pct,
                    "slope_pct": slope_pct,
                    "bars": lb,
                    "tolerance_used": tol,
                    "lookback_used": lb,
                    "source": "fallback_visual"
                }

        return None

    def _volume_ok(self, volumes: Sequence[float], ratio: float) -> bool:
        if len(volumes) < 30:
            return True
        recent = volumes[-1]
        avg = sum(volumes[-30:]) / 30
        return avg > 0 and recent / avg >= ratio

    def _confirm_break(self, price: float, box: Dict[str, float], direction: str, volumes: Sequence[float]) -> bool:
        if direction == "up" and price <= box["resistance"]:
            return False
        if direction == "down" and price >= box["support"]:
            return False
        return self._volume_ok(volumes, self.cfg["trend_break_vol_ratio"])

    def update(self, highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], volumes: Sequence[float], bar_index: int) -> Dict:
        """
        输入最新的高低收和成交量序列，返回当前模式下的意图：
        {'mode': 'grid'|'trend_long'|'trend_short', 'actions': [...], 'state': {...}}
        actions 示例：
        - grid: {'type':'grid_entry','side':'buy'|'sell','price':level,'weight':w}
        - trend: {'type':'trend_entry','side':'buy'|'sell','stop':stop_price,'trail':trail_price}
        """
        atr = self._compute_atr(highs, lows, closes)
        price = closes[-1]
        actions: List[Dict] = []

        # 箱体判定
        box = self._detect_box(highs, lows, closes, atr=atr)
        if box:
            self.state.box = box

        # 模式切换逻辑
        if self.state.mode == "grid":
            if self.state.box and self._confirm_break(price, self.state.box, "up", volumes):
                self.state.mode = "trend_long"
                self.state.last_break_dir = "up"
                self.state.last_break_bar = bar_index
                self.state.trail_stop = price - atr * self.cfg["atr_mult_trend_sl"]
            elif self.state.box and self._confirm_break(price, self.state.box, "down", volumes):
                self.state.mode = "trend_short"
                self.state.last_break_dir = "down"
                self.state.last_break_bar = bar_index
                self.state.trail_stop = price + atr * self.cfg["atr_mult_trend_sl"]
        else:
            # 回归箱体则回到 grid
            if self.state.box and self.state.box["support"] <= price <= self.state.box["resistance"]:
                self.state.mode = "grid"
                self.state.trail_stop = None
                self.state.last_break_dir = None

        # 行为生成
        if self.state.mode == "grid" and self.state.box:
            levels, weights = self._build_grid_levels(self.state.box, atr)
            for lvl, w in zip(levels, weights):
                if price <= lvl:  # 下侧做多
                    actions.append({"type": "grid_entry", "side": "buy", "price": lvl, "weight": w})
                elif price >= lvl:  # 上侧做空
                    actions.append({"type": "grid_entry", "side": "sell", "price": lvl, "weight": w})
        elif self.state.mode == "trend_long" and self.state.box:
            stop = self.state.box["resistance"] - atr * self.cfg["atr_mult_trend_sl"]
            trail = max(self.state.trail_stop or stop, price - atr * self.cfg["atr_mult_trend_trail"])
            tp = price * (1 + self.cfg["trend_tp_pct"])
            actions.append({"type": "trend_entry", "side": "buy", "stop": stop, "trail": trail, "take_profit": tp})
            self.state.trail_stop = trail
        elif self.state.mode == "trend_short" and self.state.box:
            stop = self.state.box["support"] + atr * self.cfg["atr_mult_trend_sl"]
            trail = min(self.state.trail_stop or stop, price + atr * self.cfg["atr_mult_trend_trail"])
            tp = price * (1 - self.cfg["trend_tp_pct"])
            actions.append({"type": "trend_entry", "side": "sell", "stop": stop, "trail": trail, "take_profit": tp})
            self.state.trail_stop = trail

        return {
            "mode": self.state.mode,
            "box": self.state.box,
            "atr": atr,
            "actions": actions,
            "state": self.state,
        }

    def _detect_visual_box(self, highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> Optional[Dict[str, float]]:
        """简化的“目测”箱体：固定窗口 + 剪裁均值 + 坡度与触碰约束。"""
        n = len(closes)
        lookback = int(self.cfg.get("box_visual_lookback", 180))
        if n < max(lookback, 60):
            return None
        window = min(lookback, n)
        lows_w = list(lows)[-window:]
        highs_w = list(highs)[-window:]
        closes_w = list(closes)[-window:]

        def _quantile(vals, q: float) -> float:
            if not vals:
                return 0.0
            vs = sorted(vals)
            k = max(0, min(len(vs) - 1, int(q * (len(vs) - 1))))
            return vs[k]

        q = float(self.cfg.get("box_visual_quantile", 0.1))
        support = _quantile(lows_w, q)
        resistance = _quantile(highs_w, 1 - q)
        if resistance <= support:
            return None
        mid = (support + resistance) / 2
        height_pct = (resistance - support) / max(abs(mid), 1e-12)
        max_height = float(self.cfg.get("box_visual_max_height_pct", 0.025))
        if height_pct > max_height:
            return None

        # 坡度约束：中轴回归斜率
        x_mean = (window - 1) / 2
        y_mean = sum(closes_w) / window
        num = sum((i - x_mean) * (p - y_mean) for i, p in enumerate(closes_w))
        den = sum((i - x_mean) ** 2 for i in range(window))
        slope = num / den if den else 0.0
        slope_pct = slope / max(abs(mid), 1e-12)
        if abs(slope_pct) > float(self.cfg.get("box_visual_max_slope_pct", 0.0015)):
            return None

        # 触碰次数：上/下沿各需至少 N 次触碰
        tol = max(
            float(self.cfg.get("box_visual_touch_tol_pct", 0.0015)) * abs(mid),
            float(self.cfg.get("box_tol_pct", 0.0)) * abs(mid)
        )
        min_touches = int(self.cfg.get("box_visual_min_touches", 1))
        touch_low = sum(1 for p in closes_w if abs(p - support) <= tol)
        touch_high = sum(1 for p in closes_w if abs(p - resistance) <= tol)
        if touch_low < min_touches or touch_high < min_touches:
            return None

        return {
            "support": support,
            "resistance": resistance,
            "mid": mid,
            "height_pct": height_pct,
            "slope_pct": slope_pct,
            "bars": window,
            "tolerance_used": tol / max(abs(mid), 1e-12),
            "lookback_used": window,
            "source": "visual_trimmed"
        }
