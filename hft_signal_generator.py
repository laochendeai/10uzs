# -*- coding: utf-8 -*-
"""
极简趋势跟随+订单簿确认信号生成器。

条件上限：三条核心约束
1. 双均线方向（金叉做多、死叉做空）；
2. 最近成交量需高于历史均值一定比例；
3. 订单簿不平衡需与方向一致（可配置）。
"""

import time
from datetime import datetime
from typing import Deque, Dict, Optional


class HFTSignalGenerator:
    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 60,
        volume_window: int = 40,
        recent_volume_ticks: int = 8,
        volume_ratio_threshold: float = 1.05,
        min_confidence: float = 0.6,
        orderbook_imbalance_min: float = 0.08,
        require_orderbook: bool = True,
        min_diff: float = 0.0004,
        cooldown_seconds: float = 4.0,
        use_cross_confirmation: bool = True,
        signal_mode: str = 'ma_follow',
    ):
        self.fast_window = max(fast_window, 5)
        self.slow_window = max(slow_window, self.fast_window + 1)
        self.volume_window = max(volume_window, 10)
        self.recent_volume_ticks = max(recent_volume_ticks, 1)
        self.volume_ratio_threshold = max(volume_ratio_threshold, 0.5)
        self.min_confidence = max(min_confidence, 0.5)
        self.orderbook_imbalance_min = max(orderbook_imbalance_min, 0.0)
        self.require_orderbook = require_orderbook
        self.min_diff = max(min_diff, 0.0)
        self.cooldown_seconds = max(cooldown_seconds, 0.0)
        self.use_cross_confirmation = use_cross_confirmation
        self.signal_mode = signal_mode
        self.last_debug: Dict[str, any] = {}

        # 兼容 TrueHFTEngine 其他模块引用的属性
        self.momentum_window = self.fast_window
        self.momentum_threshold = self.min_diff
        self.entry_threshold = self.min_diff

        self._prev_diff: Optional[float] = None
        self._last_signal_ts: float = 0.0

    def _avg(self, values):
        return sum(values) / len(values) if values else 0.0

    def _calc_volume_ratio(self, ticks: Deque[Dict]) -> float:
        volumes = [float(t.get('volume') or t.get('size') or 0.0) for t in list(ticks)[-self.volume_window:]]
        if not volumes:
            return 0.0
        avg_volume = self._avg(volumes)
        recent_volume = sum(volumes[-self.recent_volume_ticks:]) if volumes else 0.0
        return recent_volume / avg_volume if avg_volume > 1e-12 else 0.0

    def generate_tick_signal(
        self,
        ticks: Deque[Dict],
        trend_bias: Optional[float] = None,
        orderbook: Optional[Dict] = None,
        higher_timeframes: Optional[Dict] = None,
    ) -> Optional[Dict]:
        required_ticks = max(self.slow_window, self.volume_window)
        if len(ticks) < required_ticks:
            self.last_debug = {
                'reason': 'not_enough_ticks',
                'required': required_ticks,
                'have': len(ticks),
            }
            return None

        prices = [float(t.get('price') or 0.0) for t in list(ticks)[-self.slow_window:]]
        if any(p <= 0 for p in prices):
            self.last_debug = {'reason': 'invalid_price'}
            return None

        fast_ma = self._avg(prices[-self.fast_window:])
        slow_ma = self._avg(prices)
        diff = fast_ma - slow_ma
        diff_ratio = diff / max(abs(slow_ma), 1e-8)
        book_imbalance = None
        if orderbook is not None:
            book_imbalance = orderbook.get('imbalance')

        direction: Optional[str] = None
        trend_reason = 'no_trend'
        if self.signal_mode == 'orderbook_imbalance':
            if book_imbalance is not None:
                if book_imbalance >= self.orderbook_imbalance_min:
                    direction = 'long'
                    trend_reason = None
                elif book_imbalance <= -self.orderbook_imbalance_min:
                    direction = 'short'
                    trend_reason = None
                else:
                    trend_reason = 'orderbook_neutral'
        else:
            if diff_ratio >= self.min_diff:
                direction = 'long'
                trend_reason = None
            elif diff_ratio <= -self.min_diff:
                direction = 'short'
                trend_reason = None

        cross_ok = True
        if self.signal_mode != 'orderbook_imbalance' and self.use_cross_confirmation and direction:
            if direction == 'long':
                cross_ok = self._prev_diff is not None and self._prev_diff <= 0
            else:
                cross_ok = self._prev_diff is not None and self._prev_diff >= 0

        volume_ratio = self._calc_volume_ratio(ticks)
        volume_ok = volume_ratio >= self.volume_ratio_threshold

        orderbook_ok = True
        if self.signal_mode == 'orderbook_imbalance':
            orderbook_ok = direction is not None
        elif self.require_orderbook:
            if book_imbalance is None:
                orderbook_ok = False
            elif direction == 'long':
                orderbook_ok = book_imbalance >= self.orderbook_imbalance_min
            elif direction == 'short':
                orderbook_ok = book_imbalance <= -self.orderbook_imbalance_min

        tick_timestamp = ticks[-1].get('timestamp')
        if isinstance(tick_timestamp, datetime):
            now_ts = tick_timestamp.timestamp()
        else:
            now_ts = time.time()
        cooldown_ok = now_ts - self._last_signal_ts >= self.cooldown_seconds

        self.last_debug = {
            'reason': None,
            'direction': direction,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'diff_ratio': diff_ratio,
            'prev_diff': self._prev_diff,
            'volume_ratio': volume_ratio,
            'volume_ok': volume_ok,
            'orderbook_imbalance': book_imbalance,
            'orderbook_ok': orderbook_ok,
            'cooldown_ok': cooldown_ok,
            'trend_bias': trend_bias,
            'signal_mode': self.signal_mode,
            'trend_reason': trend_reason,
        }

        self._prev_diff = diff_ratio

        if not direction:
            self.last_debug['reason'] = trend_reason
            return None
        if self.signal_mode != 'orderbook_imbalance' and self.use_cross_confirmation and not cross_ok:
            self.last_debug['reason'] = 'no_cross'
            return None
        if not volume_ok:
            self.last_debug['reason'] = 'volume_insufficient'
            return None
        if not orderbook_ok:
            self.last_debug['reason'] = 'orderbook_reject'
            return None
        if not cooldown_ok:
            self.last_debug['reason'] = 'cooldown'
            return None

        bias_component = trend_bias or 0.0
        if direction == 'short':
            bias_component = -bias_component
        bias_penalty = 0.03 if bias_component < -0.05 else 0.0

        confidence = self.min_confidence
        intensity = abs(book_imbalance) if self.signal_mode == 'orderbook_imbalance' and book_imbalance is not None else abs(diff_ratio)
        confidence += min(intensity * 8, 0.25)
        confidence += min(max(volume_ratio - self.volume_ratio_threshold, 0.0), 1.0) * 0.1
        confidence = max(self.min_confidence, min(0.98, confidence - bias_penalty))

        timestamp = tick_timestamp
        if not isinstance(timestamp, datetime):
            timestamp = datetime.utcnow()

        self.last_debug['reason'] = 'passed'
        self._last_signal_ts = now_ts

        if self.signal_mode == 'orderbook_imbalance':
            momentum = book_imbalance if direction == 'long' else -book_imbalance if book_imbalance is not None else 0.0
        else:
            momentum = diff_ratio if direction == 'long' else -diff_ratio
        votes = {
            'ma_cross': self.signal_mode != 'orderbook_imbalance',
            'volume': volume_ok,
            'orderbook': orderbook_ok,
            'orderbook_imbalance': self.signal_mode == 'orderbook_imbalance',
        }

        return {
            'direction': direction,
            'confidence': confidence,
            'trend_diff': diff_ratio,
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'volume_ratio': volume_ratio,
            'orderbook_imbalance': book_imbalance,
            'momentum': momentum,
            'market_state': 'trend_follow',
            'timestamp': timestamp,
            'type': 'trend_follow_ma',
            'votes': votes,
            'debug': dict(self.last_debug),
        }
