# -*- coding: utf-8 -*-
"""
高频信号生成器
"""

from datetime import datetime, timedelta
from typing import Optional, List, Deque, Dict


class HFTSignalGenerator:
    def __init__(self, momentum_window: int = 10,
                 momentum_threshold: float = 0.0008,
                 volume_threshold: float = 2.5,
                 imbalance_threshold: float = 0.6,
                 enable_volatility_filter: bool = False,
                 volatility_threshold: float = 0.001,
                 require_orderbook_confirm: bool = False,
                 orderbook_ratio_threshold: float = 0.8,
                 avoid_funding_hours: bool = False,
                 funding_rate_times: Optional[List[int]] = None,
                 entry_threshold: float = 0.2,
                 market_volatility_threshold: float = 0.0003,
                 min_confidence: float = 0.5):
        self.momentum_window = momentum_window
        self.momentum_threshold = momentum_threshold
        self.volume_threshold = volume_threshold
        self.imbalance_threshold = imbalance_threshold
        self.enable_volatility_filter = enable_volatility_filter
        self.volatility_threshold = volatility_threshold
        self.require_orderbook_confirm = require_orderbook_confirm
        self.orderbook_ratio_threshold = orderbook_ratio_threshold
        self.avoid_funding_hours = avoid_funding_hours
        self.funding_rate_times = funding_rate_times or []
        self.entry_threshold = entry_threshold
        self.market_volatility_threshold = market_volatility_threshold
        self.min_confidence = min_confidence
        self.last_debug: Dict[str, any] = {}

    def generate_tick_signal(self, ticks: Deque[Dict], trend_bias: Optional[float] = None,
                             orderbook: Optional[Dict] = None,
                             higher_timeframes: Optional[Dict[str, Deque[Dict]]] = None) -> Optional[Dict]:
        self.last_debug = {}
        if len(ticks) < self.momentum_window:
            self.last_debug = {'reason': 'not_enough_ticks'}
            return None

        recent = list(ticks)[-self.momentum_window:]
        price_momentum = self._calculate_weighted_momentum(recent)
        volume_surge = self._detect_volume_surge(recent)
        order_imbalance = self._calculate_order_imbalance(recent)

        momentum_ok = abs(price_momentum) >= self.momentum_threshold
        volume_ok = volume_surge >= self.volume_threshold
        imbalance_ok = abs(order_imbalance) >= self.imbalance_threshold

        composite_info = self._calc_composite_trend(
            ticks,
            orderbook=orderbook,
            higher_timeframes=higher_timeframes or {},
            trend_bias=trend_bias
        )
        composite_trend = composite_info['score']
        orderbook_metrics = composite_info.get('orderbook_metrics')
        market_state = self._assess_market_state(
            ticks,
            orderbook_metrics,
            price_momentum
        )

        desired_direction: Optional[str] = None
        if market_state == "range":
            dynamic_threshold = self.entry_threshold * 1.5
        else:
            dynamic_threshold = self.entry_threshold

        long_threshold = dynamic_threshold
        short_threshold = dynamic_threshold * 0.9
        if composite_trend >= long_threshold:
            desired_direction = 'long'
        elif composite_trend <= -short_threshold:
            desired_direction = 'short'

        direction_consistent = self._is_direction_consistent(
            desired_direction,
            price_momentum,
            order_imbalance,
            composite_info,
            trend_bias
        )

        recent_volatility = self._recent_volatility(ticks)
        adaptive_l1_threshold = self._adaptive_momentum_threshold(
            market_state,
            composite_trend,
            recent_volatility
        )
        l1_pass = abs(price_momentum) >= adaptive_l1_threshold
        l2_pass = self._confirm_short_term_breakout(ticks, desired_direction)
        l3_pass = market_state != "idle"

        condition_count = sum([momentum_ok, volume_ok, imbalance_ok])
        self.last_debug = {
            'momentum_ok': momentum_ok,
            'volume_ok': volume_ok,
            'imbalance_ok': imbalance_ok,
            'condition_count': condition_count,
            'desired_direction': desired_direction,
            'direction_consistent': direction_consistent,
            'l1': l1_pass,
            'l2': l2_pass,
            'l3': l3_pass,
            'market_state': market_state,
            'composite': composite_trend,
            'l1_threshold': adaptive_l1_threshold,
            'recent_volatility': recent_volatility
        }

        if all([l1_pass, l2_pass, l3_pass]) and condition_count >= 2 and desired_direction and direction_consistent:
            direction = desired_direction
            if price_momentum * (1 if direction == 'long' else -1) < 0:
                self.last_debug['reason'] = 'momentum_mismatch'
                return None

            confidence_penalty = 0.0
            if trend_bias is not None:
                signal_direction = 1 if direction == 'long' else -1
                alignment = trend_bias * signal_direction
                if abs(trend_bias) > 0.5 and alignment < 0:
                    return None
                if alignment < 0:
                    confidence_penalty = 0.2

            confidence = min(abs(price_momentum) * 800 + max(volume_surge - 1.0, 0), 0.95)
            confidence = max(0.0, confidence - confidence_penalty)
            if confidence < self.min_confidence:
                return None

            if self.enable_volatility_filter and self._is_high_volatility(ticks):
                return None

            if self.require_orderbook_confirm and orderbook:
                if not self._confirm_with_orderbook(direction, orderbook):
                    return None

            if self.avoid_funding_hours and self._is_funding_time(datetime.utcnow()):
                self.last_debug['reason'] = 'funding_time'
                return None

            self.last_debug['reason'] = 'passed'
            return {
                'direction': direction,
                'confidence': confidence,
                'momentum': price_momentum,
                'volume_ratio': volume_surge,
                'order_imbalance': order_imbalance,
                'composite_trend': composite_trend,
                'composite_trend_breakdown': composite_info,
                'market_state': market_state,
                'timestamp': recent[-1]['timestamp'],
                'type': 'hft_tick_breakout'
            }
        if 'reason' not in self.last_debug:
            if not l1_pass:
                self.last_debug['reason'] = 'L1'
            elif not l2_pass:
                self.last_debug['reason'] = 'L2'
            elif not l3_pass:
                self.last_debug['reason'] = 'L3'
            elif not desired_direction:
                self.last_debug['reason'] = 'no_direction'
            elif not direction_consistent:
                self.last_debug['reason'] = 'direction_conflict'
            elif condition_count < 2:
                self.last_debug['reason'] = 'base_filters'
            else:
                self.last_debug['reason'] = 'unknown'
        return None

    def _calc_composite_trend(self, ticks: Deque[Dict], orderbook: Optional[Dict],
                              higher_timeframes: Dict[str, Deque[Dict]],
                              trend_bias: Optional[float] = None) -> Dict[str, float]:
        metrics = higher_timeframes.get('orderbook_metrics') or {}
        ob_score = self._orderbook_imbalance_score(metrics or orderbook)
        momentum_component = self._momentum_component(ticks)
        micro_state = self._microstructure_score(metrics, ticks)
        trade_quality = self._trade_quality_score(ticks)
        weights = {
            'orderbook': 0.25 * (1.5 if ob_score < 0 else 0.9),
            'momentum': 0.25 * (1.4 if momentum_component < 0 else 0.95),
            'micro_state': 0.25 * (1.3 if micro_state < 0 else 1.0),
            'trade_quality': 0.15 * (1.2 if trade_quality < 0 else 0.9),
        }
        weighted_sum = (
            weights['orderbook'] * ob_score +
            weights['momentum'] * momentum_component +
            weights['micro_state'] * micro_state +
            weights['trade_quality'] * trade_quality
        )
        weight_total = sum(weights.values()) or 1.0
        base_score = weighted_sum / weight_total
        trend_component = max(min(trend_bias or 0.0, 1.0), -1.0) * 0.5
        score = max(min(base_score + trend_component, 1.0), -1.0)
        return {
            'score': score,
            'base_score': base_score,
            'trend_bias_component': trend_component,
            'orderbook': ob_score,
            'momentum': momentum_component,
            'micro_state': micro_state,
            'trade_quality': trade_quality,
            'orderbook_metrics': metrics
        }

    def _orderbook_imbalance_score(self, orderbook: Optional[Dict]) -> float:
        if not orderbook:
            return 0.0
        bid = orderbook.get('bid_volume_top3') or orderbook.get('liquidity') or 0.0
        ask = orderbook.get('ask_volume_top3') or 0.0
        total = bid + ask
        if total <= 0:
            return 0.0
        imbalance = (bid - ask) / total
        return max(min(imbalance, 1.0), -1.0)

    def _volume_weighted_trend(self, ticks: Deque[Dict]) -> float:
        if len(ticks) < 3:
            return 0.0
        vwap_changes = []
        prev_vwap = None
        window = list(ticks)[-self.momentum_window:]
        for i in range(1, len(window) + 1):
            subset = window[:i]
            total_vol = sum(t.get('volume', 0.0) for t in subset)
            if total_vol <= 0:
                continue
            vwap = sum(t['price'] * t.get('volume', 0.0) for t in subset) / total_vol
            if prev_vwap is not None and prev_vwap > 0:
                vwap_changes.append((vwap - prev_vwap) / prev_vwap)
            prev_vwap = vwap
        if not vwap_changes:
            return 0.0
        normalized = sum(vwap_changes) / len(vwap_changes)
        return max(min(normalized * 500, 1.0), -1.0)

    def _ema_trend(self, bars: Optional[Deque[Dict]]) -> float:
        if not bars or len(bars) < 3:
            return 0.0
        closes = [bar['close'] for bar in list(bars)[-5:]]
        alpha = 2 / (len(closes) + 1)
        ema = closes[0]
        for price in closes[1:]:
            ema = alpha * price + (1 - alpha) * ema
        previous = closes[-2]
        if previous <= 0:
            return 0.0
        change = (ema - previous) / previous
        return max(min(change * 1000, 1.0), -1.0)

    def _assess_market_state(self, ticks: Deque[Dict], orderbook: Optional[Dict] = None,
                              momentum: Optional[float] = None) -> str:
        if len(ticks) < self.momentum_window:
            return "idle"
        recent_ticks = list(ticks)[-self.momentum_window:]
        prices = [tick['price'] for tick in recent_ticks]
        changes = [abs(prices[i] - prices[i - 1]) / max(prices[i - 1], 1e-8)
                   for i in range(1, len(prices)) if prices[i - 1] > 0]
        avg_volatility = sum(changes) / len(changes) if changes else 0.0

        instant_spike = abs(recent_ticks[-1]['price'] - recent_ticks[-2]['price']) / max(recent_ticks[-2]['price'], 1e-8)
        orderbook_push = abs(self._orderbook_imbalance_score(orderbook))
        liquidity = (orderbook or {}).get('liquidity', 0.0)
        if avg_volatility >= self.market_volatility_threshold or instant_spike >= self.market_volatility_threshold * 2:
            return "trending"
        if orderbook_push > 0.6 or (momentum and abs(momentum) > self.momentum_threshold * 2) or liquidity < 500:
            return "trending"
        if avg_volatility < self.market_volatility_threshold / 3 and orderbook_push < 0.2:
            return "range"
        return "range"

    def _recent_volatility(self, ticks: Deque[Dict]) -> float:
        if len(ticks) < 3:
            return 0.0
        prices = [tick['price'] for tick in list(ticks)[-self.momentum_window:]]
        changes = [
            abs(prices[i] - prices[i - 1]) / max(prices[i - 1], 1e-8)
            for i in range(1, len(prices)) if prices[i - 1] > 0
        ]
        if not changes:
            return 0.0
        return sum(changes) / len(changes)

    def _adaptive_momentum_threshold(self, market_state: str,
                                     composite_score: float,
                                     recent_volatility: float) -> float:
        base = self.momentum_threshold * 0.5
        if recent_volatility < self.momentum_threshold:
            softness = max(recent_volatility / max(self.momentum_threshold, 1e-9), 0.1)
            base *= max(0.3, softness)
        if market_state == "range":
            base *= 0.7
        elif market_state == "idle":
            base *= 0.85
        strong_signal = max(abs(composite_score) - self.entry_threshold, 0.0)
        if strong_signal > 0:
            reduction = min(strong_signal * 0.8, 0.5)
            base *= max(0.4, 1 - reduction)
        return max(base, self.momentum_threshold * 0.15)

    def _is_direction_consistent(self, desired_direction: Optional[str],
                                 price_momentum: float,
                                 order_imbalance: float,
                                 composite_info: Dict[str, float],
                                 trend_bias: Optional[float] = None) -> bool:
        if not desired_direction:
            return False
        direction = 1 if desired_direction == 'long' else -1
        votes = []
        if price_momentum:
            votes.append(1 if price_momentum > 0 else -1)
        if order_imbalance:
            votes.append(1 if order_imbalance > 0 else -1)
        if composite_info.get('orderbook'):
            votes.append(1 if composite_info['orderbook'] > 0 else -1)
        if composite_info.get('vwap'):
            votes.append(1 if composite_info['vwap'] > 0 else -1)
        if composite_info.get('ema'):
            votes.append(1 if composite_info['ema'] > 0 else -1)
        if trend_bias and abs(trend_bias) > 1e-6:
            votes.append(1 if trend_bias > 0 else -1)

        if not votes:
            return False
        alignment = sum(1 for v in votes if v == direction)
        return alignment / len(votes) >= 0.6

    def _momentum_component(self, ticks: Deque[Dict]) -> float:
        if len(ticks) < 2:
            return 0.0
        recent = list(ticks)[-self.momentum_window:]
        changes = []
        for i in range(1, len(recent)):
            prev = recent[i - 1]['price']
            curr = recent[i]['price']
            if prev > 0:
                changes.append((curr - prev) / prev)
        if not changes:
            return 0.0
        return max(min(sum(changes) * 200, 1.0), -1.0)

    def _microstructure_score(self, orderbook_metrics: Optional[Dict], ticks: Deque[Dict]) -> float:
        if not orderbook_metrics:
            return 0.0
        spread = orderbook_metrics.get('spread') or 0.0
        liquidity = orderbook_metrics.get('liquidity') or 0.0
        spread_component = max(min((0.05 - spread) * 20, 1.0), -1.0)
        liquidity_component = min(liquidity / 5000.0, 1.0)
        return max(min(spread_component + liquidity_component, 1.0), -1.0)

    def _trade_quality_score(self, ticks: Deque[Dict]) -> float:
        recent = list(ticks)[-5:]
        buy = sum(t.get('buy_volume', 0.0) for t in recent)
        sell = sum(t.get('sell_volume', 0.0) for t in recent)
        total = buy + sell
        if total <= 0:
            return 0.0
        return max(min((buy - sell) / total, 1.0), -1.0)

    def _confirm_short_term_breakout(self, ticks: Deque[Dict], direction: Optional[str]) -> bool:
        if not direction or len(ticks) < 4:
            return False
        recent = list(ticks)[-4:]
        if direction == 'long':
            return all(recent[i]['price'] >= recent[i - 1]['price'] for i in range(1, len(recent)))
        else:
            return all(recent[i]['price'] <= recent[i - 1]['price'] for i in range(1, len(recent)))

    def _calculate_weighted_momentum(self, ticks: List[Dict]) -> float:
        if len(ticks) < 2:
            return 0.0
        weighted_sum = 0.0
        total_weight = 0.0
        for i in range(1, len(ticks)):
            prev = ticks[i - 1]['price']
            curr = ticks[i]['price']
            if prev > 0:
                change = (curr - prev) / prev
                weight = i / len(ticks)
                weighted_sum += change * weight
                total_weight += weight
        return weighted_sum / total_weight if total_weight else 0.0

    def _detect_volume_surge(self, ticks: List[Dict]) -> float:
        if len(ticks) < 5:
            return 1.0
        recent_volume = sum(t['volume'] for t in ticks[-3:])
        past_volume = sum(t['volume'] for t in ticks[:-3]) / max(len(ticks) - 3, 1)
        return recent_volume / past_volume if past_volume > 0 else 1.0

    def _calculate_order_imbalance(self, ticks: List[Dict]) -> float:
        buy_volume = sum(t.get('buy_volume', 0.0) for t in ticks)
        sell_volume = sum(t.get('sell_volume', 0.0) for t in ticks)
        total = buy_volume + sell_volume
        if total == 0:
            return 0.0
        return (buy_volume - sell_volume) / total

    def _confirm_with_orderbook(self, direction: str, orderbook: Dict) -> bool:
        bid_volume = orderbook.get('bid_volume_top3')
        ask_volume = orderbook.get('ask_volume_top3')
        if bid_volume is None or ask_volume is None:
            return True

        if direction == 'long':
            return bid_volume >= ask_volume * self.orderbook_ratio_threshold
        return ask_volume >= bid_volume * self.orderbook_ratio_threshold

    def _is_high_volatility(self, ticks: Deque[Dict]) -> bool:
        if len(ticks) < 10:
            return False

        recent_prices = [tick['price'] for tick in list(ticks)[-10:]]
        price_changes = [
            abs(recent_prices[i] - recent_prices[i - 1]) / max(recent_prices[i - 1], 1e-8)
            for i in range(1, len(recent_prices))
        ]
        if not price_changes:
            return False

        avg_volatility = sum(price_changes) / len(price_changes)
        return avg_volatility > self.volatility_threshold

    def _is_funding_time(self, now: datetime) -> bool:
        if not self.funding_rate_times:
            return False

        for hour in self.funding_rate_times:
            window_center = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            window_start = window_center - timedelta(minutes=10)
            window_end = window_center + timedelta(minutes=10)
            if window_start <= now <= window_end:
                return True

        return False
