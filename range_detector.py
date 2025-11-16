# -*- coding: utf-8 -*-
"""
震荡区间检测模块
专门用于识别ETH价格震荡区间和突破信号
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from gateio_config import RANGE_LOOKBACK_CANDLES, RANGE_VOLATILITY_THRESHOLD, MIN_BREAKOUT_VOLUME_RATIO

class RangeDetector:
    """震荡区间检测器"""

    def __init__(self):
        self.current_range = None
        self.range_start_time = None
        self.breakout_candidates = []

    def detect_consolidation_range(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        检测震荡区间

        Args:
            df: K线数据DataFrame，包含['timestamp', 'open', 'high', 'low', 'close', 'volume']

        Returns:
            震荡区间信息或None
        """
        if len(df) < RANGE_LOOKBACK_CANDLES:
            return None

        # 取最近N根K线
        recent_df = df.tail(RANGE_LOOKBACK_CANDLES).copy()

        # 计算关键价格点
        recent_high = recent_df['high'].max()
        recent_low = recent_df['low'].min()
        recent_close = recent_df['close'].iloc[-1]

        # 计算区间波动率
        range_width = recent_high - recent_low
        range_volatility = range_width / recent_low

        # 计算价格在区间内的分布
        price_distribution = self._analyze_price_distribution(recent_df, recent_low, recent_high)

        # 计算成交量特征
        volume_profile = self._analyze_volume_profile(recent_df)

        # 判断是否为有效震荡区间
        is_valid_range = self._validate_consolidation(
            recent_df,
            range_volatility,
            price_distribution,
            volume_profile
        )

        if is_valid_range:
            range_info = {
                'upper_bound': recent_high,
                'lower_bound': recent_low,
                'range_width': range_width,
                'range_volatility': range_volatility,
                'mid_line': (recent_high + recent_low) / 2,
                'current_price': recent_close,
                'price_position': (recent_close - recent_low) / range_width,  # 价格在区间中的位置
                'start_time': recent_df['timestamp'].iloc[0],
                'end_time': recent_df['timestamp'].iloc[-1],
                'duration_hours': (recent_df['timestamp'].iloc[-1] - recent_df['timestamp'].iloc[0]).total_seconds() / 3600,
                'price_distribution': price_distribution,
                'volume_profile': volume_profile,
                'volume_ratio': recent_df['volume'].iloc[-1] / recent_df['volume'].mean(),  # 当前成交量与均值比
                'ema_fast': self._calculate_ema(recent_df['close'], 9),
                'ema_slow': self._calculate_ema(recent_df['close'], 21),
                'rsi': self._calculate_rsi(recent_df['close']),
            }

            self.current_range = range_info
            return range_info

        return None

    def detect_breakout_signal(self, df: pd.DataFrame, range_info: Dict) -> Optional[Dict]:
        """
        检测突破信号

        Args:
            df: K线数据
            range_info: 震荡区间信息

        Returns:
            突破信号信息或None
        """
        if len(df) < 3:
            return None

        latest_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        current_price = latest_candle['close']
        current_volume = latest_candle['volume']

        # 计算突破强度
        breakout_strength = self._calculate_breakout_strength(
            latest_candle, prev_candle, range_info
        )

        # 检测向上突破
        if (current_price > range_info['upper_bound'] and
            prev_candle['close'] <= range_info['upper_bound']):

            volume_confirmation = current_volume > (range_info['volume_profile']['avg_volume'] * MIN_BREAKOUT_VOLUME_RATIO)

            if volume_confirmation:
                return {
                    'type': 'bullish_breakout',
                    'direction': 'long',
                    'breakout_price': current_price,
                    'breakout_strength': breakout_strength,
                    'volume_confirmation': volume_confirmation,
                    'volume_ratio': current_volume / range_info['volume_profile']['avg_volume'],
                    'stop_loss': max(range_info['mid_line'], prev_candle['low'] * 0.995),
                    'target_price': current_price * (1 + 0.01),  # 1%目标
                    'risk_reward_ratio': self._calculate_risk_reward_ratio(current_price,
                                                                          current_price * 1.01,
                                                                          max(range_info['mid_line'], prev_candle['low'] * 0.995)),
                    'confidence': self._calculate_signal_confidence('bullish', breakout_strength, volume_confirmation),
                    'timestamp': latest_candle['timestamp']
                }

        # 检测向下突破
        elif (current_price < range_info['lower_bound'] and
              prev_candle['close'] >= range_info['lower_bound']):

            volume_confirmation = current_volume > (range_info['volume_profile']['avg_volume'] * MIN_BREAKOUT_VOLUME_RATIO)

            if volume_confirmation:
                return {
                    'type': 'bearish_breakout',
                    'direction': 'short',
                    'breakout_price': current_price,
                    'breakout_strength': breakout_strength,
                    'volume_confirmation': volume_confirmation,
                    'volume_ratio': current_volume / range_info['volume_profile']['avg_volume'],
                    'stop_loss': min(range_info['mid_line'], prev_candle['high'] * 1.005),
                    'target_price': current_price * (1 - 0.01),  # 1%目标
                    'risk_reward_ratio': self._calculate_risk_reward_ratio(current_price,
                                                                          current_price * 0.99,
                                                                          min(range_info['mid_line'], prev_candle['high'] * 1.005)),
                    'confidence': self._calculate_signal_confidence('bearish', breakout_strength, volume_confirmation),
                    'timestamp': latest_candle['timestamp']
                }

        return None

    def _analyze_price_distribution(self, df: pd.DataFrame, low: float, high: float) -> Dict:
        """分析价格在区间内的分布"""
        prices = df[['open', 'high', 'low', 'close']].values.flatten()

        # 将价格区间分为5个等分
        price_bins = np.linspace(low, high, 6)
        distribution, _ = np.histogram(prices, bins=price_bins)

        # 计算分布均匀性
        distribution_std = np.std(distribution)
        distribution_mean = np.mean(distribution)
        uniformity = 1 - (distribution_std / (distribution_mean + 1e-8))

        return {
            'distribution': distribution.tolist(),
            'uniformity': uniformity,
            'concentration_areas': self._find_concentration_areas(distribution, price_bins)
        }

    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """分析成交量特征"""
        volumes = df['volume'].values
        price_changes = df['close'].pct_change().fillna(0)

        return {
            'avg_volume': np.mean(volumes),
            'volume_std': np.std(volumes),
            'volume_trend': 'increasing' if volumes[-1] > volumes[-3] else 'decreasing',
            'volume_price_correlation': np.corrcoef(volumes, np.abs(price_changes))[0, 1],
            'volume_surge_ratio': volumes[-1] / np.mean(volumes[:-1]) if len(volumes) > 1 else 1.0
        }

    def _validate_consolidation(self, df: pd.DataFrame, volatility: float,
                              price_dist: Dict, volume_profile: Dict) -> bool:
        """验证是否为有效的震荡区间"""

        # 1. 波动率不能过高
        if volatility > RANGE_VOLATILITY_THRESHOLD:
            return False

        # 2. 价格分布相对均匀
        if price_dist['uniformity'] < 0.3:  # 分布过于集中
            return False

        # 3. 成交量相对稳定，没有异常放量
        if volume_profile['volume_price_correlation'] > 0.7:  # 成交量与价格变动相关性过高
            return False

        # 4. 区间内价格多次测试边界
        high_tests = (df['high'] >= df['high'].quantile(0.9)).sum()
        low_tests = (df['low'] <= df['low'].quantile(0.1)).sum()

        if high_tests < 2 or low_tests < 2:  # 边界测试次数不足
            return False

        return True

    def _calculate_breakout_strength(self, latest_candle: pd.Series, prev_candle: pd.Series,
                                   range_info: Dict) -> float:
        """计算突破强度"""
        current_price = latest_candle['close']
        prev_price = prev_candle['close']

        if current_price > range_info['upper_bound']:  # 向上突破
            penetration = (current_price - range_info['upper_bound']) / range_info['range_width']
            momentum = (current_price - prev_price) / prev_price

        else:  # 向下突破
            penetration = (range_info['lower_bound'] - current_price) / range_info['range_width']
            momentum = (prev_price - current_price) / prev_price

        # 综合评分：穿透深度 + 动量 + 成交量
        volume_factor = latest_candle['volume'] / range_info['volume_profile']['avg_volume']

        strength = (penetration * 0.4 +
                   min(momentum * 10, 0.3) +  # 限制动量影响
                   min(volume_factor / 3, 0.3))  # 限制成交量影响

        return min(strength, 1.0)  # 限制最大值为1.0

    def _calculate_risk_reward_ratio(self, entry: float, target: float, stop_loss: float) -> float:
        """计算风险回报比"""
        profit = abs(target - entry)
        risk = abs(entry - stop_loss)
        return profit / (risk + 1e-8)

    def _calculate_signal_confidence(self, direction: str, strength: float, volume_conf: bool) -> float:
        """计算信号可信度"""
        base_confidence = strength * 0.6

        # EMA趋势确认
        ema_bonus = 0.0
        if direction == 'bullish' and hasattr(self, 'current_range') and self.current_range:
            if self.current_range.get('ema_fast', 0) > self.current_range.get('ema_slow', 0):
                ema_bonus = 0.2
        elif direction == 'bearish' and hasattr(self, 'current_range') and self.current_range:
            if self.current_range.get('ema_fast', 0) < self.current_range.get('ema_slow', 0):
                ema_bonus = 0.2

        # 成交量确认
        volume_bonus = 0.2 if volume_conf else 0.0

        confidence = base_confidence + ema_bonus + volume_bonus
        return min(confidence, 1.0)

    def _calculate_ema(self, prices: pd.Series, period: int) -> float:
        """计算EMA"""
        return prices.ewm(span=period, adjust=False).mean().iloc[-1]

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def _find_concentration_areas(self, distribution: np.ndarray, price_bins: np.ndarray) -> List[Dict]:
        """找出价格集中区域"""
        concentration_areas = []
        threshold = np.mean(distribution) * 1.2  # 高于平均值20%的区域

        for i, count in enumerate(distribution):
            if count > threshold:
                concentration_areas.append({
                    'price_range': (price_bins[i], price_bins[i+1]),
                    'concentration_level': count / np.max(distribution)
                })

        return concentration_areas

    def get_range_status(self, current_price: float) -> Dict:
        """获取当前价格在区间中的状态"""
        if not self.current_range:
            return {'status': 'no_range'}

        range_info = self.current_range
        position = (current_price - range_info['lower_bound']) / range_info['range_width']

        if position < 0.1:
            status = 'near_bottom'
        elif position > 0.9:
            status = 'near_top'
        elif 0.4 <= position <= 0.6:
            status = 'middle'
        elif position < 0.5:
            status = 'lower_half'
        else:
            status = 'upper_half'

        return {
            'status': status,
            'position': position,
            'distance_to_top': (range_info['upper_bound'] - current_price) / current_price,
            'distance_to_bottom': (current_price - range_info['lower_bound']) / current_price
        }