# -*- coding: utf-8 -*-
"""
技术指标模块
提供EMA、RSI、成交量等技术分析指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class TechnicalIndicators:
    """技术指标计算器"""

    def __init__(self):
        self.indicators_cache = {}

    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """
        计算所有技术指标

        Args:
            df: K线数据DataFrame

        Returns:
            包含所有指标的字典
        """
        indicators = {}

        # 趋势指标
        indicators.update(self._calculate_trend_indicators(df))

        # 动量指标
        indicators.update(self._calculate_momentum_indicators(df))

        # 成交量指标
        indicators.update(self._calculate_volume_indicators(df))

        # 波动性指标
        indicators.update(self._calculate_volatility_indicators(df))

        # 支撑阻力指标
        indicators.update(self._calculate_support_resistance(df))

        return indicators

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> Dict:
        """计算趋势指标"""
        close = df['close']

        # EMA指标
        ema_9 = self._ema(close, 9)
        ema_21 = self._ema(close, 21)
        ema_50 = self._ema(close, 50)

        # EMA交叉信号
        ema_signal = self._get_ema_cross_signal(ema_9, ema_21)

        # MACD指标
        macd_line, macd_signal, macd_histogram = self._macd(close)

        return {
            'ema_9': ema_9,
            'ema_21': ema_21,
            'ema_50': ema_50,
            'ema_trend': self._get_ema_trend(ema_9, ema_21),
            'ema_cross_signal': ema_signal,
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram,
            'macd_trend': self._get_macd_trend(macd_line, macd_signal)
        }

    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """计算动量指标"""
        close = df['close']
        high = df['high']
        low = df['low']

        # RSI指标
        rsi = self._rsi(close, 14)
        rsi_signal = self._get_rsi_signal(rsi)

        # Stochastic指标
        stochastic_k, stochastic_d = self._stochastic(high, low, close)

        # Williams %R
        williams_r = self._williams_r(high, low, close)

        # CCI指标
        cci = self._cci(high, low, close)

        return {
            'rsi': rsi,
            'rsi_signal': rsi_signal,
            'rsi_overbought': rsi > 70,
            'rsi_oversold': rsi < 30,
            'stochastic_k': stochastic_k,
            'stochastic_d': stochastic_d,
            'williams_r': williams_r,
            'cci': cci,
            'momentum_strength': self._calculate_momentum_strength(rsi, stochastic_k)
        }

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict:
        """计算成交量指标"""
        volume = df['volume']
        close = df['close']

        # 成交量移动平均
        volume_sma_20 = volume.rolling(window=20).mean()
        volume_sma_5 = volume.rolling(window=5).mean()

        # VWAP (成交量加权平均价)
        vwap = self._vwap(df)

        # 成交量变化率
        volume_change = volume.pct_change()

        # 成交量价格趋势
        vpt = self._vpt(close, volume)

        # OBV (能量潮)
        obv = self._obv(close, volume)

        return {
            'volume_sma_20': volume_sma_20,
            'volume_sma_5': volume_sma_5,
            'volume_ratio': volume / volume_sma_20,
            'volume_surge': self._detect_volume_surge(volume, volume_sma_20),
            'vwap': vwap,
            'price_above_vwap': close.iloc[-1] > vwap,
            'volume_change': volume_change.iloc[-1] if len(volume_change) > 0 else 0,
            'vpt': vpt,
            'obv': obv,
            'volume_trend': self._get_volume_trend(volume, volume_sma_20)
        }

    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict:
        """计算波动性指标"""
        close = df['close']
        high = df['high']
        low = df['low']

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._bollinger_bands(close)

        # ATR (平均真实波幅)
        atr = self._atr(high, low, close)

        # 历史波动率
        historical_volatility = self._historical_volatility(close)

        return {
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'bb_width': (bb_upper - bb_lower) / bb_middle,
            'bb_position': (close - bb_lower) / (bb_upper - bb_lower),
            'bb_squeeze': self._detect_bb_squeeze(bb_upper, bb_lower, bb_middle),
            'atr': atr,
            'atr_ratio': atr / close,
            'historical_volatility': historical_volatility,
            'volatility_regime': self._get_volatility_regime(historical_volatility)
        }

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """计算支撑阻力位"""
        high = df['high']
        low = df['low']
        close = df['close']

        # Pivot Points
        pivots = self._pivot_points(df)

        # 高低点支撑阻力
        resistance_levels = self._find_resistance_levels(high)
        support_levels = self._find_support_levels(low)

        # 当前价格与最近支撑阻力的距离
        current_price = close.iloc[-1]
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
        nearest_support = max([s for s in support_levels if s < current_price], default=None)

        return {
            'pivot_points': pivots,
            'resistance_levels': resistance_levels,
            'support_levels': support_levels,
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'resistance_distance': (nearest_resistance - current_price) / current_price if nearest_resistance else None,
            'support_distance': (current_price - nearest_support) / current_price if nearest_support else None,
            'price_position': self._get_price_position_in_range(current_price, nearest_support, nearest_resistance)
        }

    def _ema(self, prices: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均线"""
        return prices.ewm(span=period, adjust=False).mean()

    def _get_ema_cross_signal(self, ema_fast: pd.Series, ema_slow: pd.Series) -> str:
        """获取EMA交叉信号"""
        if len(ema_fast) < 2 or len(ema_slow) < 2:
            return 'neutral'

        current_fast = ema_fast.iloc[-1]
        current_slow = ema_slow.iloc[-1]
        prev_fast = ema_fast.iloc[-2]
        prev_slow = ema_slow.iloc[-2]

        if prev_fast <= prev_slow and current_fast > current_slow:
            return 'bullish_cross'
        elif prev_fast >= prev_slow and current_fast < current_slow:
            return 'bearish_cross'
        elif current_fast > current_slow:
            return 'bullish'
        else:
            return 'bearish'

    def _get_ema_trend(self, ema_fast: pd.Series, ema_slow: pd.Series) -> str:
        """获取EMA趋势状态"""
        current_fast = ema_fast.iloc[-1]
        current_slow = ema_slow.iloc[-1]

        if current_fast > current_slow:
            return 'uptrend'
        else:
            return 'downtrend'

    def _macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD指标"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram

    def _get_macd_trend(self, macd_line: pd.Series, macd_signal: pd.Series) -> str:
        """获取MACD趋势"""
        if len(macd_line) < 2 or len(macd_signal) < 2:
            return 'neutral'

        if macd_line.iloc[-1] > macd_signal.iloc[-1]:
            if macd_line.iloc[-1] > macd_line.iloc[-2]:
                return 'strong_uptrend'
            else:
                return 'uptrend'
        else:
            if macd_line.iloc[-1] < macd_line.iloc[-2]:
                return 'strong_downtrend'
            else:
                return 'downtrend'

    def _rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _get_rsi_signal(self, rsi: pd.Series) -> str:
        """获取RSI信号"""
        current_rsi = rsi.iloc[-1]

        if current_rsi > 70:
            return 'overbought'
        elif current_rsi < 30:
            return 'oversold'
        elif current_rsi > 50:
            return 'bullish'
        else:
            return 'bearish'

    def _stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                    k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """计算Stochastic指标"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent

    def _williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """计算Williams %R指标"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return wr

    def _cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """计算CCI指标"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        return cci

    def _vwap(self, df: pd.DataFrame) -> float:
        """计算VWAP"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
        return vwap

    def _detect_volume_surge(self, volume: pd.Series, volume_sma: pd.Series, threshold: float = 1.5) -> bool:
        """检测成交量激增"""
        current_volume = volume.iloc[-1]
        avg_volume = volume_sma.iloc[-1]
        return current_volume > (avg_volume * threshold)

    def _vpt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """计算成交量价格趋势(VPT)"""
        price_change = close.pct_change()
        vpt = (price_change * volume).cumsum()
        return vpt

    def _obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """计算能量潮(OBV)"""
        price_change = close.diff()
        obv = volume.copy()
        obv[price_change < 0] = -obv[price_change < 0]
        return obv.cumsum()

    def _get_volume_trend(self, volume: pd.Series, volume_sma: pd.Series) -> str:
        """获取成交量趋势"""
        current_volume = volume.iloc[-1]
        avg_volume = volume_sma.iloc[-1]

        if current_volume > avg_volume * 1.5:
            return 'high'
        elif current_volume > avg_volume:
            return 'increasing'
        elif current_volume > avg_volume * 0.7:
            return 'normal'
        else:
            return 'decreasing'

    def _bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def _detect_bb_squeeze(self, upper: pd.Series, lower: pd.Series, middle: pd.Series, threshold: float = 0.1) -> bool:
        """检测布林带收缩"""
        current_width = (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1]
        avg_width = ((upper - lower) / middle).rolling(window=20).mean().iloc[-1]
        return current_width < (avg_width * (1 - threshold))

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """计算平均真实波幅"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def _historical_volatility(self, prices: pd.Series, period: int = 20) -> float:
        """计算历史波动率"""
        log_returns = np.log(prices / prices.shift(1))
        volatility = log_returns.rolling(window=period).std() * np.sqrt(365)  # 年化波动率
        return volatility.iloc[-1]

    def _get_volatility_regime(self, volatility: float) -> str:
        """获取波动性状态"""
        if volatility > 0.8:
            return 'high'
        elif volatility > 0.4:
            return 'normal'
        else:
            return 'low'

    def _pivot_points(self, df: pd.DataFrame) -> Dict:
        """计算枢轴点"""
        last_candle = df.iloc[-1]
        high = last_candle['high']
        low = last_candle['low']
        close = last_candle['close']

        pp = (high + low + close) / 3
        r1 = 2 * pp - low
        r2 = pp + (high - low)
        r3 = high + 2 * (pp - low)
        s1 = 2 * pp - high
        s2 = pp - (high - low)
        s3 = low - 2 * (high - pp)

        return {
            'pp': pp,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }

    def _find_resistance_levels(self, high: pd.Series, lookback: int = 20, tolerance: float = 0.005) -> List[float]:
        """寻找阻力位"""
        resistance_levels = []
        highs = high.rolling(window=lookback).max()
        unique_levels = []

        for i in range(len(highs)):
            if high.iloc[i] == highs.iloc[i]:
                level = high.iloc[i]
                # 检查是否与现有水平过于接近
                if not any(abs(level - existing) < level * tolerance for existing in unique_levels):
                    unique_levels.append(level)

        return sorted(unique_levels, reverse=True)[:5]  # 返回前5个阻力位

    def _find_support_levels(self, low: pd.Series, lookback: int = 20, tolerance: float = 0.005) -> List[float]:
        """寻找支撑位"""
        support_levels = []
        lows = low.rolling(window=lookback).min()
        unique_levels = []

        for i in range(len(lows)):
            if low.iloc[i] == lows.iloc[i]:
                level = low.iloc[i]
                # 检查是否与现有水平过于接近
                if not any(abs(level - existing) < level * tolerance for existing in unique_levels):
                    unique_levels.append(level)

        return sorted(unique_levels)[:5]  # 返回前5个支撑位

    def _get_price_position_in_range(self, price: float, support: float, resistance: float) -> str:
        """获取价格在支撑阻力区间中的位置"""
        if support is None or resistance is None:
            return 'unknown'

        position = (price - support) / (resistance - support)

        if position < 0.2:
            return 'near_support'
        elif position > 0.8:
            return 'near_resistance'
        elif 0.4 <= position <= 0.6:
            return 'middle'
        elif position < 0.5:
            return 'lower_half'
        else:
            return 'upper_half'

    def _calculate_momentum_strength(self, rsi: pd.Series, stochastic_k: pd.Series) -> float:
        """计算动量强度综合评分"""
        rsi_value = rsi.iloc[-1]
        stoch_value = stochastic_k.iloc[-1]

        # RSI评分 (50为中心，越远离50越强)
        rsi_score = abs(rsi_value - 50) / 50

        # Stochastic评分
        if stoch_value > 80:  # 超买区域，强势
            stoch_score = 0.8
        elif stoch_value > 50:
            stoch_score = stoch_value / 100
        elif stoch_value > 20:  # 超卖区域，弱势
            stoch_score = stoch_value / 100
        else:  # 极度超卖，可能反弹
            stoch_score = 0.2

        # 综合评分
        momentum_strength = (rsi_score * 0.6 + stoch_score * 0.4)
        return min(momentum_strength, 1.0)

    def get_trading_signal_summary(self, indicators: Dict) -> Dict:
        """汇总所有技术指标信号"""
        signals = {
            'trend_signal': 'neutral',
            'momentum_signal': 'neutral',
            'volume_signal': 'neutral',
            'volatility_signal': 'neutral',
            'overall_strength': 0.0,
            'recommendation': 'hold'
        }

        # 趋势信号
        ema_trend = indicators.get('ema_trend', 'neutral')
        macd_trend = indicators.get('macd_trend', 'neutral')
        if 'uptrend' in ema_trend and 'uptrend' in macd_trend:
            signals['trend_signal'] = 'strong_bullish'
        elif 'uptrend' in ema_trend or 'uptrend' in macd_trend:
            signals['trend_signal'] = 'bullish'
        elif 'downtrend' in ema_trend and 'downtrend' in macd_trend:
            signals['trend_signal'] = 'strong_bearish'
        elif 'downtrend' in ema_trend or 'downtrend' in macd_trend:
            signals['trend_signal'] = 'bearish'

        # 动量信号
        rsi_signal = indicators.get('rsi_signal', 'neutral')
        momentum_strength = indicators.get('momentum_strength', 0.5)
        if rsi_signal == 'overbought':
            signals['momentum_signal'] = 'bearish_reversal'
        elif rsi_signal == 'oversold':
            signals['momentum_signal'] = 'bullish_reversal'
        elif momentum_strength > 0.7:
            signals['momentum_signal'] = 'strong'
        elif momentum_strength > 0.5:
            signals['momentum_signal'] = 'moderate'

        # 综合强度评分
        trend_weight = 0.4
        momentum_weight = 0.3
        volume_weight = 0.3

        trend_score = self._signal_to_score(signals['trend_signal'])
        momentum_score = self._signal_to_score(signals['momentum_signal'])
        volume_score = 0.5  # 简化处理

        signals['overall_strength'] = (trend_score * trend_weight +
                                      momentum_score * momentum_weight +
                                      volume_score * volume_weight)

        # 最终建议
        if signals['overall_strength'] > 0.7:
            signals['recommendation'] = 'buy'
        elif signals['overall_strength'] < 0.3:
            signals['recommendation'] = 'sell'
        else:
            signals['recommendation'] = 'hold'

        return signals

    def _signal_to_score(self, signal: str) -> float:
        """将信号转换为数值评分"""
        signal_scores = {
            'strong_bullish': 1.0,
            'bullish': 0.75,
            'bullish_reversal': 0.6,
            'neutral': 0.5,
            'bearish_reversal': 0.4,
            'bearish': 0.25,
            'strong_bearish': 0.0,
            'strong': 0.8,
            'moderate': 0.6,
        }
        return signal_scores.get(signal, 0.5)