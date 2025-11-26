# -*- coding: utf-8 -*-
"""
风险驱动的仓位管理
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple


class AggressivePositionManager:
    """结合1%风险规则、ATR与波动率的仓位控制"""

    def __init__(self, leverage: int = 100,
                 risk_config: Optional[Dict[str, float]] = None,
                 initial_equity: float = 0.0):
        self.leverage = leverage
        self.config = risk_config or {}
        self.risk_per_trade = float(self.config.get('risk_per_trade', 0.01))
        self.daily_loss_limit = float(self.config.get('daily_loss_limit', 0.03))
        self.consecutive_loss_pause = int(self.config.get('consecutive_loss_pause', 3))
        cooldown_minutes = max(float(self.config.get('loss_pause_cooldown_minutes', 15.0)), 0.0)
        self.loss_pause_cooldown = int(cooldown_minutes * 60)
        self.min_stop_distance = float(self.config.get('min_stop_distance', 0.0009))
        self.atr_stop_multiplier = float(self.config.get('atr_stop_multiplier', 1.1))
        self.high_volatility_threshold = float(self.config.get('high_volatility_threshold', 0.0025))
        self.low_volatility_threshold = float(self.config.get('low_volatility_threshold', 0.0005))
        self.high_volatility_scale = float(self.config.get('high_volatility_scale', 0.75))
        self.low_volatility_scale = float(self.config.get('low_volatility_scale', 1.15))
        self.max_margin_ratio = float(self.config.get('max_margin_ratio', 0.25))
        self.win_streak = 0
        self.loss_streak = 0
        self.consecutive_wins = 0
        self.pause_until: Optional[datetime] = None
        self.current_day = datetime.utcnow().date()
        self.start_of_day_equity = max(initial_equity, 0.0)
        self.daily_realized = 0.0

    def _roll_day(self, total_equity: float, now: datetime):
        if now.date() != self.current_day:
            self.current_day = now.date()
            self.start_of_day_equity = max(total_equity, 0.0)
            self.daily_realized = 0.0
            self.win_streak = 0
            self.loss_streak = 0
            self.consecutive_wins = 0
            self.pause_until = None

    def can_open(self, total_equity: float, now: Optional[datetime] = None) -> Tuple[bool, str]:
        now = now or datetime.utcnow()
        self._roll_day(total_equity, now)
        if self.pause_until and now < self.pause_until:
            return False, "cooldown_active"
        if self.start_of_day_equity > 0:
            drawdown = (self.start_of_day_equity - total_equity) / max(self.start_of_day_equity, 1e-8)
            if drawdown >= self.daily_loss_limit:
                self.pause_until = now + timedelta(seconds=self.loss_pause_cooldown or 60)
                return False, "daily_loss_limit"
        if self.consecutive_loss_pause > 0 and self.loss_streak >= self.consecutive_loss_pause:
            self.pause_until = now + timedelta(seconds=self.loss_pause_cooldown or 60)
            return False, "consecutive_losses"
        return True, ""

    def determine_margin(self, total_equity: float, leverage: float, stop_ratio: float,
                         atr_value: Optional[float], price: float, volatility: float) -> Tuple[float, float]:
        effective_stop = max(stop_ratio, self.min_stop_distance)
        if atr_value and price > 0:
            atr_ratio = (atr_value / price) * self.atr_stop_multiplier
            effective_stop = max(effective_stop, atr_ratio)
        effective_stop = max(effective_stop, self.min_stop_distance)
        leverage = max(leverage, 1e-6)
        risk_capital = total_equity * self.risk_per_trade
        margin = risk_capital / max(leverage * effective_stop, 1e-8)
        if volatility > self.high_volatility_threshold:
            margin *= self.high_volatility_scale
        elif volatility < self.low_volatility_threshold:
            margin *= self.low_volatility_scale
        if self.max_margin_ratio > 0:
            margin = min(margin, total_equity * self.max_margin_ratio)
        return max(margin, 0.0), effective_stop

    def register_trade(self, pnl: float, total_equity: float, now: Optional[datetime] = None):
        now = now or datetime.utcnow()
        self._roll_day(total_equity, now)
        self.daily_realized += pnl
        if pnl > 0:
            self.consecutive_wins += 1
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.consecutive_wins = 0
            self.win_streak = 0
            self.loss_streak += 1
            if self.consecutive_loss_pause > 0 and self.loss_streak >= self.consecutive_loss_pause:
                self.pause_until = now + timedelta(seconds=self.loss_pause_cooldown or 60)

    def get_progressive_boost(self, current_capital: float) -> float:
        """保留旧接口，按连胜情况微调仓位"""
        boost_multiplier = 1.0
        if self.consecutive_wins == 1:
            boost_multiplier = 1.1
        elif self.consecutive_wins == 2:
            boost_multiplier = 1.25
        elif self.consecutive_wins >= 3:
            boost_multiplier = 1.4
        return min(current_capital * boost_multiplier, current_capital * 3)
