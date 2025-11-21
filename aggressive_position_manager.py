# -*- coding: utf-8 -*-
"""
激进仓位管理
"""

class AggressivePositionManager:
    """激进仓位管理 - 全仓搏杀"""

    def __init__(self, leverage: int = 100):
        self.leverage = leverage
        self.win_streak = 0
        self.loss_streak = 0
        self.consecutive_wins = 0

    def calculate_all_in_position(self, signal, account_balance):
        """全仓计算"""
        entry_price = signal['entry_price']
        margin_available = account_balance
        position_value = margin_available * self.leverage
        position_size = position_value / max(entry_price, 1e-8)

        return {
            'position_size': position_size,
            'leverage': self.leverage,
            'margin_required': margin_available,
            'is_all_in': True,
            'risk_percentage': 1.0
        }

    def get_progressive_boost(self, current_capital):
        """连胜加成"""
        boost_multiplier = 1.0
        if self.consecutive_wins == 1:
            boost_multiplier = 1.2
        elif self.consecutive_wins == 2:
            boost_multiplier = 1.5
        elif self.consecutive_wins >= 3:
            boost_multiplier = 2.0
        return min(current_capital * boost_multiplier, current_capital * 3)

    def update_streak(self, trade_profit: float):
        """更新连胜记录"""
        if trade_profit > 0:
            self.consecutive_wins += 1
            self.loss_streak = 0
        else:
            self.consecutive_wins = 0
            self.loss_streak += 1
