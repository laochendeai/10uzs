# -*- coding: utf-8 -*-
"""
高风险策略生存法则
"""

from typing import List, Dict


class SurvivalRules:
    """高风险策略生存法则"""

    @staticmethod
    def mandatory_rules() -> List[str]:
        return [
            "1. 永远不要追加保证金",
            "2. 单次亏损后不要立即复仇交易",
            "3. 每日绝对亏损上限：初始本金的50%",
            "4. 连续3次亏损强制停止当天交易",
            "5. 达到日目标（翻倍）立即收手",
            "6. 重大新闻发布前后30分钟不交易",
            "7. 网络延迟>100ms时不交易",
            "8. 情绪波动时手动暂停交易"
        ]

    @staticmethod
    def emergency_stop_conditions(current_capital: float,
                                  initial_capital: float,
                                  trades_today: List[Dict]) -> List[str]:
        """检查是否触发紧急停止条件"""
        conditions = []
        if current_capital < initial_capital * 0.5:
            conditions.append("亏损超过50%")

        losses = [trade for trade in trades_today if trade.get('pnl', 0) < 0]
        if len(losses) >= 3:
            conditions.append("连续3次亏损")

        if len(trades_today) >= 15:
            conditions.append("交易次数过多")

        return conditions
