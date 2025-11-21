# -*- coding: utf-8 -*-
"""
高频性能监控
"""

from collections import deque
from typing import Dict


class HFTPerformanceMonitor:
    """记录信号/执行延迟、交易频率等"""

    def __init__(self):
        self.latency_metrics = {
            'signal_generation': deque(maxlen=1000),
            'order_execution': deque(maxlen=1000),
            'total_cycle': deque(maxlen=1000)
        }
        self.trade_metrics = {
            'trades_per_minute': 0,
            'win_rate': 0.0,
            'avg_holding_time': 0.0,
            'slippage_avg': 0.0
        }

    def record_latency(self, metric: str, value: float):
        if metric in self.latency_metrics:
            self.latency_metrics[metric].append(value)

    def update_trade_metrics(self, metrics: Dict):
        self.trade_metrics.update(metrics)

    def get_summary(self) -> Dict:
        summary = {}
        for metric, data in self.latency_metrics.items():
            if data:
                summary[metric] = sum(data) / len(data)
            else:
                summary[metric] = 0.0
        summary.update(self.trade_metrics)
        return summary
