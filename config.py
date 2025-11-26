# -*- coding: utf-8 -*-
"""
运行期参数统一入口。

所有硬编码路径、冷却时间、指标确认规则、模拟回测设置在此集中配置，便于统一调整。
"""

from itertools import product
from pathlib import Path
from typing import List, Dict, Any


def _default_parameter_grid() -> List[Dict[str, Any]]:
    """自动生成≥50组参数组合，用于大规模扫描."""
    momentum_values = [0.00012, 0.00015, 0.00018, 0.00022, 0.00025]
    volume_values = [1.1, 1.3, 1.5, 1.8, 2.2]
    imbalance_values = [0.15, 0.2, 0.25]
    entry_thresholds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    vote_requirements = [2, 3, 4]
    momentum_windows = [8, 10, 12, 14]
    cooldowns = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    volatility_filters = [0.00008, 0.0001, 0.00015]

    grid: List[Dict[str, Any]] = []
    combo_iter = product(
        momentum_values,
        volume_values,
        imbalance_values,
        entry_thresholds,
        vote_requirements,
        momentum_windows,
        cooldowns,
        volatility_filters
    )
    for idx, combo in enumerate(combo_iter, start=1):
        (momentum, volume, imbalance, entry, votes,
         window, cooldown, vol_filter) = combo
        same_dir = max(cooldown * 1.5, cooldown + 0.8)
        duplicate_window = max(0.5, cooldown / 2)
        grid.append({
            'name': f"auto_{idx:02d}",
            'momentum_threshold': momentum,
            'momentum_window': window,
            'volume_spike_min': volume,
            'order_imbalance_min': imbalance,
            'composite_entry_threshold': entry,
            'direction_vote_required': votes,
            'min_reentry_seconds': cooldown,
            'same_direction_reentry_seconds': same_dir,
            'duplicate_window_seconds': duplicate_window,
            'market_volatility_threshold': vol_filter,
            'volatility_threshold': vol_filter * 3
        })
        if len(grid) >= 60:
            break
    return grid


SCALP_RISK_CONFIG: Dict[str, Any] = {
    'risk_per_trade': 0.01,
    'daily_loss_limit': 0.03,
    'consecutive_loss_pause': 5,
    'loss_pause_cooldown_minutes': 15,
    'min_stop_distance': 0.0009,
    'atr_stop_multiplier': 1.1,
    'high_volatility_threshold': 0.0025,
    'low_volatility_threshold': 0.0005,
    'high_volatility_scale': 0.75,
    'low_volatility_scale': 1.15,
    'max_margin_ratio': 0.25
}


CUSTOM_PARAMETER_PRESETS: Dict[str, Dict[str, Any]] = {
    'eth_scalp': {
        'name': 'eth_scalp',
        'momentum_threshold': 0.00018,
        'momentum_window': 12,
        'volume_spike_min': 1.4,
        'order_imbalance_min': 0.25,
        'composite_entry_threshold': 0.55,
        'min_confidence': 0.85,
        'direction_vote_required': 2,
        'direction_vote_sources': [
            'momentum',
            'volume',
            'imbalance',
            'orderbook',
            'trend_bias'
        ],
        'min_reentry_seconds': 1.2,
        'same_direction_reentry_seconds': 3.0,
        'duplicate_window_seconds': 0.6,
        'market_volatility_threshold': 0.0002,
        'volatility_threshold': 0.0008,
        'momentum_threshold_min': 0.00012,
        'momentum_threshold_max': 0.00045,
        'composite_threshold_min': 0.45,
        'composite_threshold_max': 0.75,
        'range_target_profit_ratio': 0.0018,
        'range_stop_loss_ratio': 0.0012,
        'trend_target_profit_ratio': 0.0030,
        'trend_stop_loss_ratio': 0.0015
    }
}


def get_parameter_preset(name: str) -> Dict[str, Any]:
    """按名称检索参数组合，供监控/调优脚本复用。"""
    if name in CUSTOM_PARAMETER_PRESETS:
        return dict(CUSTOM_PARAMETER_PRESETS[name])
    for combo in _default_parameter_grid():
        if combo.get('name') == name:
            return dict(combo)
    raise KeyError(f"未找到名称为 {name} 的参数预设")


TRADE_GUARD_CONFIG: Dict[str, Any] = {
    # 防重复开仓
    'min_reentry_seconds': 1.2,
    'same_direction_reentry_seconds': 2.5,
    'duplicate_window_seconds': 1.0,
    'entry_frequency_buffer': 500,
    'entry_frequency_log_threshold': 1,

    # 多重方向确认
    'direction_vote_required': 2,
    'direction_vote_sources': [
        'momentum',
        'volume',
        'imbalance',
        'composite',
        'breakout',
        'orderbook',
        'trend_bias'
    ],

    # 方向准确率统计窗口 (关闭后用于达成 75%+ 目标)
    'direction_accuracy_window': 200,

    # 回看价格的秒数（用于离线评估正确率）
    'price_lookahead_seconds': 5
}

TREND_GUARD_CONFIG: Dict[str, Any] = {
    # 趋势滤波参数
    'bias_scale': 1600.0,               # 放大多周期趋势的权重
    'trade_threshold': 0.12,            # 交易阈值
    'min_trade_threshold': 0.05,        # 最低阈值
    'neutral_tolerance': 0.02,          # 低于该值视为中性
    'fallback_threshold': 0.06,         # 允许继续交易的兜底阈值
    'strong_signal_confidence': 0.9,    # 趋势弱时允许放行的最小信心
    'strong_signal_votes': 4,           # 需要通过的投票数量
    'bias_smoothing': 0.25,             # EMA 平滑系数
    'allow_neutral_bias': True          # 趋势为空时是否允许依赖其他过滤
}

SIMULATION_CONFIG: Dict[str, Any] = {
    'historical_trade_file': str(Path('data') / 'historical_trades.csv'),
    'timestamp_field': 'timestamp',
    'price_field': 'price',
    'size_field': 'size',
    'side_field': 'side',
    'direction_field': 'direction',
    'max_rows': 15000,
    'report_path': str(Path('test') / 'backtest_report.jsonl'),
    'parameter_grid': _default_parameter_grid(),
    'price_lookahead_seconds': TRADE_GUARD_CONFIG['price_lookahead_seconds']
}

DEFAULT_MONITOR_PRESET = 'auto_20'
MONITORING_CONFIG: Dict[str, Any] = {
    'preset_name': DEFAULT_MONITOR_PRESET,
    'report_path': str(Path('test') / 'monitor_auto20.jsonl'),
    'heartbeat_interval_seconds': 10.0,
    'heartbeat_stall_warning_seconds': 30.0
}
