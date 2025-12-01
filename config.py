# -*- coding: utf-8 -*-
"""
运行期参数统一入口。

所有硬编码路径、冷却时间、指标确认规则、模拟回测设置在此集中配置，便于统一调整。
"""

from itertools import product
from pathlib import Path
from typing import List, Dict, Any

from gateio_config import MARKET_DATA_API_BASE_URL, SYMBOL, TAKER_FEE_RATE

# TUI 显示行数配置，限制 WS 行情滚动高度
UI_DISPLAY_CONFIG: Dict[str, int] = {
    'ticker_history_rows': 3,
    'order_book_rows': 3,
    'log_rows': 20,
    'tui_refresh_seconds': 1,  # TUI刷新频率，适当放缓避免刷屏
}

# 历史委托/成交拉取配置
HISTORY_FETCH_CONFIG: Dict[str, Any] = {
    'recent_hours': 8,
    'max_orders': 200,
}

# 网格下单防刷屏配置
GRID_ORDER_COOLDOWN_SECONDS = 10  # 同一价位重复挂单的冷却时间
GRID_GLOBAL_THROTTLE_SECONDS = 2   # 单侧全局节流，避免同一秒批量挂单


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
    'max_margin_ratio': 0.25,
    'disable_loss_pause': True  # 测试阶段关闭连续亏损暂停/冷却
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
    'min_reentry_seconds': 3.0,
    'same_direction_reentry_seconds': 6.0,
    'duplicate_window_seconds': 1.5,
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
    'bias_scale': 2000.0,               # 放大多周期趋势的权重
    'trade_threshold': 0.06,            # 交易阈值
    'min_trade_threshold': 0.02,        # 最低阈值
    'neutral_tolerance': 0.012,         # 低于该值视为中性
    'fallback_threshold': 0.04,         # 允许继续交易的兜底阈值
    'strong_signal_confidence': 0.8,    # 趋势弱时允许放行的最小信心
    'strong_signal_votes': 2,           # 需要通过的投票数量
    'bias_smoothing': 0.1,              # EMA 平滑系数
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

# Gate.io ETH 永续合约日内策略研究配置
GATEIO_ETH_DATA_CONFIG: Dict[str, Any] = {
    'contract': SYMBOL,
    'interval': '1m',
    'interval_seconds': 60,
    'days': 3,  # 拉取最近天数的K线数据
    'limit': 200,  # 单次API最多条数
    'api_base': MARKET_DATA_API_BASE_URL,
    'fallback_api_bases': [
        "https://fx-api.gateio.ws/api/v4",
        "https://api.gateio.ws/api/v4"
    ],
    'output_path': str(Path('data') / 'gateio_eth_perp_1m.csv'),
    'fallback_local_files': [
        str(Path('data') / 'gateio_eth_perp_1m.csv'),
        str(Path('data') / 'ethusd_bitfinex_2019_1m.csv')
    ],
    'max_bars': 12000,  # 兜底数据最多读取的K线数量，避免回测超时
    'timeout_seconds': 10,
    'sleep_seconds': 0.6,
    'use_cache': True
}

ETH_INTRADAY_RESEARCH_CONFIG: Dict[str, Any] = {
    'results_path': str(Path('test') / 'eth_intraday_research.jsonl'),
    'fee_rate': TAKER_FEE_RATE,
    'slippage_pct': 0.0002,
    'min_trades': 500,
    'win_rate_threshold': 0.70,
    'expectancy_threshold': 0.0,
    'breakout_params': {
        'lookback': [30, 45, 60],
        'buffer_pct': [0.0005, 0.0008],
        'volume_ratio_min': [1.4, 1.8],
        'take_profit_pct': [0.0025, 0.0032],
        'stop_loss_pct': [0.0015, 0.0020],
        'max_hold_bars': [40, 60],
        'risk_scale_high_vol': [0.5, 0.7]
    },
    'rsi_params': {
        'periods': [12, 14],
        'lower': [28, 32],
        'upper': [68, 72],
        'take_profit_pct': [0.0015, 0.0020],
        'stop_loss_pct': [0.0012, 0.0016],
        'max_hold_bars': [30, 45],
        'risk_scale_high_vol': [0.5, 0.7]
    },
    'volatility_filter': {
        'atr_window': 14,
        'atr_pct_threshold': 0.015
    },
    'orderbook_filter': {
        'imbalance_min': 0.15
    }
}

# 双模式（箱体网格 + 突破趋势）策略参数
DUAL_MODE_CONFIG: Dict[str, Any] = {
    'atr_window': 14,
    'atr_mult_grid': 0.8,         # 箱体容差/网格高度 ATR 放大倍数
    'atr_mult_trend_sl': 1.2,     # 趋势止损 ATR 倍数
    'atr_mult_trend_trail': 1.5,  # 趋势移动止损 ATR 倍数
    # “视觉”箱体判定（更接近人工目测）
    'box_visual_lookback': 180,          # 固定窗口长度（更贴近5m可视区间）
    'box_visual_quantile': 0.1,          # 使用分位数定位上下沿，替代剪裁均值
    'box_visual_max_height_pct': 0.025,  # 允许的最大箱体宽度（相对中轴）
    'box_visual_max_slope_pct': 0.0015,  # 中轴最大坡度
    'box_visual_touch_tol_pct': 0.0020,  # 触碰容差
    'box_visual_min_touches': 1,         # 上下沿最少触碰次数
    'box_lookback': 120,
    'box_alt_lookbacks': [],        # 保留占位，现已关闭降级窗口
    'box_alt_tol_pct': 0.006,     # 降级窗口容差 (0.6%)
    'box_min_bars': 120,
    'box_dynamic_min_lookback': 120,
    'box_dynamic_max_lookback': 360,
    'box_dynamic_step': 10,
    'box_tol_pct': 0.008,          # 放宽容差
    'box_quantile': 0.05,
    'box_max_slope_pct': 0.0015,   # 放宽坡度
    'box_atr_mult': 2.0,          # ATR 动态容差系数
    'box_fallback_height_pct': 0.012,  # 缠论回退可接受的最大箱体高度（相对均价）
    'grid_levels': [0.25, 0.5],   # 相对箱体高度的网格层，正负对称
    'grid_weights': [1.0, 0.6],   # 各层仓位权重（边界重）
    'grid_fee_buffer_pct': 0.0010,   # 费用缓冲(双边)占比，用于限制过密网格
    'grid_slippage_pct': 0.0005,     # 预估滑点占比
    'grid_min_spacing_pct': 0.0015,  # 最小层间距占比，避免箱体过窄仍强行铺网格
    'grid_min_size': 1.0,            # 网格最小下单张数
    'grid_size_cap': 100.0,          # 网格单次最大张数
    'grid_max_exposure_pct': 0.10,
    'grid_single_risk_pct': 0.005,
    'trend_break_vol_ratio': 1.5, # 突破时成交量放大量化
    'trend_confirm_bars': 3,      # 突破持续时间确认
    'trend_tp_pct': 0.008,        # 趋势止盈初始比例
    'trend_max_drawdown_pct': 0.05,
    'fund_allocation': {
        'grid': 0.6,
        'trend': 0.3,
        'cash': 0.1
    },
    # 风控与仓位
    'risk_per_trade': 0.01,            # 单笔风险占总权益比例
    'max_gross_exposure_pct': 0.3,     # 总暴露上限（绝对值）
    'direction_exposure_pct': 0.2,     # 单方向暴露上限
    'daily_loss_limit_pct': 0.05,      # 日内最大亏损
    'consecutive_loss_limit': 4,       # 连续亏损笔数暂停
    'add_on_pullback_pct': 0.003,      # 趋势加仓回踩幅度
    'max_add_positions': 2,            # 趋势最多加仓次数
    # 缠论/波段回退参数
    'swing_change_threshold': 0.004,   # 缠论分型/波段反转阈值
    'swing_min_bars_between': 3,       # 分型间最小间隔
    'swing_max_pivots': 10,            # 回退箱体使用的最大 pivot 数
    'swing_atr_mult': 0.0              # ATR 动态阈值倍数，0 为关闭
}
