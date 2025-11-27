# -*- coding: utf-8 -*-
"""
真正高频交易配置
"""

HFT_CONFIG = {
    # 数据处理频率
    'tick_processing_interval': 0.05,  # 50毫秒
    'signal_refresh_rate': 0.1,        # 100毫秒
    'max_trades_per_minute': 10,

    # 数据源
    'use_tick_data': True,
    'use_orderbook': True,

    # 信号参数
    'momentum_window': 20,
    'momentum_threshold': 0.0,
    'volume_spike_min': 1.2,
    'order_imbalance_min': 0.05,
    'min_confidence': 0.6,
    'composite_entry_threshold': 0.5,
    'market_volatility_threshold': 0.0001,
    'momentum_threshold_min': 0.00015,
    'momentum_threshold_max': 0.0005,
    'composite_threshold_min': 0.5,
    'composite_threshold_max': 0.8,
    'ema_fast_period': 9,
    'ema_slow_period': 21,
    'rsi_period': 14,
    'bollinger_period': 20,
    'bollinger_stddev': 2.0,
    'multi_timeframes': ['60s', '300s', '900s'],
    'range_volatility_ceiling': 0.0018,
    'trend_ema_threshold': 0.0004,
    'trend_rsi_upper': 58,
    'trend_rsi_lower': 42,
    'range_rsi_low': 45,
    'range_rsi_high': 55,
    'range_secondary_rsi_margin': 3,
    'range_secondary_vol_multiplier': 1.4,
    'trend_secondary_min_diff': 0.0003,
    'min_signal_quality': 0.7,
    'trend_signal_mode': 'orderbook_imbalance',
    'trend_fast_window': 8,
    'trend_slow_window': 21,
    'trend_min_diff': 0.00008,
    'trend_use_cross_confirmation': False,
    'trend_volume_window': 20,
    'trend_recent_ticks': 5,
    'trend_volume_ratio': 0.95,
    'trend_min_confidence': 0.55,
    'trend_orderbook_use': False,
    'trend_orderbook_min_imbalance': 0.0005,
    'trend_signal_cooldown': 4.0,
    'signal_only_mode': False,
    'detailed_monitoring': True,
    'signal_log_max_entries': 1000,
    'l1_floor_ratio': 0.015,
    'l1_floor_absolute': 1e-06,
    'l1_volatility_multiplier': 1.0,
    'enable_adaptive_tuning': True,
    'adaptive_trade_window': 20,
    'adaptive_win_rate_target': 0.6,
    'adaptive_win_rate_tolerance': 0.05,
    'adaptive_momentum_step': 0.00002,
    'adaptive_composite_step': 0.02,
    'same_direction_cooldown': 6.0,          # 单向基础冷却时间
    'same_direction_cooldown_min': 3.0,
    'same_direction_cooldown_max': 12.0,
    'trend_bias_trade_threshold': 0.3,
    'enable_trend_component_log': False,
    'trend_component_log_interval': 3.0,
    'auto_relax_guards': True,
    'auto_relax_block_window': 60.0,
    'auto_relax_block_threshold': 12,
    'auto_relax_idle_seconds': 120.0,
    'auto_relax_cooldown_step': 0.5,
    'auto_relax_min_cooldown': 2.0,
    'auto_relax_signal_cooldown_min': 0.5,
    'auto_relax_trend_step': 0.01,
    'auto_relax_min_trend_threshold': 0.04,
    'auto_relax_l1_step': 0.001,
    'auto_relax_min_l1_floor': 0.006,
    'auto_relax_max_vol_multiplier': 1.4,
    'auto_relax_volume_step': 0.05,
    'auto_relax_min_volume_ratio': 0.9,
    'auto_relax_min_orderbook_imbalance': 0.02,
    'auto_relax_relaxation_cooldown': 20.0,
    'secondary_confirmation_tf': '300s',
    'ema_secondary_fast': 50,
    'ema_secondary_slow': 100,

    # 执行参数
    'leverage': 5,
    'base_target_profit_ratio': 0.008,
    'base_stop_loss_ratio': 0.004,
    'target_profit_ratio': 0.008,
    'stop_loss_ratio': 0.004,
    'trend_target_profit_ratio': 0.008,
    'trend_stop_loss_ratio': 0.004,
    'range_target_profit_ratio': 0.008,
    'range_stop_loss_ratio': 0.004,
    'max_position_duration': 0,
    'slippage_tolerance': 0.0002,
    'use_limit_orders': True,
    'limit_order_premium': 0.0001,
    'limit_order_timeout': 5.0,   # 5秒未成交撤单
    'fallback_to_market': False,  # 不回落到市价，避免滑点和高手续费

    # 风险控制
    'daily_trade_limit': 300,
    'consecutive_loss_limit': 0,
    'enforce_survival_rules': True,
    'max_long_trades_per_day': 0,
    'max_short_trades_per_day': 0,

    # 信号过滤
    'require_orderbook_confirm': True,
    'orderbook_ratio_threshold': 0.8,
    'volatility_filter': True,
    'volatility_threshold': 0.001,
    'avoid_funding_hours': True,
    'enable_signal_debug_log': True,
    'enable_trend_log': True,

    # 退出优化
    'enable_trailing_stop': True,
    'trailing_activation': 0.004,
    'trailing_step': 0.003,
    'enable_partial_profits': True,
    'partial_profit_ratio': 0.5,
    'partial_profit_level': 0.006,

    # 资金使用
    'fixed_margin_ratio': 0.0,
    'fixed_margin': 0.0,
    'enable_progressive_margin': False,
    'loss_streak_reduce_start': 2,
    'loss_streak_reduce_factor': 0.5,
    'loss_streak_min_factor': 0.25,
    'min_position_hold_secs': 0.0,
    'dynamic_margin_ratio': 0.08,
    'min_contract_margin': 3.0,  # 至少满足交易所单张保证金要求
    'max_contracts_per_trade': 1,
    'max_margin_ratio': 0.25,
    'min_stop_loss_ratio': 0.002,
    'atr_timeframe': '60s',
    'atr_window': 14,
    'atr_stop_multiplier': 1.1,
    'atr_volatility_anchor': 0.001,

    # 保护性委托
    'enable_protective_stops': True,
    'enable_protective_take_profit': True,
    'stop_order_price_type': 1,  # 0 最新成交价, 1 标记价
    'stop_order_expiration': 86400,  # Gate.io 触发单要求按天为单位
    'take_profit_order_price_type': 1,
    'take_profit_order_expiration': 86400,

    # 调试开关
    'disable_high_risk_guard': True,
    'disable_survival_rules': True
}
