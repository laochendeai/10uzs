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
    'momentum_window': 50,
    'momentum_threshold': 0.0,
    'volume_spike_min': 1.5,
    'order_imbalance_min': 0.2,
    'min_confidence': 0.9,
    'composite_entry_threshold': 0.6,
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
    'trend_fast_window': 15,
    'trend_slow_window': 45,
    'trend_min_diff': 0.00015,
    'trend_use_cross_confirmation': False,
    'trend_volume_window': 40,
    'trend_recent_ticks': 8,
    'trend_volume_ratio': 1.2,
    'trend_min_confidence': 0.6,
    'trend_orderbook_use': True,
    'trend_orderbook_min_imbalance': 0.05,
    'trend_signal_cooldown': 2.0,
    'signal_only_mode': True,
    'detailed_monitoring': False,
    'signal_log_max_entries': 1000,
    'l1_floor_ratio': 0.01,
    'l1_floor_absolute': 1e-06,
    'l1_volatility_multiplier': 0.8,
    'enable_adaptive_tuning': False,
    'adaptive_trade_window': 20,
    'adaptive_win_rate_target': 0.6,
    'adaptive_win_rate_tolerance': 0.05,
    'adaptive_momentum_step': 0.00002,
    'adaptive_composite_step': 0.02,
    'same_direction_cooldown': 15.0,          # 单向基础冷却时间
    'same_direction_cooldown_min': 8.0,
    'same_direction_cooldown_max': 45.0,
    'trend_bias_trade_threshold': 0.3,
    'enable_trend_component_log': False,
    'trend_component_log_interval': 3.0,
    'secondary_confirmation_tf': '300s',
    'ema_secondary_fast': 50,
    'ema_secondary_slow': 100,

    # 执行参数
    'leverage': 10,
    'target_profit_ratio': 0.0025,
    'stop_loss_ratio': 0.0045,
    'trend_target_profit_ratio': 0.0030,
    'trend_stop_loss_ratio': 0.0030,
    'range_target_profit_ratio': 0.0018,
    'range_stop_loss_ratio': 0.0036,
    'max_position_duration': 0,
    'slippage_tolerance': 0.0002,
    'use_limit_orders': True,
    'limit_order_premium': 0.0005,
    'limit_order_timeout': 0.2,
    'fallback_to_market': True,

    # 风险控制
    'daily_trade_limit': 300,
    'consecutive_loss_limit': 5,
    'enforce_survival_rules': True,
    'max_long_trades_per_day': 0,
    'max_short_trades_per_day': 0,

    # 信号过滤
    'require_orderbook_confirm': True,
    'orderbook_ratio_threshold': 0.8,
    'volatility_filter': True,
    'volatility_threshold': 0.001,
    'avoid_funding_hours': True,
    'enable_signal_debug_log': False,
    'enable_trend_log': False,

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
    'min_position_hold_secs': 1.5,
    'dynamic_margin_ratio': 0.1,
    'min_contract_margin': 3.0,  # 至少满足交易所单张保证金要求
    'max_contracts_per_trade': 1,
    'max_margin_ratio': 0.25,
    'min_stop_loss_ratio': 0.0015,
    'atr_timeframe': '60s',
    'atr_window': 14,
    'atr_stop_multiplier': 1.1,
    'atr_volatility_anchor': 0.001,

    # 保护性委托
    'enable_protective_stops': True,
    'stop_order_price_type': 1,  # 0 最新成交价, 1 标记价
    'stop_order_expiration': 3600
}
