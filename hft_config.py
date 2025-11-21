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
    'momentum_window': 10,
    'momentum_threshold': 0.0002,
    'volume_spike_min': 1.5,
    'order_imbalance_min': 0.2,
    'min_confidence': 0.9,
    'composite_entry_threshold': 0.6,
    'market_volatility_threshold': 0.0001,
    'momentum_threshold_min': 0.00015,
    'momentum_threshold_max': 0.0005,
    'composite_threshold_min': 0.5,
    'composite_threshold_max': 0.8,
    'signal_only_mode': False,
    'signal_log_max_entries': 1000,
    'enable_adaptive_tuning': True,
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

    # 执行参数
    'leverage': 10,
    'target_profit_ratio': 0.012,
    'stop_loss_ratio': 0.006,
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
    'fixed_margin': 1.5,  # 每笔交易最大使用1.5 USDT保证金
    'enable_progressive_margin': True,
    'min_position_hold_secs': 1.5,
    'dynamic_margin_ratio': 0.1,
    'min_contract_margin': 3.0,  # 至少满足交易所单张保证金要求

    # 保护性委托
    'enable_protective_stops': True,
    'stop_order_price_type': 1,  # 0 最新成交价, 1 标记价
    'stop_order_expiration': 3600
}
