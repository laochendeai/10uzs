# -*- coding: utf-8 -*-
"""
Gate.io VIP0 ETH高频剥头皮交易配置
基于gate.io平台实际规则定制
"""

import os

# Gate.io平台配置
EXCHANGE_NAME = "gateio"
VIP_LEVEL = 0  # VIP等级
SYMBOL = "ETH_USDT"  # ETH永续合约交易对
CONTRACT_TYPE = "perpetual"  # 永续合约
MARGIN_MODE = "isolated"  # 逐仓模式
FUTURES_SETTLE = "usdt"  # 合约结算币种

# API配置
TESTNET_MODE = os.getenv("GATEIO_TESTNET", "false").lower() in ("1", "true", "yes")
_LIVE_API_BASE = "https://api.gateio.ws/api/v4"
_LIVE_WS_BASE = "wss://fx-ws.gateio.ws/v4/ws/usdt"
_DEFAULT_API_BASE = "https://api-testnet.gateapi.io/api/v4" if TESTNET_MODE else _LIVE_API_BASE
_DEFAULT_WS_BASE = "wss://fx-ws-testnet.gateio.ws/v4/ws/usdt" if TESTNET_MODE else _LIVE_WS_BASE

API_BASE_URL = os.getenv("GATEIO_API_BASE_URL", _DEFAULT_API_BASE)
WS_BASE_URL = os.getenv("GATEIO_WS_URL", _DEFAULT_WS_BASE)

MARKET_DATA_USE_MAINNET = os.getenv("GATEIO_MARKET_DATA_USE_MAINNET", "true").lower() in ("1", "true", "yes")
MARKET_DATA_API_BASE_URL = os.getenv(
    "GATEIO_MARKET_API_BASE_URL",
    _LIVE_API_BASE if MARKET_DATA_USE_MAINNET else API_BASE_URL
)
MARKET_DATA_WS_URL = os.getenv(
    "GATEIO_MARKET_WS_URL",
    _LIVE_WS_BASE if MARKET_DATA_USE_MAINNET else WS_BASE_URL
)
GATEIO_API_KEY = os.getenv("GATEIO_API_KEY", "")
GATEIO_API_SECRET = os.getenv("GATEIO_API_SECRET", "")
USE_GATEIO_MARKET_DATA = os.getenv("USE_GATEIO_MARKET_DATA", "false").lower() in ("1", "true", "yes")
ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "false").lower() in ("1", "true", "yes")
CONTRACT_VALUE = float(os.getenv("GATEIO_CONTRACT_VALUE", "0.01"))
PRICE_TICK_SIZE = float(os.getenv("GATEIO_PRICE_TICK_SIZE", "0.01"))  # 最小价格单位

# Gate.io VIP0费率配置
MAKER_FEE_RATE = -0.00025  # Maker费率 -0.025% (返还)
TAKER_FEE_RATE = 0.0005   # Taker费率 0.05%
FUNDING_RATE_INTERVAL = 8  # 资金费率间隔 (小时)
FUNDING_RATE_TIMES = [0, 8, 16]  # UTC时间收取资金费率

# 杠杆配置
MAX_LEVERAGE = 100  # 最大杠杆倍数 (VIP0限制)
DEFAULT_LEVERAGE = 75  # 默认使用杠杆，留安全边际
LEVERAGE_STEP = 25  # 杠杆调整步长

# 交易基础配置
INITIAL_CAPITAL = 10.0  # 初始资金 (USDT)
PROFIT_TARGET = 0.01   # 目标盈利 (1%)
STOP_LOSS = 0.004      # 止损幅度 (0.4%)
MAX_TRADES_PER_DAY = 3  # 每日最大交易次数
MIN_TRADE_INTERVAL = 300  # 最小交易间隔 (5分钟)

# 震荡区间检测配置 (适应gate.io行情)
RANGE_LOOKBACK_CANDLES = 24  # 回看K线数量 (15分钟图，约6小时)
RANGE_VOLATILITY_THRESHOLD = 0.015  # 震荡区间波动阈值 (1.5%)
MIN_BREAKOUT_VOLUME_RATIO = 1.5  # 最小突破成交量倍数
CONSOLIDATION_FACTOR = 0.7  # 整理因子，判断是否为有效震荡

# 技术指标配置
EMA_FAST = 9   # 快速EMA
EMA_SLOW = 21  # 慢速EMA
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MA_PERIOD = 20  # 成交量均线周期

# Gate.io交易时段配置 (24小时运行)
ACTIVE_TRADING_HOURS = [(0, 24)]

# 避险时段 (避开重要数据发布)
HIGH_RISK_HOURS = {
    "non_farm": [(8, 8, 30)],    # 非农数据 (北京时间 20:30)
    "cpi": [(13, 13, 30)],       # CPI数据 (北京时间 21:30)
    "funding_rate_times": [0, 8, 16]  # 资金费率收取时间前后的波动
}

# 细化的高风险事件窗口配置（时间为北京时间）
HIGH_RISK_WINDOWS = [
    {
        "name": "non_farm_payrolls",
        "hour": 20,
        "minute": 30,
        "pre_buffer": 5,
        "post_buffer": 15,
        "weekdays": [4]  # 周五
    },
    {
        "name": "cpi_release",
        "hour": 21,
        "minute": 30,
        "pre_buffer": 5,
        "post_buffer": 15
    },
    {
        "name": "fed_rate_decision",
        "hour": 2,
        "minute": 0,
        "pre_buffer": 15,
        "post_buffer": 20
    },
    {
        "name": "fomc_press_conference",
        "hour": 2,
        "minute": 30,
        "pre_buffer": 20,
        "post_buffer": 30
    },
    {
        "name": "us_gdp_release",
        "hour": 20,
        "minute": 30,
        "pre_buffer": 10,
        "post_buffer": 20,
        "weekdays": [3]  # 周四常规发布
    },
    {
        "name": "core_pce_release",
        "hour": 20,
        "minute": 30,
        "pre_buffer": 10,
        "post_buffer": 20,
        "weekdays": [4]
    }
]
NEWS_COOLDOWN_MINUTES = 30

# 风控配置 (考虑gate.io爆仓机制)
MAX_DAILY_LOSS_RATIO = 0.2  # 最大日亏损比例
MARGIN_CALL_RATIO = 0.005   # 最低保证金比例限制
LIQUIDATION_BUFFER = 0.002  # 爆仓缓冲距离
CONSECUTIVE_LOSS_LIMIT = 2  # 连续亏损限制
AUTO_PAUSE_ON_LOSSES = True  # 连续亏损后自动暂停

# 资金管理策略 (渐进式复利)
PROGRESSIVE_BETTING = {
    "enabled": True,
    "trades": [
        {"trade_num": 1, "capital_ratio": 1.0, "leverage": 75},   # 第1笔：10U, 75x
        {"trade_num": 2, "capital_ratio": 1.5, "leverage": 75},   # 第2笔：15U, 75x
        {"trade_num": 3, "capital_ratio": 2.5, "leverage": 75},   # 第3笔：25U, 75x
    ],
    "reset_on_loss": True,  # 亏损后重置到初始资金
    "daily_reset": True,    # 每日重置
}

# 滑点和市场深度配置
EXPECTED_SLIPPAGE = 0.0001  # 预期滑点 (0.01%)
MAX_SLIPPAGE_TOLERANCE = 0.0003  # 最大滑点容忍度 (0.03%)
MIN_MARKET_DEPTH = 10000   # 最小市场深度要求 (USDT)

# 订单执行配置
ORDER_TYPE = "market"      # 市价单 (确保快速成交)
PARTIAL_FILL_RETRY = 3     # 部分成交重试次数
ORDER_TIMEOUT = 10         # 订单超时时间 (秒)
POSITION_SIZE_PRECISION = 4  # 仓位大小精度

# API接口配置
API_RATE_LIMIT = 10        # API调用频率限制 (次/秒)
WEBSOCKET_TIMEOUT = 30     # WebSocket超时时间
RECONNECT_ATTEMPTS = 5     # 重连尝试次数
HEARTBEAT_INTERVAL = 20    # 心跳间隔 (秒)

# 日志和监控配置
LOG_LEVEL = "INFO"
TRADE_LOG_ENABLED = True
PERFORMANCE_MONITORING = True
ALERT_CONFIG = {
    "large_profit": True,
    "large_loss": True,
    "consecutive_losses": True,
    "margin_warning": True
}

# Gate.io特定规则
GATEIO_RULES = {
    "min_order_value": 5.0,        # 最小订单价值 (USDT)
    "max_position_value": 50000,   # VIP0最大仓位价值
    "price_precision": 2,          # 价格精度
    "size_precision": 4,           # 数量精度
    "force_liquidation": True,     # 是否支持强制平仓
    "partial_liquidation": True,   # 是否支持部分平仓
    "adl_ranking": True,           # 是否需要考虑ADL排名
}
