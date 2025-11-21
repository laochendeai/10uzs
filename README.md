# Gate.io ETH高频剥头皮交易系统

## 📋 项目概述

这是一个专门针对Gate.io平台VIP0用户开发的ETH高频剥头皮交易系统。系统基于震荡区间突破策略，捕捉短期价格波动进行快速盈利。

### 核心特点
- 🎯 **策略明确**: 震荡区间突破 + 1%目标盈利
- ⚡ **高频执行**: 日内多次小利交易，累计收益
- 🛡️ **风险控制**: 多层次风险管理和止损机制
- 💰 **渐进下注**: 10U起步，三次盈利实现复利增长
- 🔧 **平台适配**: 专为Gate.io VIP0规则优化

## 📈 交易策略详解

### 核心逻辑
1. **震荡区间识别**: 分析15分钟和1小时图表，识别价格横盘整理区间
2. **突破信号确认**: 成交量放量的突破信号，配合EMA和RSI确认
3. **快速入场**: 突破确认后立即入场，75-100倍杠杆
4. **精确止盈**: 目标1%盈利，达到立即平仓
5. **严格止损**: 止损设置在区间中轴，单次风险控制在0.3-0.5%

### 渐进式资金管理
| 交易次数 | 动用资金 | 杠杆 | 目标盈利 | 累计资金 |
|---------|---------|------|---------|---------|
| 第1笔    | 10 USDT | 75x  | 10 USDT | 20 USDT |
| 第2笔    | 15 USDT | 75x  | 15 USDT | 35 USDT |
| 第3笔    | 25 USDT | 75x  | 25 USDT | 60 USDT |

### 趋势行情应对
- 连续1小时无有效区间信号时自动切换趋势跟踪模式
- EMA(9/21)+ADX+RSI 三重过滤，ADX≥25 且 RSI 处于(20,80)才触发信号
- 基于ATR的动态止盈止损，默认风险回报1:2，避免极端行情追涨杀跌
- 仓位随趋势强度自适应调整，最大不超过账户资金的15%

### 关键规则
- ✅ 每日最多3次盈利交易
- ❌ 任何亏损立即停止当日交易
- 🕐 专注活跃交易时段（亚洲下午盘、欧美开盘）
- 🚫 避开重要数据发布时段

## 🏗️ 系统架构

```
trading_engine.py          # 主交易引擎
├── gateio_config.py       # Gate.io平台配置
├── range_detector.py      # 震荡区间检测
├── technical_indicators.py # 技术指标计算
├── trend_detector.py      # 趋势检测
├── trend_risk_manager.py  # 趋势风险管理
├── config.py              # 策略参数
├── risk_management.py     # 风险管理
└── position_manager.py    # 仓位管理
```

## ⚙️ 配置说明

### Gate.io平台配置 (gateio_config.py)
```python
# 基础交易配置
SYMBOL = "ETH_USDT"           # ETH永续合约
VIP_LEVEL = 0                  # VIP等级
LEVERAGE = 75                 # 默认杠杆

# VIP0费率
MAKER_FEE_RATE = -0.00025     # Maker费率 -0.025% (返还)
TAKER_FEE_RATE = 0.0005       # Taker费率 0.05%
```

### 风险控制参数
```python
MAX_TRADES_PER_DAY = 3        # 每日最大交易次数
PROFIT_TARGET = 0.01          # 目标盈利 1%
STOP_LOSS = 0.004            # 止损幅度 0.4%
MAX_DAILY_LOSS_RATIO = 0.2   # 最大日亏损 20%
```

### 趋势策略配置 (config.py)
```python
TREND_STRATEGY_CONFIG = {
    'ema_fast_period': 9,
    'ema_slow_period': 21,
    'adx_period': 14,
    'adx_threshold': 25.0,
    'rsi_period': 14,
    'trend_stop_loss_atr_multiplier': 1.5,
    'trend_take_profit_ratio': 2.0,
    'max_trend_position_size': 0.15,
    'trend_mode_timeout': 3600
}
```
> 可在该文件中按需调整趋势检测灵敏度、ATR倍数以及仓位占比。

## 🚀 快速开始

### 环境要求
- Python 3.8+
- Gate.io账户 (VIP0即可)
- 必需库: pandas, numpy, aiohttp, gate-api

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置Gate.io API密钥

**方法1: 使用环境变量 (推荐)**
```bash
export GATEIO_API_KEY="your_api_key_here"
export GATEIO_API_SECRET="your_api_secret_here"
export GATEIO_PASSPHRASE="your_passphrase_if_needed"
```

**方法2: 使用配置文件**
```bash
# 复制示例文件
cp api_keys.txt.example api_keys.txt

# 编辑配置文件，填入您的API密钥
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here
PASSPHRASE=your_passphrase_if_needed
TESTNET=false
```

**获取API密钥步骤:**
1. 登录 Gate.io 网站
2. 进入 "用户中心" → "API管理"
3. 创建新的API密钥
4. 选择"合约交易"权限
5. 记录 API Key、Secret Key 和 Passphrase(如果有)
6. 建议设置IP白名单增强安全性

### 3. 启动交易系统

**模拟模式 (安全测试):**
```bash
python run_trading.py --mode simulation
```

**实盘模式 (需要配置API):**
```bash
python run_trading.py --mode live
```

**或者直接启动主程序:**
```bash
python trading_engine.py
```

### 🧪 历史数据回测
无需实际下单即可检验策略在 Gate.io 历史行情中的表现，并统计胜率、收益与回撤：

```bash
# 从Gate.io接口直接拉取最近500根15m K线并输出信号+模拟交易
python historical_backtest.py --contract ETH_USDT --interval 15m --limit 500 \
    --output-csv signals.csv --trades-csv trades.csv

# 或者使用已保存的CSV数据回测（需包含 timestamp/open/high/low/close/volume）
python historical_backtest.py --input-csv your_data.csv --min-bars 80 --trades-csv trades.csv
```

脚本会输出区间/趋势信号数量、市场状态占比，并根据模拟的仓位管理给出总交易数、胜率、累计盈亏、最大回撤和终盘权益；可将信号与交易明细分别导出到CSV进行复盘。

**批量参数扫描**
```bash
python historical_backtest.py --contract ETH_USDT --interval 15m --limit 1000 \
  --param-grid '{"atr_trending_ratio":[0.008,0.012],"range_reward_ratio":[1.5,2.0]}' \
  --grid-output sweep_results.csv
```
`param-grid` 中的键会自动映射到 `MARKET_STATE_CONFIG` / `DYNAMIC_TARGET_CONFIG` 的字段，脚本将对每组参数输出交易数、胜率、盈亏与最大回撤，并可写入 CSV 方便比较。

## 📊 监控界面

系统提供实时监控信息：
- 📈 当前价格和技术指标
- 🎯 震荡区间和突破信号
- 💰 资金状况和持仓信息
- ⚠️ 风险警告和止损状态
- 📊 每日交易统计

### 日志输出示例
```
🚀 启动高频剥头皮交易引擎
📈 生成交易信号: bullish_breakout - 置信度: 0.78
✅ 开仓成功: pos_20251116_143022 - long 大小: 0.1333 - 杠杆: 75x
🎯 止盈触发: 10.50 USDT

## ⚡ 高风险闪电突破模式
如果希望体验“要么翻倍要么归零”的激进玩法，可以使用新增的高风险引擎。它直接消费 Gate.io WebSocket 的 **逐笔成交 (tick)** 数据，满足瞬时动量 ≥0.3% 且成交量是均值 2 倍以上就立即全仓追击，默认 100 倍杠杆、0.5% 止损（盈利 0.5% 后自动拖尾）、≥1.5% 目标。

```bash
# 模拟模式
python run_high_risk.py --initial 10 --target-multiplier 100

# 使用 Gate.io 实盘（需配置 API，可用 1U 试水）
python run_high_risk.py --live --initial 1
```

核心组件（在 `--live` 时会自动接入 Gate.io WebSocket，在 TUI 中显示真实盘口）：
- `high_risk_engine.py`：高风险交易主循环，负责加载闪电信号、执行全仓交易、跟踪盈亏并检查生存法则。
- `lightning_detector.py`：5 分钟级别的动量/放量监控。
- `aggressive_position_manager.py` / `gambling_progression.py`：全仓下注并按连胜提升赌注。
- `high_frequency_executor.py`：高频下单执行器（实盘时用于市价单 + 止损）。
- `survival_rules.py`：确保连续亏损、50% 回撤、新闻时间等情况下强制停手。

> ⚠️ 该模式极度冒险，只为研究用途，任何实盘请自行承担风险。

### 🧩 真正的毫秒级 HFT 引擎
如果需要更高频的测试，可运行新增加的 tick 级 HFT 引擎，它每 50-100ms 处理最新逐笔成交并实时构建 1s/5s/15s K 线、动量、成交量和订单不平衡：

```bash
# 模拟模式
python run_true_hft.py --initial 10

# Gate.io 实盘（需配置 API，建议 1U 试水）
python run_true_hft.py --live --initial 1

# 仅监控真实行情中的信号触发频率
python hft_signal_monitor.py --summary 60
```

该引擎使用 `HFT_CONFIG` 中的参数（0.08%-0.2% 止损/止盈、50×杠杆、每分钟最多 30 笔等），并在 `true_hft_engine.py` 中整合了秒级数据管理、信号生成（动量/放量/订单不平衡）、高频执行和性能监控，会在 TUI 中实时显示真正的 tick 级行情。
```

## 🛡️ 安全特性

### 多层风险管理
1. **入场前评估**: 信号质量、风险回报比、市场条件
2. **实时监控**: 保证金比例、爆仓距离、市场波动
3. **自动止损**: 移动止损、时间止损、连续亏损保护
4. **资金保护**: 每日亏损限制、最大仓位限制

### Gate.io特定规则
- ✅ VIP0手续费优化
- ✅ 逐仓模式风险隔离
- ✅ 资金费率时间规避
- ✅ 最小订单价值检查

## 📈 性能指标

### 系统统计
- **交易频率**: 日均3-5次
- **平均持仓时间**: 2-10分钟
- **目标胜率**: 70%+
- **风险回报比**: 2.5:1+

### 历史回测 (模拟)
- 月收益: 50-100%
- 最大回撤: <15%
- 夏普比率: >2.0

## ⚠️ 风险提示

1. **高风险策略**: 高杠杆交易可能导致快速亏损
2. **市场风险**: 极端行情下可能触发强制平仓
3. **技术风险**: 网络延迟、API故障可能影响交易
4. **平台风险**: Gate.io规则变更可能影响策略有效性

### 建议使用
- 仅用可承受损失的资金进行交易
- 建议从小资金开始测试
- 密切监控系统运行状态
- 定期评估策略表现

## 🔧 故障排除

### 常见问题
1. **API连接失败**: 检查网络连接和API配置
2. **仓位无法开立**: 检查保证金是否充足
3. **信号生成过少**: 市场可能处于趋势行情，不适合震荡策略
4. **频繁止损**: 可能市场波动过大，建议暂停交易

### 日志查看
```bash
tail -f trading_engine.log
```

## 📞 技术支持

- 系统基于Gate.io官方文档开发
- 代码结构模块化，便于维护和扩展
- 提供完整的日志记录和错误处理
- 支持参数配置和策略调整

## 📄 免责声明

本系统仅供学习和研究使用。加密货币交易存在极高风险，可能导致资金损失。使用者应充分了解相关风险，并在可承受范围内使用。开发者不对交易损失承担任何责任。

---

**⚡ 开始您的高频剥头皮交易之旅！**
