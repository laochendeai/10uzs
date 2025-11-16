# -*- coding: utf-8 -*-
"""
风险管理模块
实现针对gate.io平台的高频交易风险控制
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from gateio_config import *

@dataclass
class RiskMetrics:
    """风险指标数据类"""
    current_position_size: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    margin_ratio: float
    liquidation_price: float
    distance_to_liquidation: float
    max_drawdown: float
    consecutive_losses: int
    daily_trade_count: int
    risk_score: float

class RiskManager:
    """风险管理器"""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_start_capital = initial_capital
        self.trades_today = []
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.max_daily_loss_reached = False
        self.trading_paused = False
        self.pause_reason = None
        self.last_trade_time = None
        self.max_capital_achieved = initial_capital

        # 实时监控数据
        self.current_positions = {}
        self.open_orders = {}
        self.risk_metrics = None

        # 历史记录
        self.trade_history = []
        self.daily_performance = {}

    def evaluate_entry_risk(self, signal: Dict, current_price: float,
                           account_balance: float, leverage: int = DEFAULT_LEVERAGE) -> Tuple[bool, Dict]:
        """
        评估入场风险

        Args:
            signal: 交易信号
            current_price: 当前价格
            account_balance: 账户余额
            leverage: 杠杆倍数

        Returns:
            (是否允许交易, 风险评估详情)
        """
        risk_assessment = {
            'allowed': False,
            'position_size': 0.0,
            'stop_loss': signal.get('stop_loss', current_price * 0.996),
            'take_profit': signal.get('target_price', current_price * 1.01),
            'leverage': leverage,
            'risk_amount': 0.0,
            'potential_loss': 0.0,
            'potential_profit': 0.0,
            'risk_reward_ratio': 0.0,
            'warnings': [],
            'risk_factors': {}
        }

        # 1. 检查基础交易条件
        if not self._check_basic_trading_conditions():
            risk_assessment['warnings'].append(self.pause_reason or 'Trading paused')
            return False, risk_assessment

        # 2. 检查日交易次数限制
        if len(self.trades_today) >= MAX_TRADES_PER_DAY:
            risk_assessment['warnings'].append('Daily trade limit reached')
            return False, risk_assessment

        # 3. 检查最小交易间隔
        if not self._check_minimum_trade_interval():
            risk_assessment['warnings'].append('Minimum trade interval not met')
            return False, risk_assessment

        # 4. 检查信号质量
        signal_quality = self._evaluate_signal_quality(signal)
        if signal_quality['score'] < 0.6:  # 信号质量阈值
            risk_assessment['warnings'].append(f'Low signal quality: {signal_quality["score"]:.2f}')
            return False, risk_assessment

        # 5. 计算仓位大小
        position_size = self._calculate_position_size(
            account_balance, signal, leverage, current_price
        )

        # 6. 验证最小订单价值
        if position_size * current_price < GATEIO_RULES['min_order_value']:
            risk_assessment['warnings'].append('Position size below minimum order value')
            return False, risk_assessment

        # 7. 计算风险指标
        risk_amount = self._calculate_risk_amount(position_size, current_price, signal['stop_loss'])
        potential_loss = abs(current_price - signal['stop_loss']) / current_price
        potential_profit = abs(signal['target_price'] - current_price) / current_price
        risk_reward_ratio = potential_profit / (potential_loss + 1e-8)

        # 8. 风险控制检查
        if not self._risk_control_checks(risk_amount, account_balance, risk_reward_ratio):
            risk_assessment['warnings'].append('Risk control checks failed')
            return False, risk_assessment

        # 9. 填充风险评估
        risk_assessment.update({
            'allowed': True,
            'position_size': position_size,
            'risk_amount': risk_amount,
            'potential_loss': potential_loss,
            'potential_profit': potential_profit,
            'risk_reward_ratio': risk_reward_ratio,
            'signal_quality': signal_quality,
            'confidence_level': signal.get('confidence', 0.5)
        })

        # 10. 添加风险因素分析
        risk_assessment['risk_factors'] = self._analyze_risk_factors(signal, current_price)

        return True, risk_assessment

    def monitor_position_risk(self, position: Dict, current_price: float) -> Dict:
        """
        监控持仓风险

        Args:
            position: 持仓信息
            current_price: 当前价格

        Returns:
            风险监控结果
        """
        monitoring_result = {
            'risk_level': 'low',
            'actions_required': [],
            'stop_loss_adjustment': None,
            'position_adjustment': None,
            'emergency_close': False,
            'warnings': []
        }

        # 计算当前盈亏
        unrealized_pnl = self._calculate_unrealized_pnl(position, current_price)
        pnl_percentage = unrealized_pnl / (position['entry_price'] * position['size'])

        # 计算爆仓距离
        liquidation_distance = self._calculate_liquidation_distance(position, current_price)

        # 风险等级评估
        if liquidation_distance < 0.002:  # 0.2%以内
            monitoring_result['risk_level'] = 'critical'
            monitoring_result['emergency_close'] = True
            monitoring_result['actions_required'].append('EMERGENCY_CLOSE')

        elif liquidation_distance < 0.005:  # 0.5%以内
            monitoring_result['risk_level'] = 'high'
            monitoring_result['actions_required'].append('TIGHTEN_STOP_LOSS')

        elif pnl_percentage < -0.003:  # 亏损超过0.3%
            monitoring_result['risk_level'] = 'medium'
            monitoring_result['actions_required'].append('CONSIDER_CLOSING')

        # 动态止损调整
        if pnl_percentage > 0.005:  # 盈利超过0.5%，移动止损
            new_stop_loss = self._calculate_trailing_stop(position, current_price)
            if new_stop_loss > position.get('stop_loss', 0):
                monitoring_result['stop_loss_adjustment'] = new_stop_loss
                monitoring_result['actions_required'].append('TRAILING_STOP')

        # 时间风险检查
        position_duration = (datetime.now() - position['entry_time']).total_seconds() / 60
        if position_duration > 30:  # 持仓超过30分钟
            monitoring_result['warnings'].append('Position held too long')
            if pnl_percentage < 0:
                monitoring_result['actions_required'].append('CONSIDER_EXIT')

        return monitoring_result

    def update_trade_result(self, trade_result: Dict):
        """更新交易结果"""
        self.trades_today.append(trade_result)
        self.last_trade_time = datetime.now()

        # 更新资金
        if trade_result['pnl'] > 0:
            self.consecutive_losses = 0
            self.current_capital += trade_result['pnl']
            self.max_capital_achieved = max(self.max_capital_achieved, self.current_capital)
        else:
            self.consecutive_losses += 1
            self.current_capital += trade_result['pnl']  # pnl为负数

        self.daily_pnl += trade_result['pnl']

        # 检查是否需要暂停交易
        self._check_pause_conditions()

        # 记录到历史
        self.trade_history.append({
            **trade_result,
            'date': datetime.now().date(),
            'timestamp': datetime.now()
        })

    def get_daily_risk_report(self) -> Dict:
        """获取每日风险报告"""
        current_time = datetime.now()

        # 计算当日表现指标
        win_trades = [t for t in self.trades_today if t['pnl'] > 0]
        loss_trades = [t for t in self.trades_today if t['pnl'] < 0]

        win_rate = len(win_trades) / len(self.trades_today) if self.trades_today else 0
        avg_win = np.mean([t['pnl'] for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t['pnl'] for t in loss_trades]) if loss_trades else 0

        # 计算最大回撤
        current_drawdown = (self.max_capital_achieved - self.current_capital) / self.max_capital_achieved

        return {
            'date': current_time.date(),
            'current_capital': self.current_capital,
            'daily_pnl': self.daily_pnl,
            'daily_return': self.daily_pnl / self.daily_start_capital,
            'trade_count': len(self.trades_today),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'consecutive_losses': self.consecutive_losses,
            'max_drawdown': current_drawdown,
            'trading_paused': self.trading_paused,
            'pause_reason': self.pause_reason,
            'risk_score': self._calculate_overall_risk_score()
        }

    def _check_basic_trading_conditions(self) -> bool:
        """检查基础交易条件"""
        if self.trading_paused:
            return False

        # 检查是否在避险时段
        current_hour = datetime.now().hour
        for risk_period in HIGH_RISK_HOURS.values():
            if isinstance(risk_period, list) and len(risk_period) > 0:
                if isinstance(risk_period[0], tuple):  # 处理具体时段
                    for start_h, start_m, duration_m in risk_period:
                        if current_hour == start_h:
                            return False
                elif isinstance(risk_period[0], int):  # 处理小时
                    if current_hour in risk_period:
                        return False

        return True

    def _check_minimum_trade_interval(self) -> bool:
        """检查最小交易间隔"""
        if self.last_trade_time is None:
            return True

        time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
        return time_since_last >= MIN_TRADE_INTERVAL

    def _evaluate_signal_quality(self, signal: Dict) -> Dict:
        """评估信号质量"""
        quality_score = 0.0
        factors = {}

        # 置信度评分 (40%)
        confidence = signal.get('confidence', 0.5)
        quality_score += confidence * 0.4
        factors['confidence'] = confidence

        # 风险回报比评分 (30%)
        risk_reward = signal.get('risk_reward_ratio', 1.0)
        rr_score = min(risk_reward / 2.0, 1.0)  # 2:1风险回报比得满分
        quality_score += rr_score * 0.3
        factors['risk_reward_ratio'] = risk_reward

        # 成交量确认评分 (20%)
        volume_confirmation = signal.get('volume_confirmation', False)
        volume_score = 0.8 if volume_confirmation else 0.4
        quality_score += volume_score * 0.2
        factors['volume_confirmation'] = volume_confirmation

        # 突破强度评分 (10%)
        breakout_strength = signal.get('breakout_strength', 0.5)
        quality_score += breakout_strength * 0.1
        factors['breakout_strength'] = breakout_strength

        return {
            'score': quality_score,
            'factors': factors
        }

    def _calculate_position_size(self, account_balance: float, signal: Dict,
                                 leverage: int, reference_price: float) -> float:
        """计算仓位大小"""
        entry_price = signal.get('breakout_price', reference_price)
        if entry_price <= 0:
            return 0.0

        if PROGRESSIVE_BETTING['enabled']:
            return self._calculate_progressive_position(account_balance, leverage, entry_price)
        else:
            # 固定比例策略
            risk_per_trade = account_balance * 0.02  # 每笔交易风险2%
            stop_loss_distance = abs(signal['breakout_price'] - signal['stop_loss'])

            if stop_loss_distance == 0:
                return 0

            position_value = risk_per_trade / (stop_loss_distance / entry_price)
            return position_value / entry_price

    def _calculate_progressive_position(self, account_balance: float, leverage: int,
                                        entry_price: float) -> float:
        """计算渐进式仓位大小"""
        trade_num = len(self.trades_today) + 1

        for trade_config in PROGRESSIVE_BETTING['trades']:
            if trade_config['trade_num'] == trade_num:
                capital_ratio = trade_config['capital_ratio']
                position_leverage = min(trade_config['leverage'], MAX_LEVERAGE)

                # 计算实际仓位价值
                margin_to_use = account_balance * capital_ratio
                position_value = margin_to_use * position_leverage
                return position_value / entry_price

        # 默认使用第一笔交易的配置
        default_config = PROGRESSIVE_BETTING['trades'][0]
        margin_to_use = account_balance * default_config['capital_ratio']
        position_value = margin_to_use * min(default_config['leverage'], MAX_LEVERAGE)
        return position_value / entry_price

    def _calculate_risk_amount(self, position_size: float, current_price: float, stop_loss: float) -> float:
        """计算风险金额"""
        position_value = position_size * current_price
        risk_percentage = abs(current_price - stop_loss) / current_price
        return position_value * risk_percentage

    def _risk_control_checks(self, risk_amount: float, account_balance: float,
                            risk_reward_ratio: float) -> bool:
        """风险控制检查"""
        # 1. 单笔风险不超过账户的5%
        if risk_amount > account_balance * 0.05:
            return False

        # 2. 风险回报比不低于1.5:1
        if risk_reward_ratio < 1.5:
            return False

        # 3. 当日累计亏损不超过20%
        if self.daily_pnl < -self.daily_start_capital * MAX_DAILY_LOSS_RATIO:
            return False

        # 4. 连续亏损不超过限制
        if self.consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
            return False

        return True

    def _analyze_risk_factors(self, signal: Dict, current_price: float) -> Dict:
        """分析风险因素"""
        factors = {
            'market_conditions': 'normal',
            'volatility_risk': 'low',
            'liquidity_risk': 'low',
            'timing_risk': 'low'
        }

        # 分析市场条件
        current_hour = datetime.now().hour
        if current_hour in [0, 1, 2, 3, 4, 5]:  # 凌晨时段
            factors['timing_risk'] = 'high'
            factors['market_conditions'] = 'low_liquidity'

        # 分析信号类型风险
        if signal['type'] in ['bullish_breakout', 'bearish_breakout']:
            factors['market_conditions'] = 'high_volatility'

        return factors

    def _calculate_unrealized_pnl(self, position: Dict, current_price: float) -> float:
        """计算未实现盈亏"""
        if position['direction'] == 'long':
            return (current_price - position['entry_price']) * position['size']
        else:
            return (position['entry_price'] - current_price) * position['size']

    def _calculate_liquidation_distance(self, position: Dict, current_price: float) -> float:
        """计算距离爆仓的价格百分比"""
        leverage = position.get('leverage', DEFAULT_LEVERAGE)
        maintenance_margin = 0.005  # gate.io维持保证金率

        if position['direction'] == 'long':
            liquidation_price = position['entry_price'] * (1 - 1/leverage + maintenance_margin)
            return (current_price - liquidation_price) / current_price
        else:
            liquidation_price = position['entry_price'] * (1 + 1/leverage - maintenance_margin)
            return (liquidation_price - current_price) / current_price

    def _calculate_trailing_stop(self, position: Dict, current_price: float) -> float:
        """计算移动止损价格"""
        trailing_distance = 0.003  # 0.3%移动距离

        if position['direction'] == 'long':
            return current_price * (1 - trailing_distance)
        else:
            return current_price * (1 + trailing_distance)

    def _check_pause_conditions(self):
        """检查是否需要暂停交易"""
        # 连续亏损暂停
        if AUTO_PAUSE_ON_LOSSES and self.consecutive_losses >= CONSECUTIVE_LOSS_LIMIT:
            self.trading_paused = True
            self.pause_reason = 'consecutive_losses'
            return

        # 日亏损限制暂停
        if self.daily_pnl < -self.daily_start_capital * MAX_DAILY_LOSS_RATIO:
            self.trading_paused = True
            self.pause_reason = 'daily_loss_limit'
            return

        # 完成每日目标暂停
        if len(self.trades_today) >= MAX_TRADES_PER_DAY and all(t['pnl'] > 0 for t in self.trades_today):
            self.trading_paused = True
            self.pause_reason = 'daily_target_achieved'

    def _calculate_overall_risk_score(self) -> float:
        """计算综合风险评分"""
        risk_factors = []

        # 资金风险
        capital_risk = abs(self.daily_pnl) / self.daily_start_capital
        risk_factors.append(min(capital_risk * 5, 1.0))

        # 连续亏损风险
        loss_streak_risk = self.consecutive_losses / CONSECUTIVE_LOSS_LIMIT
        risk_factors.append(min(loss_streak_risk, 1.0))

        # 交易频率风险
        frequency_risk = len(self.trades_today) / MAX_TRADES_PER_DAY
        risk_factors.append(min(frequency_risk, 1.0))

        return np.mean(risk_factors)

    def reset_daily(self):
        """重置每日状态"""
        self.trades_today = []
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.daily_start_capital = self.current_capital
        self.max_daily_loss_reached = False
        self.trading_paused = False
        self.pause_reason = None

    def force_pause_trading(self, reason: str, duration_minutes: int = 30):
        """强制暂停交易"""
        self.trading_paused = True
        self.pause_reason = reason
        # 这里可以添加定时恢复逻辑

    def resume_trading(self):
        """恢复交易"""
        self.trading_paused = False
        self.pause_reason = None
