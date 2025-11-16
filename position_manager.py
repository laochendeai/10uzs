# -*- coding: utf-8 -*-
"""
仓位和资金管理模块
实现针对高频剥头皮的渐进式下注策略
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from gateio_config import *

@dataclass
class Position:
    """持仓数据类"""
    symbol: str
    direction: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    leverage: int
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    status: str = 'open'  # 'open', 'closed', 'liquidated'
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    trade_id: Optional[str] = None
    margin_required: float = 0.0

class PositionManager:
    """仓位管理器"""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.available_capital = initial_capital
        self.reserved_capital = 0.0
        self.total_capital = initial_capital

        # 渐进式下注状态
        self.daily_progressive_state = {
            'trade_count': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'current_betting_level': 1,
            'daily_start_capital': initial_capital,
            'max_capital_today': initial_capital,
            'betting_reset_needed': False
        }

        # 持仓管理
        self.open_positions = {}
        self.closed_positions = []
        self.position_history = []

        # 资金分配策略
        self.allocation_strategy = 'progressive'  # 'progressive', 'fixed', 'kelly'

        # 风险参数
        self.max_position_risk = 0.05  # 单仓位最大风险5%
        self.max_total_risk = 0.15     # 总仓位最大风险15%
        self.min_reserve_ratio = 0.1   # 最小保留资金比例10%

    def calculate_optimal_position(self, signal: Dict, current_price: float,
                                 account_balance: float, market_data: Dict) -> Dict:
        """
        计算最优仓位大小

        Args:
            signal: 交易信号
            current_price: 当前价格
            account_balance: 账户余额
            market_data: 市场数据

        Returns:
            仓位配置信息
        """
        position_config = {
            'position_size': 0.0,
            'leverage': DEFAULT_LEVERAGE,
            'margin_required': 0.0,
            'risk_amount': 0.0,
            'position_value': 0.0,
            'max_loss': 0.0,
            'max_profit': 0.0,
            'risk_percentage': 0.0,
            'allocation_method': self.allocation_strategy,
            'confidence_adjustment': 1.0
        }

        # 1. 根据策略类型计算基础仓位
        if self.allocation_strategy == 'progressive':
            base_position = self._calculate_progressive_position(account_balance, current_price, signal)
        elif self.allocation_strategy == 'fixed':
            base_position = self._calculate_fixed_position(account_balance, current_price)
        elif self.allocation_strategy == 'kelly':
            base_position = self._calculate_kelly_position(account_balance, current_price, signal, market_data)
        else:
            base_position = self._calculate_conservative_position(account_balance, current_price)

        # 2. 市场条件调整
        market_adjustment = self._get_market_adjustment(market_data)
        adjusted_position = base_position * market_adjustment

        # 3. 信号置信度调整
        confidence = signal.get('confidence', 0.5)
        confidence_adjustment = 0.5 + confidence  # 0.5-1.5的调整系数
        final_position = adjusted_position * confidence_adjustment

        # 4. 风险控制检查
        max_allowed_position = self._get_max_allowed_position(account_balance, current_price)
        final_position = min(final_position, max_allowed_position)

        # 5. 杠杆优化
        optimal_leverage = self._calculate_optimal_leverage(signal, final_position, current_price)

        # 6. 计算最终参数
        margin_required = (final_position * current_price) / optimal_leverage
        risk_amount = self._calculate_position_risk(final_position, current_price, signal['stop_loss'])

        # 7. 验证资金充足性
        if margin_required > self.available_capital:
            if self.available_capital <= 0:
                final_position = 0.0
                margin_required = 0.0
                risk_amount = 0.0
            else:
                shortage_ratio = margin_required / self.available_capital
                final_position /= shortage_ratio
                margin_required = self.available_capital
                risk_amount = self._calculate_position_risk(final_position, current_price, signal['stop_loss'])

        # 填充配置
        position_config.update({
            'position_size': final_position,
            'leverage': optimal_leverage,
            'margin_required': margin_required,
            'risk_amount': risk_amount,
            'position_value': final_position * current_price,
            'max_loss': abs(current_price - signal['stop_loss']) * final_position,
            'max_profit': abs(signal['target_price'] - current_price) * final_position,
            'risk_percentage': risk_amount / account_balance,
            'confidence_adjustment': confidence_adjustment,
            'market_adjustment': market_adjustment
        })

        return position_config

    def open_position(self, position_config: Dict, signal: Dict, current_price: float) -> str:
        """
        开仓

        Args:
            position_config: 仓位配置
            signal: 交易信号
            current_price: 当前价格

        Returns:
            仓位ID
        """
        position_id = f"pos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 创建持仓对象
        position = Position(
            symbol=SYMBOL,
            direction=signal['direction'],
            size=position_config['position_size'],
            entry_price=current_price,
            entry_time=datetime.now(),
            leverage=position_config['leverage'],
            stop_loss=signal['stop_loss'],
            take_profit=signal['target_price'],
            trade_id=position_id,
            margin_required=position_config['margin_required']
        )

        # 更新资金状态
        self._reserve_margin(position_config['margin_required'])
        self.daily_progressive_state['trade_count'] += 1

        # 记录持仓
        self.open_positions[position_id] = position

        return position_id

    def close_position(self, position_id: str, exit_price: float, reason: str = 'manual') -> Dict:
        """
        平仓

        Args:
            position_id: 仓位ID
            exit_price: 平仓价格
            reason: 平仓原因

        Returns:
            平仓结果
        """
        if position_id not in self.open_positions:
            raise ValueError(f"Position {position_id} not found")

        position = self.open_positions[position_id]

        # 计算盈亏
        pnl = self._calculate_position_pnl(position, exit_price)
        pnl_percentage = pnl / (position.entry_price * position.size)

        # 更新持仓状态
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.realized_pnl = pnl
        position.status = reason

        # 更新资金
        self._release_margin(position.margin_required)
        self.current_capital += pnl

        # 更新渐进式下注状态
        self._update_progressive_state(pnl > 0)

        # 移动持仓到历史记录
        self.closed_positions.append(position)
        del self.open_positions[position_id]

        # 更新统计
        self._update_statistics()

        return {
            'position_id': position_id,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'exit_price': exit_price,
            'exit_reason': reason,
            'holding_time': (position.exit_time - position.entry_time).total_seconds(),
            'current_capital': self.current_capital,
            'daily_profit': self._get_daily_profit()
        }

    def update_positions(self, current_price: float, market_data: Dict) -> Dict:
        """
        更新所有持仓状态

        Args:
            current_price: 当前价格
            market_data: 市场数据

        Returns:
            更新结果
        """
        update_results = {
            'positions_updated': 0,
            'stop_losses_triggered': [],
            'take_profits_triggered': [],
            'margin_warnings': [],
            'liquidation_warnings': []
        }

        for position_id, position in list(self.open_positions.items()):
            # 更新未实现盈亏
            position.unrealized_pnl = self._calculate_position_pnl(position, current_price)

            # 检查止损止盈
            if position.direction == 'long':
                if current_price <= position.stop_loss:
                    result = self.close_position(position_id, position.stop_loss, 'stop_loss')
                    update_results['stop_losses_triggered'].append(result)
                elif current_price >= position.take_profit:
                    result = self.close_position(position_id, position.take_profit, 'take_profit')
                    update_results['take_profits_triggered'].append(result)
            else:  # short position
                if current_price >= position.stop_loss:
                    result = self.close_position(position_id, position.stop_loss, 'stop_loss')
                    update_results['stop_losses_triggered'].append(result)
                elif current_price <= position.take_profit:
                    result = self.close_position(position_id, position.take_profit, 'take_profit')
                    update_results['take_profits_triggered'].append(result)

            # 检查保证金风险
            margin_ratio = self._calculate_margin_ratio(position, current_price)
            if margin_ratio < 0.05:  # 5%维持保证金警告
                update_results['margin_warnings'].append({
                    'position_id': position_id,
                    'margin_ratio': margin_ratio,
                    'liquidation_price': self._calculate_liquidation_price(position)
                })

            if margin_ratio < 0.02:  # 2%濒临爆仓
                update_results['liquidation_warnings'].append({
                    'position_id': position_id,
                    'margin_ratio': margin_ratio,
                    'liquidation_price': self._calculate_liquidation_price(position)
                })

            update_results['positions_updated'] += 1

        return update_results

    def _calculate_progressive_position(self, account_balance: float, current_price: float,
                                       signal: Dict) -> float:
        """计算渐进式仓位大小"""
        # 获取当前交易次数
        trade_num = self.daily_progressive_state['trade_count'] + 1
        entry_price = signal.get('breakout_price', current_price)
        if entry_price <= 0:
            return 0.0

        for betting_config in PROGRESSIVE_BETTING['trades']:
            if betting_config['trade_num'] == trade_num:
                capital_ratio = betting_config['capital_ratio']
                leverage = min(betting_config['leverage'], MAX_LEVERAGE)

                # 计算仓位价值（名义价值）
                margin_to_use = account_balance * capital_ratio
                position_value = margin_to_use * leverage
                return position_value / entry_price

        # 如果超出配置范围，使用最大配置
        max_config = PROGRESSIVE_BETTING['trades'][-1]
        margin_to_use = account_balance * max_config['capital_ratio']
        position_value = margin_to_use * min(max_config['leverage'], MAX_LEVERAGE)
        return position_value / entry_price

    def _calculate_fixed_position(self, account_balance: float, current_price: float) -> float:
        """计算固定仓位大小"""
        risk_per_trade = account_balance * 0.02  # 每笔风险2%
        estimated_loss_ratio = 0.004  # 预期亏损0.4%

        position_value = risk_per_trade / estimated_loss_ratio
        return position_value / max(current_price, 1e-8)

    def _calculate_kelly_position(self, account_balance: float, current_price: float,
                                  signal: Dict, market_data: Dict) -> float:
        """使用凯利公式计算仓位"""
        # 估算胜率
        win_probability = signal.get('confidence', 0.5)
        risk_reward_ratio = signal.get('risk_reward_ratio', 1.5)

        # 凯利公式：f = (p * b - q) / b
        # f: 仓位比例, p: 胜率, b: 赔率, q: 败率 (1-p)
        kelly_fraction = (win_probability * risk_reward_ratio - (1 - win_probability)) / risk_reward_ratio

        # 保守调整，使用半凯利
        kelly_fraction = max(kelly_fraction * 0.5, 0.01)  # 最小1%，最大50%

        position_value = account_balance * kelly_fraction
        return position_value / max(current_price, 1e-8)

    def _calculate_conservative_position(self, account_balance: float, current_price: float) -> float:
        """计算保守仓位大小"""
        position_value = account_balance * 0.1  # 最大使用10%资金
        leverage = DEFAULT_LEVERAGE * 0.5
        notional_value = position_value * leverage
        return notional_value / max(current_price, 1e-8)

    def _get_market_adjustment(self, market_data: Dict) -> float:
        """根据市场条件调整仓位"""
        adjustment = 1.0

        # 根据波动性调整
        volatility = market_data.get('volatility', 0.02)
        if volatility > 0.04:  # 高波动
            adjustment *= 0.7
        elif volatility < 0.01:  # 低波动
            adjustment *= 1.2

        # 根据流动性调整
        volume_ratio = market_data.get('volume_ratio', 1.0)
        if volume_ratio < 0.5:  # 低流动性
            adjustment *= 0.5
        elif volume_ratio > 2.0:  # 高流动性
            adjustment *= 1.1

        return max(adjustment, 0.3)  # 最小30%仓位

    def _get_max_allowed_position(self, account_balance: float, current_price: float) -> float:
        """获取最大允许仓位"""
        # 考虑gate.io VIP0限制
        gateio_max_position = GATEIO_RULES['max_position_value'] / current_price

        # 考虑可用资金
        capital_limit = self.available_capital * 0.8 / current_price  # 保留20%作为保证金

        return min(gateio_max_position, capital_limit)

    def _calculate_optimal_leverage(self, signal: Dict, position_size: float, current_price: float) -> int:
        """计算最优杠杆"""
        base_leverage = DEFAULT_LEVERAGE

        # 根据信号强度调整
        confidence = signal.get('confidence', 0.5)
        leverage_adjustment = 0.5 + confidence  # 0.5-1.5调整

        optimal_leverage = int(base_leverage * leverage_adjustment)

        # 确保在合理范围内
        optimal_leverage = max(min(optimal_leverage, MAX_LEVERAGE), 10)

        # 检查gate.io限制
        return min(optimal_leverage, MAX_LEVERAGE)

    def _calculate_position_risk(self, position_size: float, current_price: float, stop_loss: float) -> float:
        """计算仓位风险金额"""
        position_value = position_size * current_price
        risk_percentage = abs(current_price - stop_loss) / current_price
        return position_value * risk_percentage

    def _reserve_margin(self, margin_amount: float):
        """预留保证金"""
        self.reserved_capital += margin_amount
        self.available_capital = self.current_capital - self.reserved_capital

    def _release_margin(self, margin_amount: float):
        """释放保证金"""
        self.reserved_capital = max(0, self.reserved_capital - margin_amount)
        self.available_capital = self.current_capital - self.reserved_capital

    def _calculate_position_pnl(self, position: Position, exit_price: float) -> float:
        """计算持仓盈亏"""
        if position.direction == 'long':
            return (exit_price - position.entry_price) * position.size
        else:
            return (position.entry_price - exit_price) * position.size

    def _update_progressive_state(self, trade_successful: bool):
        """更新渐进式下注状态"""
        if trade_successful:
            self.daily_progressive_state['successful_trades'] += 1
        else:
            self.daily_progressive_state['failed_trades'] += 1

        # 检查是否需要重置
        if (trade_successful and
            self.daily_progressive_state['successful_trades'] >= PROGRESSIVE_BETTING['trades'][-1]['trade_num']):
            # 完成所有交易，状态归零
            self._reset_progressive_state()
        elif (not trade_successful and PROGRESSIVE_BETTING['reset_on_loss']):
            # 亏损后重置
            self._reset_progressive_state()

    def _reset_progressive_state(self):
        """重置渐进式下注状态"""
        self.daily_progressive_state.update({
            'trade_count': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'current_betting_level': 1,
            'betting_reset_needed': False
        })

    def _calculate_margin_ratio(self, position: Position, current_price: float) -> float:
        """计算保证金比例"""
        position_value = position.size * current_price
        margin_required = position_value / position.leverage
        unrealized_pnl = self._calculate_position_pnl(position, current_price)
        equity = margin_required + unrealized_pnl

        return equity / position_value

    def _calculate_liquidation_price(self, position: Position) -> float:
        """计算爆仓价格"""
        maintenance_margin = 0.005  # gate.io维持保证金率

        if position.direction == 'long':
            return position.entry_price * (1 - 1/position.leverage + maintenance_margin)
        else:
            return position.entry_price * (1 + 1/position.leverage - maintenance_margin)

    def _update_statistics(self):
        """更新统计数据"""
        self.total_capital = self.current_capital + sum(pos.unrealized_pnl for pos in self.open_positions.values())
        self.available_capital = self.current_capital - self.reserved_capital

    def _get_daily_profit(self) -> float:
        """获取当日盈利"""
        today = datetime.now().date()
        today_trades = [pos for pos in self.closed_positions if pos.exit_time.date() == today]
        return sum(pos.realized_pnl for pos in today_trades)

    def get_portfolio_summary(self) -> Dict:
        """获取投资组合摘要"""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.open_positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.closed_positions)

        return {
            'current_capital': self.current_capital,
            'available_capital': self.available_capital,
            'reserved_capital': self.reserved_capital,
            'total_capital': self.total_capital,
            'open_positions': len(self.open_positions),
            'closed_positions_today': len([pos for pos in self.closed_positions
                                         if pos.exit_time and pos.exit_time.date() == datetime.now().date()]),
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': total_realized_pnl,
            'daily_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'progressive_state': self.daily_progressive_state,
            'allocation_strategy': self.allocation_strategy
        }

    def reset_daily_state(self):
        """重置每日状态"""
        self.daily_progressive_state = {
            'trade_count': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'current_betting_level': 1,
            'daily_start_capital': self.current_capital,
            'max_capital_today': self.current_capital,
            'betting_reset_needed': False
        }

        # 清理过期的历史持仓
        cutoff_date = datetime.now() - timedelta(days=7)
        self.position_history = [pos for pos in self.position_history
                               if pos.exit_time and pos.exit_time > cutoff_date]
