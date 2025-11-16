# -*- coding: utf-8 -*-
"""
交易执行引擎
整合所有模块，实现完整的高频剥头皮交易系统
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# 导入自定义模块
from gateio_config import *
from range_detector import RangeDetector
from technical_indicators import TechnicalIndicators
from risk_management import RiskManager
from position_manager import PositionManager

class TradingEngine:
    """高频剥头皮交易引擎"""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        # 初始化各个模块
        self.range_detector = RangeDetector()
        self.technical_indicators = TechnicalIndicators()
        self.risk_manager = RiskManager(initial_capital)
        self.position_manager = PositionManager(initial_capital)

        # 市场数据
        self.current_price = 0.0
        self.market_data = {}
        self.kline_data = pd.DataFrame()

        # 交易状态
        self.is_running = False
        self.last_signal_time = None
        self.signal_cooldown = 60  # 信号冷却时间（秒）

        # 性能统计
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'avg_holding_time': 0.0,
            'signals_generated': 0,
            'signals_executed': 0
        }

        # 设置日志
        self._setup_logging()

    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trading_engine.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def start(self):
        """启动交易引擎"""
        self.logger.info("🚀 启动高频剥头皮交易引擎")
        self.is_running = True

        try:
            await self._trading_loop()
        except KeyboardInterrupt:
            self.logger.info("⏹️ 收到停止信号，正在关闭交易引擎")
        except Exception as e:
            self.logger.error(f"❌ 交易引擎运行错误: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """停止交易引擎"""
        self.logger.info("🛑 停止交易引擎")
        self.is_running = False

        # 平仓所有持仓
        await self._close_all_positions()

        # 生成最终报告
        self._generate_final_report()

    async def _trading_loop(self):
        """主交易循环"""
        self.logger.info("🔄 开始主交易循环")

        while self.is_running:
            try:
                # 1. 更新市场数据
                await self._update_market_data()

                # 2. 检查交易时段
                if not self._is_active_trading_time():
                    await asyncio.sleep(60)  # 非交易时段，等待1分钟
                    continue

                # 3. 检查风险管理状态
                if not self._check_risk_permission():
                    await asyncio.sleep(30)  # 风险控制中，等待30秒
                    continue

                # 4. 更新现有持仓
                await self._update_positions()

                # 5. 生成交易信号
                signal = await self._generate_signal()

                if signal:
                    self.stats['signals_generated'] += 1
                    self.logger.info(f"📈 生成交易信号: {signal['type']} - 置信度: {signal.get('confidence', 0):.2f}")

                    # 6. 执行交易
                    await self._execute_signal(signal)

                # 7. 短暂休息
                await asyncio.sleep(5)  # 5秒检查间隔

            except Exception as e:
                self.logger.error(f"⚠️ 交易循环错误: {e}")
                await asyncio.sleep(10)  # 错误后等待10秒

    async def _update_market_data(self):
        """更新市场数据"""
        # 这里应该连接gate.io API获取实时数据
        # 模拟数据更新
        try:
            # 获取K线数据 (15分钟)
            # 实际实现中应该调用 gate.io API
            self.kline_data = await self._fetch_kline_data('15m', 100)

            # 获取当前价格和深度
            self.current_price = self.kline_data['close'].iloc[-1]

            # 计算技术指标
            indicators = self.technical_indicators.calculate_all_indicators(self.kline_data)
            volume_ratio = indicators.get('volume_ratio', 1.0)
            if hasattr(volume_ratio, 'iloc'):
                try:
                    volume_ratio_value = float(volume_ratio.iloc[-1])
                except (IndexError, TypeError, ValueError):
                    volume_ratio_value = 1.0
            elif isinstance(volume_ratio, (list, tuple, np.ndarray)):
                volume_ratio_value = float(volume_ratio[-1]) if len(volume_ratio) else 1.0
            else:
                volume_ratio_value = float(volume_ratio) if volume_ratio not in (None, '') else 1.0

            self.market_data = {
                'current_price': self.current_price,
                'indicators': indicators,
                'volume_ratio': volume_ratio_value,
                'volatility': indicators.get('historical_volatility', 0.02),
                'trend': indicators.get('ema_trend', 'neutral'),
                'timestamp': datetime.now()
            }

        except Exception as e:
            self.logger.error(f"❌ 更新市场数据失败: {e}")

    async def _fetch_kline_data(self, interval: str, limit: int) -> pd.DataFrame:
        """
        获取K线数据 (模拟)
        实际实现中需要调用 gate.io API
        """
        # 这里应该是实际的API调用
        # 现在返回模拟数据用于演示
        np.random.seed(int(time.time()))

        timestamps = pd.date_range(end=datetime.now(), periods=limit, freq='15T')
        base_price = 3500 + np.random.randn() * 100

        # 生成模拟K线数据
        price_changes = np.random.randn(limit) * 0.005  # 0.5%波动
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        # 生成OHLCV数据
        data = []
        for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
            high = close_price * (1 + abs(np.random.randn() * 0.002))
            low = close_price * (1 - abs(np.random.randn() * 0.002))
            open_price = low if i > 0 and prices[i-1] > close_price else high
            volume = np.random.randint(1000, 5000)

            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })

        return pd.DataFrame(data)

    def _is_active_trading_time(self) -> bool:
        """检查是否为活跃交易时间"""
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute

        # 检查是否在活跃时段
        for session_name, time_ranges in ACTIVE_TRADING_HOURS.items():
            for start_hour, end_hour in time_ranges:
                if start_hour <= current_hour < end_hour:
                    return True

        # 避开资金费率收取时间前后15分钟
        for funding_hour in FUNDING_RATE_TIMES:
            if current_hour == funding_hour and current_minute < 15:
                return False

        return False

    def _check_risk_permission(self) -> bool:
        """检查是否获得风险许可"""
        risk_report = self.risk_manager.get_daily_risk_report()

        if risk_report['trading_paused']:
            self.logger.warning(f"⛔ 交易被暂停: {risk_report['pause_reason']}")
            return False

        if risk_report['risk_score'] > 0.8:
            self.logger.warning(f"⚠️ 风险评分过高: {risk_report['risk_score']:.2f}")
            return False

        return True

    async def _update_positions(self):
        """更新现有持仓"""
        if not self.position_manager.open_positions:
            return

        try:
            update_result = self.position_manager.update_positions(
                self.current_price, self.market_data
            )

            # 处理触发的止损止盈
            for stop_loss_result in update_result['stop_losses_triggered']:
                self.logger.info(f"🛑 止损触发: {stop_loss_result['pnl']:.2f} USDT")
                self._update_trade_statistics(stop_loss_result)

            for take_profit_result in update_result['take_profits_triggered']:
                self.logger.info(f"🎯 止盈触发: {take_profit_result['pnl']:.2f} USDT")
                self._update_trade_statistics(take_profit_result)

            # 处理保证金警告
            for warning in update_result['margin_warnings']:
                self.logger.warning(f"⚠️ 保证金警告: 仓位 {warning['position_id']} 保证金比例 {warning['margin_ratio']:.2%}")

            for warning in update_result['liquidation_warnings']:
                self.logger.error(f"🚨 爆仓警告: 仓位 {warning['position_id']} 即将爆仓")

        except Exception as e:
            self.logger.error(f"❌ 更新持仓失败: {e}")

    async def _generate_signal(self) -> Optional[Dict]:
        """生成交易信号"""
        # 检查信号冷却时间
        if (self.last_signal_time and
            (datetime.now() - self.last_signal_time).total_seconds() < self.signal_cooldown):
            return None

        try:
            # 1. 检测震荡区间
            range_info = self.range_detector.detect_consolidation_range(self.kline_data)

            if not range_info:
                return None

            # 2. 检测突破信号
            breakout_signal = self.range_detector.detect_breakout_signal(self.kline_data, range_info)

            if not breakout_signal:
                return None

            # 3. 技术指标确认
            indicators = self.market_data['indicators']
            signal_summary = self.technical_indicators.get_trading_signal_summary(indicators)

            # 4. 综合信号评分
            confidence = self._calculate_signal_confidence(breakout_signal, signal_summary, range_info)

            if confidence < 0.6:  # 置信度阈值
                return None

            # 5. 完善信号信息
            signal = {
                **breakout_signal,
                'range_info': range_info,
                'technical_summary': signal_summary,
                'market_data': self.market_data,
                'generated_time': datetime.now()
            }

            self.last_signal_time = datetime.now()
            return signal

        except Exception as e:
            self.logger.error(f"❌ 生成信号失败: {e}")
            return None

    def _calculate_signal_confidence(self, breakout_signal: Dict, signal_summary: Dict,
                                   range_info: Dict) -> float:
        """计算综合信号置信度"""
        confidence = 0.0

        # 突破强度 (30%)
        confidence += breakout_signal.get('breakout_strength', 0) * 0.3

        # 成交量确认 (25%)
        if breakout_signal.get('volume_confirmation', False):
            confidence += 0.25

        # 技术指标配合 (25%)
        trend_score = 0.5
        if signal_summary['trend_signal'] == breakout_signal['direction']:
            trend_score = 0.8
        confidence += trend_score * 0.25

        # 震荡区间质量 (20%)
        range_quality = min(range_info.get('price_distribution', {}).get('uniformity', 0.5) * 1.5, 1.0)
        confidence += range_quality * 0.2

        return min(confidence, 1.0)

    async def _execute_signal(self, signal: Dict):
        """执行交易信号"""
        try:
            # 1. 风险评估
            risk_allowed, risk_assessment = self.risk_manager.evaluate_entry_risk(
                signal, self.current_price, self.position_manager.current_capital
            )

            if not risk_allowed:
                self.logger.warning(f"⚠️ 风险评估未通过: {risk_assessment['warnings']}")
                return

            # 2. 计算仓位
            position_config = self.position_manager.calculate_optimal_position(
                signal, self.current_price, self.position_manager.current_capital, self.market_data
            )

            # 3. 开仓
            position_id = self.position_manager.open_position(position_config, signal, self.current_price)

            # 4. 设置止损止盈 (这里应该调用实际的API)
            await self._place_stop_loss_order(position_id, signal['stop_loss'])
            await self._place_take_profit_order(position_id, signal['target_price'])

            self.logger.info(f"✅ 开仓成功: {position_id} - {signal['direction']} "
                           f"大小: {position_config['position_size']:.4f} - 杠杆: {position_config['leverage']}x")

            self.stats['signals_executed'] += 1

        except Exception as e:
            self.logger.error(f"❌ 执行信号失败: {e}")

    async def _place_stop_loss_order(self, position_id: str, stop_loss_price: float):
        """设置止损订单 (模拟)"""
        # 实际实现中应该调用 gate.io API
        self.logger.debug(f"🔒 设置止损订单: {position_id} @ {stop_loss_price}")

    async def _place_take_profit_order(self, position_id: str, take_profit_price: float):
        """设置止盈订单 (模拟)"""
        # 实际实现中应该调用 gate.io API
        self.logger.debug(f"🎯 设置止盈订单: {position_id} @ {take_profit_price}")

    async def _close_all_positions(self):
        """平仓所有持仓"""
        if not self.position_manager.open_positions:
            return

        self.logger.info("🔄 正在平仓所有持仓")

        for position_id in list(self.position_manager.open_positions.keys()):
            try:
                # 这里应该调用实际的平仓API
                position = self.position_manager.open_positions[position_id]
                result = self.position_manager.close_position(position_id, self.current_price, 'manual_close')
                self.logger.info(f"✅ 仓位已平仓: {position_id} - 盈亏: {result['pnl']:.2f} USDT")
            except Exception as e:
                self.logger.error(f"❌ 平仓失败 {position_id}: {e}")

    def _update_trade_statistics(self, trade_result: Dict):
        """更新交易统计"""
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += trade_result['pnl']

        if trade_result['pnl'] > 0:
            self.stats['winning_trades'] += 1
            self.stats['best_trade'] = max(self.stats['best_trade'], trade_result['pnl'])
        else:
            self.stats['losing_trades'] += 1
            self.stats['worst_trade'] = min(self.stats['worst_trade'], trade_result['pnl'])

        # 更新持仓时间统计
        holding_time = trade_result['holding_time']
        total_time = self.stats['avg_holding_time'] * (self.stats['total_trades'] - 1)
        self.stats['avg_holding_time'] = (total_time + holding_time) / self.stats['total_trades']

        # 更新风险管理器
        self.risk_manager.update_trade_result(trade_result)

    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        portfolio = self.position_manager.get_portfolio_summary()
        risk_report = self.risk_manager.get_daily_risk_report()

        win_rate = (self.stats['winning_trades'] / self.stats['total_trades']
                   if self.stats['total_trades'] > 0 else 0)

        profit_factor = (abs(sum(t['pnl'] for t in self.risk_manager.trades_today if t['pnl'] > 0)) /
                        abs(sum(t['pnl'] for t in self.risk_manager.trades_today if t['pnl'] < 0))
                        if self.risk_manager.trades_today else float('inf'))

        return {
            'basic_stats': {
                'total_trades': self.stats['total_trades'],
                'win_rate': f"{win_rate:.2%}",
                'profit_factor': f"{profit_factor:.2f}",
                'total_pnl': f"{self.stats['total_pnl']:.2f} USDT",
                'best_trade': f"{self.stats['best_trade']:.2f} USDT",
                'worst_trade': f"{self.stats['worst_trade']:.2f} USDT"
            },
            'portfolio': {
                'current_capital': f"{portfolio['current_capital']:.2f} USDT",
                'daily_return': f"{portfolio['daily_return']:.2%}",
                'open_positions': portfolio['open_positions'],
                'total_unrealized_pnl': f"{portfolio['total_unrealized_pnl']:.2f} USDT"
            },
            'risk_metrics': {
                'risk_score': f"{risk_report['risk_score']:.2f}",
                'max_drawdown': f"{risk_report['max_drawdown']:.2%}",
                'consecutive_losses': risk_report['consecutive_losses'],
                'trading_paused': risk_report['trading_paused']
            },
            'signal_performance': {
                'signals_generated': self.stats['signals_generated'],
                'signals_executed': self.stats['signals_executed'],
                'execution_rate': f"{(self.stats['signals_executed'] / self.stats['signals_generated'] * 100) if self.stats['signals_generated'] > 0 else 0:.1f}%"
            }
        }

    def _generate_final_report(self):
        """生成最终报告"""
        report = self.get_performance_report()

        self.logger.info("📊 === 交易引擎最终报告 ===")
        self.logger.info(f"💰 总资金: {report['portfolio']['current_capital']}")
        self.logger.info(f"📈 总交易次数: {report['basic_stats']['total_trades']}")
        self.logger.info(f"🎯 胜率: {report['basic_stats']['win_rate']}")
        self.logger.info(f"💎 总盈亏: {report['basic_stats']['total_pnl']}")
        self.logger.info(f"⚡ 平均持仓时间: {self.stats['avg_holding_time']:.1f}秒")
        self.logger.info(f"🛡️ 最大回撤: {report['risk_metrics']['max_drawdown']}")
        self.logger.info("=" * 40)

# 主程序入口
async def main():
    """主程序入口"""
    print("🎯 Gate.io ETH高频剥头皮交易系统")
    print("=" * 50)
    print(f"💰 初始资金: {INITIAL_CAPITAL} USDT")
    print(f"🎯 目标策略: 震荡区间突破剥头皮")
    print(f"⚡ 杠杆设置: {DEFAULT_LEVERAGE}x")
    print(f"📊 技术指标: EMA(9,21) + RSI(14) + 成交量")
    print("=" * 50)

    # 创建并启动交易引擎
    engine = TradingEngine(INITIAL_CAPITAL)

    try:
        await engine.start()
    except KeyboardInterrupt:
        print("\n⏹️ 用户停止程序")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
    finally:
        print("\n👋 程序结束")

if __name__ == "__main__":
    asyncio.run(main())
