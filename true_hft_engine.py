# -*- coding: utf-8 -*-
"""
真正的高频交易引擎（Tick级）
"""

import asyncio
import math
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Deque, Tuple, List

from collections import deque

from gateio_config import SYMBOL, INITIAL_CAPITAL, TAKER_FEE_RATE, MAKER_FEE_RATE, FUNDING_RATE_TIMES, CONTRACT_VALUE
from gateio_api import GateIOAPI
from api_config import load_config
from gateio_ws import GateIOWebsocket
from hft_config import HFT_CONFIG
from hft_data_manager import HFTDataManager
from hft_signal_generator import HFTSignalGenerator
from hft_executor import HFTExecutor
from hft_performance import HFTPerformanceMonitor
from aggressive_position_manager import AggressivePositionManager
from survival_rules import SurvivalRules
from tui_display import LiveTickerDisplay


class TrueHFTEngine:
    """真正的高频 (Tick级) 交易引擎"""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL, use_real_api: bool = False):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        self.config = dict(HFT_CONFIG)
        self.signal_only_mode = self.config.get('signal_only_mode', False)
        self.use_real_api = use_real_api and not self.signal_only_mode
        self.enable_signal_debug = self.config.get('enable_signal_debug_log', True)
        self.enable_trend_log = self.config.get('enable_trend_log', True)
        self.trend_bias_threshold = float(self.config.get('trend_bias_trade_threshold', 0.0))
        self.enable_trend_component_log = self.config.get('enable_trend_component_log', False)
        self.trend_component_log_interval = float(self.config.get('trend_component_log_interval', 3.0))

        self.api_client = None
        self.ws_client: Optional[GateIOWebsocket] = GateIOWebsocket(
            contract=SYMBOL,
            order_book_interval="100ms",
            order_book_depth=20
        )
        if self.use_real_api:
            if load_config():
                self.api_client = GateIOAPI(enable_market_data=False, enable_trading=True)
            else:
                self.use_real_api = False

        self.data_manager = HFTDataManager()
        self.signal_generator = HFTSignalGenerator(
            momentum_window=self.config['momentum_window'],
            momentum_threshold=self.config['momentum_threshold'],
            volume_threshold=self.config['volume_spike_min'],
            imbalance_threshold=self.config['order_imbalance_min'],
            enable_volatility_filter=self.config.get('volatility_filter', False),
            volatility_threshold=self.config.get('volatility_threshold', 0.001),
            require_orderbook_confirm=self.config.get('require_orderbook_confirm', False),
            orderbook_ratio_threshold=self.config.get('orderbook_ratio_threshold', 0.8),
            avoid_funding_hours=self.config.get('avoid_funding_hours', False),
            funding_rate_times=FUNDING_RATE_TIMES,
            entry_threshold=self.config.get('composite_entry_threshold', 0.25),
            market_volatility_threshold=self.config.get('market_volatility_threshold', 0.0003),
            min_confidence=self.config.get('min_confidence', 0.5)
        )
        self.position_manager = AggressivePositionManager(leverage=self.config['leverage'])
        executor_client = self.api_client if self.use_real_api else None
        self.executor = HFTExecutor(api_client=executor_client, min_order_interval=0.1)
        self.performance_monitor = HFTPerformanceMonitor()
        self.survival_rules = SurvivalRules()

        self.open_positions: Dict[str, Dict] = {}
        self.trade_history = deque(maxlen=2000)
        self.trade_timestamps: Deque[datetime] = deque(maxlen=1000)
        self.last_signal_time = None
        self.last_cycle_time = time.time()
        self._trading_halted = False
        self._halt_reason = ""
        self._active_stop_orders: Dict[str, str] = {}
        self._trades_today: Deque[Dict] = deque(maxlen=500)
        self._trades_today_date = datetime.utcnow().date()
        self._last_known_price: Optional[float] = None
        self._last_rate_limit_warning = 0.0
        self._last_trend_log = 0.0
        self._last_trend_component_log = 0.0
        self._last_balance_refresh = 0.0
        self._last_signal_debug_log = 0.0

        self.stats = {
            'signals_generated': 0,
            'signals_executed': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'long_trades': 0,
            'long_wins': 0,
            'short_trades': 0,
            'short_wins': 0,
            'positions_opened': 0
        }

        self.is_running = False
        self.tui_display: Optional[LiveTickerDisplay] = None
        self._last_margin_warning_time = 0.0
        self._last_available_margin = 0.0
        self._last_account_info: Dict[str, float] = {'available': self.current_capital, 'total': self.current_capital}
        self.market_data: Dict[str, float] = {
            'current_price': self.current_capital
        }
        self.recent_logs: Deque[str] = deque(maxlen=50)
        log_capacity = int(self.config.get('signal_log_max_entries', 500))
        self.signal_events: Deque[Dict] = deque(maxlen=max(log_capacity, 50))
        self._last_trade_direction: Optional[str] = None
        self._last_trade_timestamp: float = 0.0
        self._last_trade_pnl: Optional[float] = None
        self._last_direction_block_log = 0.0
        if self.signal_only_mode:
            self._log("🧪 已启用信号记录模式，仅跟踪信号与虚拟盈亏")

    async def start(self):
        self.is_running = True
        if self.ws_client:
            await self.ws_client.start()

        if self.use_real_api and self.api_client:
            await self._apply_live_trading_settings()
            await self._refresh_account_info()

        self.logger = True
        try:
            await self._hft_trading_loop()
        finally:
            if self.ws_client:
                await self.ws_client.stop()

    async def _hft_trading_loop(self):
        self._log("🚀 启动真正高频交易循环")

        while self.is_running:
            loop_start = time.time()

            try:
                await self._handle_websocket_data()
                if self.use_real_api and time.time() - self._last_balance_refresh > 30:
                    await self._refresh_account_info()

                if len(self.data_manager.data_buffers['ticks']) >= 10 and not self._trading_halted:
                    trend_bias = self._multi_timeframe_trend_bias()
                    orderbook = None
                    if self.ws_client:
                        book_updates = self.ws_client.get_recent_order_book_updates(limit=1)
                        orderbook = book_updates[0] if book_updates else None
                    if orderbook:
                        self.data_manager.update_orderbook(orderbook)
                    higher_timeframes = {'180s': self.data_manager.data_buffers['180s']}
                    higher_timeframes['orderbook_metrics'] = self.data_manager.get_orderbook_metrics()
                    composite_info = self.signal_generator._calc_composite_trend(
                        self.data_manager.data_buffers['ticks'],
                        orderbook=orderbook,
                        higher_timeframes=higher_timeframes
                    )
                    market_state = self.signal_generator._assess_market_state(
                        self.data_manager.data_buffers['ticks'],
                        higher_timeframes.get('orderbook_metrics'),
                        None
                    )
                    now_ts = time.time()
                    if self.enable_trend_log and now_ts - self._last_trend_log > 1:
                        self._log(
                            f"📊 趋势分析: bias {trend_bias if trend_bias is not None else 'N/A'} | "
                            f"composite {composite_info.get('score', 0):.2f} | state {market_state}"
                        )
                        self._last_trend_log = now_ts
                    signal = self.signal_generator.generate_tick_signal(
                        self.data_manager.data_buffers['ticks'],
                        trend_bias=trend_bias,
                        orderbook=orderbook,
                        higher_timeframes=higher_timeframes
                    )
                    if self.enable_signal_debug:
                        debug_info = getattr(self.signal_generator, 'last_debug', {})
                        if (not signal) and debug_info and time.time() - self._last_signal_debug_log > 1:
                            comp_val = debug_info.get('composite')
                            comp_display = f"{comp_val:.2f}" if isinstance(comp_val, (int, float)) else comp_val
                            self._log(
                                f"🔍 信号调试: reason={debug_info.get('reason')} "
                                f"| L1={debug_info.get('l1')} L2={debug_info.get('l2')} L3={debug_info.get('l3')} "
                                f"| cond={debug_info.get('condition_count')} dir={debug_info.get('desired_direction')} "
                                f"| consistent={debug_info.get('direction_consistent')} "
                                f"| comp={comp_display}"
                            )
                            self._last_signal_debug_log = time.time()
                    if signal and self._confirm_trend_direction(signal, trend_bias):
                        self._log_trend_analysis(trend_bias, signal)
                        await self._execute_signal(signal)
                        self._log(
                            f"🎯 信号: {signal['direction']} 信心 {signal['confidence']:.2f} | "
                            f"trend {signal.get('composite_trend', 0):.2f} | state {signal.get('market_state')}"
                        )

                await self._monitor_positions()

                cycle_time = time.time() - loop_start
                self.performance_monitor.record_latency('total_cycle', cycle_time)
                sleep_time = max(0, self.config['tick_processing_interval'] - cycle_time)
                await asyncio.sleep(sleep_time)

            except Exception as exc:
                self._log(f"⚠️ 高频循环错误: {exc}")
                await asyncio.sleep(0.1)

    async def _handle_websocket_data(self):
        if not self.ws_client:
            return

        ticker = self.ws_client.get_latest_ticker()
        trades = self.ws_client.get_recent_trade_updates(limit=50)

        if ticker:
            await self.data_manager.process_ticker(ticker)
            self._last_known_price = ticker.get('last_price') or ticker.get('mark_price') or self._last_known_price
            if ticker.get('last_price'):
                self.market_data['current_price'] = ticker['last_price']
            elif ticker.get('mark_price'):
                self.market_data['current_price'] = ticker['mark_price']
        for trade in trades:
            await self.data_manager.process_trade(trade)
            trade_price = trade.get('price')
            if trade_price is not None:
                self._last_known_price = trade_price
                self.market_data['current_price'] = trade_price

    async def _execute_signal(self, signal: Dict):
        now = datetime.utcnow()
        if self.last_signal_time and (now - self.last_signal_time).total_seconds() < self.config['signal_refresh_rate']:
            return
        if self.stats['total_trades'] >= self.config['daily_trade_limit']:
            return
        if self.open_positions and not self.signal_only_mode:
            return
        if not self._can_submit_trade(now):
            return
        if self._direction_cooldown_active(signal['direction']):
            return

        entry_price = self._current_price()
        self._record_signal_event(signal, entry_price)
        fixed_margin = await self._fixed_margin()
        if fixed_margin is None:
            return
        if self.use_real_api:
            theoretical_min = (CONTRACT_VALUE * entry_price) / max(self.config['leverage'], 1e-8)
            min_required = max(theoretical_min, self.config.get('min_contract_margin', theoretical_min))
            available_now = self._last_available_margin
            if available_now < min_required:
                self._log(f"⚠️ 实时余额不足，跳过信号 (需要 {min_required:.4f} USDT, 可用 {available_now:.4f} USDT)")
                return
        adjusted_margin = self._apply_progressive_margin(fixed_margin)
        position_config = self._calculate_position(signal, adjusted_margin)
        if not position_config:
            return
        if self.signal_only_mode:
            result = {'status': 'simulated'}
        else:
            result = await self.executor.execute(
                signal,
                position_config,
                current_price=entry_price,
                prefer_limit=self.config.get('use_limit_orders', True),
                limit_premium=self.config.get('limit_order_premium', 0.0005),
                limit_timeout=self.config.get('limit_order_timeout', 0.1),
                fallback_to_market=self.config.get('fallback_to_market', True)
            )
        self.last_signal_time = now
        self.stats['signals_generated'] += 1
        if result.get('status') not in ('filled', 'simulated'):
            return

        if result.get('status') in ('filled', 'simulated'):
            position_id = f"hft_{now.strftime('%H%M%S_%f')}"
            stop_loss = entry_price * (1 - self.config['stop_loss_ratio']
                                       if signal['direction'] == 'long'
                                       else 1 + self.config['stop_loss_ratio'])
            target_price = entry_price * (1 + self.config['target_profit_ratio']
                                          if signal['direction'] == 'long'
                                          else 1 - self.config['target_profit_ratio'])
            position = {
                'id': position_id,
                'direction': signal['direction'],
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'size': position_config['position_size'],
                'entry_time': now,
                'remaining_ratio': 1.0,
                'partial_taken': False
            }
            self.open_positions[position_id] = position
            self.trade_timestamps.append(now)
            self.stats['signals_executed'] += 1
            self.stats['positions_opened'] += 1
            if self.signal_only_mode:
                self._log(
                    f"📘 虚拟开仓 {signal['direction']} @ {entry_price:.4f} | "
                    f"止盈 {target_price:.4f} / 止损 {stop_loss:.4f} | conf {signal['confidence']:.2f} "
                    f"| 累计 {self.stats['positions_opened']}"
                )
            if self.use_real_api and self.api_client and not self.signal_only_mode and self.config.get('enable_protective_stops', True):
                await self._register_protective_stop(position_id, signal, position)

    async def _monitor_positions(self):
        if not self.open_positions:
            return
        price = self._current_price()
        to_close = []
        duration_limit = max(self.config.get('max_position_duration', 0), 0)
        min_hold = max(self.config.get('min_position_hold_secs', 0), 0)
        for pid, position in list(self.open_positions.items()):
            elapsed = (datetime.utcnow() - position['entry_time']).total_seconds()
            stop_hit = price <= position['stop_loss'] if position['direction'] == 'long' else price >= position['stop_loss']
            target_hit = price >= position['target_price'] if position['direction'] == 'long' else price <= position['target_price']

            if self.config.get('enable_trailing_stop', False):
                new_stop = self._calculate_trailing_stop(position, price)
                if new_stop and abs(new_stop - position['stop_loss']) / max(position['stop_loss'], 1e-8) > 1e-4:
                    position['stop_loss'] = new_stop
                    self._log(f"📈 移动止损更新至 {new_stop:.2f}")

            partial_ratio = self._check_partial_profit(position, price)
            if partial_ratio:
                await self._close_partial_position(pid, partial_ratio, price)

            timeout_hit = duration_limit > 0 and elapsed > duration_limit
            should_close = stop_hit or target_hit or timeout_hit
            if should_close and not stop_hit and min_hold > 0 and elapsed < min_hold:
                continue
            if should_close:
                reason = 'stop_loss' if stop_hit else 'take_profit' if target_hit else 'timeout'
                to_close.append((pid, reason))

        for pid, reason in to_close:
            await self._close_position(pid, reason)

        if self._trading_halted and not self.open_positions:
            self.is_running = False

    async def _close_position(self, position_id: str, reason: str):
        position = self.open_positions.pop(position_id, None)
        if not position:
            return
        await self._cancel_protective_stop(position_id)
        price = self._current_price()
        remaining_ratio = max(position.get('remaining_ratio', 1.0), 0.0)
        if remaining_ratio <= 0:
            return

        pnl = self._calculate_ratio_pnl(position, price, remaining_ratio)
        fee_rate = TAKER_FEE_RATE or 0.0
        funding_rate = self._current_funding_rate()
        position_value = self.current_capital * self.config['leverage'] * remaining_ratio
        total_fee = position_value * (fee_rate + funding_rate)
        pnl -= total_fee
        self.current_capital += pnl
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += pnl
        if pnl > 0:
            self.stats['winning_trades'] += 1
        direction = position['direction']
        if direction == 'long':
            self.stats['long_trades'] += 1
            if pnl > 0:
                self.stats['long_wins'] += 1
        else:
            self.stats['short_trades'] += 1
            if pnl > 0:
                self.stats['short_wins'] += 1
        trade = {
            'position_id': position_id,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': price,
            'pnl': pnl,
            'exit_reason': reason,
            'timestamp': datetime.utcnow()
        }
        self.trade_history.append(trade)
        self.position_manager.update_streak(pnl)
        self._record_trade_for_survival(trade)
        self._enforce_survival_rules()
        self._last_trade_direction = position['direction']
        self._last_trade_timestamp = time.time()
        self._last_trade_pnl = pnl
        trade_label = "虚拟" if self.signal_only_mode else "真实"
        self._log(
            f"📙 {trade_label}平仓 {position['direction']} @ {price:.4f} | PnL {pnl:.4f} USDT | 原价 {position['entry_price']:.4f}"
        )
        total = max(self.stats['total_trades'], 1)
        win_rate = self.stats['winning_trades'] / total
        long_rate = self.stats['long_wins'] / max(self.stats['long_trades'], 1) if self.stats['long_trades'] else 0.0
        short_rate = self.stats['short_wins'] / max(self.stats['short_trades'], 1) if self.stats['short_trades'] else 0.0
        self._log(
            f"📊 当前胜率 {win_rate:.2f} ({self.stats['winning_trades']}/{self.stats['total_trades']}) | "
            f"多头 {long_rate:.2f} ({self.stats['long_wins']}/{max(self.stats['long_trades'],1)}) | "
            f"空头 {short_rate:.2f} ({self.stats['short_wins']}/{max(self.stats['short_trades'],1)}) | "
            f"累计{trade_label}仓 {self.stats['positions_opened']}"
        )
        self._auto_tune_parameters()

    async def _fixed_margin(self) -> Optional[float]:
        margin_cap = self.config.get('fixed_margin', 1.0)
        if margin_cap <= 0:
            margin_cap = self.current_capital
        available = self._last_account_info.get('available', self.current_capital)
        if self.use_real_api and self.api_client:
            try:
                account_info = await self.api_client.get_account_balance(
                    fallback=available or self.current_capital
                )
                refreshed = float(account_info.get('available', available))
                if refreshed <= 0 and self._last_available_margin > 0:
                    self._log(
                        f"⚠️ 刷新余额返回 {refreshed:.4f} USDT，沿用上次 {self._last_available_margin:.4f} USDT"
                    )
                    refreshed = self._last_available_margin
                else:
                    self._last_account_info = account_info
                available = refreshed if refreshed > 0 else available
                self._last_available_margin = available
                self._last_balance_refresh = time.time()
            except Exception as exc:
                available = self._last_available_margin or available
                self._log(f"⚠️ 刷新余额失败，沿用 {available:.4f} USDT: {exc}")
        desired_margin = margin_cap
        price = max(self._current_price(), 1e-8)
        theoretical_min = (CONTRACT_VALUE * price) / max(self.config['leverage'], 1e-8)
        config_min = self.config.get('min_contract_margin', theoretical_min)
        min_required = max(theoretical_min, config_min)
        desired = min(available, max(desired_margin, min_required))
        if desired <= 0 or available < min_required:
            now = time.time()
            if now - self._last_margin_warning_time > 1:
                self._log(f"⚠️ 可用保证金不足: 需要 {min_required:.4f} USDT, 仅有 {available:.4f} USDT")
                self._last_margin_warning_time = now
            return None
        return desired

    async def _refresh_account_info(self):
        if not self.use_real_api or not self.api_client:
            return
        account_info = await self.api_client.get_account_balance(fallback=self.current_capital)
        self._last_account_info = account_info
        self._last_available_margin = account_info.get('available', self.current_capital)
        self._last_balance_refresh = time.time()
        self._log(
            f"💰 实盘可用保证金 {self._last_available_margin:.4f} USDT (总权益 {account_info.get('total', 0):.4f})"
        )

    def _calculate_position(self, signal: Dict, fixed_margin: float) -> Optional[Dict]:
        price = max(self._current_price(), 1e-8)
        leverage = max(self.config['leverage'], 1e-8)
        raw_position_value = fixed_margin * leverage
        raw_size = raw_position_value / price

        contract_value = max(CONTRACT_VALUE, 1e-8)
        contracts = max(1, int(math.floor(abs(raw_size) / contract_value)))
        contracts = min(contracts, 1)
        if contracts <= 0:
            contracts = 1

        actual_size = contracts * contract_value
        actual_position_value = actual_size * price
        actual_margin = actual_position_value / leverage

        if actual_margin > fixed_margin * 1.05:
            self._log(f"ℹ️ 调整保证金: 目标 {fixed_margin:.4f} USDT, 实际最小合约需要 {actual_margin:.4f} USDT")
            while contracts > 1 and actual_margin > fixed_margin:
                contracts -= 1
                actual_size = contracts * contract_value
                actual_position_value = actual_size * price
                actual_margin = actual_position_value / leverage
            if actual_margin > fixed_margin * 5:
                return None

        available = self._last_available_margin if self.use_real_api else self.current_capital
        while contracts > 1 and actual_margin > available:
            contracts -= 1
            actual_size = contracts * contract_value
            actual_position_value = actual_size * price
            actual_margin = actual_position_value / leverage

        if actual_margin > available:
            now = time.time()
            if now - self._last_margin_warning_time > 1:
                self._log(f"⚠️ 可用保证金不足: 需要 {actual_margin:.4f} USDT, 仅有 {available:.4f} USDT")
                self._last_margin_warning_time = now
            return None

        if actual_margin > self.current_capital:
            self._log(f"⚠️ 当前资金不足以满足最小合约保证金 ({actual_margin:.4f} USDT)")
            return None

        return {
            'position_size': actual_size,
            'leverage': self.config['leverage'],
            'margin_required': actual_margin
        }

    def _current_price(self) -> float:
        if self.data_manager.data_buffers['ticks']:
            price = self.data_manager.data_buffers['ticks'][-1]['price']
            self.market_data['current_price'] = price
            return price
        if self.data_manager.latest_price is not None:
            self.market_data['current_price'] = self.data_manager.latest_price
            return self.data_manager.latest_price
        if self._last_known_price is not None:
            self.market_data['current_price'] = self._last_known_price
            return self._last_known_price
        fallback = max(self.current_capital, 1e-8)
        self.market_data['current_price'] = fallback
        return fallback

    def _log(self, message: str):
        print(message)
        timestamp = datetime.utcnow().strftime('%H:%M:%S')
        if len(message) > 120:
            msg = message[:117] + "..."
        else:
            msg = message
        self.recent_logs.append(f"[{timestamp}] {msg}")

    def _record_signal_event(self, signal: Dict, price: float):
        event = {
            'timestamp': datetime.utcnow(),
            'signal_timestamp': signal.get('timestamp'),
            'direction': signal.get('direction'),
            'price': price,
            'confidence': signal.get('confidence', 0.0),
            'composite': signal.get('composite_trend', 0.0),
            'market_state': signal.get('market_state')
        }
        ob_metrics = (signal.get('composite_trend_breakdown') or {}).get('orderbook_metrics') or {}
        event['orderbook_imbalance'] = ob_metrics.get('imbalance')
        event['orderbook_liquidity'] = ob_metrics.get('liquidity')
        self.signal_events.append(event)
        imbalance_display = f"{event['orderbook_imbalance']:.2f}" if event.get('orderbook_imbalance') is not None else "N/A"
        liquidity_display = f"{event['orderbook_liquidity']:.2f}" if event.get('orderbook_liquidity') is not None else "N/A"
        self._log(
            f"📝 记录信号 {event['direction']} @ {price:.4f} | conf {event['confidence']:.2f} "
            f"| comp {event['composite']:.2f} | OB失衡 {imbalance_display} | 流动性 {liquidity_display}"
        )

    def _direction_cooldown_active(self, direction: str) -> bool:
        if not direction:
            return False
        if not self._last_trade_direction or direction != self._last_trade_direction:
            return False
        if self._last_trade_timestamp <= 0:
            return False
        cooldown, volatility, pnl_ratio = self._calculate_direction_cooldown()
        elapsed = time.time() - self._last_trade_timestamp
        if elapsed >= cooldown:
            return False
        remaining = max(cooldown - elapsed, 0.0)
        if time.time() - self._last_direction_block_log > 1:
            self._log(
                f"⏸️ 同向冷却 {remaining:.1f}s / {cooldown:.1f}s | pnl_ratio {pnl_ratio:.2f} | vol {volatility:.5f}"
            )
            self._last_direction_block_log = time.time()
        return True

    def _calculate_direction_cooldown(self) -> Tuple[float, float, float]:
        base = float(self.config.get('same_direction_cooldown', 15.0))
        pnl_ratio = 0.0
        if self._last_trade_pnl is not None:
            margin_ref = max(self.config.get('fixed_margin', 1.0), 1e-6)
            pnl_ratio = self._last_trade_pnl / margin_ref
            if pnl_ratio < -0.6:
                base *= 1.9
            elif pnl_ratio < -0.2:
                base *= 1.5
            elif pnl_ratio > 0.6:
                base *= 0.6
            elif pnl_ratio > 0.2:
                base *= 0.85
        volatility = self._recent_tick_volatility()
        vol_threshold = max(self.config.get('market_volatility_threshold', 0.0003), 1e-6)
        if volatility < vol_threshold * 0.5:
            base *= 1.25
        elif volatility > vol_threshold * 1.8:
            base *= 0.75
        loss_streak = getattr(self.position_manager, 'loss_streak', 0)
        if loss_streak >= 2:
            base *= 1 + min((loss_streak - 1) * 0.15, 0.6)
        cooldown_min = float(self.config.get('same_direction_cooldown_min', 8.0))
        cooldown_max = float(self.config.get('same_direction_cooldown_max', 45.0))
        base = max(cooldown_min, min(cooldown_max, base))
        return base, volatility, pnl_ratio

    def _recent_tick_volatility(self) -> float:
        ticks = self.data_manager.data_buffers['ticks']
        window = min(len(ticks), self.signal_generator.momentum_window)
        if window < 3:
            return 0.0
        recent_prices = [tick['price'] for tick in list(ticks)[-window:]]
        changes = [
            abs(recent_prices[i] - recent_prices[i - 1]) / max(recent_prices[i - 1], 1e-8)
            for i in range(1, len(recent_prices))
            if recent_prices[i - 1] > 0
        ]
        if not changes:
            return 0.0
        return sum(changes) / len(changes)

    def _log_trend_components(self, components: List[Tuple[str, float, float]], bias: float):
        if not self.enable_trend_component_log or not components:
            return
        if time.time() - self._last_trend_component_log < self.trend_component_log_interval:
            return
        parts = [
            f"{name}:{value:.4f}(w{weight:.1f})"
            for name, value, weight in components
        ]
        self._log(f"📐 趋势构成 bias {bias:.2f} | " + " ".join(parts))
        self._last_trend_component_log = time.time()

    def _auto_tune_parameters(self):
        if not self.config.get('enable_adaptive_tuning', False):
            return
        window = max(int(self.config.get('adaptive_trade_window', 20)), 5)
        if len(self.trade_history) < window:
            self._log(
                f"⚙️ 自适应调参: 数据不足 {len(self.trade_history)}/{window}，暂不调整"
            )
            return
        recent = list(self.trade_history)[-window:]
        wins = sum(1 for trade in recent if trade.get('pnl', 0.0) > 0)
        win_rate = wins / len(recent) if recent else 0.0
        target = float(self.config.get('adaptive_win_rate_target', 0.6))
        tolerance = float(self.config.get('adaptive_win_rate_tolerance', 0.05))
        adjust_direction = 0
        if win_rate < target - tolerance:
            adjust_direction = 1  # tighten
        elif win_rate > target + tolerance:
            adjust_direction = -1  # loosen
        if adjust_direction == 0:
            self._log(
                f"⚙️ 自适应调参: 胜率 {win_rate:.2f} 在目标区间 [{target - tolerance:.2f}, {target + tolerance:.2f}]，参数保持不变"
            )
            return
        step_m = float(self.config.get('adaptive_momentum_step', 0.00002))
        step_c = float(self.config.get('adaptive_composite_step', 0.02))
        min_m = float(self.config.get('momentum_threshold_min', 0.00005))
        max_m = float(self.config.get('momentum_threshold_max', 0.001))
        min_c = float(self.config.get('composite_threshold_min', 0.15))
        max_c = float(self.config.get('composite_threshold_max', 0.6))
        momentum = float(self.config.get('momentum_threshold', self.signal_generator.momentum_threshold))
        composite = float(self.config.get('composite_entry_threshold', self.signal_generator.entry_threshold))
        new_momentum = min(max(momentum + adjust_direction * step_m, min_m), max_m)
        new_composite = min(max(composite + adjust_direction * step_c, min_c), max_c)
        if abs(new_momentum - momentum) < 1e-9 and abs(new_composite - composite) < 1e-9:
            return
        self.config['momentum_threshold'] = new_momentum
        self.config['composite_entry_threshold'] = new_composite
        self.signal_generator.momentum_threshold = new_momentum
        self.signal_generator.entry_threshold = new_composite
        direction_text = "收紧" if adjust_direction > 0 else "放宽"
        self._log(
            f"⚙️ 自适应调参({direction_text}): momentum {momentum:.5f}->{new_momentum:.5f}, "
            f"composite {composite:.2f}->{new_composite:.2f} | win_rate {win_rate:.2f}"
        )

    def _can_submit_trade(self, now: datetime) -> bool:
        if self._trading_halted:
            return False
        limit = max(int(self.config.get('max_trades_per_minute', 0)), 0)
        if limit:
            self._prune_trade_timestamps(now)
            if len(self.trade_timestamps) >= limit:
                if time.time() - self._last_rate_limit_warning > 1:
                    self._log(f"⏳ 已达到每分钟{limit}笔的限制，等待冷却")
                    self._last_rate_limit_warning = time.time()
                return False
        loss_limit = max(int(self.config.get('consecutive_loss_limit', 0)), 0)
        if loss_limit and self.position_manager.loss_streak >= loss_limit:
            if time.time() - self._last_rate_limit_warning > 1:
                self._log(f"🛑 连续亏损{self.position_manager.loss_streak}次，暂停新开仓")
                self._last_rate_limit_warning = time.time()
            return False
        return True

    def _prune_trade_timestamps(self, now: datetime):
        cutoff = now - timedelta(minutes=1)
        while self.trade_timestamps and self.trade_timestamps[0] < cutoff:
            self.trade_timestamps.popleft()

    def _apply_progressive_margin(self, margin: float) -> float:
        if margin <= 0:
            return margin
        if not self.config.get('enable_progressive_margin', False):
            return margin
        boosted = self.position_manager.get_progressive_boost(margin)
        return min(boosted, self.current_capital)

    async def _register_protective_stop(self, position_id: str, signal: Dict, position: Dict):
        if not self.api_client:
            return
        attempts = 3
        last_err = None
        for attempt in range(attempts):
            try:
                stop = await self.api_client.place_stop_order(
                    contract=SYMBOL,
                    side=signal['direction'],
                    trigger_price=position['stop_loss'],
                    price_type=self.config.get('stop_order_price_type', 1),
                    expiration=self.config.get('stop_order_expiration', 3600)
                )
                if stop and stop.get('id') is not None:
                    order_id = str(stop.get('id'))
                    self._active_stop_orders[position_id] = order_id
                    self._log(f"🛡️ 已挂出交易所止损单 #{order_id}")
                    return
            except Exception as exc:
                last_err = exc
                await asyncio.sleep(0.2 * (attempt + 1))
        if last_err:
            self._log(f"⚠️ 止损委托失败: {last_err}")

    async def _cancel_protective_stop(self, position_id: str):
        if not self.api_client:
            return
        stop_id = self._active_stop_orders.pop(position_id, None)
        if not stop_id:
            return
        try:
            await self.api_client.cancel_stop_order(stop_id)
        except Exception as exc:
            self._log(f"⚠️ 取消止损委托失败({stop_id}): {exc}")

    def _record_trade_for_survival(self, trade: Dict):
        today = datetime.utcnow().date()
        if today != self._trades_today_date:
            self._trades_today.clear()
            self._trades_today_date = today
        self._trades_today.append(trade)

    def _enforce_survival_rules(self):
        if not self.config.get('enforce_survival_rules', True):
            return
        trades_today = list(self._trades_today)
        conditions = self.survival_rules.emergency_stop_conditions(
            current_capital=self.current_capital,
            initial_capital=self.initial_capital,
            trades_today=trades_today
        )
        if conditions and not self._trading_halted:
            self._trading_halted = True
            self._halt_reason = ", ".join(conditions)
            self._log(f"🛑 生存法则触发: {self._halt_reason}，停止开仓")

    def _multi_timeframe_trend_bias(self) -> Optional[float]:
        multi_trends = self._calculate_multi_window_trends()
        base_score: Optional[float] = None
        if multi_trends:
            weighted = sum(value * weight for _, value, weight in multi_trends)
            total_weight = sum(weight for _, _, weight in multi_trends)
            base_score = weighted / total_weight if total_weight else 0.0
        else:
            short_trend = self._calculate_short_term_trend()
            medium_trend = self._calculate_medium_term_trend()
            if short_trend is None and medium_trend is None:
                return None
            weight_sum = 0.0
            blended = 0.0
            if short_trend is not None:
                blended += short_trend * 0.4
                weight_sum += 0.4
            if medium_trend is not None:
                blended += medium_trend * 0.6
                weight_sum += 0.6
            base_score = blended / weight_sum if weight_sum else 0.0

        bias = max(min(base_score * 600, 1.0), -1.0)
        self._log_trend_components(multi_trends, bias)
        return bias

    def _calculate_multi_window_trends(self) -> List[Tuple[str, float, float]]:
        configs = [
            ('1s', 60, 1.0),
            ('3s', 40, 1.2),
            ('7s', 30, 1.4),
            ('13s', 24, 1.6),
            ('5s', 24, 1.8),
            ('15s', 12, 2.0)
        ]
        results: List[Tuple[str, float, float]] = []
        for key, length, weight in configs:
            bars = self.data_manager.data_buffers.get(key)
            trend_value = self._trend_strength_from_bars(bars, length)
            if trend_value is not None:
                results.append((key, trend_value, weight))
        return results

    def _trend_strength_from_bars(self, bars: Optional[Deque[Dict]], length: int) -> Optional[float]:
        if not bars or len(bars) < length:
            return None
        recent = list(bars)[-length:]
        closes = [bar.get('close') for bar in recent if bar.get('close') is not None]
        if len(closes) < 2:
            return None
        weighted_sum = 0.0
        total_weight = 0.0
        for idx in range(1, len(closes)):
            prev = closes[idx - 1]
            curr = closes[idx]
            if not prev:
                continue
            change = (curr - prev) / prev if prev else 0.0
            weight = (idx + 1) / len(closes)
            weighted_sum += change * weight
            total_weight += weight
        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight

    def _calculate_short_term_trend(self) -> Optional[float]:
        bars = self.data_manager.data_buffers['1s']
        if len(bars) < 30:
            return None
        recent_bars = list(bars)[-30:]
        price_changes = []
        for i in range(1, len(recent_bars)):
            prev = recent_bars[i - 1]['close']
            curr = recent_bars[i]['close']
            if prev > 0:
                price_changes.append((curr - prev) / prev)

        if not price_changes:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0
        for i, change in enumerate(price_changes):
            weight = (i + 1) / len(price_changes)
            weighted_sum += change * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_medium_term_trend(self) -> Optional[float]:
        bars = self.data_manager.data_buffers['1s']
        if len(bars) < 180:
            return None
        recent_bars = list(bars)[-180:]
        x = list(range(len(recent_bars)))
        y = [bar['close'] for bar in recent_bars]
        if len(y) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        avg_price = sum_y / n
        normalized_slope = slope / avg_price if avg_price > 0 else 0.0
        return normalized_slope

    def _analyze_price_position(self) -> str:
        bars = self.data_manager.data_buffers['1s']
        if len(bars) < 100:
            return "middle"

        recent_prices = [bar['close'] for bar in list(bars)[-100:]]
        current = recent_prices[-1]
        high = max(recent_prices)
        low = min(recent_prices)
        range_size = high - low
        if range_size == 0:
            return "middle"

        position = (current - low) / range_size
        if position > 0.7:
            return "high"
        elif position < 0.3:
            return "low"
        return "middle"

    def _confirm_trend_direction(self, signal: Dict, trend_bias: Optional[float]) -> bool:
        threshold = max(float(self.trend_bias_threshold), 0.0)
        if trend_bias is None:
            if threshold > 0:
                self._log("⏸️ 无法判定趋势，跳过信号")
                return False
            return True

        signal_dir = 1 if signal['direction'] == 'long' else -1
        if abs(trend_bias) < threshold:
            self._log(
                f"⏸️ 趋势强度 {trend_bias:.2f} 低于阈值 {threshold:.2f}，暂不交易"
            )
            return False
        if trend_bias > 0 and signal_dir < 0:
            self._log(f"🚫 禁止逆势交易: 趋势 {trend_bias:.2f} 偏多，拒绝做空信号")
            return False
        if trend_bias < 0 and signal_dir > 0:
            self._log(f"🚫 禁止逆势交易: 趋势 {trend_bias:.2f} 偏空，拒绝做多信号")
            return False

        return True

    def _log_trend_analysis(self, trend_bias: Optional[float], signal: Optional[Dict]):
        if trend_bias is None:
            self._log("📊 趋势分析: 中性")
            return
        strength = "强烈" if abs(trend_bias) > 0.7 else "中等" if abs(trend_bias) > 0.3 else "微弱"
        direction = "看多" if trend_bias > 0 else "看空"
        self._log(f"📊 趋势分析: {direction}({strength}) {trend_bias:.2f}")
        if signal:
            self._log(f"🎯 信号: {signal['direction']} 信心 {signal['confidence']:.2f}")

    def _current_funding_rate(self) -> float:
        now = datetime.utcnow()
        hour = now.hour
        # 如果在资金费率收取时间前后 10 分钟，扣除 0.0008%
        if any(abs((hour - frh) % 24) < 0.2 for frh in FUNDING_RATE_TIMES):
            return 0.000008
        return 0.0

    def _calculate_trailing_stop(self, position: Dict, current_price: float) -> Optional[float]:
        activation = self.config.get('trailing_activation', 0.0)
        trail_step = self.config.get('trailing_step', 0.003)
        if activation <= 0 or trail_step <= 0:
            return None

        entry = position['entry_price']
        if position['direction'] == 'long':
            profit_pct = (current_price - entry) / max(entry, 1e-8)
            if profit_pct < activation:
                return None
            candidate = current_price * (1 - trail_step)
            return candidate if candidate > position['stop_loss'] else None
        else:
            profit_pct = (entry - current_price) / max(entry, 1e-8)
            if profit_pct < activation:
                return None
            candidate = current_price * (1 + trail_step)
            return candidate if candidate < position['stop_loss'] else None

    def _check_partial_profit(self, position: Dict, current_price: float) -> Optional[float]:
        if not self.config.get('enable_partial_profits', False):
            return None
        if position.get('partial_taken'):
            return None

        level = self.config.get('partial_profit_level', 0.0)
        if level <= 0:
            return None

        entry = position['entry_price']
        if position['direction'] == 'long':
            profit_pct = (current_price - entry) / max(entry, 1e-8)
        else:
            profit_pct = (entry - current_price) / max(entry, 1e-8)

        if profit_pct >= level:
            ratio = min(max(self.config.get('partial_profit_ratio', 0.5), 0.1), 1.0)
            return ratio
        return None

    async def _close_partial_position(self, position_id: str, ratio: float, price: float):
        position = self.open_positions.get(position_id)
        if not position:
            return
        remaining = max(position.get('remaining_ratio', 1.0), 0.0)
        ratio = min(ratio, remaining)
        if ratio <= 0:
            return

        pnl = self._calculate_ratio_pnl(position, price, ratio)
        fee_rate = TAKER_FEE_RATE or 0.0
        funding_rate = self._current_funding_rate()
        position_value = self.current_capital * self.config['leverage'] * ratio
        total_fee = position_value * (fee_rate + funding_rate)
        pnl -= total_fee

        self.current_capital += pnl
        self.stats['total_pnl'] += pnl
        position['remaining_ratio'] = remaining - ratio
        position['partial_taken'] = True
        self._log(f"🎯 部分止盈 {ratio:.2f} | 已实现收益 {pnl:.4f} USDT")

    def _calculate_ratio_pnl(self, position: Dict, price: float, ratio: float) -> float:
        direction = position['direction']
        entry = position['entry_price']
        if ratio <= 0:
            return 0.0
        delta = (price - entry) / max(entry, 1e-8) if direction == 'long' else (entry - price) / max(entry, 1e-8)
        position_value = self.current_capital * self.config['leverage'] * ratio
        return delta * position_value

    async def _apply_live_trading_settings(self):
        leverage = int(self.config.get('leverage', 10))
        try:
            await self.api_client.set_leverage(leverage, SYMBOL)
            self._log(f"🔧 已设定Gate.io杠杆为 {leverage}x")
        except Exception as exc:
            self._log(f"⚠️ 设置杠杆失败，将沿用交易所默认值: {exc}")
        self.recent_logs: Deque[str] = deque(maxlen=20)
