# -*- coding: utf-8 -*-
"""
真正的高频交易引擎（Tick级）
"""

import asyncio
import math
import time
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, Optional, Deque, Tuple, List

from collections import deque

from gateio_config import (
    SYMBOL,
    INITIAL_CAPITAL,
    TAKER_FEE_RATE,
    MAKER_FEE_RATE,
    FUNDING_RATE_TIMES,
    CONTRACT_VALUE,
    HIGH_RISK_WINDOWS,
    NEWS_COOLDOWN_MINUTES,
    MARKET_DATA_WS_URL
)
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
from config import TRADE_GUARD_CONFIG, TREND_GUARD_CONFIG, SCALP_RISK_CONFIG, MONITORING_CONFIG


class TrueHFTEngine:
    """真正的高频 (Tick级) 交易引擎"""

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        use_real_api: bool = False,
        max_long_trades: Optional[int] = None,
        max_short_trades: Optional[int] = None,
        max_contracts_per_trade: Optional[int] = None,
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        self.config = dict(HFT_CONFIG)
        self.trade_guard = dict(TRADE_GUARD_CONFIG)
        self.trend_guard = dict(TREND_GUARD_CONFIG)
        self.signal_only_mode = self.config.get('signal_only_mode', False)
        self.detailed_monitoring = bool(self.config.get('detailed_monitoring', False))
        self.use_real_api = use_real_api and not self.signal_only_mode
        self.enable_signal_debug = self.config.get('enable_signal_debug_log', True)
        self.enable_trend_log = self.config.get('enable_trend_log', True)
        default_threshold = float(self.config.get('trend_bias_trade_threshold', 0.0))
        guard_threshold = float(self.trend_guard.get('trade_threshold', default_threshold))
        min_threshold = float(self.trend_guard.get('min_trade_threshold', 0.0))
        if guard_threshold <= 0:
            guard_threshold = default_threshold
        self.trend_bias_threshold = max(min_threshold, guard_threshold)
        self.trend_bias_scale = max(float(self.trend_guard.get('bias_scale', 600.0)), 1.0)
        self.trend_neutral_tolerance = max(float(self.trend_guard.get('neutral_tolerance', 0.0)), 0.0)
        fallback_threshold = float(self.trend_guard.get('fallback_threshold', self.trend_bias_threshold))
        self.trend_fallback_threshold = max(fallback_threshold, self.trend_neutral_tolerance)
        self.trend_override_confidence = max(float(self.trend_guard.get('strong_signal_confidence', 0.95)), 0.0)
        self.trend_override_votes = max(int(self.trend_guard.get('strong_signal_votes', 3)), 1)
        smoothing = float(self.trend_guard.get('bias_smoothing', 0.0))
        self.trend_smoothing = min(max(smoothing, 0.0), 1.0)
        self.allow_neutral_bias = bool(self.trend_guard.get('allow_neutral_bias', True))
        self._smoothed_bias: Optional[float] = None
        self.enable_trend_component_log = self.config.get('enable_trend_component_log', False)
        self.trend_component_log_interval = float(self.config.get('trend_component_log_interval', 3.0))

        self.api_client = None
        self.ws_client: Optional[GateIOWebsocket] = GateIOWebsocket(
            contract=SYMBOL,
            order_book_interval="100ms",
            order_book_depth=20,
            url=MARKET_DATA_WS_URL
        )
        if self.use_real_api:
            if load_config():
                self.api_client = GateIOAPI(enable_market_data=False, enable_trading=True)
            else:
                self.use_real_api = False
        self._fee_rates = {'maker': MAKER_FEE_RATE, 'taker': TAKER_FEE_RATE}
        self._latest_funding_rate = 0.0
        self._next_funding_time: Optional[float] = None
        self._last_fee_refresh = 0.0
        self.fee_refresh_interval = float(self.config.get('fee_refresh_interval', 300.0))

        self.data_manager = HFTDataManager()
        self.signal_generator = HFTSignalGenerator(
            fast_window=int(self.config.get('trend_fast_window', 20)),
            slow_window=int(self.config.get('trend_slow_window', 60)),
            volume_window=int(self.config.get('trend_volume_window', 40)),
            recent_volume_ticks=int(self.config.get('trend_recent_ticks', 8)),
            volume_ratio_threshold=float(self.config.get('trend_volume_ratio', 1.05)),
            min_confidence=float(self.config.get('trend_min_confidence', 0.6)),
            orderbook_imbalance_min=float(self.config.get('trend_orderbook_min_imbalance', 0.08)),
            require_orderbook=bool(self.config.get('trend_orderbook_use', True)),
            min_diff=float(self.config.get('trend_min_diff', 0.0004)),
            cooldown_seconds=float(self.config.get('trend_signal_cooldown', 4.0)),
            use_cross_confirmation=bool(self.config.get('trend_use_cross_confirmation', True)),
            signal_mode=str(self.config.get('trend_signal_mode', 'ma_follow')),
        )
        self.position_manager = AggressivePositionManager(
            leverage=self.config['leverage'],
            risk_config=SCALP_RISK_CONFIG,
            initial_equity=self.current_capital
        )
        executor_client = self.api_client if self.use_real_api else None
        protective_stops_enabled = self.config.get('enable_protective_stops', True)
        self.executor = HFTExecutor(
            api_client=executor_client,
            min_order_interval=0.1,
            enable_exchange_stop_orders=protective_stops_enabled,
            stop_loss_ratio=self.config.get('stop_loss_ratio', 0.006),
            stop_order_price_type=self.config.get('stop_order_price_type', 1),
            stop_order_expiration=self.config.get('stop_order_expiration', 3600)
        )
        self.performance_monitor = HFTPerformanceMonitor()
        self.min_reentry_seconds = float(self.trade_guard.get('min_reentry_seconds', 0.0))
        self.same_direction_reentry_seconds = float(
            self.trade_guard.get('same_direction_reentry_seconds', self.min_reentry_seconds)
        )
        self.duplicate_window_seconds = max(float(self.trade_guard.get('duplicate_window_seconds', 1.0)), 0.1)
        freq_buffer = max(int(self.trade_guard.get('entry_frequency_buffer', 500)), 1)
        self._entry_log_threshold = max(int(self.trade_guard.get('entry_frequency_log_threshold', 1)), 1)
        accuracy_window = max(int(self.trade_guard.get('direction_accuracy_window', 100)), 1)
        self.direction_outcomes: Deque[int] = deque(maxlen=accuracy_window)
        self._last_entry_time: Optional[float] = None
        self._last_direction_entry_time = {'long': None, 'short': None}
        self._entry_second_window: Deque[float] = deque(maxlen=freq_buffer)
        self._signal_event_index: Dict[str, Dict] = {}
        self.survival_rules = SurvivalRules()
        self.max_long_trades = max(self._resolve_limit_value(max_long_trades, self.config.get('max_long_trades_per_day', 0)), 0)
        self.max_short_trades = max(self._resolve_limit_value(max_short_trades, self.config.get('max_short_trades_per_day', 0)), 0)
        contract_limit_default = self.config.get('max_contracts_per_trade', 1)
        contract_limit_value = self._resolve_limit_value(max_contracts_per_trade, contract_limit_default)
        self.max_contracts_per_trade = contract_limit_value if contract_limit_value > 0 else 0

        self.open_positions: Dict[str, Dict] = {}
        self.trade_history = deque(maxlen=2000)
        self.trade_timestamps: Deque[datetime] = deque(maxlen=1000)
        self.last_signal_time = None
        self.last_cycle_time = time.time()
        self._trading_halted = False
        self._halt_reason = ""
        self._active_stop_orders: Dict[str, str] = {}
        self._active_tp_orders: Dict[str, str] = {}
        self._trades_today: Deque[Dict] = deque(maxlen=500)
        self._trades_today_date = datetime.utcnow().date()
        self._long_trades_today = 0
        self._short_trades_today = 0
        self._last_known_price: Optional[float] = None
        self._last_rate_limit_warning = 0.0
        self._last_trend_log = 0.0
        self._last_trend_component_log = 0.0
        self._last_balance_refresh = 0.0
        self._engine_started_at = time.time()
        self.auto_relax_enabled = bool(self.config.get('auto_relax_guards', False))
        self.auto_relax_window = max(float(self.config.get('auto_relax_block_window', 60.0)), 5.0)
        self.auto_relax_idle_seconds = max(float(self.config.get('auto_relax_idle_seconds', 120.0)), 10.0)
        self.auto_relax_relaxation_cooldown = max(
            float(self.config.get('auto_relax_relaxation_cooldown', 20.0)),
            1.0
        )
        self._guard_block_events: Deque[Tuple[float, str]] = deque(maxlen=500)
        self._guard_block_counts: Dict[str, int] = {}
        self._last_guard_relax_time = 0.0
        self._last_signal_debug_log = 0.0
        self._smoothed_bias = None

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
            'positions_opened': 0,
            'entry_blocks': 0,
            'duplicate_blocks': 0,
            'direction_accuracy': 0.0
        }
        self.pnl_history: List[float] = []

        self.is_running = False
        self.tui_display: Optional[LiveTickerDisplay] = None
        self._last_margin_warning_time = 0.0
        self._last_available_margin = 0.0
        self._last_account_info: Dict[str, float] = {'available': self.current_capital, 'total': self.current_capital}
        self.market_data: Dict[str, float] = {
            'current_price': self.current_capital
        }
        self.recent_logs: Deque[str] = deque(maxlen=50)
        self.heartbeat_interval = float(MONITORING_CONFIG.get('heartbeat_interval_seconds', 10.0))
        stall_warning = MONITORING_CONFIG.get(
            'heartbeat_stall_warning_seconds',
            self.heartbeat_interval * 3
        )
        self.heartbeat_stall_warning = float(stall_warning)
        self._last_heartbeat_log = 0.0
        self._last_market_event_time = 0.0
        self._last_market_log = 0.0
        log_capacity = int(self.config.get('signal_log_max_entries', 500))
        self.signal_events: Deque[Dict] = deque(maxlen=max(log_capacity, 50))
        self._last_trade_direction: Optional[str] = None
        self._last_trade_timestamp: float = 0.0
        self._last_trade_pnl: Optional[float] = None
        self._last_direction_block_log = 0.0
        self._last_high_risk_warning = 0.0
        self._news_cooldown_until: Optional[datetime] = None
        if self.signal_only_mode:
            self._log("🧪 已启用信号记录模式，仅跟踪信号与虚拟盈亏")

    async def start(self):
        self.is_running = True
        if self.ws_client:
            await self.ws_client.start()

        if self.use_real_api and self.api_client:
            await self._apply_live_trading_settings()
            await self._refresh_account_info()
            await self._refresh_fee_and_funding(force=True)

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
                    higher_timeframes = None
                    now_ts = time.time()
                    if self.enable_trend_log and now_ts - self._last_trend_log > 1:
                        price_snapshot = self.market_data.get('current_price')
                        self._log(
                            f"📊 趋势分析: bias {trend_bias if trend_bias is not None else 'N/A'} | "
                            f"price {price_snapshot if price_snapshot is not None else 'N/A'}"
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
                        breakout_val = debug_info.get('breakout_strength')
                        breakout_display = f"{breakout_val:.4f}" if isinstance(breakout_val, (int, float)) else breakout_val
                        volume_ratio = debug_info.get('volume_ratio')
                        volume_display = f"{volume_ratio:.2f}" if isinstance(volume_ratio, (int, float)) else volume_ratio
                        self._log(
                            f"🔍 信号调试: reason={debug_info.get('reason')} "
                            f"| dir={debug_info.get('direction')} "
                            f"| breakout={breakout_display} "
                            f"| volume={volume_display}"
                        )
                        self._last_signal_debug_log = time.time()
                        self._register_guard_pressure(
                            self._categorize_guard_reason(debug_info.get('reason'))
                        )
                if signal and self._confirm_trend_direction(signal, trend_bias):
                    if self._is_low_volatility_blocked():
                        self._log("⏸️ 低波动过滤: 暂停开仓")
                        continue
                    self._log_trend_analysis(trend_bias, signal)
                    signal_time = signal.get('timestamp')
                    signal_time = signal_time if isinstance(signal_time, datetime) else None
                    await self._execute_signal(signal, signal_time)
                    self._log(
                        f"🎯 信号: {signal['direction']} 信心 {signal['confidence']:.2f} | "
                        f"trend {signal.get('composite_trend', 0):.2f} | state {signal.get('market_state')}"
                    )

                tick_time = None
                if self.data_manager.data_buffers['ticks']:
                    candidate = self.data_manager.data_buffers['ticks'][-1].get('timestamp')
                    if isinstance(candidate, datetime):
                        tick_time = candidate
                await self._monitor_positions(current_time=tick_time)
                self._maybe_log_heartbeat()

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
        now = time.time()
        market_updated = False

        if ticker:
            await self.data_manager.process_ticker(ticker)
            self._last_known_price = ticker.get('last_price') or ticker.get('mark_price') or self._last_known_price
            if ticker.get('last_price'):
                self.market_data['current_price'] = ticker['last_price']
            elif ticker.get('mark_price'):
                self.market_data['current_price'] = ticker['mark_price']
            market_updated = True
            if self.detailed_monitoring and now - self._last_market_log > 1.0:
                last_price = ticker.get('last_price') or ticker.get('mark_price')
                best_bid = ticker.get('best_bid')
                best_ask = ticker.get('best_ask')
                price_value = None
                bid_value = None
                ask_value = None
                try:
                    price_value = float(last_price)
                except (TypeError, ValueError):
                    price_value = None
                try:
                    bid_value = float(best_bid) if best_bid is not None else None
                except (TypeError, ValueError):
                    bid_value = None
                try:
                    ask_value = float(best_ask) if best_ask is not None else None
                except (TypeError, ValueError):
                    ask_value = None
                spread = None
                if bid_value is not None and ask_value is not None:
                    spread = ask_value - bid_value
                price_display = f"{price_value:.2f}" if price_value is not None else last_price
                parts = [
                    f"📡 行情 {price_display}" if price_display is not None else "📡 行情",
                    f"bid {bid_value:.2f}" if bid_value is not None else None,
                    f"ask {ask_value:.2f}" if ask_value is not None else None,
                    f"spread {spread:.2f}" if isinstance(spread, float) else None
                ]
                snapshot = " | ".join(filter(None, parts))
                self._log(snapshot)
                self._last_market_log = now
        for trade in trades:
            await self.data_manager.process_trade(trade)
            trade_price = trade.get('price')
            if trade_price is not None:
                self._last_known_price = trade_price
                self.market_data['current_price'] = trade_price
            market_updated = True
        if market_updated:
            self._last_market_event_time = now

    async def _execute_signal(self, signal: Dict, signal_time: Optional[datetime] = None):
        now = signal_time or datetime.utcnow()
        now_ts = now.timestamp()
        if self.last_signal_time and (now - self.last_signal_time).total_seconds() < self.config['signal_refresh_rate']:
            return
        if self.stats['total_trades'] >= self.config['daily_trade_limit']:
            return
        if not self._can_submit_trade(now, signal.get('direction')):
            return
        if self._direction_cooldown_active(signal['direction'], now_ts):
            return
        cooldown_reason = self._entry_cooldown_reason(now, signal.get('direction'))
        if cooldown_reason:
            self._record_entry_block(cooldown_reason)
            return

        entry_price = self._current_price()
        self._record_signal_event(signal, entry_price, event_time=now)
        atr_value = self.data_manager.calculate_atr(
            timeframe=self.config.get('atr_timeframe', '60s'),
            window=int(self.config.get('atr_window', 14))
        )
        signal['atr'] = atr_value
        recent_volatility = signal.get('recent_volatility')
        if recent_volatility is None:
            recent_volatility = self.data_manager.estimate_tick_volatility()
            signal['recent_volatility'] = recent_volatility
        trade_profile = self._determine_trade_profile(signal, atr_value, entry_price)
        total_equity = self._current_equity()
        allowed, block_reason = self.position_manager.can_open(total_equity, now)
        if not allowed:
            self._log(f"🚫 风险控制阻止开仓: {block_reason}")
            return
        desired_margin, effective_stop = self.position_manager.determine_margin(
            total_equity=total_equity,
            leverage=self.config['leverage'],
            stop_ratio=trade_profile['stop_ratio'],
            atr_value=atr_value,
            price=entry_price,
            volatility=recent_volatility or 0.0
        )
        trade_profile['stop_ratio'] = effective_stop
        margin_budget = await self._allocate_margin(desired_margin, entry_price)
        if margin_budget is None:
            return
        adjusted_margin = self._apply_progressive_margin(margin_budget)
        position_config = self._calculate_position(signal, adjusted_margin)
        if not position_config:
            return
        position_config['stop_loss_ratio'] = trade_profile['stop_ratio']
        signal['stop_loss_ratio'] = trade_profile['stop_ratio']
        if self.use_real_api and self.api_client and not self.signal_only_mode:
            total_equity = self._last_account_info.get('total', 0)
            available_now = self._last_available_margin
            self._log(
                f"💰 实盘可用保证金 {available_now:.4f} USDT (总权益 {total_equity:.4f})"
            )
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
        status = result.get('status')
        status_extra = result.get('message') or result.get('reason') or result.get('error')
        self._log(
            f"📘 开仓请求 {signal['direction']} | size {position_config['position_size']:.4f} @ {entry_price:.4f} "
            f"| 状态 {status or 'unknown'}{f' ({status_extra})' if status_extra else ''}"
        )
        self.last_signal_time = now
        self.stats['signals_generated'] += 1
        if result.get('status') not in ('filled', 'simulated'):
            return

        if result.get('status') in ('filled', 'simulated'):
            self._increment_direction_counter(signal.get('direction'))
            position_id = f"hft_{now.strftime('%H%M%S_%f')}"
            stop_loss = entry_price * (1 - trade_profile['stop_ratio']
                                       if signal['direction'] == 'long'
                                       else 1 + trade_profile['stop_ratio'])
            target_price = entry_price * (1 + trade_profile['target_ratio']
                                          if signal['direction'] == 'long'
                                          else 1 - trade_profile['target_ratio'])
            position = {
                'id': position_id,
                'direction': signal['direction'],
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'target_price': target_price,
                'size': position_config['position_size'],
                'entry_time': now,
                'remaining_ratio': 1.0,
                'partial_taken': False,
                'fills': result.get('fills'),
                'profile': trade_profile
            }
            if self.signal_events:
                self.signal_events[-1]['position_id'] = position_id
                self._signal_event_index[position_id] = self.signal_events[-1]
            self.open_positions[position_id] = position
            entry_ts = now_ts
            self._last_entry_time = entry_ts
            if signal['direction'] in self._last_direction_entry_time:
                self._last_direction_entry_time[signal['direction']] = entry_ts
            self._record_entry_frequency(entry_ts)
            self.trade_timestamps.append(now)
            self.stats['signals_executed'] += 1
            self.stats['positions_opened'] += 1
            if self.signal_only_mode:
                self._log(
                    f"📘 虚拟开仓 {signal['direction']} @ {entry_price:.4f} | "
                    f"止盈 {target_price:.4f} / 止损 {stop_loss:.4f} | conf {signal['confidence']:.2f} "
                    f"| 累计 {self.stats['positions_opened']}"
                )
            executor_handles_stops = getattr(self.executor, 'supports_exchange_stops', False)
            if self.use_real_api and self.api_client and not self.signal_only_mode:
                if self.config.get('enable_protective_stops', True) and not executor_handles_stops:
                    await self._register_protective_stop(position_id, signal, position)
                if self.config.get('enable_protective_take_profit', True):
                    await self._register_protective_take_profit(position_id, signal, position)

    async def _monitor_positions(self, current_time: Optional[datetime] = None):
        if not self.open_positions:
            return
        now = current_time or datetime.utcnow()
        price = self._current_price()
        to_close = []
        duration_limit = max(self.config.get('max_position_duration', 0), 0)
        min_hold = max(self.config.get('min_position_hold_secs', 0), 0)
        for pid, position in list(self.open_positions.items()):
            elapsed = (now - position['entry_time']).total_seconds()
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
            await self._close_position(pid, reason, close_time=now)

        if self._trading_halted and not self.open_positions:
            self.is_running = False

    async def _close_position(self, position_id: str, reason: str, close_time: Optional[datetime] = None):
        position = self.open_positions.pop(position_id, None)
        if not position:
            return
        await self._cancel_protective_stop(position_id)
        await self._cancel_protective_take_profit(position_id)
        await self._cancel_residual_triggers(position.get('direction'))
        price = self._current_price()
        remaining_ratio = max(position.get('remaining_ratio', 1.0), 0.0)
        if remaining_ratio <= 0:
            return

        fills = position.get('fills') or []
        pnl = self._calculate_ratio_pnl(position, price, remaining_ratio)
        fee_rate = self._fee_rates.get('taker', TAKER_FEE_RATE)
        funding_rate = self._current_funding_rate()
        notional = abs(float(position.get('size', 0.0))) * price * min(max(remaining_ratio, 0.0), 1.0)
        total_fee = notional * (fee_rate + funding_rate)
        pnl -= total_fee
        if fills:
            realized = 0.0
            total_fee_actual = 0.0
            total_size = 0.0
            entry = position['entry_price']
            for fill in fills:
                f_price = fill.get('price')
                f_size = abs(float(fill.get('size') or 0.0))
                f_fee = float(fill.get('fee') or 0.0)
                if not f_price or f_size <= 0:
                    continue
                if position['direction'] == 'long':
                    realized += (price - entry) * f_size
                else:
                    realized += (entry - price) * f_size
                total_fee_actual += f_fee
                total_size += f_size
            if total_size > 0:
                pnl = realized - total_fee_actual
        self._finalize_signal_event(position_id, pnl, reason, price)
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
        close_dt = close_time or datetime.utcnow()
        trade = {
            'position_id': position_id,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': price,
            'pnl': pnl,
            'exit_reason': reason,
            'timestamp': close_dt
        }
        self.trade_history.append(trade)
        self._update_direction_accuracy(pnl)
        self.position_manager.register_trade(pnl, self._current_equity())
        self._record_trade_for_survival(trade)
        self._enforce_survival_rules()
        self._last_trade_direction = position['direction']
        self._last_trade_timestamp = close_dt.timestamp()
        self._last_trade_pnl = pnl
        trade_label = "虚拟" if self.signal_only_mode else "真实"
        self._log(
            f"📙 {trade_label}平仓 {position['direction']} @ {price:.4f} | PnL {pnl:.4f} USDT "
            f"| 原价 {position['entry_price']:.4f} | 原因 {reason}"
        )
        total = max(self.stats['total_trades'], 1)
        win_rate = self.stats['winning_trades'] / total
        long_rate = self.stats['long_wins'] / max(self.stats['long_trades'], 1) if self.stats['long_trades'] else 0.0
        short_rate = self.stats['short_wins'] / max(self.stats['short_trades'], 1) if self.stats['short_trades'] else 0.0
        self._log(
            f"📊 当前胜率 {win_rate:.2f} ({self.stats['winning_trades']}/{self.stats['total_trades']}) | "
            f"多头 {long_rate:.2f} ({self.stats['long_wins']}/{max(self.stats['long_trades'],1)}) | "
            f"空头 {short_rate:.2f} ({self.stats['short_wins']}/{max(self.stats['short_trades'],1)}) | "
            f"方向准确率 {self.stats['direction_accuracy']:.2f} | "
            f"累计{trade_label}仓 {self.stats['positions_opened']}"
        )
        self._log_cumulative_pnl(reason)
        self._auto_tune_parameters()

    async def _available_margin_snapshot(self) -> float:
        available = self._last_available_margin or self.current_capital
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
                await self._refresh_fee_and_funding()
            except Exception as exc:
                available = self._last_available_margin or available
                self._log(f"⚠️ 刷新余额失败，沿用 {available:.4f} USDT: {exc}")
        else:
            available = self.current_capital
            self._last_available_margin = available
        return available

    async def _allocate_margin(self, desired_margin: float, entry_price: float) -> Optional[float]:
        leverage = max(self.config.get('leverage', 1.0), 1e-8)
        available = await self._available_margin_snapshot()
        theoretical_min = (CONTRACT_VALUE * entry_price) / leverage
        config_min = self.config.get('min_contract_margin', theoretical_min)
        min_required = max(theoretical_min, config_min)
        margin_cap = desired_margin if desired_margin > 0 else min_required
        margin = min(available, max(margin_cap, min_required))
        if margin <= 0 or available < min_required:
            now = time.time()
            if now - self._last_margin_warning_time > 1:
                self._log(f"⚠️ 可用保证金不足: 需要 {min_required:.4f} USDT, 仅有 {available:.4f} USDT")
                self._last_margin_warning_time = now
            return None
        return margin

    def _determine_trade_profile(self, signal: Dict, atr_value: Optional[float], entry_price: float) -> Dict[str, float]:
        mode = signal.get('market_state') or 'range'
        base_target = float(self.config.get('base_target_profit_ratio', self.config.get('target_profit_ratio', 0.0012)))
        base_stop = float(self.config.get('base_stop_loss_ratio', self.config.get('stop_loss_ratio', 0.0008)))
        fee_buffer = max(float(self._fee_rates.get('taker', 0.0)), 0.0) * 2
        target = base_target + fee_buffer
        stop = base_stop + fee_buffer
        stop = max(stop, float(self.config.get('min_stop_loss_ratio', stop)))
        if atr_value and entry_price > 0:
            atr_ratio = atr_value / entry_price
            anchor = max(float(self.config.get('atr_volatility_anchor', 0.001)), 1e-9)
            scale = min(max(atr_ratio / anchor, 0.7), 1.5)
            target *= scale
        target = max(target, stop * 1.2)
        return {
            'target_ratio': target,
            'stop_ratio': stop,
            'mode': mode or 'range'
        }

    async def _refresh_account_info(self):
        if not self.use_real_api or not self.api_client:
            return
        account_info = await self.api_client.get_account_balance(fallback=self.current_capital)
        available = float(account_info.get('available', self.current_capital))
        total_equity = float(account_info.get('total', available))
        if total_equity > 0:
            self.current_capital = total_equity
            if hasattr(self.position_manager, 'start_of_day_equity'):
                self.position_manager.start_of_day_equity = total_equity
        self._last_account_info = account_info
        self._last_available_margin = available
        self._last_balance_refresh = time.time()
        await self._refresh_fee_and_funding(force=True)

    async def _refresh_fee_and_funding(self, force: bool = False):
        if not self.use_real_api or not self.api_client:
            return
        now = time.time()
        if not force and now - self._last_fee_refresh < self.fee_refresh_interval:
            return
        try:
            fee_rates = await self.api_client.get_trading_fee_rates(force_refresh=force)
            if fee_rates:
                self._fee_rates.update(fee_rates)
        except Exception as exc:
            self._log(f"⚠️ 无法更新实时手续费，沿用缓存: {exc}")
        try:
            funding = await self.api_client.get_latest_funding_info(force_refresh=force)
            if funding:
                self._latest_funding_rate = float(funding.get('rate', self._latest_funding_rate) or 0.0)
                next_time = funding.get('next_funding_time')
                if next_time is not None:
                    try:
                        self._next_funding_time = float(next_time)
                    except (TypeError, ValueError):
                        pass
        except Exception as exc:
            self._log(f"⚠️ 无法更新资金费率，沿用估算: {exc}")
        self._last_fee_refresh = now

    def _calculate_position(self, signal: Dict, fixed_margin: float) -> Optional[Dict]:
        price = max(self._current_price(), 1e-8)
        leverage = max(self.config['leverage'], 1e-8)
        raw_position_value = fixed_margin * leverage
        raw_size = raw_position_value / price

        contract_value = max(CONTRACT_VALUE, 1e-8)
        contracts = max(1, int(math.floor(abs(raw_size) / contract_value)))
        if self.max_contracts_per_trade > 0:
            contracts = min(contracts, self.max_contracts_per_trade)
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

        capital_reference = self.current_capital
        if self.use_real_api:
            capital_reference = max(
                float(self._last_available_margin or 0.0),
                float(self._current_equity() or 0.0),
                capital_reference
            )
        if actual_margin > capital_reference:
            self._log(
                f"⚠️ 当前资金不足以满足最小合约保证金 ({actual_margin:.4f} USDT)"
                f"，可用 {capital_reference:.4f} USDT"
            )
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

    def _current_equity(self) -> float:
        if self.use_real_api and self._last_account_info:
            try:
                return float(self._last_account_info.get('total', self.current_capital))
            except (TypeError, ValueError):
                return self.current_capital
        return self.current_capital

    def _log(self, message: str):
        print(message)
        timestamp = datetime.utcnow().strftime('%H:%M:%S')
        if len(message) > 120:
            msg = message[:117] + "..."
        else:
            msg = message
        self.recent_logs.append(f"[{timestamp}] {msg}")

    def _maybe_log_heartbeat(self):
        now = time.time()
        if now - self._last_heartbeat_log < self.heartbeat_interval:
            return
        ws_status = "connected"
        if not self.ws_client or not self.ws_client.is_connected():
            ws_status = "disconnected"
        last_market_gap = None
        if self._last_market_event_time:
            last_market_gap = now - self._last_market_event_time
        market_gap_display = f"{last_market_gap:.1f}s" if last_market_gap is not None else "N/A"
        accuracy_display = "N/A"
        if self.direction_outcomes:
            accuracy_display = f"{self.stats['direction_accuracy'] * 100:.2f}%"
        last_signal_gap = None
        if self.last_signal_time:
            last_signal_gap = now - self.last_signal_time.timestamp()
        signal_gap_display = f"{last_signal_gap:.1f}s" if last_signal_gap is not None else "N/A"
        message = (
            f"❤️ 心跳 | WS:{ws_status} | ticks:{len(self.data_manager.data_buffers['ticks'])} "
            f"| last_tick:{market_gap_display} | signals:{self.stats['signals_generated']} "
            f"/exec:{self.stats['signals_executed']} | trades:{self.stats['total_trades']} "
            f"| PnL:{self.stats['total_pnl']:.4f} | accuracy:{accuracy_display} "
            f"| last_signal:{signal_gap_display} | positions:{len(self.open_positions)} "
            f"| equity:{self._current_equity():.4f}"
        )
        if last_market_gap and last_market_gap > self.heartbeat_stall_warning:
            message += " ⚠️ 行情长时间无更新"
        self._log(message)
        self._last_heartbeat_log = now

    def _resolve_limit_value(self, provided: Optional[int], default: Optional[int]) -> int:
        value = default if provided is None else provided
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _record_signal_event(self, signal: Dict, price: float, event_time: Optional[datetime] = None):
        event = {
            'timestamp': event_time or datetime.utcnow(),
            'signal_timestamp': signal.get('timestamp'),
            'direction': signal.get('direction'),
            'price': price,
            'confidence': signal.get('confidence', 0.0),
            'composite': signal.get('composite_trend', 0.0),
            'market_state': signal.get('market_state'),
            'market_mode': signal.get('market_mode'),
            'recent_volatility': signal.get('recent_volatility'),
            'position_id': None,
            'result': None
        }
        debug_snapshot = signal.get('debug') or getattr(self.signal_generator, 'last_debug', {})
        if debug_snapshot:
            event['debug'] = dict(debug_snapshot)
            event['votes'] = dict(debug_snapshot.get('votes', {})) if isinstance(debug_snapshot.get('votes'), dict) else debug_snapshot.get('votes')
            event['signal_reason'] = debug_snapshot.get('reason')
        ob_metrics = (signal.get('composite_trend_breakdown') or {}).get('orderbook_metrics') or {}
        event['orderbook_imbalance'] = ob_metrics.get('imbalance')
        event['orderbook_liquidity'] = ob_metrics.get('liquidity')
        if signal.get('multi_timeframe'):
            event['multi_timeframe'] = signal['multi_timeframe']
        self.signal_events.append(event)
        imbalance_display = f"{event['orderbook_imbalance']:.2f}" if event.get('orderbook_imbalance') is not None else "N/A"
        liquidity_display = f"{event['orderbook_liquidity']:.2f}" if event.get('orderbook_liquidity') is not None else "N/A"
        self._log(
            f"📝 记录信号 {event['direction']} @ {price:.4f} | conf {event['confidence']:.2f} "
            f"| comp {event['composite']:.2f} | OB失衡 {imbalance_display} | 流动性 {liquidity_display}"
        )
        if self.detailed_monitoring:
            self._log_signal_debug(signal, debug_snapshot)

    def _log_signal_debug(self, signal: Dict, debug_snapshot: Optional[Dict]):
        snapshot = debug_snapshot or {}
        desired = snapshot.get('desired_direction') or signal.get('direction')
        condition_count = snapshot.get('condition_count')
        market_state = snapshot.get('market_state') or signal.get('market_state') or 'unknown'
        parts = [
            f"🧠 信号依据: 市场 {market_state}",
            f"方向 {desired or 'n/a'}",
            f"条件数 {condition_count or 0}"
        ]
        l1_flag = snapshot.get('l1')
        l2_flag = snapshot.get('l2')
        l3_flag = snapshot.get('l3')
        if any(flag is not None for flag in (l1_flag, l2_flag, l3_flag)):
            l1 = '✔' if l1_flag else '✘'
            l2 = '✔' if l2_flag else '✘'
            l3 = '✔' if l3_flag else '✘'
            parts.append(f"L1 {l1} / L2 {l2} / L3 {l3}")
        self._log(" | ".join(parts))
        votes_detail = snapshot.get('votes') or signal.get('votes') or {}
        if isinstance(votes_detail, dict) and votes_detail:
            vote_icons = []
            for source, active in votes_detail.items():
                icon = '✔' if active else '✘'
                vote_icons.append(f"{source}:{icon}")
            votes_obtained = snapshot.get('votes_obtained')
            if votes_obtained is None:
                votes_obtained = sum(1 for state in votes_detail.values() if state)
            votes_required = snapshot.get('votes_required') or len(votes_detail)
            self._log(f"🗳️ 投票 {votes_obtained}/{votes_required} | " + ", ".join(vote_icons))
        metrics = []
        if signal.get('momentum') is not None:
            metrics.append(f"momentum={signal['momentum']:+.6f}")
        if signal.get('volume_ratio') is not None:
            metrics.append(f"volume={signal['volume_ratio']:.2f}x")
        if signal.get('order_imbalance') is not None:
            metrics.append(f"imbalance={signal['order_imbalance']:+.2f}")
        if signal.get('composite_trend') is not None:
            metrics.append(f"composite={signal['composite_trend']:+.2f}")
        if metrics:
            self._log("📐 指标详情: " + " | ".join(metrics))
        if snapshot.get('recent_volatility') is not None:
            threshold = snapshot.get('l1_threshold')
            threshold_text = f"{threshold:.6f}" if threshold is not None else "n/a"
            self._log(
                f"🌪️ 波动率 {snapshot['recent_volatility']:.6f} | 动态阈值 {threshold_text}"
            )
        reason = snapshot.get('reason')
        if reason:
            self._log(f"📝 信号判定: {reason}")

    def _finalize_signal_event(self, position_id: str, pnl: float, reason: str, exit_price: float):
        event = self._signal_event_index.pop(position_id, None)
        if not event:
            return
        event.update({
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': reason,
            'result': 'win' if pnl > 0 else 'loss'
        })

    def _increment_direction_counter(self, direction: Optional[str]):
        if not direction:
            return
        self._ensure_daily_counters(datetime.utcnow())
        if direction == 'long':
            self._long_trades_today += 1
        elif direction == 'short':
            self._short_trades_today += 1

    def _direction_cooldown_active(self, direction: str, reference_ts: Optional[float] = None) -> bool:
        if not direction:
            return False
        if not self._last_trade_direction or direction != self._last_trade_direction:
            return False
        if self._last_trade_timestamp <= 0:
            return False
        cooldown, volatility, pnl_ratio = self._calculate_direction_cooldown()
        now_ts = reference_ts if reference_ts is not None else time.time()
        elapsed = now_ts - self._last_trade_timestamp
        if elapsed >= cooldown:
            return False
        remaining = max(cooldown - elapsed, 0.0)
        if time.time() - self._last_direction_block_log > 1:
            self._log(
                f"⏸️ 同向冷却 {remaining:.1f}s / {cooldown:.1f}s | pnl_ratio {pnl_ratio:.2f} | vol {volatility:.5f}"
            )
            self._last_direction_block_log = time.time()
        return True

    def _log_cumulative_pnl(self, reason: str):
        self.pnl_history.append(self.stats['total_pnl'])
        if not self.detailed_monitoring:
            return
        curve = self._render_pnl_curve()
        self._log(
            f"📈 累计盈亏 {self.stats['total_pnl']:.4f} USDT | 资金 {self.current_capital:.4f} USDT "
            f"| 出场原因 {reason} | 曲线 {curve}"
        )

    def _render_pnl_curve(self, window: int = 20) -> str:
        if not self.pnl_history:
            return "-"
        history = self.pnl_history[-window:]
        if len(history) == 1:
            return "."
        min_val = min(history)
        max_val = max(history)
        if math.isclose(max_val, min_val, rel_tol=1e-9, abs_tol=1e-9):
            return "-" * len(history)
        symbols = ".__-~=^*#@"
        span = max_val - min_val
        rendered: List[str] = []
        for value in history:
            normalized = (value - min_val) / span
            idx = min(int(normalized * (len(symbols) - 1)), len(symbols) - 1)
            rendered.append(symbols[idx])
        return "".join(rendered)

    def _entry_cooldown_reason(self, now: datetime, direction: Optional[str]) -> Optional[str]:
        now_ts = now.timestamp()
        if self.min_reentry_seconds > 0 and self._last_entry_time:
            elapsed = now_ts - self._last_entry_time
            if elapsed < self.min_reentry_seconds:
                remaining = self.min_reentry_seconds - elapsed
                return f"global冷却 {remaining:.2f}s"
        same_dir_interval = max(self.same_direction_reentry_seconds, self.min_reentry_seconds)
        if direction and same_dir_interval > 0:
            last_dir_entry = self._last_direction_entry_time.get(direction)
            if last_dir_entry:
                elapsed = now_ts - last_dir_entry
                if elapsed < same_dir_interval:
                    remaining = same_dir_interval - elapsed
                    return f"{direction}冷却 {remaining:.2f}s"
        return None

    def _record_entry_block(self, reason: str):
        self.stats['entry_blocks'] += 1
        if '冷却' in reason or 'cooldown' in reason.lower():
            self.stats['duplicate_blocks'] += 1
        self.performance_monitor.record_guard_event('entry_block', reason)
        self._log(f"🚫 开仓被阻止: {reason}")
        self._register_guard_pressure(self._categorize_guard_reason(reason))

    def _record_entry_frequency(self, timestamp: float):
        self._entry_second_window.append(timestamp)
        window = max(self.duplicate_window_seconds, 0.1)
        while self._entry_second_window and timestamp - self._entry_second_window[0] > window:
            self._entry_second_window.popleft()
        count = len(self._entry_second_window)
        self.performance_monitor.record_entry_frequency(count, window)
        if count > self._entry_log_threshold:
            self._log(f"📈 最近 {window:.1f}s 内开仓尝试 {count} 次")

    def _categorize_guard_reason(self, reason: Optional[str]) -> str:
        if not reason:
            return "general"
        lowered = reason.lower()
        if "冷却" in reason or "cooldown" in lowered:
            return "cooldown"
        if "趋势" in reason or "trend" in lowered or "逆势" in reason:
            return "trend"
        if "volume" in lowered or "成交量" in reason:
            return "volume"
        if "orderbook" in lowered or "盘口" in reason or "imbalance" in lowered:
            return "orderbook"
        if "l1" in lowered or "l2" in lowered:
            return "l1"
        if "保证金" in reason or "margin" in lowered:
            return "margin"
        return "general"

    def _register_guard_pressure(self, category: str):
        if not self.auto_relax_enabled:
            return
        now = time.time()
        self._guard_block_events.append((now, category))
        self._guard_block_counts[category] = self._guard_block_counts.get(category, 0) + 1
        window = self.auto_relax_window
        while self._guard_block_events and now - self._guard_block_events[0][0] > window:
            _, old_category = self._guard_block_events.popleft()
            if old_category in self._guard_block_counts:
                self._guard_block_counts[old_category] -= 1
                if self._guard_block_counts[old_category] <= 0:
                    self._guard_block_counts.pop(old_category, None)
        block_threshold = max(int(self.config.get('auto_relax_block_threshold', 15)), 1)
        total_blocks = sum(self._guard_block_counts.values())
        idle_since = self._last_trade_timestamp or self._engine_started_at
        idle_elapsed = now - idle_since
        needs_relax = total_blocks >= block_threshold or idle_elapsed >= self.auto_relax_idle_seconds
        if not needs_relax:
            return
        if now - self._last_guard_relax_time < self.auto_relax_relaxation_cooldown:
            return
        relax_category = category if total_blocks >= block_threshold else "general"
        self._apply_guard_relaxation(relax_category)
        self._guard_block_events.clear()
        self._guard_block_counts.clear()
        self._last_guard_relax_time = now

    def _apply_guard_relaxation(self, category: str):
        adjustments: List[str] = []
        step_cd = float(self.config.get('auto_relax_cooldown_step', 0.5))
        min_cd = float(self.config.get('auto_relax_min_cooldown', self.config.get('same_direction_cooldown_min', 0.0)))
        min_signal_cd = float(self.config.get('auto_relax_signal_cooldown_min', 0.5))
        if category in ("cooldown", "general"):
            base_cd = float(self.config.get('same_direction_cooldown', step_cd))
            new_cd = max(base_cd - step_cd, min_cd)
            if new_cd < base_cd:
                self.config['same_direction_cooldown'] = new_cd
                self.same_direction_reentry_seconds = max(new_cd, self.min_reentry_seconds)
                adjustments.append(f"同向冷却 {base_cd:.1f}->{new_cd:.1f}s")
            signal_cd = getattr(self.signal_generator, 'cooldown_seconds', 0.0)
            new_signal_cd = max(signal_cd - step_cd, min_signal_cd)
            if new_signal_cd < signal_cd:
                self.signal_generator.cooldown_seconds = new_signal_cd
                adjustments.append(f"信号冷却 {signal_cd:.1f}->{new_signal_cd:.1f}s")
        if category in ("trend", "general"):
            step_trend = float(self.config.get('auto_relax_trend_step', 0.01))
            min_trend = float(self.config.get(
                'auto_relax_min_trend_threshold',
                self.trend_guard.get('min_trade_threshold', 0.05)
            ))
            current_threshold = float(self.trend_guard.get('trade_threshold', self.trend_bias_threshold))
            new_threshold = max(current_threshold - step_trend, min_trend)
            if new_threshold < current_threshold:
                self.trend_guard['trade_threshold'] = new_threshold
                self.trend_bias_threshold = new_threshold
                fallback = max(self.trend_guard.get('fallback_threshold', new_threshold), new_threshold, self.trend_neutral_tolerance)
                self.trend_guard['fallback_threshold'] = fallback
                self.trend_fallback_threshold = fallback
                adjustments.append(f"趋势阈值 {current_threshold:.2f}->{new_threshold:.2f}")
        if category in ("l1", "general"):
            step_l1 = float(self.config.get('auto_relax_l1_step', 0.001))
            min_l1 = float(self.config.get('auto_relax_min_l1_floor', 0.006))
            current_ratio = float(self.config.get('l1_floor_ratio', 0.01))
            new_ratio = max(current_ratio - step_l1, min_l1)
            if new_ratio < current_ratio:
                self.config['l1_floor_ratio'] = new_ratio
                adjustments.append(f"L1阈值 {current_ratio:.3f}->{new_ratio:.3f}")
            current_vm = float(self.config.get('l1_volatility_multiplier', 1.0))
            max_vm = float(self.config.get('auto_relax_max_vol_multiplier', 1.4))
            new_vm = min(max_vm, current_vm + step_l1 * 5)
            if new_vm > current_vm:
                self.config['l1_volatility_multiplier'] = new_vm
                adjustments.append(f"L1波动乘数 {current_vm:.2f}->{new_vm:.2f}")
        if category in ("volume", "orderbook", "general"):
            step_vol = float(self.config.get('auto_relax_volume_step', 0.05))
            min_vol = float(self.config.get('auto_relax_min_volume_ratio', 0.9))
            vol_ratio = float(getattr(self.signal_generator, 'volume_ratio_threshold', 1.0))
            new_vol_ratio = max(vol_ratio - step_vol, min_vol)
            if new_vol_ratio < vol_ratio:
                self.signal_generator.volume_ratio_threshold = new_vol_ratio
                self.config['trend_volume_ratio'] = new_vol_ratio
                adjustments.append(f"成交量阈值 {vol_ratio:.2f}->{new_vol_ratio:.2f}")
            min_ob = float(self.config.get('auto_relax_min_orderbook_imbalance', 0.02))
            ob_threshold = float(getattr(self.signal_generator, 'orderbook_imbalance_min', 0.0))
            new_ob = max(ob_threshold - step_vol / 10, min_ob)
            if new_ob < ob_threshold:
                self.signal_generator.orderbook_imbalance_min = new_ob
                self.config['trend_orderbook_min_imbalance'] = new_ob
                adjustments.append(f"盘口不平衡 {ob_threshold:.3f}->{new_ob:.3f}")
        if adjustments:
            self._log("🪄 自动放宽条件: " + " | ".join(adjustments))

    def _calculate_direction_cooldown(self) -> Tuple[float, float, float]:
        base = float(self.config.get('same_direction_cooldown', 15.0))
        pnl_ratio = 0.0
        if self._last_trade_pnl is not None:
            margin_ref = float(self.config.get('fixed_margin', 0.0) or 0.0)
            ratio = float(self.config.get('fixed_margin_ratio', 0.0) or 0.0)
            if margin_ref <= 0 and ratio > 0:
                margin_ref = self.current_capital * ratio
            margin_ref = max(margin_ref if margin_ref > 0 else 1.0, 1e-6)
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

    def _ensure_daily_counters(self, now: datetime):
        today = now.date()
        if today != self._trades_today_date:
            self._trades_today.clear()
            self._trades_today_date = today
            self._long_trades_today = 0
            self._short_trades_today = 0

    def _can_submit_trade(self, now: datetime, direction: Optional[str]) -> bool:
        self._ensure_daily_counters(now)
        if self._trading_halted:
            return False
        if self._is_high_risk_time(now):
            if time.time() - self._last_high_risk_warning > 1:
                self._log("📰 重大经济事件窗口，暂停开仓")
                self._last_high_risk_warning = time.time()
            return False
        limit = max(int(self.config.get('max_trades_per_minute', 0)), 0)
        if limit:
            self._prune_trade_timestamps(now)
            if len(self.trade_timestamps) >= limit:
                if time.time() - self._last_rate_limit_warning > 1:
                    self._log(f"⏳ 已达到每分钟{limit}笔的限制，等待冷却")
                    self._last_rate_limit_warning = time.time()
                return False
        if direction == 'long' and self.max_long_trades and self._long_trades_today >= self.max_long_trades:
            if time.time() - self._last_rate_limit_warning > 1:
                self._log(f"🚫 多单当日已达到 {self.max_long_trades} 次上限")
                self._last_rate_limit_warning = time.time()
            return False
        if direction == 'short' and self.max_short_trades and self._short_trades_today >= self.max_short_trades:
            if time.time() - self._last_rate_limit_warning > 1:
                self._log(f"🚫 空单当日已达到 {self.max_short_trades} 次上限")
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

    def _is_high_risk_time(self, now: datetime) -> bool:
        if self.config.get('disable_high_risk_guard', False):
            return False
        if self._news_cooldown_until:
            if now <= self._news_cooldown_until:
                return True
            self._news_cooldown_until = None
        if not HIGH_RISK_WINDOWS:
            return False
        beijing_now = now + timedelta(hours=8)
        cooldown_delta = timedelta(minutes=max(int(NEWS_COOLDOWN_MINUTES or 0), 0))
        for day_offset in (-1, 0, 1):
            candidate_local_dt = beijing_now + timedelta(days=day_offset)
            for window in HIGH_RISK_WINDOWS:
                weekdays = window.get('weekdays')
                if weekdays and candidate_local_dt.weekday() not in weekdays:
                    continue
                hour = int(window.get('hour', 0))
                minute = int(window.get('minute', 0))
                local_event = datetime.combine(candidate_local_dt.date(), dt_time(hour=hour, minute=minute))
                event_utc = local_event - timedelta(hours=8)
                pre = timedelta(minutes=int(window.get('pre_buffer', 0)))
                post = timedelta(minutes=int(window.get('post_buffer', 0)))
                start = event_utc - pre
                end = event_utc + post
                cooldown_end = end + cooldown_delta
                if start <= now <= cooldown_end:
                    if cooldown_delta > timedelta(0):
                        if not self._news_cooldown_until or cooldown_end > self._news_cooldown_until:
                            self._news_cooldown_until = cooldown_end
                    return True
        return False

    def _is_low_volatility_blocked(self) -> bool:
        if not self.config.get('low_volatility_block', False):
            return False
        threshold = float(self.config.get('low_volatility_threshold', 0.0))
        if threshold <= 0:
            return False
        vol = self.data_manager.estimate_tick_volatility(window=60)  # 约 1 分钟窗口
        if vol < threshold:
            return True
        return False

    def _apply_progressive_margin(self, margin: float) -> float:
        if margin <= 0:
            return margin
        start = int(self.config.get('loss_streak_reduce_start', 0))
        if start > 0:
            loss_streak = getattr(self.position_manager, 'loss_streak', 0)
            if loss_streak >= start:
                steps = loss_streak - start + 1
                reduce_factor = float(self.config.get('loss_streak_reduce_factor', 0.5))
                min_factor = float(self.config.get('loss_streak_min_factor', 0.25))
                applied = max(min_factor, reduce_factor ** steps)
                margin *= applied
                self._log(f"⚠️ 连续亏损 {loss_streak} 次，自动减仓 {applied:.2f}x")
        if self.config.get('enable_progressive_margin', False):
            boosted = self.position_manager.get_progressive_boost(margin)
            margin = min(boosted, self.current_capital)
        return margin

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

    async def _register_protective_take_profit(self, position_id: str, signal: Dict, position: Dict):
        if not self.api_client:
            return
        attempts = 3
        last_err = None
        trigger_price = position.get('target_price')
        if trigger_price is None:
            return
        for attempt in range(attempts):
            try:
                tp = await self.api_client.place_take_profit_order(
                    contract=SYMBOL,
                    side=signal['direction'],
                    trigger_price=trigger_price,
                    price_type=self.config.get('take_profit_order_price_type', 1),
                    expiration=self.config.get('take_profit_order_expiration', 3600)
                )
                if tp and tp.get('id') is not None:
                    order_id = str(tp.get('id'))
                    self._active_tp_orders[position_id] = order_id
                    self._log(f"🎯 已挂出交易所止盈单 #{order_id}")
                    return
            except Exception as exc:
                last_err = exc
                await asyncio.sleep(0.2 * (attempt + 1))
        if last_err:
            self._log(f"⚠️ 止盈委托失败: {last_err}")

    async def _cancel_protective_take_profit(self, position_id: str):
        if not self.api_client:
            return
        tp_id = self._active_tp_orders.pop(position_id, None)
        if not tp_id:
            return
        try:
            await self.api_client.cancel_stop_order(tp_id)
        except Exception as exc:
            self._log(f"⚠️ 取消止盈委托失败({tp_id}): {exc}")

    async def _cancel_residual_triggers(self, direction: Optional[str] = None):
        """
        平仓后扫尾，取消同向的剩余触发单，避免残单。
        """
        if not self.api_client:
            return
        try:
            cancelled = await self.api_client.cancel_all_reducing_triggers(direction)
            if cancelled > 0:
                self._log(f"🧹 已取消残留触发单 {cancelled} 个")
        except Exception as exc:
            self._log(f"⚠️ 残留触发单取消失败: {exc}")

    def _record_trade_for_survival(self, trade: Dict):
        timestamp = trade.get('timestamp') or datetime.utcnow()
        self._ensure_daily_counters(timestamp)
        self._trades_today.append(trade)

    def _update_direction_accuracy(self, pnl: float):
        outcome = 1 if pnl > 0 else 0
        self.direction_outcomes.append(outcome)
        accuracy = sum(self.direction_outcomes) / len(self.direction_outcomes)
        self.stats['direction_accuracy'] = accuracy
        self.performance_monitor.record_direction_accuracy(accuracy, len(self.direction_outcomes))

    def _enforce_survival_rules(self):
        if not self.config.get('enforce_survival_rules', True):
            return
        if self.config.get('disable_survival_rules', False):
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

        scaled_bias = max(min(base_score * self.trend_bias_scale, 1.0), -1.0)
        if self.trend_smoothing > 0:
            prev = scaled_bias if self._smoothed_bias is None else self._smoothed_bias
            smoothed = prev * (1 - self.trend_smoothing) + scaled_bias * self.trend_smoothing
        else:
            smoothed = scaled_bias
        self._smoothed_bias = smoothed
        bias = max(min(smoothed, 1.0), -1.0)
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
        neutral_tol = max(self.trend_neutral_tolerance, 0.0)
        fallback_threshold = max(self.trend_fallback_threshold, neutral_tol)
        votes_detail: Dict[str, bool] = {}
        raw_votes = signal.get('votes')
        if isinstance(raw_votes, dict):
            votes_detail = dict(raw_votes)
        else:
            debug_votes = (signal.get('debug') or {}).get('votes')
            if isinstance(debug_votes, dict):
                votes_detail = dict(debug_votes)
        vote_total = len(votes_detail)
        vote_pass = sum(1 for passed in votes_detail.values() if passed)
        if trend_bias is None:
            if threshold > 0 and not self.allow_neutral_bias:
                self._log("⏸️ 无法判定趋势，跳过信号")
                return False
            return True

        abs_bias = abs(trend_bias)
        if abs_bias < neutral_tol:
            if threshold > 0:
                self._log(f"ℹ️ 趋势 {trend_bias:.2f} 接近中性，依赖多因子确认继续")
            return True

        signal_dir = 1 if signal['direction'] == 'long' else -1
        if abs_bias < threshold:
            votes_text = f"{vote_pass}/{vote_total}" if vote_total else "0/0"
            strong_signal = (
                vote_total > 0
                and vote_pass >= min(self.trend_override_votes, vote_total)
                and signal.get('confidence', 0.0) >= self.trend_override_confidence
            )
            if strong_signal:
                self._log(
                    f"⚖️ 趋势偏弱({trend_bias:.2f}<{threshold:.2f})，但投票 {votes_text} "
                    f"且信心 {signal['confidence']:.2f}，允许尝试"
                )
                abs_bias = threshold
            elif abs_bias >= fallback_threshold:
                self._log(
                    f"ℹ️ 趋势 {trend_bias:.2f} 接近阈值 {threshold:.2f}，沿用其他过滤继续"
                )
            else:
                self._log(
                    f"⏸️ 趋势强度 {trend_bias:.2f} 低于阈值 {threshold:.2f} (投票 {votes_text})，暂不交易"
                )
                self._register_guard_pressure("trend")
                return False
        if trend_bias > 0 and signal_dir < 0:
            self._log(f"🚫 禁止逆势交易: 趋势 {trend_bias:.2f} 偏多，拒绝做空信号")
            self._register_guard_pressure("trend")
            return False
        if trend_bias < 0 and signal_dir > 0:
            self._log(f"🚫 禁止逆势交易: 趋势 {trend_bias:.2f} 偏空，拒绝做多信号")
            self._register_guard_pressure("trend")
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
        if self.use_real_api:
            return float(self._latest_funding_rate or 0.0)
        now = datetime.utcnow()
        hour = now.hour
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
        fee_rate = self._fee_rates.get('taker', TAKER_FEE_RATE)
        funding_rate = self._current_funding_rate()
        notional = abs(float(position.get('size', 0.0))) * price * min(max(ratio, 0.0), 1.0)
        total_fee = notional * (fee_rate + funding_rate)
        pnl -= total_fee

        self.current_capital += pnl
        self.stats['total_pnl'] += pnl
        position['remaining_ratio'] = remaining - ratio
        position['partial_taken'] = True
        self._log(f"🎯 部分止盈 {ratio:.2f} | 已实现收益 {pnl:.4f} USDT")

    def _calculate_ratio_pnl(self, position: Dict, price: float, ratio: float) -> float:
        direction = position['direction']
        entry = position['entry_price']
        size = abs(float(position.get('size', 0.0)))
        if ratio <= 0 or size <= 0:
            return 0.0
        effective_size = size * min(max(ratio, 0.0), 1.0)
        if direction == 'long':
            return (price - entry) * effective_size
        return (entry - price) * effective_size

    async def _apply_live_trading_settings(self):
        leverage = int(self.config.get('leverage', 10))
        success = False
        try:
            success = await self.api_client.set_leverage(leverage, SYMBOL)
        except Exception as exc:
            self._log(f"⚠️ 设置杠杆失败，将沿用交易所默认值: {exc}")
        else:
            if success:
                self._log(f"🔧 已设定Gate.io杠杆为 {leverage}x")
            else:
                self._log("⚠️ 杠杆设置请求被Gate.io拒绝，已沿用默认值")
        self.recent_logs: Deque[str] = deque(maxlen=20)
