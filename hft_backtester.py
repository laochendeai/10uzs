# -*- coding: utf-8 -*-
"""
离线回测与参数扫描工具。

该模块使用真实行情 CSV（默认 my_trades.csv）驱动 `TrueHFTEngine`，但只执行虚拟撮合，
用于验证多空方向的准确率、防重复开仓策略以及不同指标参数组合的表现。
"""

import argparse
import asyncio
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Any, Tuple

from config import SIMULATION_CONFIG
from true_hft_engine import TrueHFTEngine


@dataclass
class HistoricalTick:
    timestamp: datetime
    price: float
    size: float
    side: str

    def as_trade(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'price': self.price,
            'size': self.size,
            'side': self.side
        }


class HistoricalDataFeed:
    """加载真实历史成交数据，为回测提供逐条 trade。"""

    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.path = Path(settings['historical_trade_file'])
        self.timestamp_field = settings.get('timestamp_field', 'timestamp')
        self.price_field = settings.get('price_field', 'price')
        self.size_field = settings.get('size_field', 'size')
        self.side_field = settings.get('side_field', 'side')
        self.direction_field = settings.get('direction_field', 'direction')
        self.max_rows = settings.get('max_rows')
        self.records: List[HistoricalTick] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            raise FileNotFoundError(f"找不到行情数据文件: {self.path}")
        with self.path.open('r', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                if self.max_rows and idx >= self.max_rows:
                    break
                ts_raw = row.get(self.timestamp_field)
                price_raw = row.get(self.price_field) or row.get('entry_price') or row.get('last_price')
                size_raw = row.get(self.size_field) or row.get('volume') or row.get('qty') or row.get('amount')
                if not ts_raw or not price_raw or size_raw is None:
                    continue
                try:
                    price = float(price_raw)
                    size_val = float(size_raw)
                except (TypeError, ValueError):
                    continue
                side = row.get(self.side_field)
                if not side:
                    direction = row.get(self.direction_field, '')
                    if direction:
                        direction = direction.lower()
                        if 'long' in direction or direction.startswith('bull'):
                            side = 'buy'
                        elif 'short' in direction or direction.startswith('bear'):
                            side = 'sell'
                    if not side:
                        side = 'buy' if size_val >= 0 else 'sell'
                tick = HistoricalTick(
                    timestamp=self._parse_timestamp(ts_raw),
                    price=price,
                    size=abs(size_val),
                    side=side
                )
                self.records.append(tick)

    @staticmethod
    def _parse_timestamp(raw: str) -> datetime:
        if not raw:
            return datetime.utcnow()
        cleaned = raw.replace('Z', '+00:00')
        try:
            parsed = datetime.fromisoformat(cleaned)
        except ValueError:
            try:
                parsed = datetime.strptime(cleaned, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                parsed = datetime.utcnow()
        return parsed.replace(tzinfo=None)

    def iter_trades(self) -> Iterable[HistoricalTick]:
        return list(self.records)


class HFTHistoricalBacktester:
    """驱动 TrueHFTEngine 在离线数据上进行连续回测与参数扫描。"""

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = settings or SIMULATION_CONFIG
        self.feed = HistoricalDataFeed(self.settings)
        self.records = list(self.feed.iter_trades())
        self.price_series = [(tick.timestamp, tick.price) for tick in self.records]
        self.parameter_grid = self.settings.get('parameter_grid') or [{}]
        self.reports: List[Dict[str, Any]] = []
        self.lookahead_seconds = float(self.settings.get('price_lookahead_seconds', 0.0))

    async def run_all(self) -> List[Dict[str, Any]]:
        for idx, override in enumerate(self.parameter_grid):
            variant_name = override.get('name') or f"variant_{idx + 1}"
            report = await self._run_variant(variant_name, override)
            self.reports.append(report)
        self._persist_reports()
        return self.reports

    async def run_variant(self, variant_name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        report = await self._run_variant(variant_name, overrides)
        self.reports.append(report)
        return report

    async def _run_variant(self, variant_name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        engine = TrueHFTEngine(use_real_api=False)
        engine.signal_only_mode = True
        engine.is_running = True
        engine.detailed_monitoring = True
        engine.config['detailed_monitoring'] = True
        engine.trend_bias_threshold = 0.0
        engine.trend_fallback_threshold = 0.0
        engine.trend_guard['trade_threshold'] = 0.0
        engine.trend_guard['fallback_threshold'] = 0.0
        engine.allow_neutral_bias = True
        engine._log = lambda *args, **kwargs: None
        self._apply_overrides(engine, overrides)
        for tick in self.records:
            await engine.data_manager.process_trade(tick.as_trade())
            self._update_synthetic_orderbook(engine, tick)
            ticks = engine.data_manager.data_buffers['ticks']
            if len(ticks) < engine.signal_generator.momentum_window:
                continue
            trend_bias = engine._multi_timeframe_trend_bias()
            orderbook = engine.data_manager.get_orderbook_metrics()
            higher_timeframes = {
                '60s': engine.data_manager.data_buffers['60s'],
                '300s': engine.data_manager.data_buffers['300s'],
                '900s': engine.data_manager.data_buffers['900s'],
                '180s': engine.data_manager.data_buffers['180s'],
                'orderbook_metrics': orderbook
            }
            signal = engine.signal_generator.generate_tick_signal(
                ticks,
                trend_bias=trend_bias,
                orderbook=orderbook,
                higher_timeframes=higher_timeframes
            )
            if signal and engine._confirm_trend_direction(signal, trend_bias):
                signal_time = signal.get('timestamp')
                if signal_time and not isinstance(signal_time, datetime):
                    signal_time = tick.timestamp
                await engine._execute_signal(signal, signal_time)
            await engine._monitor_positions(current_time=tick.timestamp)
        final_time = self.records[-1].timestamp if self.records else None
        for position_id in list(engine.open_positions.keys()):
            await engine._close_position(position_id, 'backtest_end', close_time=final_time)
        return self._summarize_engine(engine, variant_name, overrides)

    def _apply_overrides(self, engine: TrueHFTEngine, overrides: Dict[str, Any]):
        if not overrides:
            return
        for key, value in overrides.items():
            if key == 'name':
                continue
            engine.config[key] = value
            if key == 'detailed_monitoring':
                engine.detailed_monitoring = bool(value)
            if key == 'momentum_threshold':
                engine.signal_generator.momentum_threshold = value
            elif key == 'momentum_window':
                engine.signal_generator.momentum_window = int(value)
            elif key == 'volume_spike_min':
                engine.signal_generator.volume_threshold = value
            elif key == 'order_imbalance_min':
                engine.signal_generator.imbalance_threshold = value
            elif key == 'composite_entry_threshold':
                engine.signal_generator.entry_threshold = value
            elif key == 'direction_vote_required':
                required = int(value)
                engine.signal_generator.direction_vote_required = required
                engine.trade_guard['direction_vote_required'] = required
            elif key == 'direction_vote_sources':
                sources = list(value)
                engine.signal_generator.direction_vote_sources = sources
                engine.trade_guard['direction_vote_sources'] = sources
            elif key == 'min_reentry_seconds':
                engine.trade_guard['min_reentry_seconds'] = float(value)
                engine.min_reentry_seconds = float(value)
            elif key == 'same_direction_reentry_seconds':
                engine.trade_guard['same_direction_reentry_seconds'] = float(value)
                engine.same_direction_reentry_seconds = float(value)
            elif key == 'duplicate_window_seconds':
                engine.trade_guard['duplicate_window_seconds'] = float(value)
                engine.duplicate_window_seconds = max(float(value), 0.1)
            elif key == 'market_volatility_threshold':
                engine.signal_generator.market_volatility_threshold = value
                engine.config['market_volatility_threshold'] = value
            elif key == 'volatility_threshold':
                engine.signal_generator.volatility_threshold = value
                engine.config['volatility_threshold'] = value
            elif key == 'trend_fast_window':
                fast = max(int(value), 5)
                engine.config[key] = fast
                engine.signal_generator.fast_window = fast
                engine.signal_generator.momentum_window = fast
            elif key == 'trend_slow_window':
                slow = max(int(value), engine.signal_generator.fast_window + 1)
                engine.config[key] = slow
                engine.signal_generator.slow_window = slow
            elif key == 'trend_min_diff':
                engine.config[key] = float(value)
                engine.signal_generator.min_diff = float(value)
            elif key == 'trend_volume_ratio':
                engine.config[key] = float(value)
                engine.signal_generator.volume_ratio_threshold = float(value)
            elif key == 'trend_recent_ticks':
                recent = max(int(value), 1)
                engine.config[key] = recent
                engine.signal_generator.recent_volume_ticks = recent
            elif key == 'trend_volume_window':
                window = max(int(value), 10)
                engine.config[key] = window
                engine.signal_generator.volume_window = window
            elif key == 'trend_orderbook_min_imbalance':
                engine.config[key] = float(value)
                engine.signal_generator.orderbook_imbalance_min = float(value)
            elif key == 'trend_signal_cooldown':
                engine.config[key] = float(value)
                engine.signal_generator.cooldown_seconds = float(value)
            elif key == 'trend_orderbook_use':
                val = bool(value)
                engine.config[key] = val
                engine.signal_generator.require_orderbook = val
            elif key == 'trend_use_cross_confirmation':
                val = bool(value)
                engine.config[key] = val
                engine.signal_generator.use_cross_confirmation = val
            elif key == 'trend_signal_mode':
                mode = str(value)
                engine.config[key] = mode
                engine.signal_generator.signal_mode = mode

    def _update_synthetic_orderbook(self, engine: TrueHFTEngine, tick: HistoricalTick):
        if not self.settings.get('simulate_orderbook_from_trades', True):
            return
        ticks = list(engine.data_manager.data_buffers['ticks'])
        if not ticks:
            return
        window = max(int(self.settings.get('synthetic_orderbook_window', 40)), 1)
        recent = ticks[-min(window, len(ticks)):]
        buy_volume = sum(t.get('buy_volume', 0.0) or 0.0 for t in recent)
        sell_volume = sum(t.get('sell_volume', 0.0) or 0.0 for t in recent)
        total = buy_volume + sell_volume
        imbalance = (buy_volume - sell_volume) / total if total > 0 else 0.0
        latest_price = recent[-1].get('price') or tick.price
        engine.data_manager.orderbook_state.update({
            'imbalance': imbalance,
            'spread': 0.0,
            'timestamp': tick.timestamp,
            'best_bid': latest_price,
            'best_ask': latest_price,
            'bid_volume_top3': buy_volume,
            'ask_volume_top3': sell_volume
        })

    def _summarize_engine(self, engine: TrueHFTEngine, name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        total = engine.stats['total_trades']
        wins = engine.stats['winning_trades']
        win_rate = wins / total if total else 0.0
        lookahead_accuracy, lookahead_samples = self._calc_lookahead_accuracy(engine)
        max_drawdown = self._max_drawdown(engine)
        summary = {
            'variant': name,
            'overrides': overrides,
            'signals_generated': engine.stats['signals_generated'],
            'positions_opened': engine.stats['positions_opened'],
            'total_trades': total,
            'win_rate': win_rate,
            'direction_accuracy': engine.stats.get('direction_accuracy', 0.0),
            'lookahead_accuracy': lookahead_accuracy,
            'lookahead_samples': lookahead_samples,
            'entry_blocks': engine.stats.get('entry_blocks', 0),
            'duplicate_blocks': engine.stats.get('duplicate_blocks', 0),
            'total_pnl': engine.stats.get('total_pnl', 0.0),
            'max_drawdown_abs': max_drawdown,
            'max_drawdown_ratio': max_drawdown / max(engine.initial_capital, 1.0),
            'error_cases': self._collect_error_cases(engine)
        }
        return summary

    @staticmethod
    def _max_drawdown(engine: TrueHFTEngine) -> float:
        history = engine.pnl_history or []
        equity = [engine.initial_capital] + [engine.initial_capital + pnl for pnl in history]
        peak = equity[0]
        max_dd = 0.0
        for value in equity:
            if value > peak:
                peak = value
            drawdown = peak - value
            if drawdown > max_dd:
                max_dd = drawdown
        return max_dd

    @staticmethod
    def _collect_error_cases(engine: TrueHFTEngine, limit: int = 5) -> List[Dict[str, Any]]:
        cases: List[Dict[str, Any]] = []
        for event in reversed(engine.signal_events):
            if event.get('result') != 'loss':
                continue
            debug = event.get('debug') or {}
            item = {
                'timestamp': event.get('timestamp').isoformat() if isinstance(event.get('timestamp'), datetime) else event.get('timestamp'),
                'direction': event.get('direction'),
                'confidence': event.get('confidence'),
                'composite': event.get('composite'),
                'votes': debug.get('votes'),
                'reason': debug.get('reason')
            }
            cases.append(item)
            if len(cases) >= limit:
                break
        return list(reversed(cases))

    def _calc_lookahead_accuracy(self, engine: TrueHFTEngine) -> Tuple[float, int]:
        if self.lookahead_seconds <= 0:
            return 0.0, 0
        total = 0
        correct = 0
        for event in engine.signal_events:
            timestamp = event.get('timestamp')
            direction = event.get('direction')
            entry_price = event.get('price')
            if not isinstance(timestamp, datetime) or direction not in ('long', 'short') or entry_price is None:
                continue
            future_price = self._price_after(timestamp, self.lookahead_seconds)
            if future_price is None:
                continue
            move = future_price - entry_price
            if (direction == 'long' and move > 0) or (direction == 'short' and move < 0):
                correct += 1
            total += 1
        accuracy = correct / total if total else 0.0
        return accuracy, total

    def _price_after(self, timestamp: datetime, seconds: float) -> Optional[float]:
        target = timestamp + timedelta(seconds=seconds)
        for ts, price in self.price_series:
            if ts >= target:
                return price
        return None

    def _persist_reports(self):
        path = Path(self.settings['report_path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as handle:
            for report in self.reports:
                serialized = self._serialize(report)
                handle.write(json.dumps(serialized, ensure_ascii=False) + "\n")

    @staticmethod
    def _serialize(report: Dict[str, Any]) -> Dict[str, Any]:
        def convert(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        return {key: convert(value) for key, value in report.items()}


async def main():
    parser = argparse.ArgumentParser(description="离线高频回测")
    parser.add_argument('--report', dest='report_path', default=None, help='覆盖默认报告输出路径')
    args = parser.parse_args()
    settings = dict(SIMULATION_CONFIG)
    if args.report_path:
        settings['report_path'] = args.report_path
    backtester = HFTHistoricalBacktester(settings)
    results = await backtester.run_all()
    best = max(results, key=lambda item: item.get('direction_accuracy', 0.0)) if results else None
    if best:
        print(
            f"[回测完成] 最佳组合 {best['variant']} | accuracy={best['direction_accuracy']:.2f} "
            f"| trades={best['total_trades']} | win_rate={best['win_rate']:.2f}"
        )
    else:
        print("没有可用的回测结果")


if __name__ == '__main__':
    asyncio.run(main())
