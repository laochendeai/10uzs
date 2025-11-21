# -*- coding: utf-8 -*-
"""
高频数据管理器
"""

import asyncio
from collections import deque
from datetime import datetime
from typing import Dict, Deque, List


class HFTDataManager:
    """管理tick和秒级K线的数据缓冲"""

    def __init__(self):
        self.data_buffers: Dict[str, Deque] = {
            '1s': deque(maxlen=300),
            '3s': deque(maxlen=200),
            '5s': deque(maxlen=180),
            '7s': deque(maxlen=200),
            '13s': deque(maxlen=200),
            '15s': deque(maxlen=120),
            '180s': deque(maxlen=60),
            'ticks': deque(maxlen=1000),
            'trades': deque(maxlen=1000)
        }
        self.latest_price = None
        self.orderbook_state: Dict[str, any] = {
            'bids': [],
            'asks': [],
            'spread': 0.0,
            'timestamp': None,
            'imbalance': 0.0,
            'liquidity': 0.0
        }

    async def process_ticker(self, ticker: Dict):
        tick = self._parse_ticker_data(ticker)
        self.latest_price = tick['price']

    async def process_trade(self, trade: Dict):
        self.data_buffers['trades'].append(trade)
        price = trade.get('price', self.latest_price or 0.0)
        volume = abs(trade.get('size', 0.0))
        side = trade.get('side')
        if side is None and self.latest_price is not None:
            side = 'buy' if price >= self.latest_price else 'sell'

        buy_volume = volume if side == 'buy' else 0.0
        sell_volume = volume if side == 'sell' else 0.0

        tick = {
            'timestamp': trade.get('timestamp', datetime.utcnow()),
            'price': price,
            'volume': volume,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume
        }
        self.data_buffers['ticks'].append(tick)
        await self._update_second_bars(tick)

    def update_orderbook(self, orderbook: Dict):
        if not orderbook:
            return
        bids = orderbook.get('top_bids') or []
        asks = orderbook.get('top_asks') or []
        bid_vol = sum(level.get('size', 0.0) for level in bids)
        ask_vol = sum(level.get('size', 0.0) for level in asks)
        total = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total if total > 0 else 0.0
        liquidity = total
        spread = orderbook.get('spread')
        self.orderbook_state = {
            'bids': bids,
            'asks': asks,
            'spread': spread,
            'timestamp': orderbook.get('timestamp'),
            'imbalance': imbalance,
            'liquidity': liquidity,
            'best_bid': orderbook.get('best_bid'),
            'best_ask': orderbook.get('best_ask')
        }

    def get_orderbook_metrics(self) -> Dict:
        return dict(self.orderbook_state)

    def _parse_ticker_data(self, data: Dict) -> Dict:
        ts = data.get('timestamp') or datetime.utcnow()
        return {
            'timestamp': ts if isinstance(ts, datetime) else datetime.utcnow(),
            'price': data.get('last_price', 0.0),
            'volume': data.get('volume', 0.0)
        }

    async def _update_second_bars(self, tick: Dict):
        current_second = int(tick['timestamp'].timestamp())
        if not self.data_buffers['1s'] or \
           int(self.data_buffers['1s'][-1]['timestamp'].timestamp()) < current_second:
            new_bar = {
                'timestamp': tick['timestamp'].replace(microsecond=0),
                'open': tick['price'],
                'high': tick['price'],
                'low': tick['price'],
                'close': tick['price'],
                'volume': tick.get('volume', 0),
                'tick_count': 1
            }
            self.data_buffers['1s'].append(new_bar)
        else:
            bar = self.data_buffers['1s'][-1]
            bar['high'] = max(bar['high'], tick['price'])
            bar['low'] = min(bar['low'], tick['price'])
            bar['close'] = tick['price']
            bar['volume'] += tick.get('volume', 0)
            bar['tick_count'] += 1

        await self._aggregate_higher_timeframes()

    async def _aggregate_higher_timeframes(self):
        self._aggregate_bars('1s', '3s', 3)
        self._aggregate_bars('1s', '5s', 5)
        self._aggregate_bars('1s', '7s', 7)
        self._aggregate_bars('1s', '13s', 13)
        self._aggregate_bars('5s', '15s', 3)
        self._aggregate_bars('15s', '180s', 12)

    def _aggregate_bars(self, source_key: str, target_key: str, window: int):
        source = self.data_buffers[source_key]
        target = self.data_buffers[target_key]
        if not source:
            return

        grouped = list(source)[-window:]
        if len(grouped) < window:
            return

        new_bar = {
            'timestamp': grouped[0]['timestamp'],
            'open': grouped[0]['open'],
            'high': max(bar['high'] for bar in grouped),
            'low': min(bar['low'] for bar in grouped),
            'close': grouped[-1]['close'],
            'volume': sum(bar['volume'] for bar in grouped),
            'tick_count': sum(bar['tick_count'] for bar in grouped)
        }
        if not target or target[-1]['timestamp'] != new_bar['timestamp']:
            target.append(new_bar)
