# -*- coding: utf-8 -*-
"""
高频执行器
"""

import asyncio
import time
from typing import Dict, Optional

from gateio_config import SYMBOL


class HFTExecutor:
    """高频执行引擎"""

    def __init__(self, api_client=None, min_order_interval: float = 0.1):
        self.api_client = api_client
        self.min_order_interval = min_order_interval
        self.last_order_time = 0.0

    async def execute(
        self,
        signal: Dict,
        position_config: Dict,
        current_price: Optional[float] = None,
        prefer_limit: bool = False,
        limit_premium: float = 0.0,
        limit_timeout: float = 0.1,
        fallback_to_market: bool = True
    ):
        now = time.time()
        if now - self.last_order_time < self.min_order_interval:
            return {'status': 'throttled', 'reason': 'too_frequent'}
        self.last_order_time = now

        if not self.api_client:
            return {'status': 'simulated'}

        if prefer_limit and current_price:
            limit_result = await self._place_limit_order(
                signal,
                position_config,
                current_price,
                limit_premium,
                limit_timeout
            )
            if limit_result['status'] == 'filled' or not fallback_to_market:
                return limit_result

        return await self._place_market_order(signal, position_config)

    async def _place_market_order(self, signal: Dict, position_config: Dict):
        params = {
            'contract': SYMBOL,
            'size': position_config['position_size'],
            'side': signal['direction'],
            'order_type': 'market',
            'time_in_force': 'ioc',
            'reduce_only': False
        }
        start = time.time()
        result = await self.api_client.place_order(**params)
        latency = time.time() - start

        if result.get('status') == 'filled':
            await self._set_stop_loss(signal)
            return {'status': 'filled', 'order_id': result.get('order_id'), 'latency': latency}
        return {'status': 'failed', 'reason': 'not_filled'}

    async def _place_limit_order(self, signal: Dict, position_config: Dict,
                                 current_price: float, limit_premium: float,
                                 limit_timeout: float):
        side = signal['direction']
        price_adjustment = 1 + limit_premium if side == 'long' else 1 - limit_premium
        limit_price = current_price * price_adjustment

        params = {
            'contract': SYMBOL,
            'size': position_config['position_size'],
            'side': side,
            'order_type': 'limit',
            'price': limit_price,
            'time_in_force': 'ioc',
            'reduce_only': False
        }

        start = time.time()
        try:
            result = await asyncio.wait_for(
                self.api_client.place_order(**params),
                timeout=max(limit_timeout, 0.01)
            )
        except asyncio.TimeoutError:
            return {'status': 'timeout', 'reason': 'limit_timeout'}

        latency = time.time() - start
        if result.get('status') == 'filled':
            await self._set_stop_loss(signal)
            return {'status': 'filled', 'order_id': result.get('order_id'), 'latency': latency, 'mode': 'limit'}
        return {'status': 'failed', 'reason': 'limit_not_filled'}

    async def _set_stop_loss(self, signal: Dict):
        # TODO: Implement via conditional orders when API supports it.
        pass
