# -*- coding: utf-8 -*-
"""
高频执行器
"""

import asyncio
import logging
import time
from typing import Dict, Optional

from gateio_config import SYMBOL


class HFTExecutor:
    """高频执行引擎"""

    def __init__(
        self,
        api_client=None,
        min_order_interval: float = 0.1,
        enable_exchange_stop_orders: bool = False,
        stop_loss_ratio: float = 0.006,
        stop_order_price_type: int = 1,
        stop_order_expiration: int = 3600
    ):
        self.logger = logging.getLogger(__name__)
        self.api_client = api_client
        self.min_order_interval = min_order_interval
        self.last_order_time = 0.0
        self.enable_exchange_stop_orders = enable_exchange_stop_orders
        self.default_stop_loss_ratio = max(stop_loss_ratio, 0.0)
        self.stop_order_price_type = stop_order_price_type
        self.stop_order_expiration = stop_order_expiration

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

        return await self._place_market_order(signal, position_config, current_price)

    async def _place_market_order(
        self,
        signal: Dict,
        position_config: Dict,
        reference_price: Optional[float] = None
    ):
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

        if self._order_success(result):
            stop_ratio = position_config.get('stop_loss_ratio') or signal.get('stop_loss_ratio') or self.default_stop_loss_ratio
            stop_order_id = await self._set_stop_loss(signal, result, reference_price, stop_ratio)
            fills = await self._fetch_fills(result)
            return {
                'status': 'filled',
                'order_id': result.get('id'),
                'latency': latency,
                'stop_order_id': stop_order_id,
                'fills': fills
            }
        return {'status': 'failed', 'reason': result.get('status', 'not_filled')}

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
        if self._order_success(result):
            stop_ratio = position_config.get('stop_loss_ratio') or signal.get('stop_loss_ratio') or self.default_stop_loss_ratio
            stop_order_id = await self._set_stop_loss(signal, result, limit_price, stop_ratio)
            fills = await self._fetch_fills(result)
            return {
                'status': 'filled',
                'order_id': result.get('id'),
                'latency': latency,
                'mode': 'limit',
                'stop_order_id': stop_order_id,
                'fills': fills
            }
        return {'status': 'failed', 'reason': result.get('status', 'limit_not_filled')}

    async def _set_stop_loss(
        self,
        signal: Dict,
        order_result: Dict,
        reference_price: Optional[float] = None,
        stop_ratio: Optional[float] = None
    ) -> Optional[str]:
        """在订单成交后，向交易所提交保护性止损"""
        if not self.supports_exchange_stops:
            return None
        ratio = stop_ratio if stop_ratio is not None else self.default_stop_loss_ratio
        if ratio <= 0:
            return None
        direction = signal.get('direction')
        if direction not in ('long', 'short'):
            return None
        fill_price = self._extract_fill_price(order_result) or reference_price
        if not fill_price or fill_price <= 0:
            return None
        if direction == 'long':
            trigger_price = fill_price * (1 - ratio)
        else:
            trigger_price = fill_price * (1 + ratio)
        try:
            response = await self.api_client.place_stop_order(
                contract=SYMBOL,
                side=direction,
                trigger_price=trigger_price,
                price_type=self.stop_order_price_type,
                expiration=self.stop_order_expiration
            )
            if response and isinstance(response, dict):
                stop_id = response.get('id') or response.get('order_id')
                return str(stop_id) if stop_id is not None else None
        except Exception as exc:
            self.logger.warning(f"Failed to place protective stop order: {exc}")
        return None

    def _order_success(self, response: Dict) -> bool:
        """根据Gate.io文档判定订单是否成交"""
        status = str(response.get('status', '')).lower()
        if status in ('closed', 'finished'):
            return True
        left = response.get('left')
        try:
            return left is not None and float(left) <= 0
        except (TypeError, ValueError):
            return False

    @property
    def supports_exchange_stops(self) -> bool:
        return bool(self.api_client) and self.enable_exchange_stop_orders

    async def _fetch_fills(self, order_result: Dict) -> Optional[list]:
        """尝试获取该订单的成交明细，返回列表[{price, size, fee, role}]"""
        if not self.api_client:
            return None
        order_id = order_result.get('id')
        if not order_id:
            return None
        try:
            trades = await self.api_client.get_trade_history(order_id=str(order_id))
        except Exception:
            return None
        fills = []
        for trade in trades or []:
            price = trade.get('price')
            size = trade.get('size')
            fee = trade.get('fee')
            role = trade.get('role')
            try:
                fills.append({
                    'price': float(price) if price is not None else None,
                    'size': float(size) if size is not None else None,
                    'fee': float(fee) if fee is not None else 0.0,
                    'role': role
                })
            except Exception:
                continue
        return fills or None

    def _extract_fill_price(self, order_result: Dict) -> Optional[float]:
        """Best-effort获取成交均价"""
        if not order_result:
            return None
        candidates = [
            order_result.get('fill_price'),
            order_result.get('avg_fill_price'),
            order_result.get('avg_price'),
            order_result.get('price')
        ]
        for value in candidates:
            if value in (None, ''):
                continue
            try:
                price = float(value)
                if price > 0:
                    return price
            except (TypeError, ValueError):
                continue
        return None
