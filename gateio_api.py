# -*- coding: utf-8 -*-
"""
Gate.io API封装
为交易引擎提供统一的行情与下单接口
"""

import asyncio
import functools
import logging
import math
from datetime import datetime
import time
from typing import Dict, List, Optional

from gate_api import (
    ApiClient,
    Configuration,
    FuturesApi,
    FuturesOrder,
    FuturesInitialOrder,
    FuturesPriceTrigger,
    FuturesPriceTriggeredOrder,
)

from api_config import get_config
from gateio_config import (
    API_BASE_URL,
    CONTRACT_VALUE,
    ENABLE_LIVE_TRADING,
    FUTURES_SETTLE,
    MARKET_DATA_API_BASE_URL,
    MARKET_DATA_USE_MAINNET,
    ORDER_TYPE,
    SYMBOL,
    USE_GATEIO_MARKET_DATA,
    MAKER_FEE_RATE,
    TAKER_FEE_RATE,
)


class GateIOAPI:
    """Gate.io API异步封装"""

    def __init__(
        self,
        enable_market_data: bool = USE_GATEIO_MARKET_DATA,
        enable_trading: bool = ENABLE_LIVE_TRADING,
        contract_value: float = CONTRACT_VALUE,
    ):
        self.logger = logging.getLogger(__name__)
        # 确保加载环境变量/.env
        try:
            from api_config import load_config
            load_config()
        except Exception:
            pass
        self.config = get_config()
        self.contract = self.config.contract or SYMBOL
        self.settle = self.config.settle or FUTURES_SETTLE
        self.market_data_enabled = enable_market_data
        self.trading_enabled = enable_trading and bool(self.config.api_key and self.config.api_secret)
        self.contract_value = contract_value if contract_value > 0 else 1.0

        # 强制使用测试网端点（如果 config.testnet=True），避免环境变量未生效时落到主网域名
        # 优先使用 fx-api-testnet 域名，避免 api-testnet.gateapi.io 解析不稳；仅在 testnet 且未指定使用主网行情时启用
        api_host = "https://fx-api-testnet.gateio.ws/api/v4" if (self.config.testnet and not MARKET_DATA_USE_MAINNET) else API_BASE_URL
        configuration = Configuration(
            host=api_host,
            key=self.config.api_key if self.trading_enabled else None,
            secret=self.config.api_secret if self.trading_enabled else None,
        )

        self.api_client = ApiClient(configuration)
        self.futures_api = FuturesApi(self.api_client)

        market_host = MARKET_DATA_API_BASE_URL or api_host
        if self.config.testnet and not MARKET_DATA_USE_MAINNET:
            market_host = "https://fx-api-testnet.gateio.ws/api/v4"
        if market_host == API_BASE_URL:
            self.market_api_client = self.api_client
            self.market_futures_api = self.futures_api
        else:
            market_configuration = Configuration(
                host=market_host,
                key=self.config.api_key if self.trading_enabled else None,
                secret=self.config.api_secret if self.trading_enabled else None,
            )
            self.market_api_client = ApiClient(market_configuration)
            self.market_futures_api = FuturesApi(self.market_api_client)

        now = time.time()
        self._fee_cache = {
            "maker": MAKER_FEE_RATE,
            "taker": TAKER_FEE_RATE,
            "timestamp": now - 9999,
        }
        self._funding_cache = {
            "rate": 0.0,
            "next_funding_time": None,
            "timestamp": now - 9999,
        }

    @property
    def can_use_market_data(self) -> bool:
        return self.market_data_enabled

    @property
    def can_trade(self) -> bool:
        return self.trading_enabled

    async def close(self):
        if self.api_client:
            await self._run_async(self.api_client.close)
        if getattr(self, "market_api_client", None) and self.market_api_client is not self.api_client:
            await self._run_async(self.market_api_client.close)

    async def get_recent_orders(
        self,
        hours: Optional[int] = None,
        limit: Optional[int] = None,
        contract: Optional[str] = None
    ) -> List[Dict]:
        """
        拉取近 hours 小时内的委托列表，按 Gate 提供的时间段查询接口。
        """
        if not self.can_trade:
            raise RuntimeError("未配置交易密钥，无法查询历史委托")

        from config import HISTORY_FETCH_CONFIG

        hours = int(hours if hours is not None else HISTORY_FETCH_CONFIG.get("recent_hours", 8))
        limit = int(limit if limit is not None else HISTORY_FETCH_CONFIG.get("max_orders", 200))
        to_ts = int(time.time())
        from_ts = to_ts - hours * 3600

        orders = await self._run_async(
            self.futures_api.get_orders_with_time_range,
            self.settle,
            contract=contract or self.contract,
            _from=from_ts,
            to=to_ts,
            limit=limit,
        )

        parsed: List[Dict] = []

        def _ts(val):
            if val is None:
                return None
            try:
                v = float(val)
                if v > 1e12:  # ms
                    v /= 1000.0
                return datetime.fromtimestamp(v)
            except Exception:
                return None

        for o in orders or []:
            data = o.to_dict() if hasattr(o, "to_dict") else dict(o)
            parsed.append(
                {
                    "id": str(data.get("id") or ""),
                    "contract": data.get("contract") or self.contract,
                    "side": data.get("side"),
                    "price": float(data.get("price", 0) or 0),
                    "size": float(data.get("size", 0) or 0),
                    "left": float(data.get("left", 0) or 0),
                    "fill_price": float(data.get("fill_price", 0) or 0),
                    "status": data.get("status"),
                    "finish_as": data.get("finish_as"),
                    "tif": data.get("tif"),
                    "reduce_only": data.get("is_reduce_only"),
                    "create_time": _ts(data.get("create_time") or data.get("create_time_ms")),
                    "update_time": _ts(data.get("update_time") or data.get("update_time_ms")),
                }
            )
        return parsed

    async def get_klines(self, interval: str, limit: int) -> List[Dict]:
        """获取K线数据"""
        if not self.can_use_market_data:
            raise RuntimeError("未启用Gate.io行情数据")

        try:
            candles = await self._run_async(
                self.market_futures_api.list_futures_candlesticks,
                self.settle,
                self.contract,
                limit=limit,
                interval=interval,
            )
        except Exception as exc:
            # DNS 或测试网端点不可用时回退主网行情，仅用于暖机
            if self.config.testnet:
                self.logger.warning(f"测试网行情失败，尝试主网作为暖机: {exc}")
                try:
                    fallback_client = ApiClient(Configuration(host=API_BASE_URL))
                    fallback_api = FuturesApi(fallback_client)
                    candles = await self._run_async(
                        fallback_api.list_futures_candlesticks,
                        self.settle,
                        self.contract,
                        limit=limit,
                        interval=interval,
                    )
                except Exception as exc2:
                    self.logger.error(f"主网回退也失败: {exc2}")
                    raise
            else:
                raise

        rows = []
        for candle in candles:
            rows.append(
                {
                    "timestamp": datetime.fromtimestamp(int(candle.t)),
                    "open": float(candle.o),
                    "high": float(candle.h),
                    "low": float(candle.l),
                    "close": float(candle.c),
                    "volume": float(candle.v),
                }
            )
        return rows

    async def get_ticker(self) -> Optional[Dict]:
        """获取最新ticker"""
        if not self.can_use_market_data:
            return None

        tickers = await self._run_async(
            self.market_futures_api.list_futures_tickers,
            self.settle,
            contract=self.contract,
        )
        if not tickers:
            return None

        ticker_obj = tickers[0]
        ticker = ticker_obj.to_dict() if hasattr(ticker_obj, "to_dict") else {}
        return {
            "last_price": float(ticker.get("last", ticker.get("close", 0)) or 0),
            "mark_price": float(ticker.get("mark_price", ticker.get("mark", 0)) or 0),
            "best_bid": float(ticker.get("best_bid", ticker.get("bid", 0)) or 0) or None,
            "best_ask": float(ticker.get("best_ask", ticker.get("ask", 0)) or 0) or None,
            "timestamp": datetime.fromtimestamp(int(ticker.get("t", time.time()))),
        }

    async def set_leverage(self, leverage: int, contract: Optional[str] = None):
        """设置仓位杠杆"""
        if not self.can_trade:
            return False

        leverage_value = str(int(leverage))
        available = await self.get_available_margin(fallback=0.0)
        if available <= 0:
            self.logger.warning("Skipping leverage update because futures wallet has no available balance")
            return False

        try:
            await self._run_async(
                self.futures_api.update_position_leverage,
                self.settle,
                contract or self.contract,
                leverage_value,
            )
            return True
        except Exception as exc:
            self.logger.error(f"Failed to set leverage to {leverage_value}x: {exc}")
            return False

    async def place_order(
        self,
        contract: str,
        size: float,
        side: str,
        order_type: str = ORDER_TYPE,
        time_in_force: str = "ioc",
        price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> Dict:
        """下单"""
        if not self.can_trade:
            raise RuntimeError("实盘下单未启用")

        signed_size = self._to_contracts(size, side)
        tif = time_in_force or ("ioc" if order_type == "market" else "gtc")
        price_value = price

        if order_type == "market":
            price_value = 0.0
        elif price_value is None:
            raise ValueError("限价单必须提供价格")

        order = FuturesOrder(
            contract=contract,
            size=signed_size,
            price=price_value,
            tif=tif,
            reduce_only=reduce_only,
        )

        response = await self._run_async(
            self.futures_api.create_futures_order,
            self.settle,
            order,
        )

        return response.to_dict()

    async def place_stop_order(
        self,
        contract: str,
        side: str,
        trigger_price: float,
        price_type: int = 1,
        expiration: int = 3600,
    ) -> Optional[Dict]:
        """在交易所挂出价格触发的止损单"""
        if not self.can_trade:
            return None

        return await self._place_price_triggered_order(
            contract=contract,
            side=side,
            trigger_price=trigger_price,
            price_type=price_type,
            expiration=expiration,
            tp=False
        )

    async def place_take_profit_order(
        self,
        contract: str,
        side: str,
        trigger_price: float,
        price_type: int = 1,
        expiration: int = 3600,
    ) -> Optional[Dict]:
        """在交易所挂出价格触发的止盈单"""
        if not self.can_trade:
            return None

        return await self._place_price_triggered_order(
            contract=contract,
            side=side,
            trigger_price=trigger_price,
            price_type=price_type,
            expiration=expiration,
            tp=True
        )

    async def cancel_stop_order(self, order_id: str) -> Optional[Dict]:
        """取消触发单"""
        if not self.can_trade or not order_id:
            return None
        return await self._run_async(
            self.futures_api.cancel_price_triggered_order,
            self.settle,
            order_id,
        )

    async def cancel_all_reducing_triggers(self, side: Optional[str] = None) -> int:
        """
        取消所有 reduce-only 的触发单（用于平仓后扫尾）。
        返回取消的数量。
        """
        if not self.can_trade:
            return 0
        try:
            orders = await self._run_async(
                self.futures_api.list_price_triggered_orders,
                self.settle,
                status="open"
            )
        except Exception:
            return 0
        cancelled = 0
        for order in orders or []:
            data = order.to_dict() if hasattr(order, "to_dict") else order
            initial = data.get("initial_order") or {}
            auto_size = str(initial.get("auto_size") or "").lower()
            direction = "long" if "long" in auto_size else "short" if "short" in auto_size else None
            if side and direction and direction != side.lower():
                continue
            order_id = data.get("id")
            if not order_id:
                continue
            try:
                await self.cancel_stop_order(str(order_id))
                cancelled += 1
            except Exception:
                continue
        return cancelled

    async def _place_price_triggered_order(
        self,
        contract: str,
        side: str,
        trigger_price: float,
        price_type: int,
        expiration: int,
        tp: bool = False
    ) -> Optional[Dict]:
        """
        通用的价格触发单，tp=True 表示止盈（反向条件），tp=False 表示止损。
        rule: 1 表示 >=，2 表示 <=
        """
        rule = 1 if tp else 2  # 默认假设多单止损用 <=，止盈用 >=
        side_lower = side.lower()
        if side_lower == "short":
            rule = 2 if tp else 1
        auto_size = "close_long" if side_lower == "long" else "close_short"
        trigger_price_aligned = self._align_price(trigger_price, side_lower, tp)
        trigger = FuturesPriceTrigger(
            strategy_type=0,
            price_type=price_type,
            price=str(trigger_price_aligned),
            rule=rule,
            expiration=expiration,
        )
        initial = FuturesInitialOrder(
            contract=contract,
            size=0,
            price="0",
            tif="ioc",
            reduce_only=True,
            auto_size=auto_size,
            is_reduce_only=True,
            is_close=True,
        )
        order = FuturesPriceTriggeredOrder(initial=initial, trigger=trigger)
        response = await self._run_async(
            self.futures_api.create_price_triggered_order,
            self.settle,
            order,
        )
        return response.to_dict() if response else None

    def _align_price(self, price: float, side_lower: str, tp: bool) -> float:
        """
        按最小价格单位对齐触发价。
        - 多单止损/空单止盈：触发价向下取整
        - 多单止盈/空单止损：触发价向上取整
        """
        try:
            from gateio_config import PRICE_TICK_SIZE
            tick = max(float(PRICE_TICK_SIZE), 1e-12)
        except Exception:
            tick = 0.01
        if tick <= 0:
            tick = 0.01
        if side_lower == "long":
            aligned = (int(price / tick) * tick) if not tp else ((int((price + tick - 1e-12) / tick)) * tick)
        else:
            aligned = (int((price + tick - 1e-12) / tick) * tick) if not tp else (int(price / tick) * tick)
        return round(aligned, 10)

    async def get_trade_history(
        self,
        contract: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        role: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> List[Dict]:
        """拉取账户成交记录"""
        if not self.can_trade:
            raise RuntimeError("实盘API未启用，无法查询交易记录")

        params = {
            "contract": contract or self.contract,
            "limit": min(max(limit, 1), 1000),
            "offset": max(offset, 0),
        }
        if order_id:
            params["order_id"] = order_id

        use_timerange = start_time is not None or end_time is not None or role
        if use_timerange:
            if start_time is not None:
                params["_from"] = int(start_time)
            if end_time is not None:
                params["to"] = int(end_time)
            if role:
                params["role"] = role
            trades = await self._run_async(
                self.futures_api.get_my_trades_with_time_range,
                self.settle,
                **params,
            )
        else:
            trades = await self._run_async(
                self.futures_api.get_my_trades,
                self.settle,
                **params,
            )

        records: List[Dict] = []
        for trade in trades or []:
            if hasattr(trade, "to_dict"):
                data = trade.to_dict()
            elif isinstance(trade, dict):
                data = trade
            else:
                continue

            trade_id = data.get("id") or data.get("trade_id")
            ts = float(data.get("create_time") or 0)
            direction_size = data.get("size") or 0
            price_value = data.get("price") or 0
            fee_value = data.get("fee") or 0
            record = {
                "id": str(trade_id) if trade_id is not None else "",
                "timestamp": datetime.fromtimestamp(ts) if ts else datetime.utcnow(),
                "contract": data.get("contract") or params["contract"],
                "order_id": str(data.get("order_id") or ""),
                "size": int(direction_size) if direction_size not in (None, "") else 0,
                "close_size": int(data.get("close_size") or 0),
                "price": float(price_value) if price_value not in (None, "") else 0.0,
                "role": data.get("role") or "",
                "text": data.get("text") or "",
                "fee": float(fee_value) if fee_value not in (None, "") else 0.0,
                "point_fee": float(data.get("point_fee") or 0.0),
            }
            records.append(record)

        return records

    async def close_position(self, contract: str, size: float, direction: str):
        """用反向市价单平仓"""
        if not self.can_trade:
            return None

        inverse_side = "sell" if direction == "long" else "buy"
        return await self.place_order(
            contract=contract,
            size=size,
            side=inverse_side,
            order_type="market",
            time_in_force="ioc",
        )

    async def get_order_status(self, order_id: str) -> Dict:
        """查询订单状态"""
        if not self.can_trade:
            raise RuntimeError("实盘下单未启用")

        order = await self._run_async(
            self.futures_api.get_futures_order,
            self.settle,
            order_id,
        )
        return order.to_dict()

    async def cancel_order(self, order_id: str) -> Dict:
        """取消订单"""
        if not self.can_trade:
            raise RuntimeError("实盘下单未启用")

        result = await self._run_async(
            self.futures_api.cancel_futures_order,
            self.settle,
            order_id,
        )
        return result.to_dict() if result else {}

    async def get_account_balance(self, fallback: float = 0.0) -> Dict:
        """返回逐仓账户的可用/总权益等信息"""
        if not self.can_trade:
            return {
                'available': fallback,
                'total': fallback,
                'unrealized_pnl': 0.0
            }

        try:
            accounts = await self._run_async(
                self.futures_api.list_futures_accounts,
                self.settle
            )
            account = accounts[0] if isinstance(accounts, list) and accounts else accounts
            if not account:
                return {'available': fallback, 'total': fallback, 'unrealized_pnl': 0.0}

            if hasattr(account, 'available'):
                available = float(getattr(account, 'available', fallback))
                total = float(getattr(account, 'total', available))
                unrealized = float(getattr(account, 'unrealized_pnl', 0.0))
                return {'available': available, 'total': total, 'unrealized_pnl': unrealized}

            if isinstance(account, dict):
                available = float(account.get('available', fallback))
                total = float(account.get('total', account.get('balance', available)))
                unrealized = float(account.get('unrealized_pnl', 0.0))
                return {'available': available, 'total': total, 'unrealized_pnl': unrealized}
        except AttributeError:
            self.logger.warning("Gate API SDK 缺少 list_futures_accounts，使用 fallback")
        except Exception as exc:
            self.logger.error(f"获取可用保证金失败: {exc}")

        return {'available': fallback, 'total': fallback, 'unrealized_pnl': 0.0}

    async def get_available_margin(self, fallback: float = 0.0) -> float:
        info = await self.get_account_balance(fallback=fallback)
        return info.get('available', fallback)

    def _to_contracts(self, size: float, side: str) -> int:
        # size 已经是“张数”，无需再按合约面值换算，防止下单量被放大
        contracts = max(1, int(round(abs(size))))
        if side.lower() in ("sell", "short"):
            contracts = -contracts
        return contracts

    async def get_trading_fee_rates(self, force_refresh: bool = False, ttl: float = 300.0) -> Dict[str, float]:
        now = time.time()
        cached = self._fee_cache
        if (
            not force_refresh
            and cached.get("maker") is not None
            and cached.get("taker") is not None
            and now - cached.get("timestamp", 0.0) < ttl
        ):
            return {"maker": cached["maker"], "taker": cached["taker"]}

        maker = cached.get("maker", MAKER_FEE_RATE)
        taker = cached.get("taker", TAKER_FEE_RATE)
        try:
            fee_resp = await self._run_async(
                self.futures_api.get_futures_fee,
                self.settle,
                contract=self.contract,
            )
            if fee_resp:
                data = fee_resp.to_dict() if hasattr(fee_resp, "to_dict") else dict(fee_resp)
                maker = float(data.get("maker_fee_rate", data.get("maker_fee", maker)))
                taker = float(data.get("taker_fee_rate", data.get("taker_fee", taker)))
        except Exception as exc:
            self.logger.warning(f"获取实时费率失败，使用缓存: {exc}")
            contract_info = await self._fetch_contract_snapshot()
            if contract_info:
                maker = float(contract_info.get("maker_fee_rate", maker))
                taker = float(contract_info.get("taker_fee_rate", taker))

        self._fee_cache.update({"maker": maker, "taker": taker, "timestamp": now})
        return {"maker": maker, "taker": taker}

    async def get_positions(self, contract: Optional[str] = None) -> List[Dict]:
        """查询持仓（合约）。"""
        if not self.can_trade:
            return []
        try:
            if contract:
                resp = [await self._run_async(self.futures_api.get_position, self.settle, contract)]
            else:
                resp = await self._run_async(self.futures_api.list_positions, self.settle)
        except Exception as exc:
            self.logger.error(f"获取持仓失败: {exc}")
            return []
        rows = []
        for p in resp or []:
            data = p.to_dict() if hasattr(p, "to_dict") else dict(p)
            rows.append(
                {
                    "contract": data.get("contract"),
                    "size": float(data.get("size") or 0.0),
                    "entry_price": float(data.get("entry_price") or 0.0),
                    "margin": float(data.get("margin") or 0.0),
                    "mark_price": float(data.get("mark_price") or 0.0),
                    "leverage": float(data.get("leverage") or 0.0),
                    "unrealized_pnl": float(data.get("unrealized_pnl") or 0.0),
                }
            )
        return rows

    async def get_latest_funding_info(self, force_refresh: bool = False, ttl: float = 300.0) -> Dict[str, Optional[float]]:
        now = time.time()
        cached = self._funding_cache
        if (
            not force_refresh
            and cached.get("timestamp", 0.0) > 0
            and now - cached["timestamp"] < ttl
        ):
            return dict(cached)

        contract_info = await self._fetch_contract_snapshot()
        if contract_info:
            rate = float(contract_info.get("funding_rate", cached.get("rate", 0.0)) or 0.0)
            next_time = contract_info.get("funding_next_apply")
            try:
                next_ts = float(next_time) if next_time is not None else cached.get("next_funding_time")
            except (TypeError, ValueError):
                next_ts = cached.get("next_funding_time")
            cached = {
                "rate": rate,
                "next_funding_time": next_ts,
                "timestamp": now,
            }
            self._funding_cache = cached
        return dict(self._funding_cache)

    async def get_account_ledger(self, limit: int = 50) -> List[Dict]:
        if not self.can_trade:
            raise RuntimeError("实盘API未启用，无法查询资金流水")
        limit = min(max(int(limit), 1), 200)
        entries = await self._run_async(
            self.futures_api.list_futures_account_book,
            self.settle,
            limit=limit,
        )
        rows: List[Dict] = []
        for entry in entries or []:
            data = entry.to_dict() if hasattr(entry, "to_dict") else dict(entry)
            rows.append(
                {
                    "time": datetime.fromtimestamp(float(data.get("time", 0)) or time.time()),
                    "change": float(data.get("change", 0.0)),
                    "balance": float(data.get("balance", 0.0)),
                    "type": data.get("type"),
                    "text": data.get("text"),
                }
            )
        return rows

    async def _fetch_contract_snapshot(self) -> Optional[Dict]:
        try:
            contract = await self._run_async(
                self.market_futures_api.get_futures_contract,
                self.settle,
                self.contract,
            )
        except Exception as exc:
            self.logger.warning(f"获取合约元数据失败: {exc}")
            return None
        return contract.to_dict() if hasattr(contract, "to_dict") else dict(contract or {})

    async def _run_async(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        bound = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, bound)
