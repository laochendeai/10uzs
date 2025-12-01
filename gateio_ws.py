# -*- coding: utf-8 -*-
"""
Gate.io WebSocket 行情客户端
"""

import asyncio
import json
import logging
import subprocess
import time
from collections import deque
from datetime import datetime
from typing import Optional

import socket
import websockets

from gateio_config import FUTURES_SETTLE, SYMBOL, WS_BASE_URL, MARKET_DATA_WS_URL

_DNS_CACHE = {}
_ORIGINAL_GETADDRINFO = socket.getaddrinfo


def _resolve_with_nslookup(host: str) -> Optional[str]:
    if host in _DNS_CACHE:
        return _DNS_CACHE[host]
    try:
        result = subprocess.run(
            ["nslookup", host],
            capture_output=True,
            text=True,
            timeout=2
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("Address:"):
            ip = line.split("Address:")[1].strip()
            if ip and not ip.startswith("127."):
                _DNS_CACHE[host] = ip
                return ip
    return None


def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _ORIGINAL_GETADDRINFO(host, port, family, type, proto, flags)
    except socket.gaierror:
        ip = _resolve_with_nslookup(host)
        if ip:
            return _ORIGINAL_GETADDRINFO(ip, port, family, type, proto, flags)
        raise


socket.getaddrinfo = _patched_getaddrinfo


class GateIOWebsocket:
    """简单的Gate.io永续合约行情WebSocket封装"""

    def __init__(
        self,
        contract: str = SYMBOL,
        settle: str = FUTURES_SETTLE,
        url: str = MARKET_DATA_WS_URL,
        candlestick_interval: str = "10s",
        order_book_interval: str = "100ms",
        order_book_depth: int = 20,
    ):
        self.logger = logging.getLogger(__name__)
        self.contract = contract
        self.settle = settle
        self.url = url
        self.candle_interval = str(candlestick_interval)
        self.order_book_interval = order_book_interval
        self.order_book_depth = order_book_depth

        self._task: Optional[asyncio.Task] = None
        self._ws = None
        self._stop = False
        self._connected = False

        self.last_ticker: Optional[dict] = None
        self.last_candle: Optional[dict] = None
        self.last_order_book: Optional[dict] = None
        self._update_event = asyncio.Event()
        self._ticker_updates = deque(maxlen=100)
        self._order_book_updates = deque(maxlen=500)
        self._trade_updates = deque(maxlen=1000)

    async def start(self):
        """启动WebSocket监听"""
        if self._task and not self._task.done():
            return
        self._stop = False
        self._task = asyncio.create_task(self._run(), name="gateio-ws-listener")

    async def stop(self):
        """停止WebSocket"""
        self._stop = True
        if self._ws:
            await self._ws.close()
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._connected = False

    def is_connected(self) -> bool:
        """返回当前连接状态，用于监控心跳."""
        return self._connected

    def get_latest_ticker(self) -> Optional[dict]:
        """获取最近一次ticker"""
        return self.last_ticker

    def get_latest_candle(self) -> Optional[dict]:
        return self.last_candle

    def get_recent_ticker_updates(self, limit: int = 10):
        """返回最近的ticker更新列表（最新在前）"""
        if not self._ticker_updates:
            return []
        return list(self._ticker_updates)[:limit]

    def get_recent_order_book_updates(self, limit: int = 20):
        """返回最近的订单簿更新（最新在前）"""
        if not self._order_book_updates:
            return []
        return list(self._order_book_updates)[:limit]

    def get_recent_trade_updates(self, limit: int = 100):
        """返回最近的成交记录"""
        if not self._trade_updates:
            return []
        return list(self._trade_updates)[:limit]

    async def wait_for_update(self, timeout: float = 5.0) -> bool:
        """等待下一次行情更新，超时返回False"""
        try:
            await asyncio.wait_for(self._update_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            self._update_event.clear()

    async def _run(self):
        """维持WebSocket连接并处理推送"""
        attempt = 0
        while not self._stop:
            attempt += 1
            try:
                self.logger.info(f"🌐 WebSocket连接尝试 #{attempt} -> {self.url}")
                async with websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    max_queue=None,
                    logger=self.logger,
                    open_timeout=10
                ) as ws:
                    self.logger.info("✅ Gate.io WebSocket已连接")
                    self._ws = ws
                    self._connected = True
                    await self._subscribe(ws)

                    async for message in ws:
                        await self._handle_message(message)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error(f"❌ WebSocket连接错误: {exc}")
                await asyncio.sleep(min(5 + attempt, 30))
            finally:
                self._connected = False
        self.logger.info("🛑 Gate.io WebSocket已停止")

    async def _subscribe(self, ws):
        """订阅ticker与candlestick"""
        timestamp = int(time.time())
        contract = self.contract
        # Gate 官方文档：WS 基础 URL 已含结算币，无需再传 settle；各频道 payload 需符合最新参数顺序
        order_book_limit = str(self.order_book_depth)
        order_book_interval = "0"  # futures.order_book 仅接受 interval=0（不做档位聚合）
        subscribe_msgs = [
            {
                "time": timestamp,
                "channel": "futures.tickers",
                "event": "subscribe",
                "payload": [contract],
            },
            {
                "time": timestamp,
                "channel": "futures.book_ticker",
                "event": "subscribe",
                "payload": [contract],
            },
            {
                "time": timestamp,
                "channel": "futures.candlesticks",
                "event": "subscribe",
                "payload": [self.candle_interval, contract],
            },
            {
                "time": timestamp,
                "channel": "futures.order_book",
                "event": "subscribe",
                "payload": [contract, order_book_limit, order_book_interval],
            },
            {
                "time": timestamp,
                "channel": "futures.trades",
                "event": "subscribe",
                "payload": [contract],
            },
        ]

        for msg in subscribe_msgs:
            await ws.send(json.dumps(msg))

    async def _handle_message(self, raw: str):
        """处理推送数据"""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.warning(f"无法解析的WebSocket消息: {raw[:80]}")
            return

        channel = msg.get("channel")
        event = msg.get("event")
        result = msg.get("result")

        if event == "subscribe":
            self.logger.debug(f"WebSocket订阅成功: {channel}")
            return
        if event == "error":
            self.logger.error(f"WebSocket返回错误: {msg}")
            return

        if event not in ("update", "all"):
            return

        updated = False

        if channel == "futures.tickers":
            data = self._extract_data(result)
            if not data:
                return
            try:
                ticker = {
                    "last_price": float(data.get("last", data.get("close", 0))),
                    "mark_price": float(data.get("mark_price", data.get("mark_price", 0))),
                    "timestamp": datetime.utcnow(),
                }
                self.last_ticker = ticker
                self._ticker_updates.appendleft(ticker)
                updated = True
            except Exception:
                pass
        elif channel == "futures.candlesticks":
            data = self._extract_data(result)
            if not data:
                return
            try:
                timestamp_value = data.get("t") or data.get("time") or data.get("timestamp")
                ts = datetime.fromtimestamp(int(timestamp_value)) if timestamp_value else datetime.utcnow()
                self.last_candle = {
                    "timestamp": ts,
                    "open": float(data.get("o")),
                    "high": float(data.get("h")),
                    "low": float(data.get("l")),
                    "close": float(data.get("c")),
                    "volume": float(data.get("v", data.get("volume", 0))),
                }
                updated = True
            except Exception:
                pass
        elif channel == "futures.order_book":
            ob_data = self._parse_order_book(result)
            if not ob_data:
                return
            self.last_order_book = ob_data
            self._order_book_updates.appendleft(ob_data)
            updated = True
        elif channel == "futures.trades":
            trades = self._parse_trades(result)
            if trades:
                for trade in trades:
                    self._trade_updates.appendleft(trade)
                updated = True
        elif channel == "futures.book_ticker":
            ob_data = self._parse_book_ticker(result)
            if not ob_data:
                return
            self.last_order_book = ob_data
            self._order_book_updates.appendleft(ob_data)
            updated = True

        if updated:
            self._update_event.set()

    def _extract_data(self, result):
        """兼容不同返回结构"""
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            first = result[0] if result else None
            if isinstance(first, dict):
                return first
            if isinstance(result, list) and len(result) >= 6:
                keys = ["t", "o", "h", "l", "c", "v"]
                return dict(zip(keys, result[:6]))
        return None

    def _parse_order_book(self, result):
        """解析订单簿更新"""
        payload = None

        if isinstance(result, dict):
            payload = result
        elif isinstance(result, list):
            # 结构可能是 [contract, interval, depth, data_dict]
            if len(result) >= 4 and isinstance(result[3], dict):
                payload = result[3]
            else:
                for item in result:
                    if isinstance(item, dict) and ("bids" in item or "asks" in item):
                        payload = item
                        break

        if not payload:
            return None

        bids = payload.get("bids") or payload.get("bid") or payload.get("b") or []
        asks = payload.get("asks") or payload.get("ask") or payload.get("a") or []

        def _extract_price(levels):
            try:
                if not levels:
                    return None
                level = levels[0]
                if isinstance(level, (list, tuple)):
                    return float(level[0])
                if isinstance(level, dict):
                    return float(level.get("p") or level.get("price"))
                return float(level)
            except (TypeError, ValueError):
                return None

        best_bid = _extract_price(bids)
        best_ask = _extract_price(asks)

        def _parse_levels(levels, limit=5):
            parsed = []
            for level in levels[:limit]:
                price = None
                size = None
                if isinstance(level, (list, tuple)) and len(level) >= 2:
                    price, size = level[0], level[1]
                elif isinstance(level, dict):
                    price = level.get("p") or level.get("price")
                    size = level.get("s") or level.get("size") or level.get("volume")
                if price is None or size is None:
                    continue
                try:
                    parsed.append({
                        "price": float(price),
                        "size": abs(float(size))
                    })
                except (TypeError, ValueError):
                    continue
            return parsed

        bid_levels = _parse_levels(bids, limit=5)
        ask_levels = _parse_levels(asks, limit=5)
        bid_volume_top3 = sum(level["size"] for level in bid_levels) if bid_levels else None
        ask_volume_top3 = sum(level["size"] for level in ask_levels) if ask_levels else None

        timestamp_value = payload.get("t") or payload.get("time") or payload.get("timestamp")
        ts = datetime.utcnow()
        if timestamp_value is not None:
            try:
                timestamp_float = float(timestamp_value)
                if timestamp_float > 1e12:  # 毫秒时间戳
                    timestamp_float /= 1000.0
                ts = datetime.fromtimestamp(timestamp_float)
            except (ValueError, OSError):
                pass

        depth = payload.get("depth") or self.order_book_depth

        return {
            "timestamp": ts,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": (best_ask - best_bid) if all(v is not None for v in (best_bid, best_ask)) else None,
            "depth": depth,
            "bid_volume_top3": bid_volume_top3,
            "ask_volume_top3": ask_volume_top3,
            "top_bids": bid_levels,
            "top_asks": ask_levels
        }

    def _parse_trades(self, result):
        """解析成交记录"""
        trades = []
        items = []
        if isinstance(result, list):
            items = result
        elif isinstance(result, dict):
            payload = result.get('result') or result.get('data') or result
            if isinstance(payload, list):
                items = payload
            else:
                items = [payload]
        else:
            return trades

        for item in items:
            trade = self._normalize_trade_entry(item)
            if trade:
                trades.append(trade)
        return trades

    def _normalize_trade_entry(self, entry):
        """规范化成交结构"""
        if isinstance(entry, dict):
            price = entry.get("price") or entry.get("p")
            size = entry.get("size") or entry.get("s") or entry.get("amount")
            side = entry.get("side")
            trade_id = entry.get("id") or entry.get("trade_id")
            ts_value = entry.get("create_time") or entry.get("time") or entry.get("t")
        elif isinstance(entry, list) and len(entry) >= 5:
            # 部分API返回格式: [contract, size, price, time, side]
            price = entry[2]
            size = entry[1]
            ts_value = entry[3]
            side = entry[4] if len(entry) > 4 else None
            trade_id = None
        else:
            return None

        if price is None or size is None:
            return None

        try:
            price = float(price)
            size = abs(float(size))
        except (TypeError, ValueError):
            return None

        ts = datetime.utcnow()
        if ts_value is not None:
            try:
                ts_float = float(ts_value)
                if ts_float > 1e12:
                    ts_float /= 1000.0
                ts = datetime.fromtimestamp(ts_float)
            except (ValueError, OSError):
                pass

        return {
            "id": trade_id,
            "price": price,
            "size": size,
            "side": side,
            "timestamp": ts
        }

    def _parse_book_ticker(self, result):
        """解析最优买卖价推送"""
        data = self._extract_data(result)
        if not data:
            return None

        try:
            bid = data.get("b")
            ask = data.get("a")
            if isinstance(bid, str):
                bid = float(bid)
            elif bid is None:
                bid = float(data.get("bid", data.get("best_bid", 0)) or 0)
            else:
                bid = float(bid)

            if isinstance(ask, str):
                ask = float(ask)
            elif ask is None:
                ask = float(data.get("ask", data.get("best_ask", 0)) or 0)
            else:
                ask = float(ask)
        except (TypeError, ValueError):
            return None

        if bid <= 0 and ask <= 0:
            return None

        timestamp_value = data.get("t") or data.get("time") or data.get("timestamp")
        ts = datetime.utcnow()
        if timestamp_value is not None:
            try:
                ts = datetime.fromtimestamp(int(timestamp_value))
            except (ValueError, OSError):
                pass

        return {
            "timestamp": ts,
            "best_bid": bid if bid > 0 else None,
            "best_ask": ask if ask > 0 else None,
            "spread": (ask - bid) if bid and ask else None,
            "depth": data.get("depth") or 1,
            "bid_volume_top3": float(data.get("B") or 0) or None,
            "ask_volume_top3": float(data.get("A") or 0) or None
        }
