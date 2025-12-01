# -*- coding: utf-8 -*-
"""
基于 Gate.io 测试网的双模式自适应交易脚本（箱体网格 + 突破趋势）。
仅做演示，默认使用测试网，需提前在 .env 中配置 GATEIO_API_KEY/SECRET 且 GATEIO_TESTNET=true。
"""

import argparse
import asyncio
import logging
import os
import time
from datetime import datetime
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

from api_config import load_config
from dual_mode_strategy import DualModeStrategy
from gateio_api import GateIOAPI
from gateio_config import SYMBOL, FUTURES_SETTLE, CONTRACT_VALUE, PRICE_TICK_SIZE
from gateio_ws import GateIOWebsocket
from rest_data_feed import RESTDataFeed
from config import GRID_ORDER_COOLDOWN_SECONDS, GRID_GLOBAL_THROTTLE_SECONDS


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class DualModeRunner:
    def __init__(
        self,
        contract: str = SYMBOL,
        settle: str = FUTURES_SETTLE,
        interval: str = "10s",
        warmup_bars: int = 5,
        history_bars: int = 100,
        enable_tui: bool = False,
    ):
        # 强制使用测试网行情，避免主网 DNS 失败；需要已有测试网行情可用
        os.environ.setdefault("GATEIO_MARKET_DATA_USE_MAINNET", "false")
        os.environ.setdefault("GATEIO_TESTNET", "true")
        load_config()  # 确保 .env 已加载
        self.api = GateIOAPI(enable_market_data=True, enable_trading=True)
        # 主用 WS（10s K 线），REST 为兜底
        self.ws = GateIOWebsocket(contract=contract, settle=settle, candlestick_interval=interval)
        self.rest_feed = RESTDataFeed(self.api, interval=interval, poll_seconds=5.0, max_rows=500)
        self.strategy = DualModeStrategy()
        self.interval = interval
        self.warmup_bars = warmup_bars
        self.history_bars = max(
            history_bars,
            self.strategy.cfg.get("box_lookback", history_bars),
            self.strategy.cfg.get("box_min_bars", history_bars),
            self.strategy.cfg.get("atr_window", history_bars),
            self.strategy.cfg.get("box_dynamic_max_lookback", history_bars),
            *self.strategy.cfg.get("box_alt_lookbacks", []),
        )
        self.enable_tui = enable_tui
        self.stats = {
            "signals_generated": 0,
            "signals_executed": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0
        }
        self.market_data: Dict = {}
        self.recent_logs: Deque[str] = deque(maxlen=50)
        self.ui_box_info: Dict = {}
        self.ui_grid_levels: List[float] = []
        self._active_grid_levels: Dict[str, float] = {}
        self.ui_recent_orders: Deque[Dict] = deque(maxlen=20)
        self._order_seq: int = 0
        self.display = None
        self.highs: Deque[float] = deque(maxlen=500)
        self.lows: Deque[float] = deque(maxlen=500)
        self.closes: Deque[float] = deque(maxlen=500)
        self.volumes: Deque[float] = deque(maxlen=500)
        self.position: Dict[str, float] = {"long": 0.0, "short": 0.0}
        self.equity: float = 0.0
        self.initial_equity: float = 0.0
        self.last_total_equity: float = 0.0
        self.available: float = 0.0
        self.daily_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.last_balance_refresh: float = 0.0
        # 默认执行下单（测试网），如需仅打印可设环境变量
        self.execute_orders: bool = os.getenv("DUAL_MODE_EXECUTE", "true").lower() in ("1", "true", "yes")
        self.grid_cooldown = max(float(GRID_ORDER_COOLDOWN_SECONDS), 0.0)
        self.tick_size = float(PRICE_TICK_SIZE or 0.01)
        self._last_grid_order_ts = {"buy": 0.0, "sell": 0.0}

    async def start(self):
        self._log(
            f"WS started; REST fallback polling... interval={self.interval}, warmup={self.warmup_bars}, "
            f"history={self.history_bars}",
            level="info"
        )
        warmup_count = await self._warmup_history()
        bar_index = len(self.closes)
        if warmup_count == 0:
            self._log("[warmup-fallback] REST 历史拉取失败，改用 WS 累积至 120 根再判箱体", level="warning")

        # 启动即预演一次箱体/网格，仅用于展示，不下单
        self._refresh_box_preview(bar_index)

        if self.enable_tui:
            from tui_display import LiveTickerDisplay
            self.display = LiveTickerDisplay(engine=self, ws_client=self.ws, refresh_interval=0.5)
            await self.display.start()

        async def process_bar(candle: Dict):
            nonlocal bar_index
            self._append_candle(candle)
            if len(self.closes) < max(self.warmup_bars, self.strategy.cfg.get("box_dynamic_min_lookback", 120)):
                self._log(f"[warmup] bars={len(self.closes)} close={self.closes[-1]:.4f}", level="info")
                return
            if bar_index == len(self.closes) - 1:
                # 首次具备实时信号前刷新一次余额，避免 equity=0 导致暴露判定过严
                await self._refresh_balance()
            bar_index += 1
            self.market_data["current_price"] = self.closes[-1]
            result = self.strategy.update(
                highs=list(self.highs),
                lows=list(self.lows),
                closes=list(self.closes),
                volumes=list(self.volumes),
                bar_index=bar_index
            )
            self.stats["signals_generated"] += 1
            if result.get("box"):
                self.ui_box_info = result["box"]
            else:
                # 无箱体时清空网格并停止网格下单
                self.ui_box_info = {}
                self.ui_grid_levels = []
                self._active_grid_levels.clear()
                result["actions"] = [a for a in result.get("actions", []) if a.get("type") != "grid_entry"]
            grid_actions = [a for a in result.get("actions", []) if a.get("type") == "grid_entry"]
            if grid_actions:
                prices = sorted(set(a.get("price") for a in grid_actions if a.get("price") is not None))
                self.ui_grid_levels = prices
            else:
                self.ui_grid_levels = []
            self._handle_actions(result)
            if not result.get("box"):
                self._log("箱体判定失败：数据不足或波动/坡度超阈值")

        # 启动 WS
        await self.ws.start()

        # REST fallback task
        async def rest_loop():
            while True:
                bar = await self.rest_feed.poll_once()
                if bar:
                    await process_bar(bar)
                await asyncio.sleep(self.rest_feed.poll_seconds)

        asyncio.create_task(rest_loop())

        # 主循环：等待 WS 推送
        while True:
            ok = await self.ws.wait_for_update(timeout=5.0)
            if ok:
                candle = self.ws.get_latest_candle()
                if candle:
                    await process_bar({
                        "timestamp": candle["timestamp"],
                        "open": candle["open"],
                        "high": candle["high"],
                        "low": candle["low"],
                        "close": candle["close"],
                        "volume": candle["volume"],
                    })
            else:
                self._log("等待 WS 行情更新，REST 兜底运行中...", level="info")

    def _append_candle(self, candle: Dict):
        # 兼容 REST/WS 字段：{'timestamp': ts, 'open':..., 'high':..., 'low':..., 'close':..., 'volume':...}
        self.highs.append(float(candle.get("high") or candle.get("h") or 0.0))
        self.lows.append(float(candle.get("low") or candle.get("l") or 0.0))
        self.closes.append(float(candle.get("close") or candle.get("c") or 0.0))
        self.volumes.append(float(candle.get("volume") or candle.get("v") or 0.0))

    def _quantize_price(self, price: Optional[float]) -> Optional[float]:
        if price is None:
            return None
        try:
            step = self.tick_size if self.tick_size > 0 else 0.01
            return round(float(price) / step) * step
        except Exception:
            return price

    def _prune_grid_cooldowns(self):
        """清理过期的网格冷却记录，防止字典膨胀。"""
        if not self._active_grid_levels or self.grid_cooldown <= 0:
            return
        now = time.time()
        expire_after = self.grid_cooldown * 2
        stale_keys = [k for k, ts in self._active_grid_levels.items() if now - ts > expire_after]
        for k in stale_keys:
            self._active_grid_levels.pop(k, None)

    def _pick_nearest_grid_per_side(self, actions: List[Dict], price: float) -> List[Dict]:
        """同侧多条网格时，仅保留最贴近当前价的一条，避免同刻批量挂单。"""
        buys = [a for a in actions if a.get("type") == "grid_entry" and a.get("side") == "buy"]
        sells = [a for a in actions if a.get("type") == "grid_entry" and a.get("side") == "sell"]
        picked: List[Dict] = []

        if buys:
            # 选离当前价最近的买单（价位 >= 现价）
            buys_sorted = sorted(buys, key=lambda a: abs((a.get("price") or 0) - price))
            picked.append(buys_sorted[0])
        if sells:
            # 选离当前价最近的卖单（价位 <= 现价）
            sells_sorted = sorted(sells, key=lambda a: abs((a.get("price") or 0) - price))
            picked.append(sells_sorted[0])

        # 追加非网格类动作
        picked.extend([a for a in actions if a.get("type") != "grid_entry"])
        return picked

    def _handle_actions(self, result: Dict):
        mode = result["mode"]
        actions = result["actions"]
        box = result.get("box")
        box_brief = None
        if box:
            box_brief = f"[{box.get('support', 0):.2f} , {box.get('resistance', 0):.2f}] w={box.get('height_pct', 0)*100:.3f}%"
        action_brief = f"{len(actions)} actions" if actions else "0 actions"
        self._log(f"[mode={mode}] price={self.closes[-1]:.4f} box={box_brief or 'N/A'} {action_brief}", level="info")
        if not self.execute_orders or not self.api.can_trade:
            return
        # 若有多个网格同侧信号，仅保留离当前价最近的买/卖各一条，减少同刻多单
        actions = self._pick_nearest_grid_per_side(actions, self.closes[-1])
        # 刷新一次余额（节流）
        now = time.time()
        if now - self.last_balance_refresh > 30:
            asyncio.create_task(self._refresh_balance())
            self.last_balance_refresh = now

        # 风控：日内亏损/连续亏损熔断
        cfg = self.strategy.cfg
        if self.daily_pnl <= -cfg["daily_loss_limit_pct"] * max(self.equity, 1e-9):
            self._log("日内亏损达到上限，暂停下单", level="warning")
            return
        if self.consecutive_losses >= cfg["consecutive_loss_limit"]:
            self._log("连续亏损达到上限，暂停下单", level="warning")
            return

        self._prune_grid_cooldowns()

        for act in actions:
            if act["type"] == "grid_entry":
                side = act["side"]
                now_ts = time.time()
                # 全局节流：同侧在窗口内只下1单
                if now_ts - self._last_grid_order_ts.get(side, 0) < GRID_GLOBAL_THROTTLE_SECONDS:
                    continue
                price_level = self._quantize_price(act["price"])
                weight = act["weight"]
                # 避免同一价位重复提交网格单
                if price_level is not None:
                    key = f"{side}:{price_level}"
                    last_ts = self._active_grid_levels.get(key)
                    if last_ts and time.time() - last_ts < self.grid_cooldown:
                        continue
                size = self._calc_size(weight=weight, mode="grid", stop_distance=self._grid_stop_distance(box))
                if self._exposure_ok(side, size):
                    order_seq = self._next_order_seq()
                    self._record_order(seq=order_seq, side=side, size=size, price=price_level, order_type="grid", status="提交中")
                    if price_level is not None:
                        self._active_grid_levels[f"{side}:{price_level}"] = time.time()
                    # 网格统一使用限价单，禁止市价，减缓手续费
                    asyncio.create_task(self._place_limit(side, size, price_level, order_seq))
                    self._last_grid_order_ts[side] = now_ts
            elif act["type"] == "trend_entry":
                side = act["side"]
                stop_distance = abs(act["stop"] - self.closes[-1])
                size = self._calc_size(weight=1.0, mode="trend", stop_distance=stop_distance)
                if self._exposure_ok(side, size):
                    order_seq = self._next_order_seq()
                    self._record_order(seq=order_seq, side=side, size=size, price=self.closes[-1], order_type="trend", status="提交中")
                    asyncio.create_task(self._handle_trend_entry(side, size, act, order_seq))

    async def _refresh_balance(self):
        try:
            info = await self.api.get_account_balance(fallback=self.equity)
            self.available = info.get("available", self.available)
            total = info.get("total", self.available)
            if self.initial_equity == 0.0:
                self.initial_equity = total
            if self.last_total_equity > 0 and total < self.last_total_equity:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            self.daily_pnl = total - self.initial_equity
            self.equity = total
            self.last_total_equity = total
            # 同步持仓（仅用于暴露统计）
            positions = await self.api.get_positions(contract=None)
            long_size = 0.0
            short_size = 0.0
            for p in positions:
                size = p.get("size", 0.0)
                if size > 0:
                    long_size += size
                elif size < 0:
                    short_size += abs(size)
            self.position["long"] = long_size
            self.position["short"] = short_size
        except Exception as exc:
            self._log(f"获取余额失败: {exc}", level="warning")

    def _calc_size(self, weight: float, mode: str, stop_distance: float = None) -> float:
        equity = self.equity or 10.0  # fallback
        cfg = self.strategy.cfg
        price = self.closes[-1] if self.closes else 0.0
        price_per_contract = max(price * CONTRACT_VALUE, 1e-9)
        if mode == "grid":
            exposure_cap_value = equity * cfg["fund_allocation"]["grid"] * cfg["grid_max_exposure_pct"]
            risk_budget = equity * cfg["risk_per_trade"]
            dist = stop_distance or self._real_atr() or 1.0
            contracts_risk = risk_budget / max(dist * price_per_contract, 1e-9)
            contracts_cap = exposure_cap_value / price_per_contract
            contracts = max(1, int(min(contracts_risk, contracts_cap) * weight))
            return float(contracts)
        else:
            exposure_cap = equity * cfg["fund_allocation"]["trend"]
            atr = self._real_atr()
            risk_budget = equity * cfg["risk_per_trade"]
            dist = stop_distance or atr or 1.0
            contracts_risk = risk_budget / max(dist * price_per_contract, 1e-9)
            contracts_cap = (equity * cfg["fund_allocation"]["trend"]) / price_per_contract
            contracts = max(1, int(min(contracts_risk, contracts_cap) * weight))
            return float(contracts)

        # 网格尺寸下限/上限
        if mode == "grid":
            min_size = float(cfg.get("grid_min_size", 1.0))
            cap_size = float(cfg.get("grid_size_cap", 0.0))
            contracts = max(min_size, float(contracts))
            if cap_size > 0:
                contracts = min(contracts, cap_size)
        return float(contracts)

    def _real_atr(self) -> float:
        n = len(self.closes)
        if n < 2:
            return 0.0
        window = self.strategy.cfg.get("atr_window", 14)
        if n < window + 1:
            return 0.0
        trs = []
        for i in range(n - window, n):
            tr = max(
                self.highs[i] - self.lows[i],
                abs(self.highs[i] - self.closes[i - 1]),
                abs(self.lows[i] - self.closes[i - 1]),
            )
            trs.append(tr)
        return sum(trs) / window if trs else 0.0

    def _grid_stop_distance(self, box: Dict) -> float:
        return (box["resistance"] - box["support"]) if box else None

    def _exposure_ok(self, side: str, size: float) -> bool:
        cfg = self.strategy.cfg
        equity_base = max(self.equity, self.initial_equity, 10.0)  # fallback 避免0导致过度阻挡
        price = self.closes[-1] if self.closes else 1
        total_exposure_value = (abs(self.position.get("long", 0)) + abs(self.position.get("short", 0)) + size) * CONTRACT_VALUE * price
        if total_exposure_value > cfg["max_gross_exposure_pct"] * equity_base:
            self._log("总暴露超限，拒绝下单", level="warning")
            return False
        direction = "long" if side == "buy" else "short"
        dir_size = self.position.get(direction, 0)
        if (dir_size + size) * CONTRACT_VALUE * price > cfg["direction_exposure_pct"] * equity_base:
            self._log("方向暴露超限，拒绝下单", level="warning")
            return False
        # 锁仓防护：禁止与当前净仓位相反的网格单
        net_pos = self.position.get("long", 0) - self.position.get("short", 0)
        side_sign = 1 if side == "buy" else -1
        min_lock = float(cfg.get("grid_min_size", 1.0))
        if net_pos * side_sign < 0 and abs(net_pos) >= min_lock:
            self._log("净仓位相反，拒绝锁仓网格单", level="warning")
            return False
        return True

    def _refresh_box_preview(self, bar_index: int):
        min_bars = max(self.warmup_bars, self.strategy.cfg.get("box_dynamic_min_lookback", 120))
        if len(self.closes) < min_bars:
            return
        self.market_data["current_price"] = self.closes[-1] if self.closes else 0.0
        result = self.strategy.update(
            highs=list(self.highs),
            lows=list(self.lows),
            closes=list(self.closes),
            volumes=list(self.volumes),
            bar_index=bar_index
        )
        if result.get("box"):
            self.ui_box_info = result["box"]
        grid_actions = [a for a in result.get("actions", []) if a.get("type") == "grid_entry"]
        if grid_actions:
            prices = sorted(set(a.get("price") for a in grid_actions if a.get("price") is not None))
            self.ui_grid_levels = prices
        else:
            self.ui_grid_levels = []

    def _log(self, message: str, level: str = "info"):
        level = (level or "info").lower()
        if level in ("warning", "error", "critical"):
            getattr(logging, level, logging.warning)(message)
        timestamp = datetime.utcnow().strftime("%H:%M:%S")
        short = message if len(message) <= 140 else message[:137] + "..."
        self.recent_logs.append(f"[{timestamp}] {short}")

    async def _warmup_history(self) -> int:
        """按启动周期预拉历史K线，填充暖机数据"""
        limit = self.highs.maxlen or self.history_bars
        limit = min(limit, self.history_bars)
        try:
            candles = await self.api.get_klines(interval=self.interval, limit=limit)
        except Exception as exc:
            self._log(f"预拉历史K线失败: {exc}", level="warning")
            return 0

        if not candles:
            self._log("预拉历史K线为空，继续实时暖机", level="warning")
            return 0

        candles_sorted = sorted(
            candles,
            key=lambda c: c.get("timestamp")
        )
        for candle in candles_sorted:
            self._append_candle(candle)

        last_ts = candles_sorted[-1].get("timestamp")
        if last_ts:
            try:
                self.rest_feed._last_ts = int(time.mktime(last_ts.timetuple()))
            except Exception:
                pass

        logging.info(
            f"[warmup] 已加载历史K线 {len(candles_sorted)} 条 interval={self.interval} limit={limit}"
        )
        return len(candles_sorted)

    async def _place_limit(self, side: str, size: float, price: float, seq: Optional[int] = None):
        try:
            await self.api.place_order(contract=SYMBOL, size=size, side=side, order_type="limit", price=price, time_in_force="gtc")
            self._record_order(seq=seq, side=side, size=size, price=price, order_type="grid", status="已下单")
            self.stats["signals_executed"] += 1
        except Exception as exc:
            logging.error(f"limit order failed: {exc}")
            self._record_order(seq=seq, side=side, size=size, price=price, order_type="grid", status=f"失败: {exc}")
        finally:
            # 记录时间戳，冷却期后再允许同价位重复挂单
            if price is not None:
                self._active_grid_levels[f"{side}:{price}"] = time.time()

    async def _place_market(self, side: str, size: float, seq: Optional[int] = None):
        try:
            await self.api.place_order(contract=SYMBOL, size=size, side=side, order_type="market")
            self._record_order(seq=seq, side=side, size=size, price=self.closes[-1] if self.closes else 0.0, order_type="trend", status="已下单")
            self.stats["signals_executed"] += 1
        except Exception as exc:
            logging.error(f"market order failed: {exc}")
            self._record_order(seq=seq, side=side, size=size, price=self.closes[-1] if self.closes else 0.0, order_type="trend", status=f"失败: {exc}")

    async def _handle_trend_entry(self, side: str, size: float, act: Dict, seq: Optional[int] = None):
        try:
            await self._place_market(side, size, seq=seq)
            # 清理旧触发单（按方向）
            await self.api.cancel_all_reducing_triggers(side="long" if side == "buy" else "short")
            if side == "buy":
                await self.api.place_stop_order(SYMBOL, "long", act["stop"])
                await self.api.place_take_profit_order(SYMBOL, "long", act["take_profit"])
            else:
                await self.api.place_stop_order(SYMBOL, "short", act["stop"])
                await self.api.place_take_profit_order(SYMBOL, "short", act["take_profit"])
            self._record_order(seq=seq, side=side, size=size, price=self.closes[-1] if self.closes else 0.0, order_type="trend", status="止盈止损已挂")
        except Exception as exc:
            logging.error(f"trend entry failed: {exc}")
            self._record_order(seq=seq, side=side, size=size, price=self.closes[-1] if self.closes else 0.0, order_type="trend", status=f"失败: {exc}")

    def _next_order_seq(self) -> int:
        self._order_seq += 1
        return self._order_seq

    def _record_order(self, seq: Optional[int], side: str, size: float, price: float, order_type: str, status: str):
        entry = {
            "seq": seq or self._next_order_seq(),
            "time": datetime.utcnow().strftime("%H:%M:%S"),
            "type": order_type,
            "side": side,
            "size": size,
            "price": price,
            "status": status
        }
        self.ui_recent_orders.append(entry)
        self.market_data["last_order"] = entry
        if order_type == "grid" and status.startswith(("提交", "已下单")) and price is not None:
            self._active_grid_levels[f"{side}:{price}"] = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate.io 双模式自适应交易（WS+REST兜底）")
    parser.add_argument("--interval", default="10s", help="K线周期，例如 10s/1m/5m/15m/1h")
    parser.add_argument("--warmup", type=int, default=5, help="暖机所需K线数量")
    parser.add_argument("--no-exec", action="store_true", help="仅打印信号，不下单")
    parser.add_argument("--tui", action="store_true", help="启用终端TUI界面")
    args = parser.parse_args()

    if args.no_exec:
        os.environ["DUAL_MODE_EXECUTE"] = "false"

    runner = DualModeRunner(interval=args.interval, warmup_bars=args.warmup, enable_tui=args.tui)
    try:
        asyncio.run(runner.start())
    except KeyboardInterrupt:
        pass
