# -*- coding: utf-8 -*-
"""
终端行情展示模块
"""

import asyncio
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


class LiveTickerDisplay:
    """实时行情终端显示"""

    def __init__(self, engine, ws_client=None, refresh_interval: float = 0.5):
        self.engine = engine
        self.ws_client = ws_client
        self.refresh_interval = refresh_interval
        self._task: Optional[asyncio.Task] = None
        self._stop = False
        self.console = Console()

    async def start(self):
        if self._task and not self._task.done():
            return
        self._stop = False
        self._task = asyncio.create_task(self._run(), name="live-ticker-display")

    async def stop(self):
        self._stop = True
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self):
        with Live(self._render_layout(), console=self.console, refresh_per_second=int(1 / self.refresh_interval) if self.refresh_interval > 0 else 2, screen=False) as live:
            while not self._stop:
                live.update(self._render_layout())
                await asyncio.sleep(self.refresh_interval)

    def _render_layout(self) -> Layout:
        layout = Layout()
        layout.split_row(Layout(name="ticker"), Layout(name="right"))
        layout["ticker"].update(self._render_ticker_panel())
        layout["right"].split_column(
            Layout(name="stats", ratio=2),
            Layout(name="logs", ratio=1)
        )
        layout["right"]["stats"].update(self._render_stats_panel())
        layout["right"]["logs"].update(self._render_logs_panel())
        return layout

    def _render_ticker_panel(self) -> Panel:
        if not self.ws_client:
            placeholder = Table.grid(expand=True)
            placeholder.add_row("状态", "未启用WebSocket")
            return Panel(placeholder, title="实时行情 (WebSocket)", border_style="cyan")

        updates = self.ws_client.get_recent_ticker_updates(limit=12)
        if not updates:
            waiting_table = Table.grid(expand=True)
            waiting_table.add_row("状态", "等待WebSocket数据...")
            return Panel(waiting_table, title="实时行情 (WebSocket)", border_style="cyan")

        latest = updates[0]
        summary = Table.grid(expand=True)
        summary.add_column(justify="left", ratio=1)
        summary.add_column(justify="right", ratio=1)
        summary.add_row("最后成交价", f"{latest.get('last_price', 0):,.2f}")
        summary.add_row("标记价格", f"{latest.get('mark_price', 0):,.2f}")
        ts = latest.get("timestamp")
        ts_str = ts.strftime("%H:%M:%S.%f")[:-3] if isinstance(ts, datetime) else "-"
        summary.add_row("最新更新时间", ts_str)

        history = Table(show_header=True, header_style="bold cyan")
        history.add_column("时间", justify="left", style="cyan")
        history.add_column("成交价", justify="right")
        history.add_column("标记价", justify="right")
        history.add_column("间隔", justify="right")

        prev_time = None
        for update in updates:
            timestamp = update.get("timestamp")
            time_str = timestamp.strftime("%H:%M:%S.%f")[:-3] if isinstance(timestamp, datetime) else "-"
            interval = "-"
            if prev_time and timestamp:
                delta_ms = (prev_time - timestamp).total_seconds() * 1000
                interval = f"{delta_ms:.0f} ms"
            last_price = update.get("last_price", 0)
            mark_price = update.get("mark_price", 0)
            history.add_row(
                time_str,
                f"{last_price:,.2f}",
                f"{mark_price:,.2f}",
                interval,
            )
            prev_time = timestamp

        order_updates = self.ws_client.get_recent_order_book_updates(limit=12)
        order_table = Table(show_header=True, header_style="bold green")
        order_table.add_column("OB时间", justify="left", style="green")
        order_table.add_column("买一", justify="right")
        order_table.add_column("卖一", justify="right")
        order_table.add_column("价差", justify="right")

        if order_updates:
            for ob in order_updates:
                ts_ob = ob.get("timestamp")
                ob_time = ts_ob.strftime("%H:%M:%S.%f")[:-3] if isinstance(ts_ob, datetime) else "-"
                bid = ob.get("best_bid")
                ask = ob.get("best_ask")
                spread = ob.get("spread")
                order_table.add_row(
                    ob_time,
                    f"{bid:,.2f}" if bid else "-",
                    f"{ask:,.2f}" if ask else "-",
                    f"{spread:,.2f}" if spread else "-",
                )
        else:
            order_table.add_row("-", "-", "-", "等待订单簿推送...")

        container = Table.grid(expand=True)
        container.add_row(summary)
        container.add_row(history)
        container.add_row(order_table)

        return Panel(container, title="实时行情 (WebSocket)", border_style="cyan")

    def _render_stats_panel(self) -> Panel:
        stats_table = Table.grid(expand=True)
        stats_table.add_column(justify="left")
        stats_table.add_column(justify="right")

        stats = self.engine.stats
        market_data = self.engine.market_data or {}

        stats_table.add_row("当前价格", f"{market_data.get('current_price', 0):,.2f}")
        stats_table.add_row("信号生成", str(stats.get('signals_generated', 0)))
        stats_table.add_row("执行信号", str(stats.get('signals_executed', 0)))
        stats_table.add_row("胜率", f"{(stats.get('winning_trades', 0) / stats.get('total_trades', 1) * 100):.1f}%"
                            if stats.get('total_trades') else "0%")
        stats_table.add_row("总盈亏", f"{stats.get('total_pnl', 0):,.2f} USDT")

        return Panel(stats_table, title="策略状态", border_style="magenta")

    def _render_logs_panel(self) -> Panel:
        log_table = Table.grid(expand=True)
        log_table.add_column(justify="left")
        logs = getattr(self.engine, 'recent_logs', [])
        if logs:
            for entry in reversed(logs):
                log_table.add_row(entry)
        else:
            log_table.add_row("暂无日志")
        return Panel(log_table, title="最新日志", border_style="yellow")
