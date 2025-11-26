# -*- coding: utf-8 -*-
"""
Gate.io 模拟账户体检脚本。

输出内容：
1. 账户余额
2. 持仓信息
3. 当前委托 & 历史委托
4. 最近成交
5. 资金流水（ledgers）

运行前请确保已 export GATEIO_API_KEY/GATEIO_API_SECRET 以及测试网域名。
"""

import asyncio
from typing import Iterable, Any

from api_config import load_config, get_config
from gateio_api import GateIOAPI


def _to_dict(item: Any) -> dict:
    if item is None:
        return {}
    if hasattr(item, "to_dict"):
        return item.to_dict()
    if isinstance(item, dict):
        return item
    return {}


def _print_section(title: str):
    print(f"\n=== {title} ===")


def _print_table(rows: Iterable[dict], keys: Iterable[str]):
    for row in rows:
        line = ", ".join(f"{key}={row.get(key)}" for key in keys)
        print(f"- {line}")


async def fetch_positions(api: GateIOAPI, limit: int = 10):
    method = getattr(api.futures_api, "list_positions", None)
    if not method:
        print("⚠️ SDK 不支持 list_positions")
        return
    try:
        positions = await api._run_async(method, api.settle)
    except Exception as exc:
        print(f"⚠️ 获取持仓失败: {exc}")
        return
    rows = []
    for pos in positions or []:
        data = _to_dict(pos)
        rows.append({
            "contract": data.get("contract"),
            "size": data.get("size"),
            "entry_price": data.get("entry_price"),
            "mark_price": data.get("mark_price"),
            "leverage": data.get("leverage"),
            "unrealised_pnl": data.get("unrealised_pnl"),
        })
    if not rows:
        print("无持仓")
        return
    _print_table(rows[:limit], ["contract", "size", "entry_price", "mark_price", "leverage", "unrealised_pnl"])


async def fetch_open_orders(api: GateIOAPI, limit: int = 20):
    try:
        orders = await api._run_async(
            api.futures_api.list_futures_orders,
            api.settle,
            contract=api.contract,
            status="open"
        )
    except Exception as exc:
        print(f"⚠️ 获取当前委托失败: {exc}")
        return
    rows = []
    for order in orders or []:
        data = _to_dict(order)
        rows.append({
            "id": data.get("id"),
            "side": data.get("side"),
            "size": data.get("size"),
            "price": data.get("price"),
            "tif": data.get("tif"),
            "create_time": data.get("create_time")
        })
    if not rows:
        print("无当前委托")
        return
    _print_table(rows[:limit], ["id", "side", "size", "price", "tif", "create_time"])


async def fetch_order_history(api: GateIOAPI, limit: int = 20):
    method = getattr(api.futures_api, "get_orders_with_time_range", None)
    if not method:
        print("⚠️ SDK 不支持 get_orders_with_time_range")
        return
    try:
        orders = await api._run_async(
            method,
            api.settle,
            contract=api.contract,
            limit=limit
        )
    except Exception as exc:
        print(f"⚠️ 获取历史委托失败: {exc}")
        return
    rows = []
    for order in orders or []:
        data = _to_dict(order)
        rows.append({
            "id": data.get("id"),
            "side": data.get("side"),
            "size": data.get("size"),
            "price": data.get("price"),
            "status": data.get("status"),
            "finish_as": data.get("finish_as"),
            "update_time": data.get("update_time")
        })
    if not rows:
        print("历史委托为空")
        return
    _print_table(rows[:limit], ["id", "side", "size", "price", "status", "finish_as", "update_time"])


async def fetch_trades(api: GateIOAPI, limit: int = 20):
    try:
        trades = await api.get_trade_history(limit=limit)
    except Exception as exc:
        print(f"⚠️ 获取成交记录失败: {exc}")
        return
    if not trades:
        print("暂无成交记录")
        return
    _print_table(trades[:limit], ["id", "direction", "size", "price", "fee", "timestamp"])


async def fetch_ledgers(api: GateIOAPI, limit: int = 20):
    method = getattr(api.futures_api, "list_futures_insurance_ledger", None)
    if not method:
        print("⚠️ SDK 不提供账户资金流水接口 (仅保险基金 ledger)")
        return
    print("⚠️ 资金流水接口仅曝光保险基金账本，Gate 当前未提供账户层级 ledger，无法实际查询。")


async def main():
    if not load_config():
        print("❌ 未能加载 Gate API 配置，请先 export GATEIO_API_KEY/SECRET")
        return
    cfg = get_config()
    print(f"连接主机: {cfg.api_host} | 合约: {cfg.contract} | settle: {cfg.settle} | testnet={cfg.testnet}")
    api = GateIOAPI(enable_market_data=False, enable_trading=True)

    _print_section("账户余额")
    balance = await api.get_account_balance()
    for key, val in balance.items():
        print(f"- {key}: {val}")

    _print_section("持仓")
    await fetch_positions(api)

    _print_section("当前委托")
    await fetch_open_orders(api)

    _print_section("历史委托 (最近20笔)")
    await fetch_order_history(api)

    _print_section("成交记录 (最近20笔)")
    await fetch_trades(api)

    _print_section("资金流水 (最近20条)")
    await fetch_ledgers(api)


if __name__ == "__main__":
    asyncio.run(main())
