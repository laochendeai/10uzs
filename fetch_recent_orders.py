# -*- coding: utf-8 -*-
"""
拉取近 N 小时的合约委托，便于复盘网格下单逻辑。
"""

import argparse
import asyncio
from collections import Counter
from typing import List, Dict

from gateio_api import GateIOAPI
from config import HISTORY_FETCH_CONFIG


def summarize(orders: List[Dict]) -> Dict:
    stats = {
        "total": len(orders),
        "status": Counter(o.get("status") for o in orders),
        "finish_as": Counter(o.get("finish_as") for o in orders),
        "sides": Counter("buy" if o.get("size", 0) > 0 else "sell" if o.get("size", 0) < 0 else "unknown" for o in orders),
        "tif": Counter(o.get("tif") for o in orders),
        "price_zero": sum(1 for o in orders if float(o.get("price", 0) or 0) == 0.0),
    }
    return stats


async def main():
    parser = argparse.ArgumentParser(description="拉取近 N 小时的 Gate.io 合约委托")
    parser.add_argument("--hours", type=int, default=HISTORY_FETCH_CONFIG.get("recent_hours", 8), help="回看小时数")
    parser.add_argument("--limit", type=int, default=HISTORY_FETCH_CONFIG.get("max_orders", 200), help="最大返回条数")
    parser.add_argument("--contract", default=None, help="合约代码，默认读取配置")
    args = parser.parse_args()

    api = GateIOAPI(enable_market_data=False, enable_trading=True)
    orders = await api.get_recent_orders(hours=args.hours, limit=args.limit, contract=args.contract)

    stats = summarize(orders)
    print(f"=== 近 {args.hours} 小时委托（最多 {args.limit} 条） ===")
    print(f"总数: {stats['total']}")
    print(f"按状态: {dict(stats['status'])}")
    print(f"按成交类型(finish_as): {dict(stats['finish_as'])}")
    print(f"按方向: {dict(stats['sides'])}")
    print(f"TIF 分布: {dict(stats['tif'])}")
    print(f"价格=0(市价单)数量: {stats['price_zero']}")
    print("\n最近 10 条：")
    for row in orders[-10:]:
        side = "buy" if row.get("size", 0) > 0 else "sell"
        print(
            f"{row.get('create_time')} | {side} | price={row.get('price')} size={row.get('size')} "
            f"left={row.get('left')} status={row.get('status')} finish_as={row.get('finish_as')} tif={row.get('tif')} "
            f"reduce_only={row.get('reduce_only')}"
        )

    await api.close()


if __name__ == "__main__":
    asyncio.run(main())
