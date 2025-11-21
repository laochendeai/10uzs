# -*- coding: utf-8 -*-
"""
运行真正的高频交易引擎
"""

import argparse
import asyncio

from true_hft_engine import TrueHFTEngine


def parse_args():
    parser = argparse.ArgumentParser(description="运行真正的高频Tick级交易引擎")
    parser.add_argument("--initial", type=float, default=10.0, help="初始资金")
    parser.add_argument("--live", action="store_true", help="是否使用真实Gate.io API")
    return parser.parse_args()


async def main():
    args = parse_args()
    engine = TrueHFTEngine(initial_capital=args.initial, use_real_api=args.live)
    await engine.start()


if __name__ == "__main__":
    asyncio.run(main())
