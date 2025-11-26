# -*- coding: utf-8 -*-
"""
快速验证 Gate.io 模拟合约账户余额。
确保已在当前终端 export GATEIO_API_KEY/SECRET 以及
GATEIO_API_BASE_URL, GATEIO_TESTNET 等变量。
"""

import asyncio
from gateio_api import GateIOAPI
from api_config import load_config, get_config


async def main():
    if not load_config():
        print("❌ 未能加载 API 密钥，请确认已 export GATEIO_API_KEY/SECRET")
        return
    cfg = get_config()
    print(f"连接主机: {cfg.api_host} | 合约: {cfg.contract} | settle: {cfg.settle} | testnet={cfg.testnet}")
    api = GateIOAPI(enable_market_data=False, enable_trading=True)
    try:
        balance = await api.get_account_balance()
        print("✅ 合约账户余额:")
        for key, val in balance.items():
            print(f"  {key}: {val}")
    except Exception as exc:
        print(f"⚠️ 查询失败: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
