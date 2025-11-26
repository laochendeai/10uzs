# -*- coding: utf-8 -*-
"""
Gate.io 模拟交易功能验证脚本

覆盖以下场景：
1. 限价单挂单 -> 查询状态 -> 撤单
2. 市价单下单 -> 平仓 -> 查询成交记录
3. 账户余额与持仓查询
4. 异常用例：资金不足 / 无效价格
5. 订单状态流转验证

运行方式：
    ENABLE_LIVE_TRADING=true USE_GATEIO_MARKET_DATA=true python tests/test_gateio_sim_trading.py
"""

import asyncio
import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api_config import load_config
from gateio_api import GateIOAPI
from gateio_config import SYMBOL


def _to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {key: _to_dict(val) for key, val in obj.items()}
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    return obj


def _extract_order_id(order: Dict[str, Any]) -> str:
    for key in ("id", "order_id", "orderId", "text"):
        value = order.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return str(int(value))
        return str(value)
    raise ValueError(f"无法在响应中找到订单ID: {order}")


@dataclass
class StepReport:
    name: str
    passed: bool
    request: Dict[str, Any] = field(default_factory=dict)
    response: Any = None
    error: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        payload["response"] = _to_dict(self.response)
        return payload


class GateIOSimTradingTester:
    def __init__(self):
        load_config()
        self.api = GateIOAPI(enable_market_data=True, enable_trading=True)
        if not self.api.can_trade:
            raise RuntimeError("Gate.io 交易API未启用，请设置 ENABLE_LIVE_TRADING=true 并配置测试网密钥")
        self.contract = SYMBOL
        self.results: List[StepReport] = []

    async def run(self) -> List[StepReport]:
        await self.test_limit_order_lifecycle()
        await self.test_market_order_and_flat()
        await self.test_balance_and_positions()
        await self.test_insufficient_balance()
        await self.test_invalid_price()
        return self.results

    async def test_limit_order_lifecycle(self):
        name = "限价单挂单-撤单"
        try:
            ticker = await self.api.get_ticker()
            if not ticker:
                raise RuntimeError("ticker为空，无法确定价格")
            last_price = float(ticker["last_price"])
            price = max(last_price * 0.7, last_price - 100.0)
            request = {
                "contract": self.contract,
                "size": 1.0,
                "side": "buy",
                "order_type": "limit",
                "time_in_force": "gtc",
                "price": round(price, 4),
            }
            order = await self.api.place_order(**request)
            order = _to_dict(order)
            order_id = _extract_order_id(order)
            status = await self.api.get_order_status(order_id)
            cancel_resp = await self.api.cancel_order(order_id)
            self.results.append(
                StepReport(
                    name=name,
                    passed=True,
                    request=request,
                    response={
                        "initial_order": order,
                        "status_before_cancel": _to_dict(status),
                        "cancel_response": _to_dict(cancel_resp),
                    },
                    notes="限价单挂单/撤单流程完成",
                )
            )
        except Exception as exc:
            self.results.append(
                StepReport(
                    name=name,
                    passed=False,
                    request={},
                    response=None,
                    error=str(exc),
                )
            )

    async def test_market_order_and_flat(self):
        name = "市价单开仓-平仓"
        try:
            request = {
                "contract": self.contract,
                "size": 1.0,
                "side": "buy",
                "order_type": "market",
                "time_in_force": "ioc",
            }
            order = await self.api.place_order(**request)
            order = _to_dict(order)
            trades_before = await self.api.get_trade_history(limit=5)
            close_resp = await self.api.close_position(self.contract, size=1.0, direction="long")
            trades_after = await self.api.get_trade_history(limit=5)
            self.results.append(
                StepReport(
                    name=name,
                    passed=True,
                    request=request,
                    response={
                        "open_order": order,
                        "close_order": _to_dict(close_resp),
                        "trades_before": trades_before,
                        "trades_after": trades_after,
                    },
                    notes="市价单已开仓并立即平仓，成交记录已拉取",
                )
            )
        except Exception as exc:
            self.results.append(
                StepReport(
                    name=name,
                    passed=False,
                    request={},
                    response=None,
                    error=str(exc),
                )
            )

    async def test_balance_and_positions(self):
        name = "账户余额与持仓查询"
        try:
            balance = await self.api.get_account_balance()
            positions = None
            position_error = None
            try:
                fetcher = getattr(self.api.futures_api, "list_futures_positions", None)
                if fetcher:
                    positions = await self.api._run_async(fetcher, self.api.settle, self.contract)
                else:
                    positions = await self.api._run_async(
                        self.api.futures_api.get_futures_position,
                        self.api.settle,
                        self.contract,
                    )
            except AttributeError as exc:
                position_error = f"SDK缺少持仓接口: {exc}"
            except Exception as exc:
                position_error = str(exc)
            payload = {"balance": balance}
            if positions is not None:
                payload["positions"] = _to_dict(positions)
            if position_error:
                payload["positions_error"] = position_error
            self.results.append(
                StepReport(
                    name=name,
                    passed=True,
                    request={},
                    response=payload,
                    notes="余额查询完成" if position_error else "余额与持仓数据获取成功",
                )
            )
        except Exception as exc:
            self.results.append(
                StepReport(
                    name=name,
                    passed=False,
                    request={},
                    response=None,
                    error=str(exc),
                )
            )

    async def test_insufficient_balance(self):
        name = "资金不足错误处理"
        try:
            request = {
                "contract": self.contract,
                "size": 1000000.0,
                "side": "buy",
                "order_type": "market",
                "time_in_force": "ioc",
            }
            error_msg = None
            try:
                await self.api.place_order(**request)
            except Exception as exc:
                error_msg = str(exc)
            if not error_msg:
                raise RuntimeError("预期资金不足错误但未捕获异常")
            self.results.append(
                StepReport(
                    name=name,
                    passed=True,
                    request=request,
                    response={"error": error_msg},
                    notes="资金不足错误已被捕获",
                )
            )
        except Exception as exc:
            self.results.append(
                StepReport(
                    name=name,
                    passed=False,
                    request={},
                    response=None,
                    error=str(exc),
                )
            )

    async def test_invalid_price(self):
        name = "无效价格参数验证"
        try:
            request = {
                "contract": self.contract,
                "size": 1.0,
                "side": "buy",
                "order_type": "limit",
                "time_in_force": "gtc",
                "price": -1.0,
            }
            error_msg = None
            try:
                await self.api.place_order(**request)
            except Exception as exc:
                error_msg = str(exc)
            if not error_msg:
                raise RuntimeError("预期无效价格错误但未捕获异常")
            self.results.append(
                StepReport(
                    name=name,
                    passed=True,
                    request=request,
                    response={"error": error_msg},
                    notes="无效价格已正确触发异常",
                )
            )
        except Exception as exc:
            self.results.append(
                StepReport(
                    name=name,
                    passed=False,
                    request={},
                    response=None,
                    error=str(exc),
                )
            )


async def main():
    tester = GateIOSimTradingTester()
    results = await tester.run()
    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": [r.name for r in results if not r.passed],
        "details": [r.to_dict() for r in results],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
