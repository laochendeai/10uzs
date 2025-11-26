# -*- coding: utf-8 -*-
"""
运行真正的高频交易引擎
"""

import argparse
import asyncio
import logging
import sys
from typing import Any, Optional

from config import TRADE_GUARD_CONFIG, get_parameter_preset
from hft_config import HFT_CONFIG
from true_hft_engine import TrueHFTEngine


def parse_args():
    parser = argparse.ArgumentParser(description="运行真正的高频Tick级交易引擎")
    parser.add_argument("--initial", type=float, default=10.0, help="初始资金")
    parser.add_argument("--live", action="store_true", help="是否使用真实Gate.io API")
    parser.add_argument("--max-long-trades", type=int, default=None, help="当日允许开多单的最大次数（默认取配置）")
    parser.add_argument("--max-short-trades", type=int, default=None, help="当日允许开空单的最大次数（默认取配置）")
    parser.add_argument("--max-contracts", type=int, default=None, help="单次开仓最多合约张数（默认取配置）")
    parser.add_argument("--preset", type=str, help="使用 config.py 中定义的参数预设名称（如 auto_20）")
    parser.add_argument("--detailed-monitor", action="store_true", help="开启详细监控日志输出")
    parser.add_argument("--no-interactive", action="store_true", help="跳过交互式提示，完全使用命令行参数")
    parser.add_argument("--debug", action="store_true", help="启用调试日志，输出更详细的监控信息")
    return parser.parse_args()


def _safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def _prompt_float(prompt: str, default: float) -> float:
    raw = _safe_input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("⚠️ 输入无效，使用默认值")
        return default


def _prompt_int(prompt: str, default: Optional[int]) -> Optional[int]:
    placeholder = default if default is not None else "不限"
    raw = _safe_input(f"{prompt} [{placeholder}]: ").strip()
    if not raw:
        return default
    if raw.lower() in ("不限", "inf", "none"):
        return None
    try:
        value = int(raw)
        return value
    except ValueError:
        print("⚠️ 输入无效，使用默认值")
        return default


def _prompt_bool(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = _safe_input(f"{prompt} ({suffix}): ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true", "t")


def maybe_interactive(args) -> Any:
    if args.no_interactive or not sys.stdin.isatty():
        return args
    print("=== 交互式启动配置（按回车使用括号内默认值）===")
    args.initial = _prompt_float("初始资金", args.initial)
    if not args.live:
        args.live = _prompt_bool("是否启用实盘（Gate.io真实 API）", args.live)
    if args.max_long_trades is None:
        default_long = HFT_CONFIG.get('max_long_trades_per_day', None)
        args.max_long_trades = _prompt_int("当日最多可开多单次数，0/空表示不限", default_long)
    if args.max_short_trades is None:
        default_short = HFT_CONFIG.get('max_short_trades_per_day', None)
        args.max_short_trades = _prompt_int("当日最多可开空单次数，0/空表示不限", default_short)
    if args.max_contracts is None:
        default_contracts = HFT_CONFIG.get('max_contracts_per_trade', None)
        args.max_contracts = _prompt_int("单次开仓最多合约张数，0/空表示不限", default_contracts)
    print("======================================")
    return args


def _apply_preset_to_configs(preset: str):
    overrides = get_parameter_preset(preset)
    for key, value in overrides.items():
        if key == 'name':
            continue
        if key in HFT_CONFIG:
            HFT_CONFIG[key] = value
        elif key in TRADE_GUARD_CONFIG:
            TRADE_GUARD_CONFIG[key] = value
        else:
            HFT_CONFIG[key] = value


def _configure_logging(debug_enabled: bool):
    """统一初始化日志输出，确保WebSocket/行情日志可见。"""
    level = logging.DEBUG if debug_enabled else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
        force=True
    )


def _configure_runtime(args):
    if args.preset:
        try:
            _apply_preset_to_configs(args.preset)
            print(f"✅ 已应用参数预设 {args.preset}")
        except KeyError as exc:
            print(f"❌ 无法找到参数预设: {exc}")
            sys.exit(1)
    if args.live:
        HFT_CONFIG['signal_only_mode'] = False
    if args.detailed_monitor:
        HFT_CONFIG['detailed_monitoring'] = True
    if args.debug:
        HFT_CONFIG['enable_signal_debug_log'] = True
        HFT_CONFIG['enable_trend_log'] = True
        HFT_CONFIG['detailed_monitoring'] = True


async def main():
    args = parse_args()
    _configure_logging(args.debug)
    args = maybe_interactive(args)
    _configure_runtime(args)
    engine = TrueHFTEngine(
        initial_capital=args.initial,
        use_real_api=args.live,
        max_long_trades=args.max_long_trades,
        max_short_trades=args.max_short_trades,
        max_contracts_per_trade=args.max_contracts,
    )
    await engine.start()


if __name__ == "__main__":
    asyncio.run(main())
