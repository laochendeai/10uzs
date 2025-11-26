# -*- coding: utf-8 -*-
"""
自动化趋势策略调参与验证。

逻辑：
1. 针对趋势参数网格（min_diff、快慢均线、成交量阈值）逐个回测；
2. 若连续3组无信号或全部不达标，则自动切换到订单簿不平衡策略网格；
3. 目标：胜率≥65%、最大回撤≤8%、信号数≥10；
4. 输出最佳结果以及全部测试记录至 logs。
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from config import SIMULATION_CONFIG
from hft_backtester import HFTHistoricalBacktester

TARGET_ACCURACY = 0.65
MAX_DRAWDOWN_RATIO = 0.08
MIN_SIGNALS = 10
ZERO_SIGNAL_LIMIT = 3
RESULT_PATH = Path("logs/trend_optimizer_report.json")
DETAIL_PATH = Path("logs/trend_optimizer_results.jsonl")


def build_trend_grid() -> List[Dict]:
    min_diffs = [0.00025, 0.00018, 0.00012]
    fast_windows = [15, 22, 30]
    slow_windows = [45, 60, 75]
    volume_ratios = [1.30, 1.15]
    combos: List[Dict] = []
    idx = 1
    for diff in min_diffs:
        for fast in fast_windows:
            for slow in slow_windows:
                if slow <= fast:
                    continue
                for vol in volume_ratios:
                    combos.append({
                        'name': f"trend_{idx:03d}",
                        'trend_signal_mode': 'ma_follow',
                        'trend_min_diff': diff,
                        'trend_fast_window': fast,
                        'trend_slow_window': slow,
                        'trend_volume_ratio': vol,
                        'trend_volume_window': 40,
                        'trend_recent_ticks': 8,
                        'trend_orderbook_min_imbalance': 0.0,
                        'trend_orderbook_use': False,
                        'trend_use_cross_confirmation': False,
                        'trend_signal_cooldown': 2.5
                    })
                    idx += 1
    return combos


def build_orderbook_grid() -> List[Dict]:
    imbalance_levels = [0.08, 0.05, 0.03]
    volume_ratios = [1.20, 1.10]
    cooldowns = [2.0, 3.0]
    combos: List[Dict] = []
    idx = 1
    for imb in imbalance_levels:
        for volume in volume_ratios:
            for cooldown in cooldowns:
                combos.append({
                    'name': f"orderbook_{idx:03d}",
                    'trend_signal_mode': 'orderbook_imbalance',
                    'trend_orderbook_min_imbalance': imb,
                    'trend_volume_ratio': volume,
                    'trend_orderbook_use': True,
                    'trend_use_cross_confirmation': False,
                    'trend_signal_cooldown': cooldown,
                    'trend_fast_window': 15,
                    'trend_slow_window': 45,
                    'trend_min_diff': 0.00015
                })
                idx += 1
    return combos


def meets_target(report: Dict) -> bool:
    return (
        report.get('signals_generated', 0) >= MIN_SIGNALS
        and report.get('win_rate', 0.0) >= TARGET_ACCURACY
        and report.get('max_drawdown_ratio', 1.0) <= MAX_DRAWDOWN_RATIO
    )


def write_detail(report: Dict):
    DETAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DETAIL_PATH.open("a", encoding="utf-8") as handle:
        serialized = json.dumps(report, ensure_ascii=False)
        handle.write(serialized + "\n")


async def run_grid(tester: HFTHistoricalBacktester, grid: List[Dict], allow_zero_break: bool) -> Dict:
    zero_streak = 0
    best: Dict = {}
    for combo in grid:
        report = await tester.run_variant(combo['name'], combo)
        enriched = dict(report)
        enriched['tested_at'] = datetime.now(timezone.utc).isoformat()
        if combo.get('trend_signal_mode') == 'orderbook_imbalance':
            enriched['strategy_mode'] = 'orderbook'
        else:
            enriched['strategy_mode'] = 'trend'
        write_detail(enriched)
        if meets_target(report):
            return report
        if report.get('signals_generated', 0) == 0:
            zero_streak += 1
        else:
            zero_streak = 0
        if not best or report.get('win_rate', 0.0) > best.get('win_rate', 0.0):
            best = report
        if allow_zero_break and zero_streak >= ZERO_SIGNAL_LIMIT:
            break
    return best


async def main():
    settings = dict(SIMULATION_CONFIG)
    settings['parameter_grid'] = []
    tester = HFTHistoricalBacktester(settings)
    DETAIL_PATH.unlink(missing_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    trend_best = await run_grid(tester, build_trend_grid(), allow_zero_break=True)
    final_report = trend_best
    fallback_used = False
    if not meets_target(trend_best):
        fallback_used = True
        orderbook_best = await run_grid(tester, build_orderbook_grid(), allow_zero_break=False)
        final_report = orderbook_best

    output = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'target_accuracy': TARGET_ACCURACY,
        'max_drawdown_limit': MAX_DRAWDOWN_RATIO,
        'min_signals': MIN_SIGNALS,
        'fallback_used': fallback_used,
        'result': final_report,
        'meets_target': meets_target(final_report),
    }
    RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if output['meets_target']:
        print(
            f"[优化完成] 模式={'orderbook' if fallback_used else 'trend'} | "
            f"accuracy={final_report.get('win_rate', 0.0):.2f} | "
            f"drawdown={final_report.get('max_drawdown_ratio', 0.0):.2%} | "
            f"signals={final_report.get('signals_generated', 0)}"
        )
    else:
        print(
            f"[优化未达标] 最佳 accuracy={final_report.get('win_rate', 0.0):.2f} | "
            f"drawdown={final_report.get('max_drawdown_ratio', 0.0):.2%} | "
            f"signals={final_report.get('signals_generated', 0)}"
        )


if __name__ == "__main__":
    asyncio.run(main())
