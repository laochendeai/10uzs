# -*- coding: utf-8 -*-
"""
以指定参数组合运行离线观察模式，输出完整的信号与交易生命周期。
"""

import argparse
import asyncio
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any

from config import SIMULATION_CONFIG, MONITORING_CONFIG, get_parameter_preset
from hft_backtester import HFTHistoricalBacktester


def _build_settings(preset_name: str, historical_file: Path, max_rows: int, report_path: Path) -> Dict[str, Any]:
    settings = deepcopy(SIMULATION_CONFIG)
    settings['historical_trade_file'] = str(historical_file)
    if max_rows > 0:
        settings['max_rows'] = max_rows
    overrides = get_parameter_preset(preset_name)
    overrides['name'] = preset_name
    overrides['detailed_monitoring'] = True
    settings['parameter_grid'] = [dict(overrides)]
    settings['report_path'] = str(report_path)
    return settings


async def run_monitor(args):
    preset = args.preset or MONITORING_CONFIG.get('preset_name')
    if not preset:
        raise ValueError("必须指定参数预设名称")
    historical_file = Path(args.historical_file or SIMULATION_CONFIG['historical_trade_file'])
    if not historical_file.exists():
        raise FileNotFoundError(f"找不到行情数据文件: {historical_file}")
    report_path = Path(args.report_path or MONITORING_CONFIG.get('report_path', 'test/monitor_auto.jsonl'))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    settings = _build_settings(
        preset_name=preset,
        historical_file=historical_file,
        max_rows=args.max_rows,
        report_path=report_path
    )
    print("进入详细观测模式，仅使用模拟撮合，所有日志均输出至控制台")
    print(f"使用参数预设 {preset} | 数据源 {historical_file} | 报告 {report_path}")
    backtester = HFTHistoricalBacktester(settings)
    results = await backtester.run_all()
    if not results:
        print("⚠️ 未生成交易结果，请检查数据或参数")
        return
    summary = results[0]
    accuracy = summary.get('direction_accuracy', 0.0)
    trades = summary.get('total_trades', 0)
    pnl = summary.get('total_pnl', 0.0)
    print(
        f"观测完成 | accuracy={accuracy:.2%} | trades={trades} | pnl={pnl:.4f} "
        f"| 报告 {report_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="运行 auto_20 等参数组合的详细监控模式")
    parser.add_argument('--preset', type=str, default=MONITORING_CONFIG.get('preset_name'), help='参数预设名称')
    parser.add_argument('--historical-file', type=str, help='行情 CSV 路径，默认沿用 SIMULATION_CONFIG')
    parser.add_argument('--max-rows', type=int, default=SIMULATION_CONFIG.get('max_rows', 0),
                        help='限制读取的行数，0 表示不限制')
    parser.add_argument('--report-path', type=str, default=MONITORING_CONFIG.get('report_path'),
                        help='回测报告输出路径')
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_monitor(args))


if __name__ == '__main__':
    main()
