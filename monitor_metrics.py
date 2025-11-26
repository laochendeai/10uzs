# -*- coding: utf-8 -*-
"""
后台运行的实时监控脚本。

每隔固定时间读取 run_true_hft.py 生成的日志，解析最新的 ❤️ 心跳行，
并把关键指标（信号数/执行数/交易数/正确率/PnL/权益等）写入 JSONL 文件。
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


HEARTBEAT_PATTERN = re.compile(
    r"signals:(?P<signals>\d+)\s+/exec:(?P<executed>\d+)\s+\|\s+"
    r"trades:(?P<trades>\d+)\s+\|\s+PnL:(?P<pnl>-?\d+\.\d+)\s+\|\s+"
    r"accuracy:(?P<accuracy>(?:N/A|[\d.]+%))\s+\|\s+last_signal:(?P<last_signal>[^|]+)"
    r"\|\s+positions:(?P<positions>\d+)\s+\|\s+equity:(?P<equity>-?\d+\.\d+)"
)


def _parse_heartbeat(line: str) -> Optional[Dict]:
    match = HEARTBEAT_PATTERN.search(line)
    if not match:
        return None
    groups = match.groupdict()
    accuracy_raw = groups["accuracy"].strip()
    accuracy = None
    if accuracy_raw.endswith("%"):
        try:
            accuracy = float(accuracy_raw.rstrip("%"))
        except ValueError:
            accuracy = None
    payload = {
        "signals": int(groups["signals"]),
        "executed": int(groups["executed"]),
        "trades": int(groups["trades"]),
        "pnl": float(groups["pnl"]),
        "accuracy_percent": accuracy,
        "accuracy_raw": accuracy_raw,
        "last_signal_gap": groups["last_signal"].strip(),
        "positions": int(groups["positions"]),
        "equity": float(groups["equity"]),
    }
    return payload


def _latest_heartbeat(log_path: Path) -> Optional[Dict]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except FileNotFoundError:
        return None
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if "❤️ 心跳" not in line:
            continue
        parsed = _parse_heartbeat(line)
        if not parsed:
            continue
        parsed["line_number"] = idx + 1
        parsed["raw_line"] = line.strip()
        return parsed
    return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def monitor_loop(log_path: Path, output_path: Path, interval: int, iterations: int, idle_retry: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for idx in range(1, iterations + 1):
        snapshot: Optional[Dict] = None
        for _ in range(idle_retry):
            snapshot = _latest_heartbeat(log_path)
            if snapshot:
                break
            time.sleep(5)
        record = {
            "checkpoint": idx,
            "timestamp_utc": _timestamp(),
            "log_path": str(log_path),
            "heartbeat": snapshot,
        }
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if snapshot:
            print(
                f"[monitor] #{idx} signals={snapshot['signals']} trades={snapshot['trades']} "
                f"accuracy={snapshot['accuracy_raw']} pnl={snapshot['pnl']:.4f} equity={snapshot['equity']:.4f}"
            )
        else:
            print(f"[monitor] #{idx} 尚未找到心跳，等待下一轮")
        if idx < iterations:
            time.sleep(max(interval, 1))


def parse_args():
    parser = argparse.ArgumentParser(description="定时抓取 run_true_hft 心跳指标")
    parser.add_argument("--log", required=True, help="run_true_hft.py 日志路径")
    parser.add_argument("--output", default="logs/monitor_metrics.jsonl", help="指标输出 JSONL")
    parser.add_argument("--interval", type=int, default=900, help="采样间隔（秒），默认15分钟")
    parser.add_argument("--iterations", type=int, default=8, help="采样轮数，默认2小时")
    parser.add_argument("--idle-retry", type=int, default=6, help="找不到心跳时的重试次数")
    return parser.parse_args()


def main():
    args = parse_args()
    monitor_loop(
        log_path=Path(args.log),
        output_path=Path(args.output),
        interval=args.interval,
        iterations=args.iterations,
        idle_retry=max(args.idle_retry, 1),
    )


if __name__ == "__main__":
    main()
