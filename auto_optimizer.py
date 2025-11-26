# -*- coding: utf-8 -*-
"""
自动化参数扫描与迭代调优脚本。

流程：
1. 从 config._default_parameter_grid 获取 >=50 组参数；
2. 分批运行 HFTHistoricalBacktester，收集正确率并记录报告；
3. 若最佳方向正确率 < 目标值，则基于当前最优组合按顺序（单参数）微调，生成新的组合加入搜索池；
4. 对所有 direction_accuracy < 0.5 且有错误样本的组合，整理典型错例，供 thinking MCP 使用。
"""

import asyncio
import json
import time
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Dict, List, Any, Tuple, Optional

from config import SIMULATION_CONFIG, TRADE_GUARD_CONFIG, _default_parameter_grid
from hft_backtester import HFTHistoricalBacktester


class AutoOptimizer:
    def __init__(self,
                 target_accuracy: float = 0.75,
                 max_iterations: int = 30,
                 chunk_size: int = 20,
                 stall_limit: int = 10,
                 convergence_patience: int = 10,
                 improvement_threshold: float = 0.01,
                 max_runtime_hours: float = 48.0):
        self.target_accuracy = target_accuracy
        self.max_iterations = max_iterations
        self.chunk_size = chunk_size
        self.stall_limit = stall_limit
        self.convergence_patience = convergence_patience
        self.improvement_threshold = improvement_threshold
        self.max_runtime_hours = max_runtime_hours
        self.base_settings = deepcopy(SIMULATION_CONFIG)
        self.guard_defaults = deepcopy(TRADE_GUARD_CONFIG)
        self.parameter_pool: List[Dict[str, Any]] = _default_parameter_grid()
        self.cursor = 0
        self.iteration = 0
        self.best_accuracy = 0.0
        self.stall_counter = 0
        self.no_progress_rounds = 0
        self.param_cycle = [
            'composite_entry_threshold',
            'direction_vote_required',
            'min_reentry_seconds',
            'momentum_threshold',
            'volume_spike_min'
        ]
        self.param_index = 0
        self.log_path = Path('test') / 'auto_optimizer_log.jsonl'
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_variant: Optional[Dict[str, Any]] = None
        self.best_overrides: Optional[Dict[str, Any]] = None
        self.stop_reason: Optional[str] = None
        self.start_time = time.time()

    def _next_grid(self) -> List[Dict[str, Any]]:
        if self.cursor >= len(self.parameter_pool):
            self.cursor = 0
        end = min(self.cursor + self.chunk_size, len(self.parameter_pool))
        subset = self.parameter_pool[self.cursor:end]
        if not subset:
            subset = self.parameter_pool[:self.chunk_size]
        self.cursor = end
        return deepcopy(subset)

    async def run(self):
        history: List[Dict[str, Any]] = []
        print("🚀 启动自动化参数优化")
        while self.iteration < self.max_iterations:
            grid = self._next_grid()
            settings = deepcopy(self.base_settings)
            settings['parameter_grid'] = grid
            backtester = HFTHistoricalBacktester(settings)
            results = await backtester.run_all()
            iteration_best, iteration_variant = self._track_best(results)
            poor_cases = self._collect_poor_variants(results)
            history.append({
                'iteration': self.iteration,
                'best_accuracy': iteration_best,
                'results': results,
                'poor_cases': poor_cases
            })
            self._append_log(history[-1])
            self._report_iteration(iteration_best, results)
            self._update_best(iteration_best, iteration_variant, results)
            if self._should_stop(iteration_best):
                break
            if self.stall_counter >= self.stall_limit:
                # 提前调整策略：降低部分约束，重置 stall
                self._nudge_defaults()
                self.stall_counter = 0
            self.iteration += 1
        if self.stop_reason:
            print(f"🛑 优化终止: {self.stop_reason}")
        if self.best_variant:
            print(
                f"🏁 最佳组合 {self.best_variant.get('variant')} "
                f"| accuracy={self.best_variant.get('direction_accuracy', 0.0):.2%} "
                f"| trades={self.best_variant.get('total_trades', 0)}"
            )
            print(f"⚙️ 对应参数: {json.dumps(self.best_overrides, ensure_ascii=False)}")
            await self._monitor_best_variant()
        return history

    def _track_best(self, results: List[Dict[str, Any]]) -> Tuple[float, Optional[Dict[str, Any]]]:
        valid = [r for r in results if r.get('total_trades', 0) > 0]
        if not valid:
            return 0.0, None
        best_variant = max(valid, key=lambda r: r.get('direction_accuracy', 0.0))
        return best_variant.get('direction_accuracy', 0.0), best_variant

    def _collect_poor_variants(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        poor: List[Dict[str, Any]] = []
        for result in results:
            accuracy = result.get('direction_accuracy', 0.0)
            samples = result.get('error_cases') or []
            if accuracy >= 0.5 or len(samples) < 3:
                continue
            poor.append({
                'variant': result.get('variant'),
                'accuracy': accuracy,
                'overrides': result.get('overrides'),
                'error_cases': samples[:3]
            })
        return poor

    def _append_log(self, record: Dict[str, Any]):
        serializable = json.dumps(record, ensure_ascii=False)
        with self.log_path.open('a', encoding='utf-8') as handle:
            handle.write(serializable + '\n')

    def _inject_mutation(self, best_variant: Dict[str, Any]):
        overrides = dict(best_variant.get('overrides') or {})
        if not overrides:
            return
        param = self.param_cycle[self.param_index % len(self.param_cycle)]
        self.param_index += 1
        mutated = dict(overrides)
        original_value = overrides.get(param)
        if param == 'composite_entry_threshold':
            mutated[param] = max(0.35, float(original_value or 0.5) - 0.05)
        elif param == 'direction_vote_required':
            mutated[param] = max(2, min(4, int(original_value or 3) - 1))
        elif param == 'min_reentry_seconds':
            mutated[param] = max(0.5, float(original_value or 1.0) - 0.3)
            same_dir = overrides.get('same_direction_reentry_seconds', mutated[param] * 1.5)
            mutated['same_direction_reentry_seconds'] = max(mutated[param] + 0.5, float(same_dir))
        elif param == 'momentum_threshold':
            mutated[param] = max(0.00008, float(original_value or 0.0002) * 0.9)
        elif param == 'volume_spike_min':
            mutated[param] = max(1.0, float(original_value or 1.5) * 0.9)
        mutated['name'] = f"tuned_iter{self.iteration}_{param}"
        self.parameter_pool.insert(0, mutated)

    def _nudge_defaults(self):
        """在长时间无改进的情况下放宽基础限制。"""
        for combo in self.parameter_pool[:10]:
            combo['composite_entry_threshold'] = max(0.35, combo.get('composite_entry_threshold', 0.5) - 0.02)
            combo['direction_vote_required'] = max(2, int(combo.get('direction_vote_required', 3)) - 1)
            combo['min_reentry_seconds'] = max(0.5, combo.get('min_reentry_seconds', 1.0) - 0.2)

    def _report_iteration(self, iteration_best: float, results: List[Dict[str, Any]]):
        elapsed_hours = (time.time() - self.start_time) / 3600
        best_display = self.best_accuracy
        delta = iteration_best - best_display
        monitor_flag = "⏳"
        print(
            f"{monitor_flag} 轮次 {self.iteration:02d} | 当前 {iteration_best:.2%} | "
            f"历史最佳 {best_display:.2%} | 改进 {delta*100:.2f}pp | "
            f"停滞 {self.no_progress_rounds}/{self.convergence_patience} | "
            f"耗时 {elapsed_hours:.2f}h"
        )

    def _update_best(self, iteration_best: float, iteration_variant: Optional[Dict[str, Any]], results: List[Dict[str, Any]]):
        delta = iteration_best - self.best_accuracy
        if iteration_variant and iteration_best > self.best_accuracy:
            self.best_accuracy = iteration_best
            self.best_variant = iteration_variant
            self.best_overrides = dict(iteration_variant.get('overrides') or {})
            self.stall_counter = 0
            self._inject_mutation(iteration_variant)
        else:
            self.stall_counter += 1
        if delta > self.improvement_threshold:
            self.no_progress_rounds = 0
        else:
            self.no_progress_rounds += 1

    def _should_stop(self, iteration_best: float) -> bool:
        elapsed_hours = (time.time() - self.start_time) / 3600
        if iteration_best >= self.target_accuracy:
            self.stop_reason = f"达到目标正确率 {iteration_best:.2%}"
            return True
        if self.no_progress_rounds >= self.convergence_patience:
            self.stop_reason = (
                f"连续 {self.convergence_patience} 轮提升 < {self.improvement_threshold:.2%}"
            )
            return True
        if elapsed_hours >= self.max_runtime_hours:
            self.stop_reason = f"运行时间超过 {self.max_runtime_hours:.1f} 小时"
            return True
        return False

    async def _monitor_best_variant(self):
        if not self.best_variant or not self.best_overrides:
            print("⚠️ 没有可供稳态观察的最佳组合")
            return
        monitor_settings = deepcopy(self.base_settings)
        monitor_settings['parameter_grid'] = [dict(self.best_overrides)]
        monitor_settings['report_path'] = str(Path('test') / 'monitor_report.jsonl')
        print(
            f"📡 进入稳态观察模式，固定组合 {self.best_variant.get('variant')}，"
            "按 Ctrl+C 结束观察"
        )
        monitor_cycle = 0
        try:
            while True:
                if (time.time() - self.start_time) / 3600 >= self.max_runtime_hours:
                    print("⏱️ 达到最大运行时长，结束稳态观察")
                    break
                monitor_cycle += 1
                monitor = HFTHistoricalBacktester(monitor_settings)
                results = await monitor.run_all()
                summary = results[0] if results else {}
                acc = summary.get('direction_accuracy', 0.0)
                trades = summary.get('total_trades', 0)
                pnl = summary.get('total_pnl', 0.0)
                print(f"[monitor {monitor_cycle:02d}] accuracy={acc:.2%} trades={trades} pnl={pnl:.4f}")
        except KeyboardInterrupt:
            print("📝 手动停止稳态观察")


async def main():
    optimizer = AutoOptimizer()
    history = await optimizer.run()
    if history:
        print(f"[auto_optimizer] 已完成 {len(history)} 轮，历史最佳正确率 {optimizer.best_accuracy:.2%}")
    else:
        print("[auto_optimizer] 未能生成可用的历史记录")


if __name__ == '__main__':
    asyncio.run(main())
