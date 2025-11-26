# Repository Guidelines
始终用中文和我沟通并告诉我你正在做的事情

# AGENTS.md - 本项目 Codex 指令

## 通用分析模板
每次收到自然语言需求时，**默认按以下顺序调用 MCP 工具**：

1. 扫描/搜索  
   用 MCP 的 `idx` 做全文/符号搜索；如要查文档再用 `context7`。
2. 推理/规划  
   用 MCP 的 `thinking` 把搜索结果按「风险/成本/优先级」三档输出表格，并给出修复方案。
3. 文件修改/提交  
   用 MCP 的 `fs` 写回磁盘；如需 commit 再用 `github` 创建分支 & PR。
4. 结果格式  
   最后给我「报告 + diff + PR 链接」三选一，不要只给结论。

## 本仓库专属规则
- 所有交流、状态更新和总结必须使用中文，禁止输出英文。
- 搜索范围限定 `~/10uzs`；不把结果写死到 README。
- 所有硬编码路径 / 魔数 / except-pass 必须集中放到 `config.py`。
- 改完代码后自动 `make lint && make test`（如有 Makefile）。
- PR 标题格式：`[&lt;模块&gt;] &lt;动词&gt;: &lt;一句话&gt;`。

## 快捷命令示例
&gt; 用 MCP 的 idx 搜索所有 TODO，用 thinking 评估风险，用 fs 把修复写回并给我 diff。  
&gt; 用 context7 查 Gate.io 最新官方文档，用 fs 重写 `gateio_api.py`，用 github 提 PR。

## MCP GitHub 默认规则
- 创建 PR 时 `base` 必须显式写 `main`，不要省略或用旧分支名。
- 令牌已配置在 `GITHUB_TOKEN`，无需再输入。
- 成功返回 PR 链接；失败给出手动 URL。


## Project Structure & Module Organization
Core runtime files live at the repo root: `run_true_hft.py` bootstraps the asyncio loop, `true_hft_engine.py` orchestrates scheduling, and `gateio_ws.py` / `gateio_api.py` encapsulate market data and REST trading. Strategy logic resides in `hft_signal_generator.py`, execution hooks in `hft_executor.py`, and guardrails in `survival_rules.py` plus `aggressive_position_manager.py`. Configuration is centralized in `hft_config.py` and `gateio_config.py`; credentials are loaded through `api_config.py` (optionally `api_keys.txt`). Data artifacts such as `trades.csv` and visualization assets under `png/` are disposable outputs.

## Build, Test, and Development Commands
Install once with `pip install -r requirements.txt` (Python 3.9+). Run a dry simulation: `python run_true_hft.py --initial 50`. Switch to live trading with `python run_true_hft.py --live --initial 10` after exporting Gate.io keys. For focused modules, use `python -m hft_signal_generator` or `python -m gateio_ws` style ad-hoc scripts when adding diagnostics—you can call them through `python -m module` to benefit from relative imports.

## Coding Style & Naming Conventions
Follow standard Black/PEP8 spacing (4-space indents, 100-char soft limit) even though Black is not enforced yet. Prefer type hints for new functions (`-> Dict[str, float]`) and snake_case for functions/variables; classes remain PascalCase. Preserve and extend the existing bilingual (Chinese+English) inline commentary rather than replacing it. Avoid hard-coding credentials—point to `api_config` helpers.

## Testing Guidelines
No dedicated `tests/` suite exists, so add targeted `pytest` modules next to the feature under test (e.g., `tests/test_signal_generator.py`). Run via `python -m pytest`. When touching execution paths, also run the simulator command above to confirm market-data loops and Gate.io SDK wiring still succeed. Capture any latency-sensitive change with brief logging diff screenshots in PRs.

## Commit & Pull Request Guidelines
Recent history uses `type: summary` (e.g., `chore: prune unused scripts`); keep the same tense and scope. Commits should isolate logical chunks (config tweak, signal refactor, etc.). PRs must describe the scenario, include reproduction or run commands, mention whether `--live` was exercised, and link the relevant issue/task. Screenshots or logs are encouraged for UI or execution output, and note any breaking configuration changes explicitly.

## Security & Configuration Tips
Never commit `api_keys.txt`; rely on environment variables where possible. Document any new config knobs inside `hft_config.py` docstrings and update README if user action is required. When testing live requests, throttle according to `API_RATE_LIMIT` and log only masked key fragments.
