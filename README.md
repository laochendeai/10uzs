# Gate.io Tick-Level HFT Engine

This repository now focuses on a single ultra-lean trading loop that consumes Gate.io futures tick data, generates high-frequency scalping signals, and optionally executes them live via the official `gate-api` SDK. The default entry point matches your production command:

```bash
python run_true_hft.py --live --initial 10
```

If the `--live` flag is omitted the engine runs in a fully simulated mode so you can profile logic without touching the exchange.

## Key Capabilities
- **Real tick + order-book feed** – `gateio_ws.py` subscribes to `futures.tickers`, `futures.order_book`, `futures.candlesticks`, and `futures.trades`, exposing microstructure metrics to the engine.
- **Composite HFT signals** – `hft_signal_generator.py` blends momentum, volume spikes, order-book imbalance, adaptive thresholds, and market-state filters to create long/short entries with confidence scores.
- **Aggressive execution** – `HFTExecutor` in `hft_executor.py` prefers IOC limit orders with millisecond timeouts and seamlessly falls back to market orders; all requests go through the official Gate.io Python SDK.
- **Survival rules & risk gating** – `survival_rules.py` and `aggressive_position_manager.py` enforce capital usage policies, leverage limits, and stop trading when loss caps hit.
- **Rich terminal UI** – `tui_display.py` uses `rich` to stream ticks, current stats, and recent logs directly in the terminal while the strategy is running.

## Project Layout
```
run_true_hft.py          # CLI wrapper (argparse + asyncio)
true_hft_engine.py       # Core event loop and state machine
hft_config.py            # All tunable parameters (leverage, filters, safeguards)
gateio_ws.py             # WebSocket market-data client
gateio_api.py            # REST trading helper built on gate-api SDK
hft_data_manager.py      # Rolling buffers for ticks, order book, and derived frames
hft_signal_generator.py  # Signal construction logic
hft_executor.py          # Order placement / throttling / stop hooks
aggressive_position_manager.py
survival_rules.py
hft_performance.py       # Optional live metrics aggregation
tui_display.py           # Streaming dashboard
```

## Requirements
- Python 3.9+
- Gate.io API key with **Futures trading** permissions
- Dependencies in `requirements.txt` (notably `gate-api`, `websockets`, `rich`)

Install dependencies once:
```bash
pip install -r requirements.txt
```

## Configure Gate.io Credentials
The engine uses `api_config.py` to load credentials. Either:

**A. Environment variables (recommended)**
```bash
export GATEIO_API_KEY="your_api_key"
export GATEIO_API_SECRET="your_api_secret"
export GATEIO_PASSPHRASE="optional_passphrase"
export ENABLE_LIVE_TRADING=true
```

**B. Local `api_keys.txt`**
```
API_KEY=your_api_key
API_SECRET=your_api_secret
PASSPHRASE=optional_passphrase
TESTNET=false
```
Copy the template with `cp api_keys.txt.example api_keys.txt` and fill in your values.

> ⚠️ The leverage API requires the futures wallet to have available margin. If you see `INSUFFICIENT_AVAILABLE`, transfer funds to the contract account or reduce the configured leverage in `hft_config.py`.

## Running the Engine
```bash
# Simulation with virtual capital
python run_true_hft.py --initial 50

# Live trading (uses Gate.io REST + WebSocket)
python run_true_hft.py --live --initial 10
```
- `--initial` only affects statistics; in live mode actual account balances are fetched through `/api/v4/futures/accounts`.
- When `--live` is set, the engine:
  1. Starts the WebSocket client and signal loop.
  2. Loads API credentials and attempts to set leverage via `update_position_leverage` (skipped if the wallet is empty).
  3. Pulls available margin every ~30 seconds for survival checks.
  4. Routes filled orders to the executor, which tracks latency and attaches protective stops when enabled.

## Customising Behaviour
All knobs live in `hft_config.py`. Notable settings:
- `momentum_*`, `order_imbalance_min`, `composite_entry_threshold` – signal sensitivity.
- `leverage`, `fixed_margin`, `min_contract_margin` – capital allocation per trade.
- `enable_trailing_stop`, `trailing_*`, `partial_profit_*` – exit logic.
- `enable_protective_stops`, `stop_order_price_type`, `stop_order_expiration` – exchange stop order parameters.

Change values, then restart the script. The engine logs active parameters on boot to help with back-to-back tweaks.

## Troubleshooting
| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `INVALID_KEY` on startup | Wrong API key/secret or missing Futures permission | Regenerate credentials on Gate.io and update env / `api_keys.txt`. |
| `INSUFFICIENT_AVAILABLE` when setting leverage | Futures wallet empty | Transfer USDT to the contract account or disable `--live`. |
| No ticks in UI | Firewall blocking WebSocket or wrong contract symbol in `gateio_config.py` | Ensure outbound `wss://fx-ws.gateio.ws` is reachable and symbol matches `ETH_USDT` (default). |
| Orders stay `open` | `hft_executor` hit IOC timeout before fill | Increase `limit_order_timeout` or disable `use_limit_orders`. |

## Disclaimer
This codebase is strictly for educational and research purposes. Cryptocurrency derivatives are highly risky. Use only capital you can afford to lose, and comply with all applicable regulations.
