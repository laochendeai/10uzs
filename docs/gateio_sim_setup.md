# Gate.io 模拟合约联调手把手教程

本文档用于指导同事快速跑通 Gate.io 模拟账户的余额查询与实盘模拟交易流程，仅依赖 4 个环境变量（API Key、Secret、REST 域名、WebSocket 域名）。

## 1. 准备测试网 API Key

1. 登录 Gate.io 测试网（fx-trade-testnet.gateio.ws），在「API 管理」创建新密钥。
2. 勾选「合约交易」权限，关闭 IP 白名单或添加服务器 IP。
3. 记录生成的 `API Key` 与 `API Secret`。

## 2. 设置环境变量

在 **当前 shell** 输入以下命令（务必 `export`，否则子进程看不到）：

```bash
export GATEIO_API_KEY="你的测试网 API Key"
export GATEIO_API_SECRET="你的测试网 API Secret"
export GATEIO_API_BASE_URL="https://api-testnet.gateapi.io/api/v4"
export GATEIO_WS_URL="wss://fx-ws-testnet.gateio.ws/v4/ws/usdt"

# 可选：快速检查是否生效
env | grep GATEIO_API
```

如果命令输出中能看到 `GATEIO_API_KEY/SECRET`，说明变量已经成功注入。

## 3. 验证余额接口 (`fetch_balance.py`)

1. 进入项目根目录 `~/10uzs`。
2. 执行 `python fetch_balance.py`。  
   该脚本会调用 `GateIOAPI.get_account_balance()` 查询测试网合约账户的 `available/total/unrealized_pnl`。
3. 看到类似以下输出即表示 REST 接口鉴权成功：

```
连接主机: https://api-testnet.gateapi.io/api/v4 | 合约: ETH_USDT | settle: usdt | testnet=True
✅ 合约账户余额:
  available: 49900.159444875
  total: 49999.48509175
  unrealized_pnl: 0.0
```

如果仍提示 `INVALID_KEY`：
- 重复确认密钥在测试网生成且开启合约权限；
- 再次 `env | grep GATEIO_API`，确保 KEY/SECRET 已 export。

## 4. 启动实盘模拟策略 (`run_true_hft.py`)

余额验证通过后，可以直接运行策略：

```bash
python run_true_hft.py --live --preset auto_20 --detailed-monitor --no-interactive
```

参数说明：
- `--live`：启用 Gate.io 接口（会自动切到真实下单模式）；
- `--preset auto_20`：加载自动调参得到的最优组合；
- `--detailed-monitor`：输出信号投票、指标明细、累计盈亏曲线；
- `--no-interactive`：跳过交互式输入，适合线上环境。

运行过程中应看到：
- `📘 开仓请求 ...` / `📙 平仓 ...`：真实的模拟持仓建立与释放；
- `🧠 信号依据 ...`、`🗳️ 投票 ...`：信号触发细节；
- `📈 累计盈亏 ...`：实时资金曲线与资金变动；
- Gate.io 模拟账户网页会同步显示资金与持仓的变化。

## 5. 常见问题

| 现象 | 可能原因 | 处理方式 |
| --- | --- | --- |
| `{"label":"INVALID_KEY"}` | 没有 export KEY/SECRET；密钥没有合约权限；IP 白名单拦截 | 重复执行 `export ...`；到测试网后台检查权限/白名单 |
| DNS 无法解析 `api-testnet.gateapi.io` | 服务器网络受限 | 确保服务器能访问 Gate 测试网域名 |
| 策略日志只有“虚拟”开仓 | 未加 `--live`；或 KEY/SECRET 未正确注入 | 启动命令必须带 `--live`，并确认第 2 步的 `env` 结果 |

按照以上步骤，先跑 `fetch_balance.py` 再跑 `run_true_hft.py`，即可在 Gate.io 模拟账户看到真实的合约买卖过程。若需要进一步诊断，保留控制台日志并联系维护人员。 
