#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate.io ETH高频剥头皮交易系统启动脚本
"""

import asyncio
import sys
import os
import argparse
from datetime import datetime

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_engine import TradingEngine
from gateio_config import INITIAL_CAPITAL, DEFAULT_LEVERAGE

def print_banner():
    """打印系统启动横幅"""
    print("🎯" + "=" * 60)
    print("🚀" + " " * 20 + "Gate.io ETH高频剥头皮交易系统" + " " * 20 + "🚀")
    print("🎯" + "=" * 60)
    print(f"💰 初始资金: {INITIAL_CAPITAL} USDT")
    print(f"⚡ 交易杠杆: {DEFAULT_LEVERAGE}x")
    print(f"🎯 策略类型: 震荡区间突破剥头皮")
    print(f"📊 目标收益: 每笔1% | 每日3笔")
    print(f"🛡️ 风险控制: 0.4%止损 | 20%日损限制")
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯" + "=" * 60)

def print_system_info():
    """打印系统信息"""
    print("📋 系统配置信息:")
    print(f"   • 交易对: ETH/USDT 永续合约")
    print(f"   • 平台: Gate.io VIP0")
    print(f"   • 手续费: Taker 0.05% | Maker -0.025%")
    print(f"   • 保证金模式: 逐仓")
    print(f"   • 技术指标: EMA(9,21) + RSI(14) + 成交量分析")
    print(f"   • 资金管理: 渐进式复利策略")
    print()

def check_system_requirements():
    """检查系统要求"""
    print("🔍 系统检查:")

    # 检查Python版本
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"   ✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"   ❌ Python版本过低: {python_version.major}.{python_version.minor}.{python_version.micro} (需要 >= 3.8)")
        return False

    # 检查必需的库
    required_packages = ['pandas', 'numpy', 'asyncio']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (缺失)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n💡 请安装缺失的库: pip install {' '.join(missing_packages)}")
        return False

    return True

def run_simulation_mode():
    """运行模拟模式"""
    print("🎮 模拟模式已启动 (使用模拟数据进行测试)")
    print("⚠️  注意: 这不是真实交易，不会产生实际盈亏")
    print("=" * 50)
    return True

def run_live_mode():
    """运行实盘模式"""
    print("⚠️  实盘模式警告:")
    print("   • 将使用真实资金进行交易")
    print("   • 请确保已正确配置Gate.io API")
    print("   • 请确保有足够的保证金")
    print()

    # 确认继续
    confirm = input("❓ 确认启动实盘交易? (输入 'YES' 确认): ")
    if confirm.upper() != 'YES':
        print("❌ 用户取消启动")
        return False

    print("✅ 实盘模式确认，准备启动...")
    return True

async def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='Gate.io ETH高频剥头皮交易系统')
    parser.add_argument('--mode', choices=['simulation', 'live'], default='simulation',
                       help='运行模式: simulation(模拟) 或 live(实盘)')
    parser.add_argument('--capital', type=float, default=INITIAL_CAPITAL,
                       help=f'初始资金 (默认: {INITIAL_CAPITAL} USDT)')
    parser.add_argument('--leverage', type=int, default=DEFAULT_LEVERAGE,
                       help=f'杠杆倍数 (默认: {DEFAULT_LEVERAGE}x)')

    args = parser.parse_args()

    # 打印启动信息
    print_banner()
    print_system_info()

    # 系统检查
    if not check_system_requirements():
        print("\n❌ 系统检查失败，请解决上述问题后重试")
        sys.exit(1)

    # 模式确认
    if args.mode == 'simulation':
        if not run_simulation_mode():
            return
    else:
        if not run_live_mode():
            return

    # 创建交易引擎
    engine = TradingEngine(initial_capital=args.capital)

    try:
        print("\n🚀 正在启动交易引擎...")
        print("📡 连接市场数据...")
        print("🔄 初始化技术指标...")
        print("🛡️ 启动风险监控...")
        print("✅ 系统启动完成!\n")

        # 启动主循环
        await engine.start()

    except KeyboardInterrupt:
        print("\n⏹️  收到用户停止信号")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 交易引擎已停止")
        print("📊 生成最终报告...")

        # 显示性能报告
        try:
            report = engine.get_performance_report()
            print("\n📈 === 交易统计报告 ===")
            print(f"💰 总资金: {report['portfolio']['current_capital']}")
            print(f"📊 总交易: {report['basic_stats']['total_trades']}")
            print(f"🎯 胜率: {report['basic_stats']['win_rate']}")
            print(f"💎 总盈亏: {report['basic_stats']['total_pnl']}")
            print(f"🛡️ 风险评分: {report['risk_metrics']['risk_score']}")
            print("=" * 30)
        except Exception as e:
            print(f"⚠️  生成报告失败: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n💥 程序崩溃: {e}")
        sys.exit(1)