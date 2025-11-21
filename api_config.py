# -*- coding: utf-8 -*-
"""
Gate.io API配置文件
用于存储API密钥和连接参数
"""

import os
from typing import Optional

class GateAPIConfig:
    """Gate.io API配置类"""

    def __init__(self):
        # API密钥配置
        self.api_key: str = ""
        self.api_secret: str = ""
        self.passphrase: Optional[str] = None  # 如果需要

        # API服务器配置
        self.testnet: bool = False  # 是否使用测试网
        self.api_host: str = "https://api.gateio.ws"  # 主网API
        self.ws_host: str = "wss://api.gateio.ws/ws/v4/"  # WebSocket

        # 连接参数
        self.timeout: int = 30
        self.retry_count: int = 3
        self.retry_delay: float = 1.0

        # 交易配置
        self.settle: str = "usdt"  # 结算货币
        self.contract: str = "ETH_USDT"  # 合约交易对

    def load_from_env(self):
        """从环境变量加载配置"""
        self.api_key = os.getenv('GATEIO_API_KEY', '')
        self.api_secret = os.getenv('GATEIO_API_SECRET', '')
        self.passphrase = os.getenv('GATEIO_PASSPHRASE')

        # 测试网配置
        testnet_env = os.getenv('GATEIO_TESTNET', 'false').lower()
        self.testnet = testnet_env in ('true', '1', 'yes')

        if self.testnet:
            self.api_host = "https://fx-api-testnet.gateio.ws"
            self.ws_host = "wss://fx-api-testnet.gateio.ws/ws/v4/"

    def load_from_file(self, config_file: str = "api_keys.txt"):
        """从配置文件加载API密钥"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('API_KEY='):
                            self.api_key = line.split('=', 1)[1].strip()
                        elif line.startswith('API_SECRET='):
                            self.api_secret = line.split('=', 1)[1].strip()
                        elif line.startswith('PASSPHRASE='):
                            self.passphrase = line.split('=', 1)[1].strip()
                        elif line.startswith('TESTNET='):
                            testnet_val = line.split('=', 1)[1].strip().lower()
                            self.testnet = testnet_val in ('true', '1', 'yes')

        except Exception as e:
            print(f"⚠️ 读取API配置文件失败: {e}")

    def validate(self) -> bool:
        """验证配置是否完整"""
        if not self.api_key or not self.api_secret:
            return False
        return True

    def get_auth_info(self) -> dict:
        """获取认证信息"""
        return {
            'api_key': self.api_key,
            'api_secret': self.api_secret,
            'passphrase': self.passphrase
        }

# 全局配置实例
gate_config = GateAPIConfig()

def load_config():
    """加载API配置"""
    # 首先尝试从环境变量加载
    gate_config.load_from_env()

    # 如果环境变量为空，尝试从文件加载
    if not gate_config.validate():
        gate_config.load_from_file()

    if not gate_config.validate():
        print("❌ 未找到有效的API配置")
        print("请设置环境变量 GATEIO_API_KEY 和 GATEIO_API_SECRET")
        print("或创建 api_keys.txt 文件，内容如下:")
        print("API_KEY=your_api_key_here")
        print("API_SECRET=your_api_secret_here")
        print("PASSPHRASE=your_passphrase_if_needed")
        print("TESTNET=false")
        return False

    return True

def get_config() -> GateAPIConfig:
    """获取配置实例"""
    return gate_config