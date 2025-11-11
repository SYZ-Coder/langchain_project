# 配置管理 config.py 文件用于管理应用程序的配置设置，主要通过环境变量和 pydantic-settings 库来加载和验证配置参数。

import os # Python 标准库，用于访问操作系统功能
from dotenv import load_dotenv #  从 .env 文件加载环境变量
from pydantic_settings import BaseSettings #Pydantic 的基础设置类，用于配置管理


load_dotenv()  # 用: 加载项目根目录下的 .env 文件中的环境变量到系统环境中

#作用: 定义应用配置类，继承自 BaseSettings 用法: 通过此类可以集中管理所有应用配置项
class Settings(BaseSettings):
    """应用配置"""
    # 作用: 定义 DeepSeek API 密钥配置项
    # 类型为字符串 (str)
    # 从环境变量 DEEPSEEK_API_KEY 中获取值
    # 如果环境变量不存在，则为 None
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY")


    # 模型配置
    # 作用: 定义 DeepSeek 模型名称配置项
    # 类型为字符串
    # 默认值为 "deepseek-chat"
    # 用于指定要使用的 DeepSeek 模型
    DEEPSEEK_MODEL: str = "deepseek-chat"


    # 服务器配置
    # 作用: 定义服务器主机和端口配置
    # HOST: 服务器监听地址，默认为 "0.0.0.0"（监听所有网络接口）
    # PORT: 服务器端口号，类型为整数，默认值为 8000
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    #作用: pydantic-settings 的内部配置类 用法: 指定环境变量文件为 .env
    class Config:
        env_file = ".env"

#作用: 创建 Settings 类的实例 用法: 在其他模块中导入 settings 对象来访问配置值
settings = Settings()

# 主要特性
# 环境变量支持: 通过 load_dotenv() 自动加载 .env 文件
# 类型安全: 使用 Pydantic 进行配置项类型验证
# 默认值: 为配置项提供合理的默认值
# 集中管理: 所有配置集中在 Settings 类中统一管理