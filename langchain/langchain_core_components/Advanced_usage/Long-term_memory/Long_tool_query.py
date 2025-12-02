# from dataclasses import dataclass: 导入Python的数据类装饰器，用于简化类的创建
# from dotenv import load_dotenv: 导入环境变量加载函数，用于读取.env文件中的配置
# from langchain.agents import create_agent: 导入创建AI代理的函数
# from langchain.tools import tool, ToolRuntime: 导入工具装饰器和运行时类型
# from langgraph.store.memory import InMemoryStore: 导入内存存储实现
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
load_dotenv()

# @dataclass: 将Context类标记为数据类，自动生成常用方法
# class Context:: 定义上下文数据结构
# user_id: str: 声明用户ID字段，用于标识用户身份
@dataclass
class Context:
    user_id: str

# store = InMemoryStore(): 创建内存存储实例，用于临时保存数据
store = InMemoryStore()
# store.put(...): 向存储中写入示例用户数据
# ("users",): 命名空间，用于组织相关数据
# "user_123": 键名，用户唯一标识符
# 字典数据: 包含用户名和语言偏好
store.put(
    ("users",),  # Namespace to group related data together (users namespace for user data)
    "user_123",  # Key within the namespace (user ID as key)
    {
        "name": "John Smith",
        "language": "English",
    }  # Data to store for the given user
)
# @tool: 装饰器，将普通函数转换为LangChain工具
# def get_user_info(runtime: ToolRuntime[Context]) -> str:: 定义查询用户信息的工具
# runtime: ToolRuntime[Context]: 运行时参数，包含存储和上下文信息
# store = runtime.store: 获取共享存储实例
# user_id = runtime.context.user_id: 从上下文中提取用户ID
# store.get(("users",), user_id): 查询用户数据
# 返回用户信息或默认提示
@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """Look up user info."""
    # Access the store - same as that provided to `create_agent`
    store = runtime.store
    user_id = runtime.context.user_id
    # Retrieve data from store - returns StoreValue object with value and metadata
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"
# agent = create_agent(...): 创建AI代理实例
# "deepseek-chat": 指定使用的语言模型
# tools=[get_user_info]: 注册可用工具列表
# store=store: 提供存储实例给代理使用
# context_schema=Context: 指定上下文数据结构
agent = create_agent(
    "deepseek-chat",
    tools=[get_user_info],
    # Pass store to agent - enables agent to access store when running tools
    store=store,
    context_schema=Context
)

# agent.invoke(...): 调用代理执行任务
# 消息参数: 用户请求"look up user information"
# context=Context(user_id="user_123"): 提供执行所需的上下文信息
result = agent.invoke(
    {"messages": [{"role": "user", "content": "look up user information"}]},
    context=Context(user_id="user_123")
)
print("result", result)

# 功能概述
# 此代码实现了基于长期记忆的AI助手功能：
# 数据持久化: 使用InMemoryStore保存用户信息
# 上下文管理: 通过Context类传递用户身份信息
# 工具集成: 创建get_user_info工具供代理调用
# 智能交互: 代理根据用户请求自动查询相关信息
# 模块化设计: 各组件职责清晰，易于扩展维护


