from dataclasses import dataclass #导入Python标准库中的dataclass装饰器，用于创建数据类

from dotenv import load_dotenv #导入load_dotenv函数，用于加载环境变量
from typing_extensions import TypedDict #导入TypedDict，用于定义字典结构类型提示

from langchain.agents import create_agent #导入create_agent函数，用于创建代理
from langchain.tools import tool, ToolRuntime #导入工具相关组件，包括tool装饰器和ToolRuntime类型
from langgraph.store.memory import InMemoryStore #导入内存存储实现
load_dotenv()

store = InMemoryStore() # 创建内存存储实例，用于保存持久化数据（生产环境中应替换为数据库存储）

@dataclass #装饰器，将Context类标记为数据类
class Context: #定义上下文数据结构，包含user_id字段
    user_id: str

class UserInfo(TypedDict): #定义用户信息的类型结构，指定包含name字段的字典
    name: str

# 装饰器，将函数转换为LangChain工具
@tool
#定义保存用户信息的工具函数
#def save_user_info(...): 定义保存用户信息的工具函数
# user_info: UserInfo: 接收用户信息参数
# runtime: ToolRuntime[Context]: 接收运行时上下文，包含存储和用户ID信息
# 函数内部通过runtime.store访问存储，使用runtime.context.user_id获取用户标识，调用store.put()保存数据
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """Save user info."""
    # Access the store - same as that provided to `create_agent`
    store = runtime.store
    user_id = runtime.context.user_id
    # Store data in the store (namespace, key, data)
    store.put(("users",), user_id, user_info)
    print(f"Saved user info for user {user_id}.")
    return "Successfully saved user info."
# agent = create_agent(...): 创建AI代理
# "deepseek-chat": 指定使用的LLM模型
# tools=[save_user_info]: 注册可用工具
# store=store: 传递存储实例
# context_schema=Context: 指定上下文数据结构类型
agent = create_agent(
    "deepseek-chat",
    tools=[save_user_info],
    store=store,
    context_schema=Context
)
# agent.invoke(...): 调用代理执行任务
# 传递用户消息"My name is John Smith"
# 通过context=Context(user_id="user_123")提供上下文信息
# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "My name is John Smith"}]},
    # user_id passed in context to identify whose information is being updated
    context=Context(user_id="user_123")
)

# 直接从存储中检索用户数据并获取其值
cacheVal=store.get(("users",), "user_123").value
print("缓存的值",cacheVal)
#这段代码演示了一个完整的AI代理工作流程：
# 设置持久化存储
# 定义数据结构和工具
# 创建具备存储操作能力的AI代理
# 执行代理任务（提取并保存用户姓名）
# 验证数据存储结果