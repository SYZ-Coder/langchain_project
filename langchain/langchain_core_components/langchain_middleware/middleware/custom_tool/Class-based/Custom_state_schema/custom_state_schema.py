# 中间件可以使用自定义属性扩展代理的状态。定义自定义状态类型并将其设置为
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, AgentMiddleware
from langchain_core.messages import HumanMessage
from typing_extensions import NotRequired
from typing import Any

load_dotenv()  # 加载 .env 文件中的环境变量

# 继承关系
#   CustomState 类继承自 AgentState 基类
#    用于扩展代理的状态模式，添加自定义属性
# 属性定义
#   model_call_count: NotRequired[int]
#     类型为整数，用于记录模型调用次数
#     使用 NotRequired 标记为可选属性
#   user_id: NotRequired[str]
#     类型为字符串，用于存储用户标识符
#     同样标记为可选属性

# 作用说明
# 状态扩展: 为代理提供额外的状态存储能力
# 调用计数: 通过 model_call_count 跟踪模型调用频率
# 用户识别: 通过 user_id 实现用户会话区分
# 灵活配置: 使用 NotRequired 允许部分属性在初始化时缺失
# 这个自定义状态模式允许中间件在代理执行过程中维护和更新这些额外的属性值。
class CustomState(AgentState):
    model_call_count: NotRequired[int]
    user_id: NotRequired[str]
# 定义了一个名为 CallCounterMiddleware 的中间件类
# 继承自 AgentMiddleware，并指定使用 CustomState 作为状态类型
class CallCounterMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState
    # 指定中间件使用的状态模式为 CustomState
    # 该模式允许在代理状态中存储自定义属性

    # 作用时机: 每次调用模型之前执行
    # 核心功能:
    # 从状态中获取 model_call_count 计数器（默认为0）
    # 当计数超过10次时，返回跳转指令终止流程
    # 实现调用次数限制，防止无限循环
    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        # Access custom state properties
        count = state.get("model_call_count", 0)

        if count > 10:
            return {"jump_to": "end"}

        return None

# 作用时机: 每次调用模型之后执行
    # 核心功能:
    # 将 model_call_count 计数器递增1
    # 更新代理状态中的调用次数统计
    def after_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        # Update custom state
        return {"model_call_count": state.get("model_call_count", 0) + 1}

agent = create_agent(
    model="deepseek-chat",
    middleware=[CallCounterMiddleware()],
    tools=[],
)

# Invoke with custom state
result = agent.invoke({
    "messages": [HumanMessage("Hello")],
    "model_call_count": 0,
    "user_id": "user-123",
})

print(result)