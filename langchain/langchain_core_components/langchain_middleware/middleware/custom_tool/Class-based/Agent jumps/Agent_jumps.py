# 中间件可以使用自定义属性扩展代理的状态。定义自定义状态类型并将其设置为
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, AgentMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from typing_extensions import NotRequired
from typing import Any
from langgraph.runtime import Runtime
load_dotenv()  # 加载 .env 文件中的环境变量

from langchain.agents.middleware import AgentMiddleware, hook_config
from typing import Any

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None

# 在 ConditionalMiddleware 类上方添加以下函数定义
def some_condition(state: AgentState) -> bool:
    # 示例逻辑：如果消息数量大于5，则触发跳转
    return len(state.get("messages", [])) > 5

class ConditionalMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end", "tools"])
    def after_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        if some_condition(state):
            return {"jump_to": "end"}
        return None


class CustomState(AgentState):
    model_call_count: NotRequired[int]
    user_id: NotRequired[str]
# 定义了一个名为 CallCounterMiddleware 的中间件类
# 继承自 AgentMiddleware，并指定使用 CustomState 作为状态类型
class CallCounterMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        # Access custom state properties
        count = state.get("model_call_count", 0)

        if count > 10:
            return {"jump_to": "end"}

        return None

    def after_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        # Update custom state
        return {"model_call_count": state.get("model_call_count", 0) + 1}

@tool("my_tool")
def my_tool():
    """
这里添加函数的详细描述
说明函数的作用、参数和返回值等信息
"""
    print("测试工具")
    pass


agent = create_agent(
    model="deepseek-chat",
    middleware=[ConditionalMiddleware(),CallCounterMiddleware()],
    tools=[my_tool],
)

# Invoke with custom state
result = agent.invoke({
    "messages": [HumanMessage("Hello")],
    "model_call_count": 0,
    "user_id": "user-123",
})

print(result)