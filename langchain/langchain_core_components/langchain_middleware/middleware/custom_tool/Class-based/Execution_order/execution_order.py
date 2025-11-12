# 中间件可以使用自定义属性扩展代理的状态。定义自定义状态类型并将其设置为
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, AgentMiddleware
from langchain_core.messages import HumanMessage
from typing_extensions import NotRequired
from typing import Any
from langgraph.runtime import Runtime
load_dotenv()  # 加载 .env 文件中的环境变量


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


class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None

agent = create_agent(
    model="deepseek-chat",
    middleware=[LoggingMiddleware(),CallCounterMiddleware()],
    tools=[],
)

# Invoke with custom state
result = agent.invoke({
    "messages": [HumanMessage("Hello")],
    "model_call_count": 0,
    "user_id": "user-123",
})

print(result)