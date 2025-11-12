from langchain.agents.middleware import before_model, after_model, wrap_model_call
from langchain.agents.middleware import AgentState, ModelRequest, ModelResponse, dynamic_prompt
from langchain.messages import AIMessage
from langchain.agents import create_agent
from langgraph.runtime import Runtime
from typing import Any, Callable

# 可用装饰器
# 节点式 (Node-style)（在特定的执行点运行）：
#
    # @before_agent - 代理启动前（每次调用一次）
    # @before_model - 每次模型调用前
    # @after_model - 每次模型响应后
    # @after_agent - 代理完成时（每次调用最多一次）
# 包装式 (Wrap-style)（拦截并控制执行）：
    #
    # @wrap_model_call - 每次模型调用周围
    # @wrap_tool_call - 每次工具调用周围
# 便利装饰器 (Convenience decorators)：
    #
    # @dynamic_prompt - 生成动态系统提示词（相当于修改提示词的 @wrap_model_call）


# 节点式 (Node-style)：模型调用前的日志记录
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"About to call model with {len(state['messages'])} messages")
    return None

# 节点式 (Node-style)：模型调用后的验证
@after_model(can_jump_to=["end"])
def validate_output(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_message = state["messages"][-1]
    if "BLOCKED" in last_message.content:
        return {
            "messages": [AIMessage("I cannot respond to that request.")],
            "jump_to": "end"
        }
    return None

# 包装式 (Wrap-style)：重试逻辑
@wrap_model_call
def retry_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")

# 包装式 (Wrap-style)：动态提示词
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.get("user_id", "guest")
    return f"You are a helpful assistant for user {user_id}. Be concise and friendly."

# 在代理中使用装饰器
agent = create_agent(
    model="openai:gpt-4o",
    middleware=[log_before_model, validate_output, retry_model, personalized_prompt],
    tools=[],
)