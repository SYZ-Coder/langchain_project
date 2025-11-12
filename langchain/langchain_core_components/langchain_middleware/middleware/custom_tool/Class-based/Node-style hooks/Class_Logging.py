from typing import Callable

from langchain.agents.middleware import before_model, after_model, wrap_model_call, ModelRequest, ModelResponse, \
    AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from langchain.agents import create_agent, AgentState
from dotenv import load_dotenv
from typing_extensions import Any

load_dotenv()  # 加载 .env 文件中的环境变量

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from typing import Any

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None
# 1. 定义输出结构
class ContactInfo(BaseModel):
    """联系人信息"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱")
    phone: str = Field(description="电话")
    status: str = Field(description="状态")

# 3. 创建代理
agent = create_agent(
    model="deepseek-chat",  # 使用聊天模型
    tools=[],  # 可选工具
    middleware=[LoggingMiddleware()], #使用自定义中间件
    response_format=ContactInfo  # 指定输出格式
)

# 4. 使用代理
result = agent.invoke({
    "messages": [{"role": "user", "content": "提取联系人信息：张三，zhangsan@example.com，13800138000 ,BLOCKED"}]
})

print(result["messages"])
# 输出：ContactInfo(name='张三', email='zhangsan@example.com', phone='13800138000')