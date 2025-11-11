from typing import Callable

from langchain.agents.middleware import before_model, after_model, wrap_model_call, ModelRequest, ModelResponse, \
    dynamic_prompt
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from langchain.agents import create_agent, AgentState
from dotenv import load_dotenv
from typing_extensions import Any

load_dotenv()  # 加载 .env 文件中的环境变量
# 1. 定义输出结构
class ContactInfo(BaseModel):
    """联系人信息"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱")
    phone: str = Field(description="电话")
    status: str = Field(description="状态")

# 2. Wrap-style: dynamic prompts
# @dynamic_prompt 覆盖了系统提示词
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    user_name = "guest"

    if request.runtime and request.runtime.context:
        user_name = request.runtime.context.get("user_name", "guest")

    print(f"User ID: {user_name}")
    return f"请将用户输入中的姓名'张三'替换为'{user_name}'，然后提取联系人信息"

# 3. 创建代理
agent = create_agent(
    model="deepseek-chat",  # 使用聊天模型
    tools=[],  # 可选工具
    system_prompt="请把用户说的名字张三改成李四",
    # middleware=[log_before_model, validate_output, retry_model, personalized_prompt],
    middleware=[personalized_prompt], #使用自定义中间件
    response_format=ContactInfo  # 指定输出格式
)

# 4. 使用代理
result = agent.invoke({
    "messages": [{"role": "user", "content": "提取联系人信息：张三，zhangsan@example.com，13800138000 ,BLOCKED"}]

}, runtime=Runtime(
    context={
        "user_id": "12345",
        "session_id": "session_001"
    }
))

print(result["messages"][-1])
# 输出：ContactInfo(name='张三', email='zhangsan@example.com', phone='13800138000')