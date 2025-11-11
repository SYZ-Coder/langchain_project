from langchain.agents.middleware import before_model, after_model
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

# 2. Node-style: validation after model calls
@after_model(can_jump_to=["end"])
def validate_output(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print("validate_output",state)
    last_message = state["messages"][-1]
    print("last_message",last_message)
    print("last_message.content",last_message.content)
    if "BLOCKED" in last_message.content:
        print("内容违规")
        return {
            "messages": [AIMessage("I cannot respond to that request.")],
            "jump_to": "end"
        }
    return None

# 3. 创建代理
agent = create_agent(
    model="deepseek-chat",  # 使用聊天模型
    tools=[],  # 可选工具
    middleware=[validate_output], #使用自定义中间件
)

# 4. 使用代理
result = agent.invoke({
    "messages": [{"role": "user", "content": "提取联系人信息：张三，zhangsan@example.com，13800138000 ,BLOCKED"}]
})
print(result["messages"][-1])
# 输出：ContactInfo(name='张三', email='zhangsan@example.com', phone='13800138000')