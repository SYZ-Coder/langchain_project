from langchain.agents.middleware import before_model
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


# 节点式 (Node-style)：模型调用前的日志记录
@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"About to call model with {len(state['messages'])} messages")
    return None
# 2. 创建代理
agent = create_agent(
    model="deepseek-chat",  # 使用聊天模型
    tools=[],  # 可选工具
    middleware=[log_before_model], #使用自定义中间件
)

# 3. 使用代理
result = agent.invoke({
    "messages": [{"role": "user", "content": "提取联系人信息：张三，zhangsan@example.com，13800138000"}]
})

print(result)
# 输出：ContactInfo(name='张三', email='zhangsan@example.com', phone='13800138000')