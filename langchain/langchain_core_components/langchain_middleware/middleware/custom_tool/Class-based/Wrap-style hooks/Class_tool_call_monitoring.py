from pydantic import BaseModel, Field
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.tools.tool_node import ToolCallRequest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing import Callable
from langchain.tools import tool
load_dotenv()  # 加载 .env 文件中的环境变量

class ToolMonitoringMiddleware(AgentMiddleware):
    def wrap_tool_call(
            self,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        print(f"Executing tool: {request.tool_call['name']}")
        print(f"Arguments: {request.tool_call['args']}")

        try:
            result = handler(request)
            print(f"Tool completed successfully")
            return result
        except Exception as e:
            print(f"Tool failed: {e}")
            raise

# 1. 定义输出结构
class ContactInfo(BaseModel):
    """联系人信息"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱")
    phone: str = Field(description="电话")
    status: str = Field(description="状态")

# 定义一个示例工具函数
@tool
def get_user_info(query: str) -> dict:

    """根据用户输入提取联系人信息"""
    # 这里可以是调用数据库、API等逻辑
    return {
        "name": "李四",
        "email": "lisi@example.com",
        "phone": "13500138000",
        "status": "PASS"
    }

# 3. 创建代理
agent = create_agent(
    model="deepseek-chat",  # 使用聊天模型
    tools=[get_user_info],  # 可选工具
    middleware=[ToolMonitoringMiddleware()], #使用自定义中间件
    response_format=ContactInfo  # 指定输出格式
)

# 4. 使用代理
result = agent.invoke({
    "messages": [{"role": "user", "content": "提取联系人信息：张三，zhangsan@example.com，13800138000 ,BLOCKED"}]
})

print(result["messages"])
# 输出：ContactInfo(name='张三', email='zhangsan@example.com', phone='13800138000')