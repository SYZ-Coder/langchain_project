from pydantic import BaseModel, Field
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy


# Tool Calling Strategy（工具调用策略）
# 适用场景： 所有支持工具调用的模型


load_dotenv()  # 加载 .env 文件中的环境变量
# 1. 定义输出结构
class ContactInfo(BaseModel):
    """联系人信息"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱")
    phone: str = Field(description="电话")

# 2. 创建代理
agent = create_agent(
    model="deepseek-chat",  # 使用聊天模型
    tools=[],  # 可选工具
    response_format=ToolStrategy(ContactInfo)  # 显式指定工具策略
)

# 3. 使用代理
result = agent.invoke({
    "messages": [{"role": "user", "content": "提取联系人信息：张三，zhangsan@example.com，13800138000"}]
})

print(result["structured_response"])
# 输出：ContactInfo(name='张三', email='zhangsan@example.com', phone='13800138000')