from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field,SecretStr
from langchain.agents import create_agent


# 1. 类型方式 (type[StructuredResponseT])
# 作用
# 最简单的结构化输出，直接指定 Pydantic 模型类

# 定义响应模型
class PersonInfo(BaseModel):
    name: str = Field(description="人物姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")

# 创建代理
llm = ChatDeepSeek(
    model="deepseek-chat",  # 使用聊天模型
    api_key=SecretStr(""),  # 使用 SecretStr 包装# 替换为你的密钥
    temperature=0.7
)
agent = create_agent(
    model=llm,
    tools=[],  # 你的工具列表
    response_format=PersonInfo  # 直接传入模型类
)
# 定义输入模型
class AgentInput(BaseModel):
    input: str = Field(description="用户输入内容")

# 使用

# 使用 Pydantic 模型封装输入
input_data = AgentInput(input="介绍一个30岁的软件工程师张三")
result = agent.invoke(input_data)

print(result.output)
# 输出: name="张三" age=30 occupation="软件工程师"