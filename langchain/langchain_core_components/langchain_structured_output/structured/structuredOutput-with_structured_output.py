from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 这是最简洁的方法，LangChain帮你处理了绑定和解析的所有中间步骤。

# 1. 定义输出Schema
class ResponseFormatter(BaseModel):
    answer: str = Field(description="对用户问题的解答")
    followup_question: str = Field(description="用户可以提出的后续问题")

# 2. 初始化模型并直接启用结构化输出
model = ChatDeepSeek(
    model="deepseek-chat",  # 使用聊天模型
    api_key="",  # 替换为你的密钥
    temperature=0.7
)
structured_model = model.with_structured_output(ResponseFormatter)

# 3. 调用模型，直接获得Pydantic对象！
result = structured_model.invoke("细胞的动力工厂是什么？")

print(type(result)) # 输出：<class '__main__.ResponseFormatter'>
print(result.answer) # 输出：细胞的动力工厂是线粒体...
print(result.followup_question) # 输出：ATP在细胞中的功能是什么？