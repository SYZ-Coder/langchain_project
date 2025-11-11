from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 方法1：使用工具调用 (Tool Calling)
# 这种方法将格式Schema作为工具绑定给模型，模型通过tool_calls参数返回结构化的字典。

# 1. 定义输出Schema
class ResponseFormatter(BaseModel):
    answer: str = Field(description="对用户问题的解答")
    followup_question: str = Field(description="用户可以提出的后续问题")

# 2. 初始化模型并绑定Schema

model = ChatDeepSeek(
    model="deepseek-chat",  # 使用聊天模型
    api_key="",  # 替换为你的密钥
    temperature=0.7
)

model_with_tools = model.bind_tools([ResponseFormatter])

# 3. 调用模型并解析输出
ai_msg = model_with_tools.invoke("细胞的动力工厂是什么？")

# 4. 从工具调用参数中提取并转换为Pydantic对象
tool_args = ai_msg.tool_calls[0]["args"]
pydantic_object = ResponseFormatter.model_validate(tool_args)

print(pydantic_object.answer) # 输出：细胞的动力工厂是线粒体...
print(pydantic_object.followup_question) # 输出：ATP在细胞中的功能是什么？