from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List

# 定义AI应该返回的结构
class CustomerSupportResponse(BaseModel):
    """客服AI响应格式"""
    issue_category: str = Field(description="问题分类", example="billing")
    priority: str = Field(description="紧急程度: low/medium/high", example="medium")
    solution_steps: List[str] = Field(description="解决步骤列表")
    escalation_required: bool = Field(description="是否需要人工介入")

# LangChain使用这个schema来约束AI输出
llm = ChatDeepSeek(
    model="deepseek-chat",  # 使用聊天模型
    api_key="",  # 替换为你的密钥
    temperature=0.7
)
structured_llm = llm.with_structured_output(CustomerSupportResponse)

# AI现在会严格按照定义的结构返回数据
response = structured_llm.invoke("我的账单有问题，已经扣款但服务没激活")
print(response.issue_category)  # 输出: billing
print(response.solution_steps)  # 输出: ['步骤1', '步骤2', ...]