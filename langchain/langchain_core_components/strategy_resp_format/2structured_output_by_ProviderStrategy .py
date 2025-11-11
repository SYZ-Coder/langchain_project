from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
# 作用
# 针对不同 LLM 提供商进行优化，处理提供商特定的结构化输出格式
load_dotenv()
class WeatherResponse(BaseModel):
    location: str = Field(description="地点")
    temperature: float = Field(description="温度")
    condition: str = Field(description="天气状况")

# 为不同提供商创建策略
provider_strategy = ProviderStrategy(
    schema=WeatherResponse,
    # 可以针对不同提供商设置特定参数
    # 使用正确的参数名

)

# 创建代理

llm = init_chat_model(
    "deepseek-chat",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)

agent = create_agent(
    model=llm,
    tools=[],
    response_format=provider_strategy
)

if __name__ == "__main__":
    # 自动处理不同提供商的结构化输出差异
    result = agent.invoke({
        "messages": [{"role": "user", "content": "分析苹果公司的业务状况"}]
    })