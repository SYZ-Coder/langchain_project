from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field,SecretStr
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
load_dotenv()  # 加载 .env 文件中的环境变量
# 综合演示
# 使用场景总结
# type[]: 简单场景，只需基本结构化
#
# ProviderStrategy: 多提供商环境，需要优化不同 LLM 的输出
#
# ToolStrategy: 复杂工具调用场景，需要结合工具使用结构化数据
#
# 选择哪种方式取决于你的具体需求复杂度和对不同 LLM 提供商的支持需求。

# 定义数据模型
class BusinessAnalysis(BaseModel):
    company_name: str = Field(description="公司名称")
    revenue_estimate: float = Field(description="收入估算")
    growth_trend: str = Field(description="增长趋势")
    risk_factors: list[str] = Field(description="风险因素")

def market_research_tool(query: str) -> str:
    """模拟市场研究工具"""
    return "市场数据: 科技行业年增长15%，竞争激烈"

def financial_analysis_tool(query: str) -> str:
    """模拟财务分析工具"""
    return "财务指标: 利润率20%，现金流稳定"

tools = [
    Tool(name="market_research", func=market_research_tool,
         description="获取市场研究数据"),
    Tool(name="financial_analysis", func=financial_analysis_tool,
         description="进行财务分析")
]

# 方式1: 直接使用类型
# llm = ChatDeepSeek(
#     model="deepseek-chat",  # 使用聊天模型
#     api_key=SecretStr(""),  # 使用 SecretStr 包装# 替换为你的密钥
#     temperature=0.7
# )

model = init_chat_model(
    "deepseek-chat",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)
simple_agent = create_agent(
    model=model,
    tools=tools,
    response_format=BusinessAnalysis  # 模式类型 - 根据模型能力自动选择最佳策略
    # response_format=ProviderStrategy(  # ProviderStrategy适用于支持原生结构化输出的模型（例如 OpenAI、Grok)
    #     schema=BusinessAnalysis
    # )
)

# 方式2: 使用 ProviderStrategy
provider_agent = create_agent(
    model=model,
    tools=tools,
    response_format=ProviderStrategy( # ProviderStrategy适用于支持原生结构化输出的模型（例如 OpenAI、Grok)
        schema=BusinessAnalysis,
        method="json_mode"  # 或
    )
)

# 方式3: 使用 ToolStrategy
tool_agent = create_agent(
    model=model,
    tools=tools,
    response_format=ToolStrategy(  # ToolStrategy适用于所有其他型号
        schema=BusinessAnalysis
    )
)
# 定义输入模型（新增）
class AgentInput(BaseModel):
    input: str

# 正确的调用方式 - 传入字典格式
# result = simple_agent.invoke({"input": "分析苹果公司的业务状况"})
#
# 或者使用定义好的 AgentInput 模型（更规范）
# result = simple_agent.invoke(AgentInput(input="分析苹果公司的业务状况"))

# 方法1: 使用标准的消息格式（推荐）
# result = simple_agent.invoke({
#     "messages": [{"role": "user", "content": "分析苹果公司的业务状况"}]
# })
#
# # 方法2: 检查是否需要同时提供 input 和 messages
# result = simple_agent.invoke({
#     "input": "分析苹果公司的业务状况",
#     "messages": [{"role": "user", "content": "分析苹果公司的业务状况"}]
# })

# result = simple_agent.invoke({
#     "messages": [{"role": "user", "content": "分析苹果公司的业务状况"}]
# })
# print("直接使用类型结构化输出:", result)

result1 = provider_agent.invoke({
    "messages": [{"role": "user", "content": "分析苹果公司的业务状况"}]
})
print("使用 ProviderStrategy结构化输出:", result1)


# result2 = tool_agent.invoke({
#     "messages": [{"role": "user", "content": "分析苹果公司的业务状况"}]
# })
# print("使用ToolStrategy结构化输出:", result2)