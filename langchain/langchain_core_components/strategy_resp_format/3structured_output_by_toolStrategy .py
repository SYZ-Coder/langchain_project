from langchain.agents import create_agent
from langchain.agents.structured_output import  ToolStrategy
from pydantic import BaseModel, Field

class CalculationResult(BaseModel):
    operation: str = Field(description="运算类型")
    result: float = Field(description="计算结果")
    steps: list[str] = Field(description="计算步骤")

def calculator_tool(expression: str) -> str:
    """简单的计算器工具"""
    try:
        result = eval(expression)
        return f"结果: {result}"
    except:
        return "计算错误"

# 创建工具
tools = [
    Tool(
        name="calculator",
        func=calculator_tool,
        description="用于数学计算"
    )
]

# 使用 ToolStrategy
tool_strategy = ToolStrategy(
    model=CalculationResult,
    tool_choice="auto",  # 自动选择工具
    structured_outputs=True
)

# 创建代理
llm = ChatOpenAI(model="gpt-3.5-turbo")
agent = create_agent(
    llm=llm,
    tools=tools,
    response_format=tool_strategy
)

# 使用 - 会自动调用工具并结构化输出
result = agent.invoke({"input": "计算 (15 + 25) * 2 的结果"})
print(result.output)
# 输出: operation="multiplication" result=80.0 steps=["15 + 25 = 40", "40 * 2 = 80"]