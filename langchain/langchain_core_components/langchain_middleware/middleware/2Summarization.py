from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool

load_dotenv()  # 加载 .env 文件中的环境变量

# 1. 定义输出结构
class ContactInfo(BaseModel):
    """联系人信息"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱")
    phone: str = Field(description="电话")

# 2. 定义工具函数
@tool
def weather_tool(location: str) -> str:
    """
    查询指定地点的天气信息

    Args:
        location: 地点名称

    Returns:
        str: 天气信息描述
    """
    # 这里模拟天气查询，实际应用中可以接入真实的天气API
    return f"{location}的天气情况：晴天，温度25°C，湿度60%，风力3级。"

@tool
def calculator_tool(expression: str) -> str:
    """
    计算数学表达式

    Args:
        expression: 数学表达式字符串

    Returns:
        str: 计算结果
    """
    try:
        # 安全地计算表达式，这里仅作示例，实际应用中需要更安全的计算方式
        allowed_chars = set('0123456789+-*/(). ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"{expression} = {result}"
        else:
            return "表达式包含非法字符"
    except Exception as e:
        return f"计算出错: {str(e)}"

# 初始化 DeepSeek 模型实例
llm = ChatDeepSeek(
    model="deepseek-chat",  # 使用聊天模型
    api_key="",  # 替换为你的密钥
    temperature=0.7
)

# 3. 创建代理
agent = create_agent(
    model="deepseek-chat",  # 使用聊天模型
    tools=[weather_tool, calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            max_tokens_before_summary=4000,  # 在 4000 个 token 时触发摘要
            messages_to_keep=20,  # 摘要后保留最近 20 条消息
            summary_prompt="Custom prompt for summarization...",  # 可选
        ),
    ],
)

# 测试工具
if __name__ == "__main__":
    # 测试天气工具
    result1 = agent.invoke({
        "messages": [{"role": "user", "content": "北京的天气怎么样？"}]
    })
    print("天气查询结果:", result1)

    # 测试计算器工具
    result2 = agent.invoke({
        "messages": [{"role": "user", "content": "计算 125 + 345 等于多少？"}]
    })
    print("计算器结果:", result2)
