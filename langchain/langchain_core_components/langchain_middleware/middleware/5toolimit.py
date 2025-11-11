from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv

load_dotenv()
# 限制所有工具调用
global_limiter = ToolCallLimitMiddleware(thread_limit=20, run_limit=10)

# 正确示例：
@tool("my_tool")
def my_tool():
    """
这里添加函数的详细描述
说明函数的作用、参数和返回值等信息
"""
    print("测试工具")
    pass


# 限制特定工具
search_limiter = ToolCallLimitMiddleware(
    tool_name="my_tool",
    thread_limit=3,
    run_limit=3,
)
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7
)


agent = create_agent(
    model=llm,
    tools=[my_tool],
    middleware=[global_limiter, search_limiter],
)

if __name__ == "__main__":

    for i in range(5):  # 超出 run_limit 的次数
        try:
            response = agent.invoke({
                "messages": [HumanMessage(content=f"测试消息{i+1}")]
            })
            print(f"调用 {i+1}: 成功")
        except Exception as e:
            print(f"调用 {i+1}: 被限流 - {str(e)}")  # 应该在第3次后出现
