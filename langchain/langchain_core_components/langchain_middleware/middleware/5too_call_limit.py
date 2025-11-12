from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv

load_dotenv()
# 限制所有工具调用
# global_limiter = ToolCallLimitMiddleware(thread_limit=2, run_limit=2)

# 正确示例：
@tool("my_tool")
def my_tool():
    """
你是一个机器小助手
"""
    print("测试工具是否限流")
    pass


# 限制特定工具
search_limiter = ToolCallLimitMiddleware(
    tool_name="search_tool",
    thread_limit=3,
    run_limit=3,
)


agent = create_agent(
    model="deepseek-chat",
    tools=[my_tool],
    middleware=[search_limiter],
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
