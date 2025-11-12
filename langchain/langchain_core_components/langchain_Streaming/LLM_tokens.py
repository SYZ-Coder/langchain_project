from dotenv import load_dotenv
from langchain.agents import create_agent
load_dotenv()  # 加载 .env 文件中的环境变量
# 要流式传输由 LLM 生成的令牌，请使用 。您可以在下面看到代理流式处理工具调用的输出和最终响应。stream_mode="messages"
def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"

agent = create_agent(
    model="deepseek-chat",  # 使用聊天模型
    tools=[get_weather],
)
for token, metadata in agent.stream(
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
        stream_mode="messages",
):
    print(f"node: {metadata['langgraph_node']}")
    print(f"content: {token.content_blocks}")
    print("\n")