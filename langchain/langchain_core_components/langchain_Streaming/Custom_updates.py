from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.config import get_stream_writer
load_dotenv()  # 加载 .env 文件中的环境变量
# 若要在执行工具时从工具流式传输更新，可以使用 get_stream_writer。
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    # stream any arbitrary data
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="deepseek-chat",  # 使用聊天模型
    tools=[get_weather],
)

for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
        stream_mode="custom"
):
    print(chunk)