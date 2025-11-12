# 您可以通过将流模式作为列表传递来指定多种流模式： stream_mode=["updates", "custom"]
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.config import get_stream_writer
load_dotenv()  # 加载 .env 文件中的环境变量

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="deepseek-chat",  # 使用聊天模型
    tools=[get_weather],
)

for stream_mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
        stream_mode=["updates", "custom"]
):
    print(f"stream_mode: {stream_mode}")
    print(f"content: {chunk}")
    print("\n")