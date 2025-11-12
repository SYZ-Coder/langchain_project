from dotenv import load_dotenv
from langchain.agents import create_agent
load_dotenv()  # 加载 .env 文件中的环境变量

def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"

agent = create_agent(
    model="deepseek-chat",  # 使用聊天模型
    tools=[get_weather],
)
for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
        stream_mode="updates",
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")