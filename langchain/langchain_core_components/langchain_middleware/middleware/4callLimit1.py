from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv

load_dotenv()

# 创建一个简单的工具
def simple_tool(query: str) -> str:
    return f"工具响应: {query}"

tools = [
    Tool(
        name="simple_tool",
        func=simple_tool,
        description="一个简单的工具"
    )
]

model = init_chat_model(
    "deepseek-chat",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)


agent = create_agent(
    model=model,
    tools=tools,  # 提供实际工具
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=3,
            run_limit=2,
            exit_behavior="error",
        ),
    ],
)

if __name__ == "__main__":
    print("=== 使用工具测试 ===")

    for i in range(5):
        try:
            response = agent.invoke(
                {"messages": [HumanMessage(content=f"使用工具处理消息{i+1}")]},
                config={"configurable": {"thread_id": "tool_test"}}
            )
            print(f"调用 {i+1}: 成功")
            # 打印响应结构
            if 'messages' in response:
                last_msg = response['messages'][-1]
                print(f"最后消息: {last_msg.content[:100]}...")
        except Exception as e:
            print(f"调用 {i+1}: 失败 - {e}")