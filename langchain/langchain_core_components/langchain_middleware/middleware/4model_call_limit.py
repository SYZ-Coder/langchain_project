from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.prompts import  PromptTemplate
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
load_dotenv()  # 加载 .env 文件中的环境变量

prompt_template = """
{input}
{agent_scratchpad}
"""


# 确保Prompt模板中包含agent_scratchpad变量
prompt_template = """You are a helpful assistant.
Human: {input}
{agent_scratchpad}

Assistant:"""

prompt = PromptTemplate.from_template(prompt_template)

# 创建工具列表（如果没有实际工具，使用空列表）
tools = []


agent = create_agent(
    model="deepseek-chat",  # 使用聊天模型
    tools=[],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=1,
            run_limit=1,
            exit_behavior="error",
        )
    ]
)




if __name__ == "__main__":
    # 检查 agent 是否包含中间件
    print(f"Agent info: {agent}")
    print(f"Agent type: {type(agent)}")

    # 测试限流
    for i in range(5):
        try:
            response = agent.invoke({
                "messages": [HumanMessage(content=f"这是第{i+1}次调用")]
            })
            print(f"调用 {i+1}: 成功")
        except Exception as e:
            print(f"调用 {i+1}: 被限流 - {str(e)}")