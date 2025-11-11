from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

prompt_template = """
{input}
{agent_scratchpad}
"""

# 初始化 DeepSeek 模型
# llm = ChatDeepSeek(
#     model="deepseek-chat",
#     temperature=0.7
# )

llm = init_chat_model(
    "deepseek-chat",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)


# 确保Prompt模板中包含agent_scratchpad变量
prompt_template = """You are a helpful assistant.
Human: {input}
{agent_scratchpad}

Assistant:"""

prompt = PromptTemplate.from_template(prompt_template)

# 创建工具列表（如果没有实际工具，使用空列表）
tools = []


# 创建 AgentExecutor 并添加中间件
agent = create_agent(
    model=llm,  # 使用聊天模型
    tools=tools,
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=1,
            run_limit=2,
            exit_behavior="error",
        ),
    ]
)



from langchain_core.messages import HumanMessage
if __name__ == "__main__":
    # 检查 agent 是否包含中间件
    print(f"Agent middlewares: {getattr(agent, 'ModelCallLimitMiddleware', 'Not found')}")

    # 测试限流
    for i in range(5):
        try:
            response = agent.invoke({
                "messages": [HumanMessage(content=f"这是第{i+1}次调用")]
            })
            print(f"调用 {i+1}: 成功")
        except Exception as e:
            print(f"调用 {i+1}: 被限流 - {str(e)}")