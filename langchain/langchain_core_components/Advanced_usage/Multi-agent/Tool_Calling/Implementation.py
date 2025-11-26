from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
load_dotenv()  # 加载 .env 文件中的环境变量

subagent1 = create_agent(model="deepseek-chat", tools=[])

@tool(
    "subagent1_name",
    description="subagent1_description"
)
def call_subagent1(query: str):
    print("执行子代理工具")
    result = subagent1.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

agent = create_agent(model="deepseek-chat", tools=[call_subagent1])

response = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "请使用 subagent1_name 工具来处理这个问题"
    },{
        "role": "user",
        "content": "今天你好吗"
    }]
})

print("response",response)
# 在这种模式中：
# 当主智能体确定任务与子智能体描述相符时，调用该任务。call_subagent1
# 子智能体独立运行并返回结果。
# 主代理接收到结果并继续编排。