from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
load_dotenv()  # 加载 .env 文件中的环境变量

# 方式一
agent = create_agent(
    "deepseek-chat",
    tools=[]
)


# 方式二 灵活控制模型
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=1000,
    timeout=30
    # ... (other params)
)
agent1 = create_agent(model, tools=[])

response = agent.invoke({
    "messages": [{"role": "user", "content": "提取联系人信息：张三，zhangsan@example.com，13800138000"}]

})

print("response",response)

response1 = agent1.invoke({
    "messages": [{"role": "user", "content": "提取联系人信息：张三，zhangsan@example.com，13800138000"}]

})

print("response1",response1)