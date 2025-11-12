from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
load_dotenv()  # 加载 .env 文件中的环境变量
# 动态模型在运行基于当前州和上下文。这可以实现复杂的路由逻辑和成本优化。
# 要使用动态模型，请使用修改请求中模型的@wrap_model_call装饰器创建中间件：


basic_model = ChatDeepSeek(model="deepseek-chat")
advanced_model = ChatDeepSeek(model="deepseek-coder")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    return handler(request)

agent = create_agent(
    model=basic_model,  # Default model
    tools=[],
    middleware=[dynamic_model_selection]
)
response = agent.invoke({
    "messages": [{"role": "user", "content": "提取联系人信息：张三，zhangsan@example.com，13800138000"}]

})

print("response1",response)