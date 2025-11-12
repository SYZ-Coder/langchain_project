from langchain.agents.middleware import wrap_model_call
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable
load_dotenv()  # 加载 .env 文件中的环境变量

class DynamicModelMiddleware(AgentMiddleware):
    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # Use different model based on conversation length
        if len(request.messages) > 10:
            print("使用模型deepseek-math")
            request.model = init_chat_model(    "deepseek-chat",
                                                temperature=0.7,
                                                timeout=15,
                                                max_tokens=2000)
        else:

            print("使用模型deepseek-coder")
            request.model = init_chat_model(    "deepseek-coder",
                                                temperature=0.3,
                                                timeout=15,
                                                max_tokens=2000)
        return handler(request)

# 1. 定义输出结构
class ContactInfo(BaseModel):
    """联系人信息"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱")
    phone: str = Field(description="电话")
    status: str = Field(description="状态")

# 2. Wrap-style: retry logic
@wrap_model_call
def retry_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")

# 3. 创建代理
agent = create_agent(
    model="deepseek-chat",  # 使用聊天模型
    tools=[],  # 可选工具
    middleware=[DynamicModelMiddleware()], #使用自定义中间件
    response_format=ContactInfo  # 指定输出格式
)

# 4. 使用代理
result = agent.invoke({
    "messages": [{"role": "user", "content": "你是一个编程小助手"},
                 {"role": "user", "content": "请实现一个抽奖小游戏代码，语言java"}]
})

print(result["messages"])
# 输出：ContactInfo(name='张三', email='zhangsan@example.com', phone='13800138000')