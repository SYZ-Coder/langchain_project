from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware



load_dotenv()  # 加载 .env 文件中的环境变量
# 1. 定义输出结构
class ContactInfo(BaseModel):
    """联系人信息"""
    name: str = Field(description="姓名")
    email: str = Field(description="邮箱")
    phone: str = Field(description="电话")

# 初始化 DeepSeek 模型实例
llm = ChatDeepSeek(
    model="deepseek-chat",  # 使用聊天模型
    api_key="",  # 替换为你的密钥
    temperature=0.7
)

# 2. 创建代理
agent = create_agent(
    model="deepseek-chat",  # 使用聊天模型
    tools=[],  # 可选工具
    # 通过在创建 agent 时传入 middleware 参数，将这两个中间件应用到代理中。它们会在代理处理请求的过程中自动生效：
    # SummarizationMiddleware() 会自动管理对话历史的长度
    # HumanInTheLoopMiddleware() 会在适当的时候暂停执行，等待人工介入
    # 支持的中间件类型
    # 1. 内置中间件
    # SummarizationMiddleware: 对话摘要中间件
    # HumanInTheLoopMiddleware: 人工介入中间件
    # 2. 自定义中间件
    # 可以继承 BaseMiddleware 类创建自定义中间件，实现：
    # on_request: 请求前处理
    # on_response: 响应后处理
    # 3. 中间件链
    # 支持多个中间件组合使用，按顺序执行：middleware=[Middleware1(), Middleware2(), HumanInTheLoopMiddleware(interrupt_on="final_answer")]
    middleware=[SummarizationMiddleware(model=llm), HumanInTheLoopMiddleware(
        interrupt_on={
            # 要求对发送邮件进行批准、编辑或拒绝
            "send_email_tool": {
                "allowed_decisions": ["approve", "edit", "reject"],
            },
            # 自动批准读取邮件
            "read_email_tool": False,
        }
    )],
)

# 3. 使用代理
result = agent.invoke({
    "messages": [{"role": "user", "content": "提取联系人信息：张三，zhangsan@example.com，13800138000"}]
})

print(result["structured_response"])
# 输出：ContactInfo(name='张三', email='zhangsan@example.com', phone='13800138000')