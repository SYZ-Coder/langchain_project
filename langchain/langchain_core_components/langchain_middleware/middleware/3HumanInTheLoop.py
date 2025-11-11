from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware

from langchain.tools import tool
from typing import List, Optional

# 邮件数据模拟存储
email_database = [
    {"id": 1, "subject": "会议邀请", "sender": "boss@company.com", "content": "明天下午3点会议室A开会"},
    {"id": 2, "subject": "项目进度", "sender": "colleague@company.com", "content": "项目已完成80%"},
]

@tool
def read_email_tool(email_id: Optional[int] = None) -> str:
    """
    读取邮件工具
    :param email_id: 邮件ID，如果为None则读取所有邮件
    :return: 邮件内容
    """
    if email_id:
        email = next((e for e in email_database if e["id"] == email_id), None)
        if email:
            return f"邮件ID: {email['id']}\n主题: {email['subject']}\n发件人: {email['sender']}\n内容: {email['content']}"
        else:
            return "未找到指定邮件"
    else:
        result = "邮件列表:\n"
        for email in email_database:
            result += f"- ID: {email['id']}, 主题: {email['subject']}, 发件人: {email['sender']}\n"
        return result

@tool
def send_email_tool(recipient: str, subject: str, content: str) -> str:
    """
    发送邮件工具
    :param recipient: 收件人邮箱
    :param subject: 邮件主题
    :param content: 邮件内容
    :return: 发送结果
    """
    # 模拟发送邮件
    new_email = {
        "id": len(email_database) + 1,
        "subject": subject,
        "sender": "current_user@company.com",
        "content": content
    }
    email_database.append(new_email)
    return f"邮件已发送至 {recipient}，主题: {subject}"

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
    tools=[read_email_tool, send_email_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # 要求对发送邮件进行批准、编辑或拒绝
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                # 自动批准读取邮件
                "read_email_tool": False,
            }
        ),
    ],
)


# # 触发 read_email_tool（不会中断，自动执行）
# result1 = agent.invoke({
#     "messages": [{"role": "user", "content": "请读取所有邮件"}]
# })
# print("read_email_tool"+result1)
# 触发 send_email_tool（会中断，等待人工确认）
result2 = agent.invoke({
    "messages": [{"role": "user", "content": "请发送邮件给boss@company.com，主题是汇报，内容是项目已完成"}]
})
print(result2)