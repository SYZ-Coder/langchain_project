import requests # 导入 HTTP 请求库，用于发起网络请求
from dotenv import load_dotenv
from langchain.agents import create_agent #  导入创建智能代理的函数
from langchain.messages import HumanMessage # 导入表示人类用户消息的类
from langchain.tools import tool # 导入工具装饰器，用于创建可被代理调用的工具
from markdownify import markdownify # 导入 HTML 转 Markdown 的转换函数
load_dotenv()
# 整个系统实现了基于代理的 RAG（Retrieval-Augmented Generation）模式，能够根据用户查询动态获取相关文档并生成准确的技术答案。

ALLOWED_DOMAINS = ["https://langchain-ai.github.io/"] # 定义允许访问的域名白名单，限制安全范围
LLMS_TXT = 'https://langchain-ai.github.io/langgraph/llms.txt' # 定义文档索引文件的 URL 地址

# 使用 @tool 装饰器定义一个文档获取工具：
# 首先检查 URL 是否在允许的域名范围内
# 如果不在允许范围内，返回错误信息
# 使用 requests.get() 获取网页内容
# 使用 markdownify() 将 HTML 内容转换为 Markdown 格式返回
@tool
def fetch_documentation(url: str) -> str:
    """Fetch and convert documentation from a URL"""
    if not any(url.startswith(domain) for domain in ALLOWED_DOMAINS):
        return (
            "Error: URL not allowed. "
            f"Must start with one of: {', '.join(ALLOWED_DOMAINS)}"
        )
    response = requests.get(url, timeout=10.0)
    response.raise_for_status()
    return markdownify(response.text)


# 我们将获取 llms.txt 的内容，这样可以在不需要 LLM 请求的情况下提前完成
# 预先获取文档索引内容，避免运行时延迟
llms_txt_content = requests.get(LLMS_TXT).text

# System prompt for the agent
# 定义代理的行为准则：
# 明确代理的角色定位（Python 开发专家）
# 规定必须使用 fetch_documentation 工具查询文档的场景
# 强调只能访问允许的域名
# 要求答案清晰、简洁且技术准确
# 嵌入预加载的文档索引内容
system_prompt = f"""
你是一位专业的Python开发人员和技术助手。
你的主要职责是帮助用户解答有关LangGraph及相关工具的问题。

使用说明：

1. 如果用户提出你不熟悉的问题——或者涉及API使用、行为或配置的问题——你必须使用 [fetch_documentation](file://D:\spring_AI\langchain\projcet_github\langchain_project\langchain\langchain_core_components\Advanced_usage\Retrieval\Agentic_RAG.py#L12-L21) 工具来查阅相关文档。
2. 引用文档时，需清晰总结并包含内容中的相关上下文。
3. 不要使用允许域之外的任何URL。
4. 如果文档获取失败，请告知用户并凭借你的专业理解继续回答。

你可以从以下批准的来源访问官方文档：

{llms_txt_content}

在回答用户关于LangGraph的问题之前，你必须查阅文档以获得最新的文档资料。

你的答案应当清晰、简洁且技术上准确。
"""
# tools = [fetch_documentation]: 定义代理可用的工具列表
tools = [fetch_documentation]
# agent = create_agent(...): 创建代理实例
# 使用 gpt-3.5-turbo 模型
# 注册 fetch_documentation 工具
# 设置系统提示词
# 命名为 "Agentic RAG"
agent = create_agent(
    model="deepseek-chat",
    tools=tools,
    system_prompt=system_prompt,
    name="Agentic RAG",
)
# 向代理发起查询请求：
# 使用 HumanMessage 包装用户问题
# 调用 agent.invoke() 执行代理推理
# 查询内容是要求编写一个使用预构建 create react agent 的 LangGraph 代理示例
response = agent.invoke({
    'messages': [
        HumanMessage(content=(
            "Write a short example of a langgraph agent using the "
            "prebuilt create react agent. the agent should be able "
            "to look up stock pricing information."
        ))
    ]
})
# print(response['messages'][-1].content): 打印代理返回的最终答案内容
print(response['messages'][-1].content)