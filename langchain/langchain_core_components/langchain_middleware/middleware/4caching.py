from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量
LONG_PROMPT = """
Please be a helpful assistant.

<Lots more context ...>
"""
# 通过使用 Anthropic 模型，缓存重复的提示词前缀以降低成本。
# 初始化 DeepSeek 模型实例 主要开发了 Claude 系列模型   下面演示无效
# llm = ChatDeepSeek(
#     model="deepseek-chat",  # 使用聊天模型
#     api_key="",  # 替换为你的密钥
#     temperature=0.7
# )
from langchain.chat_models import init_chat_model



agent = create_agent(
    model="deepseek-chat",  # 使用聊天模型
    system_prompt=LONG_PROMPT,
    middleware=[AnthropicPromptCachingMiddleware(ttl="5m")],
)

# 缓存存储
agent.invoke({"messages": [HumanMessage("Hi, my name is Bob")]})

# 缓存命中，系统提示词被缓存
agent.invoke({"messages": [HumanMessage("What's my name?")]})


# AnthropicPromptCachingMiddleware缓存中间件只支持Anthropic模型，但代码中使用了DeepSeek模型，导致类型不匹配警告。