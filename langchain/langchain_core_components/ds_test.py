from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

# 1. 初始化 DeepSeek 模型
llm = ChatDeepSeek(
    model="deepseek-chat",  # 使用聊天模型
    api_key="",  # 替换为你的密钥
    temperature=0.7
)

# 2. 创建提示词模板
prompt = ChatPromptTemplate.from_template(
    "你是一位{role}。请用{style}风格回答以下问题：{question}"
)

# 3. 创建链式调用
chain = prompt | llm

# 4. 调用模型
response = chain.invoke({
    "role": "历史学家",
    "style": "幽默风趣",
    "question": "秦始皇为什么统一六国？"
})

print(response.content)



