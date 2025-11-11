from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
import json # 需要手动解析JSON

# 方法2：使用JSON模式 (JSON Mode)
# 这种方法通过模型参数强制模型输出JSON字符串，然后你需要手动将其解析为Python对象。
# 注意：使用此方法时，你的提示词最好明确要求模型返回JSON，否则效果可能不理想。

model = ChatDeepSeek(
    model="deepseek-chat",  # 使用聊天模型
    api_key="",  # 替换为你的密钥
    temperature=0.7,
    model_kwargs={
        "response_format": { "type": "json_object" } # 启用JSON模式
    }

)
ai_msg = model.invoke("返回一个JSON对象，包含'random_ints'键，其值是10个0-99的随机整数。")

# 模型返回的是字符串，需要手动解析
json_string = ai_msg.content
json_object = json.loads(json_string)

print(json_object["random_ints"]) # 输出: [23, 47, 89, ...]