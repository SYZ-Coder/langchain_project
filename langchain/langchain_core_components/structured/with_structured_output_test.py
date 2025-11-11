# 安装必要库
# pip install langchain-openai pydantic

# 以下是一个可以直接运行的完整示例，它演示了如何使用推荐的with_structured_output方法来创建一个结构化输出的问答程序。
# 运行这个Demo，你会直接得到一个格式完美的QuizQuestion对象，可以直接访问它的各个属性，极大地简化了后续处理流程。

import os

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field # BaseModel, Field: 来自 pydantic 库，用于定义结构化数据模型


# 1. 定义我们希望得到的精确输出结构
class QuizQuestion(BaseModel): # 定义一个名为 QuizQuestion 的 Pydantic 数据模型：
    # 继承自 BaseModel
    # 包含三个字段：
    # question: 字符串类型，表示题目内容
    # correct_answer: 正确答案字符串
    # wrong_answers: 错误答案列表，每个元素为字符串
    # 使用 Field(...) 添加字段描述信息，有助于 LLM 更好理解输出结构要求
    question: str = Field(description="生成的测验问题")
    correct_answer: str = Field(description="问题的正确答案")
    wrong_answers: list[str] = Field(description="三个错误选项的列表")

# 2. 初始化模型并启用结构化输出
model = ChatDeepSeek(
    model="deepseek-chat",  # 使用聊天模型
    api_key="",  # 替换为你的密钥
    temperature=0.7
)
# 调用 with_structured_output() 方法指定输出结构为 QuizQuestion 类型：
# 这使得 LLM 输出必须符合该结构
# 返回一个新的 LLM 对象 structured_llm
structured_llm = model.with_structured_output(QuizQuestion)

# 3. 构建提示词并调用模型
# 设置输入内容和提示语句：
# user_input: 用户关心的主题
# prompt: 向 LLM 发送的具体指令模板
user_input = "关于pydantic的基础知识"
prompt = f"请根据以下主题，生成一个多项选择题：{user_input}"

# 直接获取结构化的QuizQuestion对象！
# 调用模型执行推理任务：
# 将 prompt 输入给封装好的结构化 LLM (structured_llm)
# 得到的结果自动解析成 QuizQuestion 类实例赋值给 quiz_data
quiz_data = structured_llm.invoke(prompt)

# 4. 使用结构化后的数据
print("【生成的问题】")
print(quiz_data.question)
print("\n【正确答案】")
print(f"A. {quiz_data.correct_answer}")
print("\n【错误答案】")
for i, wrong_answer in enumerate(quiz_data.wrong_answers, start=1):
    print(f"{chr(66+i)}. {wrong_answer}") # 66是'B'的ASCII码

# 你也可以轻松地将它转为字典或JSON，用于存储或API传输
# 提供备选方案说明如何进一步序列化输出结果：
# .dict(): 将 Pydantic 对象转化为 Python 字典
# .json(): 将其转为 JSON 字符串形式
print("\n【quiz_dict】")
quiz_dict = quiz_data.model_dump()
print(quiz_dict)
import json;
print("\n【quiz_json】")
quiz_json = quiz_data.model_dump_json()
print(quiz_json)