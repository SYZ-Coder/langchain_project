from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field
from typing import List
load_dotenv()  # 加载 .env 文件中的环境变量
class BusinessCard(BaseModel):
    """从文本中提取的名片信息"""
    name: str = Field(description="姓名")
    title: str = Field(description="职位")
    company: str = Field(description="公司名称")
    email: str = Field(description="邮箱")
    phone: str = Field(description="电话号码")
    skills: List[str] = Field(description="提到的技能列表")

def analyze_strategy(model, model_name):
    print(f"\n=== 测试模型: {model_name} ===")

    try:
        structured_llm = model.with_structured_output(BusinessCard)

        # 查看使用的策略
        strategy = getattr(structured_llm, '_structured_output_method', 'unknown')
        print(f"使用的策略: {strategy}")

        result = structured_llm.invoke(
            "我是李四，高级工程师，在ABC科技工作。"
            "邮箱lisi@abctech.com，电话13800138000。"
            "擅长Python、机器学习和云计算。"
        )
        print(f"提取结果: {result}")

    except Exception as e:
        print(f"错误: {e}")
if __name__ == "__main__":
    # 测试不同模型
    analyze_strategy(ChatDeepSeek(model="deepseek-chat"), "deepseek-chat")
    # 可以继续测试其他支持工具调用的模型