import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

load_dotenv()

# 定义结构化输出 schema
class Person(BaseModel):
    """个人信息"""
    name: str = Field(description="用户姓名")
    age: int = Field(description="用户年龄", ge=0, le=150)
    email: str = Field(description="邮箱地址")
    hobbies: list[str] = Field(description="兴趣爱好列表")

class Company(BaseModel):
    """公司信息"""
    name: str = Field(description="公司名称")
    industry: str = Field(description="所属行业")
    employee_count: int = Field(description="员工数量")
    founded_year: int = Field(description="成立年份")

def demo_manual_strategy():
    """演示手动设置结构化输出策略"""

    # 初始化模型
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 测试文本
    test_text = """
    用户信息：张三，28岁，邮箱是zhangsan@email.com。
    他的爱好包括篮球、阅读和旅行。他在一家科技公司工作。
    """

    print("=" * 50)
    print("LangChain 结构化输出策略演示")
    print("=" * 50)

    # 方法1：直接使用 method 参数
    print("\n1. 使用 method='tool_calling' 策略:")
    try:
        structured_model = model.with_structured_output(
            Person,
            method="function_calling"
        )
        result = structured_model.invoke(test_text)
        print(f"结果: {result}")
        print(f"类型: {type(result)}")
    except Exception as e:
        print(f"错误: {e}")

    # 方法2：使用配置字典
    print("\n2. 使用配置字典策略:")
    try:
        config = {"structured_output_method": "function_calling"}
        structured_model = model.with_structured_output(Person)
        result = structured_model.invoke(test_text, config=config)
        print(f"结果: {result}")
    except Exception as e:
        print(f"错误: {e}")

    # 方法3：使用 RunnableConfig
    print("\n3. 使用 RunnableConfig 策略:")
    try:
        config = RunnableConfig(
            configurable={
                "structured_output_method": "tool_calling"
            }
        )
        structured_model = model.with_structured_output(Person)
        result = structured_model.invoke(test_text, config=config)
        print(f"结果: {result}")
    except Exception as e:
        print(f"错误: {e}")

    # 方法4：比较不同策略
    print("\n4. 比较不同输出方法:")
    methods = ["tool_calling", "json_mode"]

    for method in methods:
        print(f"\n尝试方法: {method}")
        try:
            structured_model = model.with_structured_output(
                Person,
                method=method
            )
            result = structured_model.invoke(test_text)
            print(f"  成功 - 姓名: {result.name}, 年龄: {result.age}")
            print(f"  邮箱: {result.email}, 爱好: {result.hobbies}")
        except Exception as e:
            print(f"  失败: {e}")

def demo_multiple_schemas():
    """演示多个schema的切换"""

    print("\n" + "=" * 50)
    print("多 Schema 切换演示")
    print("=" * 50)

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 公司信息文本
    company_text = """
    Apple Inc. 是一家科技公司，主要专注于消费电子、软件和在线服务。
    公司成立于1976年，目前拥有约16万名员工。
    """

    # 创建两个不同的结构化输出
    person_model = model.with_structured_output(
        Person,
        method="tool_calling"
    )

    company_model = model.with_structured_output(
        Company,
        method="tool_calling"
    )

    # 分别调用
    print("提取个人信息:")
    person_result = person_model.invoke("John, 35岁, john@test.com, 喜欢音乐和运动")
    print(f"Person: {person_result}")

    print("\n提取公司信息:")
    company_result = company_model.invoke(company_text)
    print(f"Company: {company_result}")

def demo_error_handling():
    """演示错误处理"""

    print("\n" + "=" * 50)
    print("错误处理演示")
    print("=" * 50)

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 不完整的文本
    incomplete_text = "这个人叫李四"

    try:
        structured_model = model.with_structured_output(
            Person,
            method="tool_calling"
        )
        result = structured_model.invoke(incomplete_text)
        print(f"结果: {result}")
    except Exception as e:
        print(f"捕获到错误: {e}")
        print("建议：提供更完整的输入信息")

def demo_advanced_config():
    """演示高级配置"""

    print("\n" + "=" * 50)
    print("高级配置演示")
    print("=" * 50)

    # 带更多模型参数的配置
    advanced_config = RunnableConfig(
        configurable={
            "structured_output_method": "tool_calling",
        }
    )

    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=1000
    )

    structured_model = model.with_structured_output(
        Person,
        method="tool_calling"
    )

    complex_text = """
    王五，42岁，资深软件工程师。
    邮箱：wangwu@company.org
    爱好：编程（Python、JavaScript）、开源项目贡献、技术博客写作、徒步旅行。
    他还喜欢阅读科技新闻和参加技术会议。
    """

    result = structured_model.invoke(complex_text, config=advanced_config)
    print(f"复杂信息提取结果:")
    print(f"姓名: {result.name}")
    print(f"年龄: {result.age}")
    print(f"邮箱: {result.email}")
    print(f"爱好: {result.hobbies}")

if __name__ == "__main__":
    # 运行所有演示
    demo_manual_strategy()
    demo_multiple_schemas()
    demo_error_handling()
    demo_advanced_config()

    print("\n" + "=" * 50)
    print("演示完成!")
    print("=" * 50)