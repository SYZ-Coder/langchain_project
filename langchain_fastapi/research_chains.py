# 调研链定义

from langchain_core.prompts import ChatPromptTemplate  # ChatPromptTemplate: 用于创建聊天提示模板
from langchain_core.output_parsers import StrOutputParser # StrOutputParser: 将模型输出解析为字符串
from langchain_core.runnables import RunnableLambda, RunnableParallel # RunnableLambda, RunnableParallel: LangChain 的可运行组件
from langchain_community.chat_models import ChatDeepSeek, ChatOpenAI # ChatDeepSeek, ChatOpenAI: DeepSeek 和 OpenAI 的聊天模型接口
from config import settings # settings: 从配置文件导入设置参数
import json  # json, re: Python 标准库，用于 JSON 处理和正则表达式
import re

# 用于创建多种针对 DeepSeek AI 模型的研究分析链。这些链可以帮助分析模型能力、进行性能测试、成本分析等。
class DeepSeekResearchChains:
    """DeepSeek 调研链集合"""

    def __init__(self):    # 定义__init__ 方法 初始化 DeepSeek 模型实例   用法：创建类实例时自动调用，配置模型参数
        # 初始化模型
        self.deepseek_model = ChatDeepSeek(
            model=settings.DEEPSEEK_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            temperature=0.7
        )
        # 逐行解析:
        # 创建 deepseek_model 属性，实例化 ChatDeepSeek 模型
        # 使用配置中的模型名称、API 密钥
        # 设置温度参数为 0.7（控制输出随机性）


    #作用: 创建一个分析 DeepSeek 模型能力的链条 用法: 调用此方法获取能力分析链，然后传入问题进行分析
    def get_capability_analysis_chain(self):
        """能力分析链"""
        prompt = ChatPromptTemplate.from_template("""
        请从以下维度分析 DeepSeek 模型的能力：

        用户问题：{question}
        
        请按以下结构分析：
        1. **技术能力**：模型在处理这类问题时的技术表现
        2. **回答质量**：准确性、相关性和完整性
        3. **响应特点**：回答风格、创造性、逻辑性
        4. **改进建议**：可能的改进空间
        
        请提供详细的分析报告。
        """)

        return prompt | self.deepseek_model | StrOutputParser()
      # 定义提示模板，包含能力分析的四个维度
      # 使用管道操作符 | 组合 prompt、deepseek_model 和 StrOutputParser
      # 返回组合后的可执行链条
    def get_comparison_chain(self):
        """模型对比链"""
        comparison_prompt = ChatPromptTemplate.from_template("""
        请对比分析 DeepSeek 和 OpenAI 模型在以下问题上的表现：

        问题：{question}
        
        DeepSeek 回答：{deepseek_response}
        OpenAI 回答：{openai_response}
        
        请从以下角度进行对比分析：
        1. **回答质量**：准确性、深度、完整性
        2. **响应风格**：语言风格、专业性、亲和力
        3. **技术特点**：推理能力、知识广度、创造性
        4. **适用场景**：各自适合的应用场景
        5. **性价比**：成本效益分析
        
        提供详细的对比报告。
        """)

        return comparison_prompt | self.deepseek_model | StrOutputParser()
   # 逐行解析:
    # 创建对比分析的提示模板
    # 模板包含问题和两个模型的回答作为输入
    # 定义五个对比维度
    # 返回组合后的对比分析链
    def get_performance_test_chain(self):
        """性能测试链"""
        test_prompt = ChatPromptTemplate.from_template("""
        请对 DeepSeek 模型进行以下类型的性能测试：

        测试类型：{test_type}
        测试内容：{content}
        
        请评估：
        1. **响应时间感知**：响应速度的主观感受
        2. **回答质量**：准确性、相关性和实用性
        3. **稳定性**：回答的一致性和可靠性
        4. **特殊能力**：在处理这类任务时的独特优势
        
        提供详细的测试报告。
        """)

        return test_prompt | self.deepseek_model | StrOutputParser()

    def get_cost_analysis_chain(self):
        """成本分析链"""
        cost_prompt = ChatPromptTemplate.from_template("""
        请分析 DeepSeek 模型的成本效益：

        使用场景：{scenario}
        预估调用量：{volume}
        
        请分析：
        1. **定价结构**：API 调用成本
        2. **性价比**：与同类模型的成本对比
        3. **适用规模**：适合的个人/企业使用规模
        4. **成本优化建议**：降低使用成本的策略
        
        提供详细的成本分析报告。
        """)

        return cost_prompt | self.deepseek_model | StrOutputParser()
       # 作用: 创建性能测试链 用法: 测试 DeepSeek 模型在特定任务下的性能表现 逐行解析:
    # 定义性能测试提示模板
    # 接收测试类型和内容作为参数
    # 定义四个评估维度
    # 返回性能测试链
    def get_technical_specs_chain(self):
        """技术规格调研链"""
        specs_prompt = ChatPromptTemplate.from_template("""
        请调研 DeepSeek 模型的技术规格：

        调研重点：{focus_area}
        
        请收集以下信息：
        1. **模型架构**：技术架构和设计特点
        2. **训练数据**：数据规模和质量
        3. **性能指标**：官方发布的性能数据
        4. **特色功能**：独特的功能和技术创新
        5. **限制条件**：使用限制和注意事项
        
        提供详细的技术规格报告。
        """)

        return specs_prompt | self.deepseek_model | StrOutputParser()

# 创建链实例
research_chains = DeepSeekResearchChains()

# 作用: 创建 DeepSeekResearchChains 类的全局实例 用法: 在其他模块中导入 research_chains 来使用各种分析链