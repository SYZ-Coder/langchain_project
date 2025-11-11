from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
from datetime import datetime
from enum import Enum

# 1. AI输出结构化 - 最核心的应用
class AIAnalysisResult(BaseModel):
    """AI分析结果结构化输出"""
    sentiment: str = Field(description="情感倾向: positive/negative/neutral")
    confidence: float = Field(ge=0, le=1, description="置信度")
    key_phrases: List[str] = Field(description="关键短语列表")
    topics: List[str] = Field(description="涉及主题")
    summary: str = Field(description="内容摘要")

    @field_validator('sentiment')
    def validate_sentiment(cls, v):
        allowed = ['positive', 'negative', 'neutral']
        if v not in allowed:
            raise ValueError(f'情感倾向必须是: {allowed}')
        return v

def ai_structured_output_demo():
    """AI输出结构化 - LangChain集成"""
    print("🎯 1. AI输出结构化 (LangChain核心应用)")
    print("=" * 50)

    # 模拟AI的原始输出（通常是不可预测的文本）
    raw_ai_output = """
    这是一段关于产品评论的分析：
    情感：积极正面
    置信度：0.92
    关键点：质量很好, 送货快, 包装精美
    主题：购物体验, 产品质量
    摘要：用户对产品质量和配送服务非常满意
    """

    # 使用Pydantic结构化后的AI输出
    structured_output = AIAnalysisResult(
        sentiment="positive",
        confidence=0.92,
        key_phrases=["质量很好", "送货快", "包装精美"],
        topics=["购物体验", "产品质量"],
        summary="用户对产品质量和配送服务非常满意"
    )

    print("📊 结构化后的AI分析结果:")
    print(f"情感: {structured_output.sentiment}")
    print(f"置信度: {structured_output.confidence:.2f}")
    print(f"关键短语: {', '.join(structured_output.key_phrases)}")
    print(f"可编程使用: if result.confidence > 0.8: process_high_confidence()")

# 2. 智能对话系统
class DialogState(str, Enum):
    GREETING = "greeting"
    QUESTION = "question"
    COMPLAINT = "complaint"
    SUPPORT = "support"
    CLOSING = "closing"

class ConversationTurn(BaseModel):
    """对话轮次结构化"""
    user_input: str = Field(description="用户输入")
    intent: DialogState = Field(description="对话意图")
    entities: Dict[str, Any] = Field(description="提取的实体")
    response: str = Field(description="AI回复")
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_chat_history(self):
        """转换为聊天历史格式"""
        return {
            "role": "user",
            "content": self.user_input,
            "intent": self.intent.value,
            "entities": self.entities,
            "timestamp": self.timestamp.isoformat()
        }

def conversation_system_demo():
    """智能对话系统"""
    print("\n💬 2. 智能对话系统")
    print("=" * 50)

    # AI分析用户输入后的结构化结果
    conversation = ConversationTurn(
        user_input="我的订单12345为什么还没发货？",
        intent=DialogState.COMPLAINT,
        entities={
            "order_id": "12345",
            "issue_type": "delivery_delay",
            "urgency": "high"
        },
        response="非常抱歉给您带来不便，我立即为您查询订单12345的状态。"
    )

    print("🗣️ 对话分析结果:")
    print(f"用户意图: {conversation.intent.value}")
    print(f"提取实体: {conversation.entities}")
    print(f"聊天历史: {conversation.to_chat_history()}")

# 3. 内容生成和审核
class ContentGenerationRequest(BaseModel):
    """内容生成请求"""
    topic: str = Field(description="生成主题")
    style: str = Field(description="写作风格", examples=["专业", "轻松", "正式"])
    length: str = Field(description="内容长度", examples=["short", "medium", "long"])
    keywords: List[str] = Field(description="需要包含的关键词")
    avoid_topics: List[str] = Field(description="需要避免的话题")

    @field_validator('length')
    def validate_length(cls, v):
        if v not in ['short', 'medium', 'long']:
            raise ValueError('长度必须是: short/medium/long')
        return v

class GeneratedContent(BaseModel):
    """生成的内容结果"""
    title: str = Field(description="标题")
    content: str = Field(description="正文内容")
    quality_score: float = Field(ge=0, le=1, description="质量评分")
    readability_level: str = Field(description="可读性级别")
    suggested_improvements: List[str] = Field(description="改进建议")

def content_generation_demo():
    """AI内容生成"""
    print("\n📝 3. AI内容生成与审核")
    print("=" * 50)

    # 内容生成请求
    request = ContentGenerationRequest(
        topic="人工智能在教育中的应用",
        style="专业",
        length="medium",
        keywords=["AI", "教育", "个性化学习", "教学效率"],
        avoid_topics=["数据隐私", "失业风险"]
    )

    # AI生成的内容（结构化输出）
    generated = GeneratedContent(
        title="人工智能如何变革现代教育体系",
        content="AI技术通过个性化学习路径...",
        quality_score=0.88,
        readability_level="大学",
        suggested_improvements=["增加具体案例", "补充数据支持"]
    )

    print("🎨 内容生成请求:")
    print(f"主题: {request.topic}, 风格: {request.style}")
    print(f"生成结果 - 质量评分: {generated.quality_score}")
    print(f"改进建议: {generated.suggested_improvements}")

# 4. 数据标注和训练数据管理
class TrainingExample(BaseModel):
    """训练数据样本"""
    text: str = Field(description="原始文本")
    labels: Dict[str, Any] = Field(description="标注标签")
    metadata: Dict[str, Any] = Field(description="元数据")
    created_by: str = Field(description="标注人员")
    created_at: datetime = Field(default_factory=datetime.now)

    def to_training_format(self, format_type: str = "huggingface"):
        """转换为不同训练框架的格式"""
        if format_type == "huggingface":
            return {
                "text": self.text,
                "labels": self.labels,
                "metadata": self.metadata
            }
        elif format_type == "spacy":
            return (self.text, {"entities": self.labels})

class DatasetStatistics(BaseModel):
    """数据集统计信息"""
    total_examples: int = Field(ge=0, description="总样本数")
    label_distribution: Dict[str, int] = Field(description="标签分布")
    average_text_length: float = Field(ge=0, description="平均文本长度")
    data_quality_score: float = Field(ge=0, le=1, description="数据质量评分")

def data_annotation_demo():
    """AI数据标注"""
    print("\n🏷️ 4. 数据标注和训练数据管理")
    print("=" * 50)

    # 标注数据样本
    example = TrainingExample(
        text="苹果公司发布了新款iPhone，搭载了更强大的AI芯片。",
        labels={
            "entities": {
                "ORG": ["苹果公司"],
                "PRODUCT": ["iPhone", "AI芯片"]
            },
            "sentiment": "positive"
        },
        metadata={"domain": "科技", "language": "zh"},
        created_by="annotator_001"
    )

    # 数据集统计
    stats = DatasetStatistics(
        total_examples=10000,
        label_distribution={"positive": 6000, "negative": 3000, "neutral": 1000},
        average_text_length=45.6,
        data_quality_score=0.94
    )

    print("📊 训练数据管理:")
    print(f"样本标注: {example.labels}")
    print(f"HuggingFace格式: {example.to_training_format('huggingface')}")
    print(f"数据集统计: {stats.total_examples}个样本, 质量评分: {stats.data_quality_score}")

# 5. AI评估和监控
class ModelPerformance(BaseModel):
    """模型性能评估"""
    accuracy: float = Field(ge=0, le=1, description="准确率")
    precision: float = Field(ge=0, le=1, description="精确率")
    recall: float = Field(ge=0, le=1, description="召回率")
    f1_score: float = Field(ge=0, le=1, description="F1分数")
    inference_speed: float = Field(ge=0, description="推理速度(ms)")

    @property
    def is_production_ready(self):
        """判断是否达到生产标准"""
        return self.f1_score > 0.85 and self.inference_speed < 100

class AIPrediction(BaseModel):
    """AI预测结果"""
    input_data: Dict[str, Any] = Field(description="输入数据")
    prediction: Any = Field(description="预测结果")
    confidence: float = Field(ge=0, le=1, description="置信度")
    model_version: str = Field(description="模型版本")
    processing_time: float = Field(ge=0, description="处理时间(秒)")

def ai_evaluation_demo():
    """AI模型评估"""
    print("\n📈 5. AI模型评估和监控")
    print("=" * 50)

    # 模型性能评估
    performance = ModelPerformance(
        accuracy=0.92,
        precision=0.89,
        recall=0.94,
        f1_score=0.915,
        inference_speed=45.2
    )

    # AI预测结果
    prediction = AIPrediction(
        input_data={"text": "这个产品非常好用！"},
        prediction="positive",
        confidence=0.96,
        model_version="sentiment-v2.1",
        processing_time=0.12
    )

    print("🔍 模型评估:")
    print(f"F1分数: {performance.f1_score}, 生产就绪: {performance.is_production_ready}")
    print(f"预测结果: {prediction.prediction} (置信度: {prediction.confidence})")

# 6. 与LangChain深度集成
class LangChainStructuredOutput(BaseModel):
    """LangChain结构化输出模板"""
    analysis: str = Field(description="核心分析内容")
    reasoning: List[str] = Field(description="推理过程")
    confidence: float = Field(ge=0, le=1, description="分析置信度")
    sources: List[str] = Field(description="参考来源")
    limitations: List[str] = Field(description="分析局限性")

    def to_llm_prompt(self):
        """转换为LLM提示词格式"""
        return f"""
分析结果: {self.analysis}
推理过程: {'; '.join(self.reasoning)}
置信度: {self.confidence}
参考来源: {', '.join(self.sources)}
        """.strip()

def langchain_integration_demo():
    """LangChain深度集成"""
    print("\n🔗 6. 与LangChain深度集成")
    print("=" * 50)

    # 在LangChain中使用Pydantic确保AI输出质量
    structured_output = LangChainStructuredOutput(
        analysis="该评论表达了用户对产品质量的满意",
        reasoning=[
            "用户使用了'质量很好'等正面词汇",
            "提到了多个产品优点",
            "没有负面情绪词汇"
        ],
        confidence=0.93,
        sources=["情感词典", "产品知识库"],
        limitations=["无法确认用户的具体使用场景"]
    )

    print("🚀 LangChain结构化输出:")
    print(f"分析: {structured_output.analysis}")
    print(f"推理步骤: {len(structured_output.reasoning)}个")
    print(f"提示词格式:\n{structured_output.to_llm_prompt()}")

if __name__ == "__main__":
    # 运行所有AI领域应用演示
    ai_structured_output_demo()
    conversation_system_demo()
    content_generation_demo()
    data_annotation_demo()
    ai_evaluation_demo()
    langchain_integration_demo()

    print("\n" + "=" * 60)
    print("🎯 Pydantic在AI领域的核心价值总结:")
    print("  ✅ 结构化AI输出 - 让不可预测的文本变可编程数据")
    print("  ✅ 数据质量保证 - 自动验证和清洗AI生成内容")
    print("  ✅ 评估和监控 - 标准化性能指标和预测结果")
    print("  ✅ 训练数据管理 - 类型安全的标注和数据统计")
    print("  ✅ 生产就绪 - 构建可靠的企业级AI应用")
    print("  ✅ LangChain集成 - 结构化输出的核心基础设施")
    print("=" * 60)