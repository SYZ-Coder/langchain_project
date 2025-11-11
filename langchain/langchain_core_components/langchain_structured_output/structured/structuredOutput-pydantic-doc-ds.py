from pydantic import BaseModel, Field, validator, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage
import json

# 1. 定义DeepSeek客户端
def create_deepseek_client():
    """创建DeepSeek客户端"""
    model = ChatDeepSeek(
        model="deepseek-chat",
        api_key="",  # 请替换为你的真实密钥
        temperature=0.7,
        model_kwargs={
            "response_format": { "type": "json_object" }  # 启用JSON模式
        }
    )
    return model

# 2. 定义结构化输出模型
class CustomerServiceResponse(BaseModel):
    """客服AI响应结构"""
    issue_category: str = Field(description="问题分类",  json_schema_extra={"example": "billing/technical/sales"})
    solution: str = Field(description="解决方案")
    priority: str = Field(description="优先级", json_schema_extra={"example": "low/medium/high/urgent"})
    confidence: float = Field(ge=0, le=1, description="解决信心度")
    follow_up_questions: List[str] = Field(description="后续问题建议")

    @field_validator('priority')
    def validate_priority(cls, v):
        allowed = ['low', 'medium', 'high', 'urgent']
        if v not in allowed:
            raise ValueError(f'优先级必须是: {allowed}')
        return v
class ContentAnalysisResult(BaseModel):
    """内容分析结果"""
    sentiment: str = Field(description="情感倾向",   json_schema_extra={"example": "positive/negative/neutral"})
    key_points: List[str] = Field(description="关键要点")
    summary: str = Field(description="内容摘要")
    tone: str = Field(description="语气风格", json_schema_extra={"example": "formal/casual/enthusiastic"})
    word_count: int = Field(ge=0, description="字数统计")

class CodeReviewResult(BaseModel):
    """代码审查结果"""
    overall_quality: str = Field(description="整体质量", json_schema_extra={"example": "excellent/good/fair/poor"})
    issues: List[Dict] = Field(description="发现问题列表")
    suggestions: List[str] = Field(description="改进建议")
    complexity: str = Field(description="代码复杂度",   json_schema_extra={"example": "low/medium/hig"})
    security_concerns: List[str] = Field(description="安全顾虑")

class ProductReviewAnalysis(BaseModel):
    """产品评论分析"""
    overall_sentiment: str = Field(description="整体情感")
    product_ratings: Dict[str, float] = Field(description="各维度评分", example={"quality": 4.5, "price": 3.8})
    pros: List[str] = Field(description="优点列表")
    cons: List[str] = Field(description="缺点列表")
    recommendation: bool = Field(description="是否推荐")

# 3. 结构化输出处理器
class DeepSeekStructuredProcessor:
    def __init__(self):
        self.client = create_deepseek_client()

    def get_structured_response(self, user_prompt: str, response_model: BaseModel) -> BaseModel:
        """获取结构化响应"""

        # 构建系统提示词
        system_prompt = f"""
        你是一个专业的AI助手。请严格按照以下JSON格式返回响应，不要添加任何其他内容。
        
        JSON Schema:
        {response_model.schema_json()}
        
        要求：
        1. 只返回纯JSON格式的数据
        2. 不要包含任何解释性文字
        3. 确保所有字段都符合Schema定义
        4. 字段值要准确反映分析结果
        """

        try:
            # 发送消息到DeepSeek
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = self.client.invoke(messages)

            # 解析JSON响应
            content = response.content
            if isinstance(content, str):
                # 尝试解析JSON
                data = json.loads(content)
            else:
                data = content

            # 转换为Pydantic模型
            return response_model(**data)

        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始响应: {response.content}")
            raise
        except Exception as e:
            print(f"处理错误: {e}")
            raise

# 4. 业务场景演示
def demo_customer_service(processor: DeepSeekStructuredProcessor):
    """演示客服场景"""
    print("🎯 1. 智能客服场景")
    print("=" * 50)

    user_query = """
    我的订单号ORD-2024-00123已经付款3天了，但状态还是待发货。
    我急需这个商品，请帮我处理一下！
    """

    prompt = f"""
    请分析以下客户问题并提供结构化响应：
    
    客户问题：{user_query}
    """

    try:
        result = processor.get_structured_response(prompt, CustomerServiceResponse)

        print("🤖 DeepSeek客服分析结果:")
        print(f"   问题分类: {result.issue_category}")
        print(f"   解决方案: {result.solution}")
        print(f"   优先级: {result.priority}")
        print(f"   信心度: {result.confidence:.2f}")
        print(f"   后续问题: {result.follow_up_questions}")

        # 基于结构化结果执行业务逻辑
        if result.priority in ['high', 'urgent']:
            print("   🚨 高优先级问题，需要立即处理！")
        if result.confidence < 0.7:
            print("   ⚠️  低信心度，建议转人工客服")

    except Exception as e:
        print(f"错误: {e}")

def demo_content_analysis(processor: DeepSeekStructuredProcessor):
    """演示内容分析场景"""
    print("\n📊 2. 内容分析场景")
    print("=" * 50)

    content = """
    我们最新发布的AI产品获得了用户的高度评价！
    用户反馈界面友好、功能强大、响应速度快。
    特别是在处理复杂任务时表现优异，大大提升了工作效率。
    当然，也有一些用户建议增加更多的自定义选项。
    总体来看，这是一个成功的产品发布。
    """

    prompt = f"""
    请分析以下内容并提供结构化分析结果：
    
    {content}
    """

    try:
        result = processor.get_structured_response(prompt, ContentAnalysisResult)

        print("📈 DeepSeek内容分析结果:")
        print(f"   情感倾向: {result.sentiment}")
        print(f"   语气风格: {result.tone}")
        print(f"   字数统计: {result.word_count}")
        print(f"   内容摘要: {result.summary}")
        print(f"   关键要点:")
        for i, point in enumerate(result.key_points, 1):
            print(f"     {i}. {point}")

    except Exception as e:
        print(f"错误: {e}")

def demo_code_review(processor: DeepSeekStructuredProcessor):
    """演示代码审查场景"""
    print("\n💻 3. 代码审查场景")
    print("=" * 50)

    python_code = """
    def calculate_average(numbers):
        total = 0
        count = 0
        for num in numbers:
            total += num
            count += 1
        average = total / count
        return average
    
    def process_user_data(user_input):
        data = eval(user_input)
        return data
    """

    prompt = f"""
    请审查以下Python代码并提供结构化审查结果：
    
    ```python
    {python_code}
    ```
    """

    try:
        result = processor.get_structured_response(prompt, CodeReviewResult)

        print("🔍 DeepSeek代码审查结果:")
        print(f"   整体质量: {result.overall_quality}")
        print(f"   代码复杂度: {result.complexity}")
        print(f"   安全顾虑: {result.security_concerns}")
        print(f"   改进建议:")
        for i, suggestion in enumerate(result.suggestions, 1):
            print(f"     {i}. {suggestion}")
        print(f"   发现问题: {len(result.issues)}个")

    except Exception as e:
        print(f"错误: {e}")

def demo_product_review_analysis(processor: DeepSeekStructuredProcessor):
    """演示产品评论分析"""
    print("\n🛍️ 4. 产品评论分析场景")
    print("=" * 50)

    review = """
    我刚买了这款智能手机，使用了一周后来评价：
    优点：屏幕显示效果很棒，拍照质量超出预期，电池续航能够满足一天使用
    缺点：价格有点高，充电速度没有宣传的那么快，系统偶尔会卡顿
    总体来说，如果预算充足的话还是值得购买的。
    """

    prompt = f"""
    请分析以下产品评论并提供结构化分析结果：
    
    {review}
    """

    try:
        result = processor.get_structured_response(prompt, ProductReviewAnalysis)

        print("📊 DeepSeek产品评论分析:")
        print(f"   整体情感: {result.overall_sentiment}")
        print(f"   是否推荐: {'✅ 推荐' if result.recommendation else '❌ 不推荐'}")
        print(f"   维度评分:")
        for dimension, score in result.product_ratings.items():
            print(f"     {dimension}: {score}/5.0")
        print(f"   优点:")
        for pro in result.pros:
            print(f"     ✅ {pro}")
        print(f"   缺点:")
        for con in result.cons:
            print(f"     ❌ {con}")

    except Exception as e:
        print(f"错误: {e}")

# 5. 批量处理演示
def demo_batch_processing(processor: DeepSeekStructuredProcessor):
    """演示批量处理"""
    print("\n📦 5. 批量评论分析")
    print("=" * 50)

    reviews = [
        "这个产品太棒了！质量很好，送货也快，完全超出预期！",
        "不太满意，产品有瑕疵，客服处理慢，不会再买了。",
        "一般般吧，没什么特别的感觉，对得起这个价格。"
    ]

    print("开始批量分析评论...")

    for i, review in enumerate(reviews, 1):
        try:
            prompt = f"分析以下产品评论: {review}"
            result = processor.get_structured_response(prompt, ProductReviewAnalysis)

            print(f"\n📝 评论 {i}:")
            print(f"   情感: {result.overall_sentiment}")
            print(f"   推荐: {'✅' if result.recommendation else '❌'}")
            print(f"   平均评分: {sum(result.product_ratings.values())/len(result.product_ratings):.1f}/5.0")

        except Exception as e:
            print(f"评论 {i} 分析失败: {e}")

# 6. 模拟处理器（用于测试，避免真实API调用）
class MockDeepSeekProcessor:
    """模拟处理器，用于演示"""

    def get_structured_response(self, user_prompt: str, response_model: BaseModel) -> BaseModel:
        """模拟获取结构化响应"""

        if response_model == CustomerServiceResponse:
            return CustomerServiceResponse(
                issue_category="order_delivery",
                solution="立即查询订单状态并联系仓库优先处理，同时向客户发送状态更新邮件",
                priority="high",
                confidence=0.88,
                follow_up_questions=[
                    "您是否需要加急配送？",
                    "请问您的订单号是否正确？",
                    "您希望我们如何联系您？"
                ]
            )
        elif response_model == ContentAnalysisResult:
            return ContentAnalysisResult(
                sentiment="positive",
                key_points=[
                    "AI产品获得用户高度评价",
                    "界面友好、功能强大、响应速度快",
                    "处理复杂任务表现优异",
                    "提升工作效率明显"
                ],
                summary="AI产品获得积极用户反馈，特别是在功能性能和用户体验方面表现突出",
                tone="enthusiastic",
                word_count=85
            )
        elif response_model == CodeReviewResult:
            return CodeReviewResult(
                overall_quality="fair",
                issues=[
                    {
                        "type": "security",
                        "description": "使用eval()函数处理用户输入存在安全风险",
                        "line": 10,
                        "severity": "high"
                    }
                ],
                suggestions=[
                    "使用ast.literal_eval()替代eval()",
                    "使用sum(numbers)/len(numbers)计算平均值",
                    "添加输入验证和错误处理"
                ],
                complexity="low",
                security_concerns=["eval函数使用"]
            )
        elif response_model == ProductReviewAnalysis:
            return ProductReviewAnalysis(
                overall_sentiment="positive",
                product_ratings={"quality": 4.5, "price": 3.8, "performance": 4.2},
                pros=["屏幕显示效果很棒", "拍照质量超出预期", "电池续航能够满足一天使用"],
                cons=["价格有点高", "充电速度没有宣传的那么快", "系统偶尔会卡顿"],
                recommendation=True
            )
        else:
            raise ValueError(f"不支持的响应模型: {response_model}")

if __name__ == "__main__":
    print("🚀 ChatDeepSeek + Pydantic 实战演示")
    print("=" * 60)

    # 选择使用真实处理器还是模拟处理器
    use_real_api = False  # 设置为True使用真实API，False使用模拟数据

    if use_real_api:
        try:
            processor = DeepSeekStructuredProcessor()
            print("✅ 使用真实DeepSeek API")
        except Exception as e:
            print(f"❌ 无法连接DeepSeek API: {e}")
            print("🔄 切换到模拟模式...")
            processor = MockDeepSeekProcessor()
    else:
        processor = MockDeepSeekProcessor()
        print("🔧 使用模拟模式（避免真实API调用）")

    # 运行演示
    demo_customer_service(processor)
    demo_content_analysis(processor)
    demo_code_review(processor)
    demo_product_review_analysis(processor)
    demo_batch_processing(processor)

    print("\n" + "=" * 60)
    print("🎯 ChatDeepSeek + Pydantic 核心优势:")
    print("  ✅ 原生JSON支持 - 利用DeepSeek的response_format参数")
    print("  ✅ 类型安全 - Pydantic自动验证和类型转换")
    print("  ✅ 结构化输出 - 非结构化文本变结构化数据")
    print("  ✅ 业务集成 - 直接用于业务逻辑和决策")
    print("  ✅ 错误处理 - 自动捕获JSON解析错误")
    print("  ✅ 可维护性 - 清晰的Schema定义")
    print("=" * 60)