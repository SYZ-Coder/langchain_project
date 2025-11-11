# FastAPI 主应用
# app.py 是基于 FastAPI 构建的主应用文件，提供了一个 DeepSeek 模型调研平台的 Web API 服务。该应用整合了之前定义的各种研究链，并通过 RESTful API 接口对外提供服务

from fastapi import FastAPI, HTTPException  # FastAPI 框架核心组件和异常处理
from langserve import add_routes  # LangServe 库用于添加链路由
from pydantic import BaseModel # Pydantic 的基础模型类，用于定义请求/响应模型
from typing import Dict, Any, Optional # 类型注解工具
import asyncio  # 异步编程支持

from research_chains import research_chains # 导入之前定义的 research_chains 实例
from config import settings  # 导入配置 settings 实例

# 创建 FastAPI 应用
# 作用: 创建 FastAPI 应用实例并配置基本信息 逐行解析:
# 设置应用标题为"DeepSeek 模型调研平台"
# 设置应用描述信息
# 设置版本号为"1.0.0"
# 配置 API 文档路径为 /docs 和 /redoc
app = FastAPI(
    title="DeepSeek 模型调研平台",
    description="基于 LangChain 的 DeepSeek 模型能力调研和评估平台",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 请求模型定义
# 作用: 定义能力分析请求的数据模型 用法: 用于验证 /research/capability-analysis 端点的请求参数 逐行解析:
# 继承自 BaseModel
# 定义必填字段 question（字符串类型）
# 定义可选字段 analysis_dimensions（列表类型，默认值为技术分析维度
class CapabilityAnalysisRequest(BaseModel):
    question: str
    analysis_dimensions: Optional[list] = ["技术", "质量", "风格", "改进"]

# 作用: 定义模型对比请求的数据模型 用法: 用于验证 /research/comparison 端点的请求参数 逐行解析:
# 定义必填字段 question
# 定义布尔字段 compare_with_openai，默认为 True
class ComparisonRequest(BaseModel):
    question: str
    compare_with_openai: bool = True
# 作用: 定义性能测试请求的数据模型 用法: 用于验证 /research/performance-test 端点的请求参数 逐行解析:
# 定义测试类型字段 test_type
# 定义测试内容字段 content
# 定义迭代次数字段 iterations，默认为 1
class PerformanceTestRequest(BaseModel):
    test_type: str  # "creative", "technical", "reasoning", "knowledge"
    content: str
    iterations: int = 1

# 作用: 定义成本分析请求的数据模型 用法: 用于验证成本分析相关端点的请求参数 逐行解析:
# 定义使用场景字段 scenario
# 定义预估调用量字段 volume
class CostAnalysisRequest(BaseModel):
    scenario: str
    volume: str  # "low", "medium", "high", "enterprise"

# 作用: 定义技术规格调研请求的数据模型 用法: 用于验证技术规格调研相关端点的请求参数 逐行解析:
# 定义调研重点字段 focus_area
class TechnicalSpecsRequest(BaseModel):
    focus_area: str  # "architecture", "training", "performance", "features"

# 添加调研链路由
# 作用: 为能力分析链添加 API 路由 用法: 通过 POST /research/capability-analysis 访问 逐行解析:
# 使用 add_routes 函数添加路由
# 指定 FastAPI 应用实例 app
# 获取能力分析链实例
# 设置路由路径为 /research/capability-analysis
# 添加描述信息
add_routes(
    app,
    research_chains.get_capability_analysis_chain(),
    path="/research/capability-analysis",
    description="DeepSeek 模型能力分析"
)

# 作用: 为技术规格调研链添加 API 路由 用法: 通过 POST /research/technical-specs 访问
add_routes(
    app,
    research_chains.get_technical_specs_chain(),
    path="/research/technical-specs",
    description="DeepSeek 技术规格调研"
)

add_routes(
    app,
    research_chains.get_cost_analysis_chain(),
    path="/research/cost-analysis",
    description="DeepSeek 成本效益分析"
)

# 自定义端点
# 作用: 提供模型对比分析功能 用法: 发送 POST 请求到 /research/comparison，传入对比请求参数 逐行解析:
# 使用 @app.post 装饰器定义 POST 路由
# 定义异步函数 model_comparison，接收 ComparisonRequest 类型的请求参数
# 调用能力分析链获取 DeepSeek 模型的回答
# 根据 compare_with_openai 字段决定是否获取 OpenAI 回答
# 调用对比分析链生成对比报告
# 返回包含所有结果的 JSON 响应
# 异常处理：捕获错误并返回 500 状态码
@app.post("/research/comparison")
async def model_comparison(request: ComparisonRequest):
    """模型对比分析"""
    try:
        # 获取 DeepSeek 回答
        deepseek_response = await research_chains.get_capability_analysis_chain().ainvoke({
            "question": request.question
        })

        if request.compare_with_openai:
            # 获取 OpenAI 回答
            openai_response = research_chains.openai_model.invoke(request.question)
            openai_content = openai_response.content
        else:
            openai_content = "未进行对比"

        # 进行对比分析
        comparison_result = await research_chains.get_comparison_chain().ainvoke({
            "question": request.question,
            "deepseek_response": deepseek_response,
            "openai_response": openai_content
        })

        return {
            "status": "success",
            "deepseek_response": deepseek_response,
            "openai_response": openai_content if request.compare_with_openai else None,
            "comparison_analysis": comparison_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对比分析失败: {str(e)}")

# 作用: 提供性能测试功能 用法: 发送 POST 请求到 /research/performance-test，传入性能测试请求参数 逐行解析:
# 定义异步函数 performance_test，接收 PerformanceTestRequest 类型的请求参数
# 根据 iterations 字段进行多次测试
# 每次调用性能测试链获取结果
# 将所有结果收集到 results 列表中
# 返回包含测试类型、迭代次数和所有结果的 JSON 响应
@app.post("/research/performance-test")
async def performance_test(request: PerformanceTestRequest):
    """性能测试端点"""
    try:
        results = []
        for i in range(request.iterations):
            result = await research_chains.get_performance_test_chain().ainvoke({
                "test_type": request.test_type,
                "content": request.content
            })
            results.append({
                "iteration": i + 1,
                "result": result
            })

        return {
            "status": "success",
            "test_type": request.test_type,
            "iterations": request.iterations,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"性能测试失败: {str(e)}")

@app.get("/research/test-cases")
async def get_test_cases():
    """获取预定义的测试用例"""
    test_cases = {
        "creative_writing": {
            "name": "创意写作测试",
            "description": "测试模型的创意和文学能力",
            "questions": [
                "写一个关于人工智能的短篇科幻故事",
                "创作一首关于春天的诗歌",
                "为一个新产品写广告文案"
            ]
        },
        "technical_reasoning": {
            "name": "技术推理测试",
            "description": "测试模型的技术理解和推理能力",
            "questions": [
                "解释Transformer架构的工作原理",
                "比较深度学习和机器学习的区别",
                "设计一个简单的推荐系统架构"
            ]
        },
        "knowledge_qa": {
            "name": "知识问答测试",
            "description": "测试模型的知识广度和准确性",
            "questions": [
                "简述量子计算的基本原理",
                "介绍中国古代四大发明",
                "解释区块链技术的工作原理"
            ]
        },
        "code_generation": {
            "name": "代码生成测试",
            "description": "测试模型的编程能力",
            "questions": [
                "用Python写一个快速排序算法",
                "实现一个简单的神经网络",
                "写一个爬取网页数据的脚本"
            ]
        }
    }
    return test_cases

@app.get("/research/metrics")
async def get_research_metrics():
    """获取调研指标定义"""
    metrics = {
        "quality_metrics": [
            "准确性", "相关性", "完整性", "一致性", "实用性"
        ],
        "performance_metrics": [
            "响应速度", "稳定性", "可扩展性", "资源效率"
        ],
        "cost_metrics": [
            "API调用成本", "性价比", "规模适应性", "ROI"
        ],
        "technical_metrics": [
            "模型规模", "训练数据", "架构创新", "多语言支持"
        ]
    }
    return metrics

# 健康检查
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "DeepSeek Research Platform",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {
        "message": "DeepSeek 模型调研平台",
        "endpoints": {
            "能力分析": "/research/capability-analysis",
            "技术规格": "/research/technical-specs",
            "成本分析": "/research/cost-analysis",
            "模型对比": "/research/comparison",
            "性能测试": "/research/performance-test",
            "测试用例": "/research/test-cases",
            "调研指标": "/research/metrics",
            "API文档": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=True,  # 开发模式
        log_level="info"
    )
# 作用: 启动 FastAPI 应用 用法: 直接运行此文件启动服务 逐行解析:
# 检查是否为主模块运行
# 导入 uvicorn ASGI 服务器
# 使用 uvicorn.run 启动应用
# 从配置中获取主机和端口
# 启用重载模式（开发环境）
# 设置日志级别为 info