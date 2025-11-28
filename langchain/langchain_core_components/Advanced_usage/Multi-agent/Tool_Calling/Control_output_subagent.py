from typing import Annotated, List, Dict, Any, Union

from dotenv import load_dotenv
from langchain_classic.agents import create_tool_calling_agent
from langchain_deepseek import ChatDeepSeek
from typing_extensions import TypedDict
import operator
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langgraph.types import Command


load_dotenv()  # 加载 .env 文件中的环境变量

# 定义状态类型
# 作用: 定义多代理系统中状态的数据结构
# 用法: messages字段使用operator.add进行合并，支持消息历史的累积；example_state_key用于传递自定义状态信息
class AgentState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage, ToolMessage]], operator.add]
    example_state_key: str

# 初始化模型
llm = ChatDeepSeek(temperature=0, model="deepseek-chat")

# 创建子代理1 - 天气查询代理
# 作用: 通过精细化提示词指导子代理输出格式和内容
# 用法: 在system提示中明确要求输出规范，包括必要信息和格式示例
weather_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个天气查询助手。当用户提供城市名称时，请给出该城市的天气情况。
    
    重要说明：
    1. 必须在最终回复中包含具体的温度和天气状况
    2. 如果缺少城市信息，请主动询问用户
    3. 回复应该简洁明了，避免冗余信息
    
    示例回复格式：
    "北京今天的天气是晴天，温度为25°C。" """),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])
# # 作用: 创建专门处理特定任务的子代理（天气查询和新闻摘要）
# # 用法: 使用create_tool_calling_agent工厂函数，传入LLM、工具列表和专用提示词模板
weather_agent = create_tool_calling_agent(llm, [], weather_prompt)

# 创建子代理2 - 新闻摘要代理
news_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个新闻摘要助手。当用户提供主题时，请给出相关的最新新闻摘要。
    
    重要说明：
    1. 必须在最终回复中包含至少2条具体新闻标题和简要内容
    2. 需要标明新闻来源和时间（如果有的话）
    3. 所有关键信息必须在一条消息中完整呈现给用户
    
    示例回复格式：
    "关于科技的最新新闻：
    1. 苹果发布新款iPhone - 据报道，苹果公司今天发布了新一代iPhone...
    2. AI技术取得重大突破 - 研究人员在自然杂志上发表了..." """),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

news_agent = create_tool_calling_agent(llm, [], news_prompt)

# 封装子代理的执行逻辑
def run_subagent(agent, input_query: str) -> Dict[str, Any]:
    """运行子代理并返回结果"""
    result = agent.invoke({
        "input": input_query,
        "chat_history": [],
        "agent_scratchpad": []
    })

    # 模拟添加一些状态信息
    additional_state = f"processed_{input_query.replace(' ', '_')}"

    return {
        "messages": [AIMessage(content=result["output"])],
        "example_state_key": additional_state
    }
# 作用: 将子代理封装为可被主代理调用的工具
# 用法: 使用@tool装饰器标记函数为工具，返回Command对象以更新状
@tool(
    "weather_agent",
    description="查询指定城市的天气情况"
)
def call_weather_agent(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[AgentState]:
    """调用天气查询子代理"""
    result = run_subagent(weather_agent, query)

    return Command(
        update={
            "example_state_key": result["example_state_key"],
            "messages": [
                ToolMessage(
                    content=result["messages"][0].content,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )

@tool(
    "news_agent",
    description="获取指定主题的新闻摘要"
)
def call_news_agent(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[AgentState]:
    """调用新闻摘要子代理"""
    result = run_subagent(news_agent, query)
     # 作用: 在返回主代理前调整和丰富子代理响应
    # 用法: 使用Command包装结果，同时传递文本内容和自定义状态键
    return Command(
        update={
            "example_state_key": result["example_state_key"],
            "messages": [
                ToolMessage(
                    content=result["messages"][0].content,
                    tool_call_id=tool_call_id
                )
            ]
        }
    )

# 主代理提示词 - 展示如何控制子代理输出
main_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个任务协调员，负责根据用户需求调用合适的工具。
    
    可用工具：
    1. weather_agent: 查询天气信息
    2. news_agent: 获取新闻摘要
    
    调用工具时请注意：
    - 提供清晰明确的查询参数
    - 工具会返回完整的结果，你只需要转发给用户即可
    
    用户只能看到你的最终回复，所以请确保将工具返回的所有重要信息都包含在内。"""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 创建主代理
# 作用: 创建中央协调器，根据用户请求调用适当工具
# 用法: 注册所有子代理工具，通过主提示词控制整体行为
main_agent = create_tool_calling_agent(
    llm,
    [call_weather_agent, call_news_agent],
    main_prompt
)

# 演示函数
def demo():
    print("=== 多代理系统演示 ===")

    # 测试用例1：天气查询
    print("\n1. 查询天气:")
    query1 = "北京的天气怎么样？"
    print(f"用户提问: {query1}")

    result1 = main_agent.invoke({
        "input": query1,
        "chat_history": [],
        "agent_scratchpad": [],
        "intermediate_steps": []  # 添加这一行以防止 KeyError
    })

    # 方法2：如果result1确实是列表，修改访问方式
    print(f"主代理回复: {result1[0]}")  # 访问第一个元素

    # 方法3：添加类型检查和调试信息
    print(f"result1类型: {type(result1)}")
    print(f"result1内容: {result1}")
    if isinstance(result1, dict):
        print(f"主代理回复: {result1['output']}")
    elif isinstance(result1, list):
        print(f"主代理回复: {result1[0] if result1 else '空列表'}")

    # 测试用例2：新闻查询
    print("\n2. 查询新闻:")
    query2 = "最近的科技新闻"
    print(f"用户提问: {query2}")

    result2 = main_agent.invoke({
        "input": query2,
        "chat_history": [],
        "agent_scratchpad": [],
        "intermediate_steps": []  # 添加这一行以防止 KeyError
    })

    print(f"主代理回复: {result1[0]}")  # 访问第一个元素

    # 方法3：添加类型检查和调试信息
    print(f"result1类型: {type(result1)}")
    print(f"result1内容: {result1}")
    if isinstance(result1, dict):
        print(f"主代理回复: {result1['output']}")
    elif isinstance(result1, list):
        print(f"主代理回复: {result1[0] if result1 else '空列表'}")


if __name__ == "__main__":
    demo()


# 这种设计实现了两个重要的输出控制策略：
# 预定义输出规范: 通过提示词约束子代理行为
# 后处理增强: 通过Command结构丰富返回信息

# 塑造主代理从子代理处获得回报的两种常见策略：
# 修改提示词——细化子代理的提示，明确应返回什么。
# 当输出不完整、冗长或缺少关键细节时非常有用。
# 常见的失败模式是子代理执行工具调用或推理，但最终消息中没有包含结果。提醒它控制器（和用户）只看到最终输出，因此所有相关信息必须包含在其中。
# 自定义输出格式——在交还主代理之前，调整或丰富子代理的代码响应。
# 举例：除了最终文本外，还将特定的状态键传回主代理。
# 这需要将结果包裹在命令（或等效结构）中，以便将自定义状态与子代理的响应合并。