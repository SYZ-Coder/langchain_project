from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool, ToolRuntime
from langchain_deepseek import ChatDeepSeek
load_dotenv()  # 加载 .env 文件中的环境变量

# 定义主代理的自定义状态类，继承自 AgentState
# 添加了 example_state_key 和 task_context 两个状态字段
class CustomState(AgentState):
    """自定义状态类，扩展基础AgentState"""
    example_state_key: str = "1"
    task_context: str = "我是用户"

# 定义子代理的状态类，包含 example_state_key 字段
class SubAgentState(AgentState):
    """子代理的状态类"""
    example_state_key: str = ""

# 模拟子代理1 - 处理数学计算任务
# 创建数学计算子代理的函数
# 定义专门的提示词，限定只能处理数学问题
def create_math_subagent():
    """创建一个专门处理数学计算的子代理"""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是一个专业的数学计算器。只能回答数学相关的问题，其他问题请说明你无法处理。"),
        ("placeholder", "{messages}")
    ])

    llm =  ChatDeepSeek(temperature=0, model="deepseek-chat")
    chain = prompt | llm  # 使用管道操作符组合提示词和LLM

    def math_agent(state: SubAgentState):
        response = chain.invoke({"messages": state["messages"]})
        return {"messages": [response]} # 定义实际的代理函数，接收状态并返回响应

    return math_agent # 返回创建好的子代理函数

# 模拟子代理2 - 处理天气查询任务
# 创建天气查询子代理，结构与数学代理类似
def create_weather_subagent():
    """创建一个专门处理天气查询的子代理"""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是一个天气查询助手。只能回答天气相关的问题，其他问题请说明你无法处理。"
                              "对于天气查询，请提供城市名称和日期。"),
        ("placeholder", "{messages}")
    ])

    llm = ChatDeepSeek(temperature=0, model="deepseek-chat")
    chain = prompt | llm

    def weather_agent(state: SubAgentState):
        response = chain.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    return weather_agent

# 实例化子代理  实例化两个子代理，供后续工具调用
math_subagent = create_math_subagent()
weather_subagent = create_weather_subagent()

def process_query_for_subagent(query: str, messages: List, agent_type: str) -> List:
    #根据代理类型处理发送给子代理的输入消息
    """
    处理发送给子代理的消息
    根据不同类型的任务准备不同的输入
    """

    if agent_type == "math": # 数学代理只需要当前查询，不需要历史上下文
        # 对于数学任务，只需要最新的查询
        return [HumanMessage(content=f"请计算: {query}")]
    elif agent_type == "weather":  # 天气代理需要上下文信息，拼接历史对话内容
        # 对于天气任务，添加上下文信息
        context = "\n".join([msg.content for msg in messages if hasattr(msg, 'content')]) if messages else ""
        return [
            HumanMessage(content=f"根据以下对话历史，请回答天气查询: {query}\n对话历史: {context}")
        ]
    else:
        # 默认情况，传递原始消息
        return [HumanMessage(content=query)]

@tool(
    "math_calculator",
    description="用于解决数学计算问题的工具。当用户询问数学问题时调用此工具。"
)
def call_math_subagent(query: str, runtime: ToolRuntime[None, CustomState]):
    """调用数学计算子代理"""
    # 处理输入，只传递相关消息给子代理
    subagent_input = process_query_for_subagent(  # 调用输入处理函数，准备发送给数学子代理的消息
        query,
        runtime.state.get("messages", []),
        "math"
    )

    try:
        # 调用数学子代理并返回结果，包含错误处理
        result = math_subagent({
            "messages": subagent_input,
            "example_state_key": runtime.state.get("example_state_key", "")
        })
        return result["messages"][-1].content
    except Exception as e:
        return f"数学计算工具出现错误: {str(e)}"
# 定义天气查询工具，结构与数学工具类似
@tool(
    "weather_assistant",
    description="用于查询天气信息的工具。当用户询问天气相关问题时调用此工具。"
)
def call_weather_subagent(query: str, runtime: ToolRuntime[None, CustomState]):
    """调用天气查询子代理"""
    # 处理输入，传递上下文给子代理
    subagent_input = process_query_for_subagent(
        query,
        runtime.state.get("messages", []),
        "weather"
    )

    try:
        result = weather_subagent({
            "messages": subagent_input,
            "example_state_key": runtime.state.get("example_state_key", "")
        })
        return result["messages"][-1].content
    except Exception as e:
        return f"天气查询工具出现错误: {str(e)}"

# 创建主代理
def create_main_agent():
    """创建主代理，能够调用子代理"""
    # 定义工具列表 收集所有可用工具
    tools = [call_math_subagent, call_weather_subagent]

    # 主代理提示词
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是一个智能助手，可以根据用户请求决定是否需要调用专门的工具。"
                              "如果是数学问题，请使用数学计算器工具；如果是天气问题，请使用天气助手工具。"),
        ("placeholder", "{messages}")
    ])
    #定义主代理的提示词，指导其如何选择和使用工具
    llm =  ChatDeepSeek(temperature=0, model="deepseek-chat")
    llm_with_tools = llm.bind_tools(tools)
    # 绑定工具到LLM，使其能够调用这些工具
    chain = prompt | llm_with_tools

    # 组合提示词和带工具的LLM
    def agent_func(state: CustomState):
        response = chain.invoke({"messages": state["messages"]})
        print("response:", response)
        return {"messages": [response]}

    return agent_func

# 演示测试
def demo_test():
    """演示测试函数"""
    print("=== 多代理系统演示 ===\n")

    # 创建主代理
    main_agent = create_main_agent()  # 创建主代理实例

    # 测试用例1: 数学计算
    print("测试1: 数学计算")
    print("-" * 30)
    # 定义测试用例，涵盖数学和天气两类问题
    test_cases = [
        "123 + 456 等于多少？",
        "北京今天天气怎么样？",
        "计算 2 的 10 次方",
        "上海明天会下雨吗？"
    ]

    #定义初始状态
    initial_state = {
        "messages": [],
        "example_state_key": "test_session_001",
        "task_context": "用户日常咨询"
    }
# 遍历测试用例，调用主代理并输出结果
    for i, query in enumerate(test_cases, 1):
        print(f"\n问题 {i}: {query}")
        try:
            result = main_agent.invoke({
                "input": query,
                **initial_state
            })
            print(f"回答: {result['output'].content}")
        except Exception as e:
            print(f"执行出错: {str(e)}")
        print("-" * 30)

if __name__ == "__main__":
    demo_test()


# 控制对子代理的输入
# 有两个主要杠杆用于控制主代理传递给子代理的输入：
# 修改提示词——调整主代理的提示词或工具元数据（即子代理的名称和描述），以更好地指导何时以及如何调用子代理。
# 上下文注入——通过调整工具调用，从代理状态中提取，添加不适合静态提示捕获的输入（例如完整消息历史、之前结果、任务元数据）。