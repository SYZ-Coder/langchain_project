# 2-step_RAG1.py
import os
from dotenv import load_dotenv
from langchain.agents import create_agent # 从 langchain.agents 模块导入 create_agent 函数，用于创建智能代理
from langchain.agents.middleware import dynamic_prompt, ModelRequest #  从中间件模块导入装饰器和请求对象，用于自定义代理行为
from langchain_community.document_loaders import TextLoader #  导入文本加载器，用于加载本地文本文件
from langchain_community.vectorstores import FAISS #导入 FAISS 向量存储，用于高效相似度搜索
from langchain_openai import OpenAIEmbeddings # 导入 OpenAI 嵌入模型，用于将文本转换为向量表示
from langchain_text_splitters import RecursiveCharacterTextSplitter #导入递归字符文本分割器，用于将长文本切分成小块

# 加载环境变量
load_dotenv()

# 初始化DeepSeek嵌入模型 创建 OpenAI 嵌入模型实例，用于后续文档向量化
embeddings = OpenAIEmbeddings()

# 构建向量存储 (假设已有文本数据文件 data.txt)  创建文本加载器实例，准备加载名为 data.txt 的文件
loader = TextLoader("data.txt")  # 替换为你自己的文档路径
documents = loader.load() # 实际加载并解析文档内容
# 创建文本分割器，设置每个文本块大小为 500 字符，重叠 50 字符
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# 将加载的文档按照设定规则进行分割
splits = text_splitter.split_documents(documents)
# 使用分割后的文档和嵌入模型创建 FAISS 向量存储
vector_store = FAISS.from_documents(splits, embeddings)

#这是一个带 @dynamic_prompt 装饰器的中间件函数，用于动态修改发送给模型的提示词：
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    # 提取用户最新消息的内容
    last_query = request.state["messages"][-1]["content"]
    # 在向量存储中搜索与用户查询最相关的文档
    retrieved_docs = vector_store.similarity_search(last_query)
    #  将检索到的相关文档内容合并成一个字符串
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    # 构造系统消息，将检索到的上下文信息注入到提示词中，指导模型基于这些上下文回答问题
    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message


# 创建带有自定义中间件的 Agent
agent = create_agent(model="gpt-3.5-turbo", tools=[], middleware=[prompt_with_context])
# 定义查询问题："What is task decomposition?"
# 使用 agent.stream() 方法以流式方式处理用户查询
# stream_mode="values" 设置流式传输模式
# step["messages"][-1].pretty_print() 打印每一步的响应结果
if __name__ == '__main__':
    query = "What is task decomposition?"
    for step in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
# 这是一个完整的 RAG（Retrieval-Augmented Generation）系统实现，能够根据用户查询从本地文档中检索相关信息，并结合这些信息生成更准确的回答。