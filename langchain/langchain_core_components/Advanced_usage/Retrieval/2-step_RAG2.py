from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent # 导入创建代理的函数
from langchain_community.document_loaders import TextLoader # 导入文本文件加载器
from langchain_community.embeddings import OpenAIEmbeddings # 导入OpenAI嵌入模型
from langchain_community.vectorstores import FAISS # : 导入FAISS向量存储
from langchain_core.documents import Document #  导入文档核心类
from langchain.agents.middleware import AgentMiddleware, AgentState # 导入中间件基类和状态类
from langchain_text_splitters import RecursiveCharacterTextSplitter #  导入文本分割器

# 在load_dotenv()之后添加以下代码
load_dotenv()

# 初始化嵌入模型 初始化OpenAI嵌入模型实例
embeddings = OpenAIEmbeddings()

# 创建示例文档（实际使用时替换为真实文档路径） 创建文本加载器，指向文档文件
loader = TextLoader("your_document.txt")
documents = loader.load() #  加载文档内容
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) #  创建文本分割器，设置块大小和重叠
splits = text_splitter.split_documents(documents) # 将文档分割成小块

# 初始化向量存储 使用分割的文档和嵌入模型创建向量存储
vector_store = FAISS.from_documents(splits, embeddings)

# 定义自定义状态类，扩展基础 AgentState，添加 context 字段存储检索到的文档
class State(AgentState):
    context: list[Document]

# 创建自定义中间件类，继承 AgentMiddleware 并指定状态模式：
# before_model 方法在模型调用前执行：
# 获取最后一条消息
# 在向量存储中搜索相关文档
# 将检索到的文档内容合并
# 构造增强后的消息内容
# 返回更新后的消息和上下文
class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        retrieved_docs = vector_store.similarity_search(last_message.text)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        augmented_message_content = (
            f"{last_message.text}\n\n"
            "Use the following context to answer the query:\n"
            f"{docs_content}"
        )
        return {
            "messages": [last_message.model_copy(update={"content": augmented_message_content})],
            "context": retrieved_docs,
        }


agent = create_agent(
    model="gpt-3.5-turbo",
    tools=[],
    middleware=[RetrieveDocumentsMiddleware()],
)
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is task decomposition?"}
    ]
})