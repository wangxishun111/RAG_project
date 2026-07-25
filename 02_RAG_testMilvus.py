import os
from typing import Any, List

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from pymilvus import DataType, MilvusClient
except ImportError as exc:
    raise ImportError(
        "当前脚本使用 MilvusClient，请先安装依赖：pip install -U pymilvus"
    ) from exc


load_dotenv()


# =========================
# 1) 基础配置
# =========================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError("请先在环境变量或 .env 中配置 DEEPSEEK_API_KEY")


# =========================
# 2) 文档与 Milvus 配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SUPPORTED_EXTENSIONS = ["*.txt", "*.md"]

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_documents")

PRIMARY_FIELD = "pk"
TEXT_FIELD = "text"
VECTOR_FIELD = "vector"
FILENAME_FIELD = "filename"
SOURCE_FIELD = "source"
VECTOR_DIMENSION = 512
TOP_K = 3


# =========================
# 3) 文档加载与切分
# =========================
def build_documents(data_dir: str = DATA_DIR) -> List[Document]:
    """加载支持的文档，并补充文件名元数据。"""
    all_documents = []

    for ext in SUPPORTED_EXTENSIONS:
        loader = DirectoryLoader(
            data_dir,
            glob=f"**/{ext}",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
            show_progress=True,
            use_multithreading=True,
            recursive=True,
        )
        try:
            docs = loader.load()
            all_documents.extend(docs)
            print(f"  ✓ 加载 {ext} 文件 {len(docs)} 个")
        except Exception as exc:
            print(f"  - 跳过 {ext} 格式（{exc}）")

    if not all_documents:
        formats = ", ".join(SUPPORTED_EXTENSIONS)
        raise ValueError(f"在目录 {data_dir} 下没有找到可加载的文档（支持格式: {formats}）")

    for doc in all_documents:
        source_path = doc.metadata.get("source", "")
        doc.metadata[FILENAME_FIELD] = os.path.basename(source_path)

    print(f"\n共加载文档数: {len(all_documents)}")
    print("文件清单（前5个）:")
    seen_names = set()
    for doc in all_documents:
        filename = doc.metadata.get(FILENAME_FIELD, "unknown")
        if filename not in seen_names:
            seen_names.add(filename)
            print(f"  - {filename}")
            if len(seen_names) >= 5:
                break
    return all_documents


def create_embeddings() -> HuggingFaceEmbeddings:
    """创建中文文本 Embedding 模型。"""
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")


# =========================
# 4) MilvusClient 集合构建与写入
# =========================
def build_milvus_client() -> MilvusClient:
    """创建新版 MilvusClient。"""
    connection_args: dict[str, Any] = {"uri": MILVUS_URI}
    if MILVUS_TOKEN:
        connection_args["token"] = MILVUS_TOKEN
    if MILVUS_DB_NAME:
        connection_args["db_name"] = MILVUS_DB_NAME
    return MilvusClient(**connection_args)


def create_collection(client: MilvusClient) -> None:
    """删除旧集合并创建带文本、来源和向量字段的新集合。"""
    if client.has_collection(MILVUS_COLLECTION_NAME):
        client.drop_collection(MILVUS_COLLECTION_NAME)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(PRIMARY_FIELD, DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(TEXT_FIELD, DataType.VARCHAR, max_length=65535)
    schema.add_field(FILENAME_FIELD, DataType.VARCHAR, max_length=1024)
    schema.add_field(SOURCE_FIELD, DataType.VARCHAR, max_length=4096)
    schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=VECTOR_FIELD,
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=MILVUS_COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )


def build_vectorstore(data_dir: str = DATA_DIR) -> tuple[MilvusClient, HuggingFaceEmbeddings]:
    """加载、切分、向量化文档，并用 MilvusClient 写入集合。"""
    documents = build_documents(data_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    chunks = splitter.split_documents(documents)
    embeddings = create_embeddings()

    print(f"\n正在重建 Milvus 集合: {MILVUS_COLLECTION_NAME}")
    print(f"Milvus URI: {MILVUS_URI}")
    client = build_milvus_client()
    create_collection(client)

    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)
    records = [
        {
            TEXT_FIELD: chunk.page_content,
            FILENAME_FIELD: chunk.metadata.get(FILENAME_FIELD, "未知文件"),
            SOURCE_FIELD: chunk.metadata.get(SOURCE_FIELD, ""),
            VECTOR_FIELD: vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    client.insert(collection_name=MILVUS_COLLECTION_NAME, data=records)
    client.load_collection(MILVUS_COLLECTION_NAME)
    print(f"已写入文本块数: {len(records)}")
    return client, embeddings


# =========================
# 5) RAG 检索与回答
# =========================
def search_documents(
    question: str,
    client: MilvusClient,
    embeddings: HuggingFaceEmbeddings,
) -> List[Document]:
    """把问题转为向量，并从 Milvus 检索最相关的文本块。"""
    question_vector = embeddings.embed_query(question)
    search_results = client.search(
        collection_name=MILVUS_COLLECTION_NAME,
        data=[question_vector],
        anns_field=VECTOR_FIELD,
        limit=TOP_K,
        output_fields=[TEXT_FIELD, FILENAME_FIELD, SOURCE_FIELD],
        search_params={"metric_type": "COSINE", "params": {}},
    )

    documents = []
    for hit in search_results[0]:
        entity = hit.get("entity", {})
        documents.append(
            Document(
                page_content=entity.get(TEXT_FIELD, ""),
                metadata={
                    FILENAME_FIELD: entity.get(FILENAME_FIELD, "未知文件"),
                    SOURCE_FIELD: entity.get(SOURCE_FIELD, ""),
                    "score": hit.get("distance"),
                },
            )
        )
    return documents


def ask_question(
    question: str,
    client: MilvusClient,
    embeddings: HuggingFaceEmbeddings,
) -> str:
    docs = search_documents(question, client, embeddings)
    if not docs:
        return "未找到相关信息。"

    formatted_docs = []
    for doc in docs:
        filename = doc.metadata.get(FILENAME_FIELD, "未知文件")
        formatted_docs.append(f"[来源: {filename}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(formatted_docs)

    llm = ChatDeepSeek(
        api_key=DEEPSEEK_API_KEY,
        model="deepseek-v4-flash",
        temperature=0,
        max_retries=2,
        streaming=False,
    )
    prompt_text = (
        "你是一个中文 RAG 智能问答助手。请严格根据给定上下文回答问题。"
        "如果上下文中没有相关信息，请直接回答：未找到相关信息。"
        "回答要简洁、准确、自然。"
        "在查询到的内容开头标引用的文件名，格式如 [来源: xxx.txt]\n[答复内容]。\n\n"
        f"上下文：\n{context}\n\n"
        f"问题：{question}"
    )
    response = llm.invoke(prompt_text)
    return response.content if hasattr(response, "content") else str(response)


# =========================
# 6) 主程序：命令行交互
# =========================
def main() -> None:
    print("=" * 60)
    print("RAG 智能问答助手已启动（MilvusClient 向量库）")
    print("输入问题后回车提问，输入 exit 或 quit 退出。")
    print("=" * 60)

    client, embeddings = build_vectorstore(DATA_DIR)
    print("-" * 60)
    while True:
        question = input("\n请输入你的问题：").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break

        try:
            answer = ask_question(question, client, embeddings)
            print("\n助手回答：")
            print(answer)
        except Exception as exc:
            print(f"\n发生错误：{exc}")


if __name__ == "__main__":
    main()
