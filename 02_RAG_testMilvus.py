import hashlib
import json
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
MANIFEST_PATH = os.path.join(BASE_DIR, "data_manifest.json")

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_documents_new")

PRIMARY_FIELD = "pk"
TEXT_FIELD = "text"
VECTOR_FIELD = "vector"
FILENAME_FIELD = "filename"
SOURCE_FIELD = "source"
DOC_ID_FIELD = "doc_id"
CHUNK_ID_FIELD = "chunk_id"
CHUNK_INDEX_FIELD = "chunk_index"
VERSION_FIELD = "version"
CONTENT_HASH_FIELD = "content_hash"
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
        doc.metadata[SOURCE_FIELD] = source_path

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
# 4) Manifest 管理
# =========================
def compute_file_hash(file_path: str) -> str:
    """计算文件内容哈希，用于检测文档变更。"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_manifest() -> dict[str, Any]:
    """加载本地文档索引清单。"""
    if not os.path.exists(MANIFEST_PATH):
        return {"documents": {}}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict[str, Any]) -> None:
    """保存本地文档索引清单。"""
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def scan_disk_documents(data_dir: str = DATA_DIR) -> dict[str, dict[str, Any]]:
    """扫描磁盘上的文档并生成文件级元信息。"""
    disk_docs: dict[str, dict[str, Any]] = {}
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if not any(file_name.lower().endswith(ext.replace("*", "")) for ext in (".txt", ".md")):
                continue
            source_path = os.path.join(root, file_name)
            doc_id = os.path.relpath(source_path, data_dir)
            disk_docs[doc_id] = {
                "doc_id": doc_id,
                "filename": file_name,
                "source": source_path,
                "content_hash": compute_file_hash(source_path),
            }
    return disk_docs


def diff_documents(
    disk_docs: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """对比磁盘文档与 manifest，返回新增、更新、删除列表。"""
    manifest_docs = manifest.get("documents", {})
    added: list[dict[str, Any]] = []
    updated: list[dict[str, Any]] = []
    deleted: list[dict[str, Any]] = []

    for doc_id, info in disk_docs.items():
        if doc_id not in manifest_docs:
            added.append(info)
        elif manifest_docs[doc_id].get("content_hash") != info["content_hash"]:
            updated.append(info)

    for doc_id, info in manifest_docs.items():
        if doc_id not in disk_docs:
            deleted.append({"doc_id": doc_id, **info})

    return added, updated, deleted


# =========================
# 5) MilvusClient 集合构建
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
    """删除旧集合并创建带文档版本信息的新集合。"""
    if client.has_collection(MILVUS_COLLECTION_NAME):
        client.drop_collection(MILVUS_COLLECTION_NAME)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(PRIMARY_FIELD, DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(DOC_ID_FIELD, DataType.VARCHAR, max_length=1024)
    schema.add_field(CHUNK_ID_FIELD, DataType.VARCHAR, max_length=1024)
    schema.add_field(CHUNK_INDEX_FIELD, DataType.INT64)
    schema.add_field(VERSION_FIELD, DataType.INT64)
    schema.add_field(FILENAME_FIELD, DataType.VARCHAR, max_length=1024)
    schema.add_field(SOURCE_FIELD, DataType.VARCHAR, max_length=4096)
    schema.add_field(CONTENT_HASH_FIELD, DataType.VARCHAR, max_length=128)
    schema.add_field(TEXT_FIELD, DataType.VARCHAR, max_length=65535)
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


# =========================
# 6) 增量写入 / 删除
# =========================
def build_chunk_records(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings,
    doc_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """为单个文档生成 chunk 记录。"""
    texts = [doc.page_content for doc in documents]
    vectors = embeddings.embed_documents(texts)
    version = int(doc_meta.get("version", 0))

    records = []
    for chunk_index, (chunk, vector) in enumerate(zip(documents, vectors)):
        chunk_id = f"{doc_meta['doc_id']}::v{version}::c{chunk_index}"
        records.append(
            {
                DOC_ID_FIELD: doc_meta["doc_id"],
                CHUNK_ID_FIELD: chunk_id,
                CHUNK_INDEX_FIELD: chunk_index,
                VERSION_FIELD: version,
                FILENAME_FIELD: doc_meta["filename"],
                SOURCE_FIELD: doc_meta["source"],
                CONTENT_HASH_FIELD: doc_meta["content_hash"],
                TEXT_FIELD: chunk.page_content,
                VECTOR_FIELD: vector,
            }
        )
    return records


def delete_document_chunks(client: MilvusClient, doc_id: str) -> None:
    """删除某个文档对应的全部 chunk。"""
    if client.has_collection(MILVUS_COLLECTION_NAME):
        client.delete(
            collection_name=MILVUS_COLLECTION_NAME,
            filter=f'{DOC_ID_FIELD} == "{doc_id}"',
        )


def upsert_document(
    client: MilvusClient,
    embeddings: HuggingFaceEmbeddings,
    doc_meta: dict[str, Any],
) -> None:
    """插入或更新单个文档的 chunk 向量。"""
    loader = TextLoader(doc_meta["source"], encoding="utf-8", autodetect_encoding=True)
    docs = loader.load()
    for doc in docs:
        doc.metadata[FILENAME_FIELD] = doc_meta["filename"]
        doc.metadata[SOURCE_FIELD] = doc_meta["source"]

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=60)
    chunks = splitter.split_documents(docs)

    version = int(doc_meta.get("version", 0)) + 1
    doc_meta["version"] = version

    records = build_chunk_records(chunks, embeddings, doc_meta)
    client.insert(collection_name=MILVUS_COLLECTION_NAME, data=records)
    print(f"  ✓ 写入/更新文档: {doc_meta['doc_id']}，chunk 数: {len(records)}，version: {version}")


def sync_documents_to_milvus(
    client: MilvusClient,
    embeddings: HuggingFaceEmbeddings,
    data_dir: str = DATA_DIR,
) -> dict[str, Any]:
    """扫描本地文档并做新增、更新、删除同步。"""
    manifest = load_manifest()
    disk_docs = scan_disk_documents(data_dir)
    added, updated, deleted = diff_documents(disk_docs, manifest)

    if not client.has_collection(MILVUS_COLLECTION_NAME):
        create_collection(client)

    for doc in added:
        doc["version"] = 1
        upsert_document(client, embeddings, doc)
        manifest.setdefault("documents", {})[doc["doc_id"]] = doc

    for doc in updated:
        old_meta = manifest["documents"].get(doc["doc_id"], {})
        old_version = int(old_meta.get("version", 0))
        doc["version"] = old_version
        delete_document_chunks(client, doc["doc_id"])
        upsert_document(client, embeddings, doc)
        manifest.setdefault("documents", {})[doc["doc_id"]] = doc

    for doc in deleted:
        delete_document_chunks(client, doc["doc_id"])
        manifest.get("documents", {}).pop(doc["doc_id"], None)
        print(f"  ✓ 删除文档: {doc['doc_id']}")

    save_manifest(manifest)
    client.load_collection(MILVUS_COLLECTION_NAME)

    summary = {
        "added": len(added),
        "updated": len(updated),
        "deleted": len(deleted),
        "total": len(manifest.get("documents", {})),
    }
    print(
        f"\n增量同步完成：新增 {summary['added']}，更新 {summary['updated']}，"
        f"删除 {summary['deleted']}，当前保留 {summary['total']} 个文档。"
    )
    return summary


def build_vectorstore(data_dir: str = DATA_DIR) -> tuple[MilvusClient, HuggingFaceEmbeddings]:
    """同步增量文档并返回可检索的 MilvusClient 与 Embedding 模型。"""
    print(f"\nMilvus URI: {MILVUS_URI}")
    client = build_milvus_client()
    embeddings = create_embeddings()
    sync_documents_to_milvus(client, embeddings, data_dir)
    return client, embeddings


# =========================
# 7) RAG 检索与回答
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
        output_fields=[TEXT_FIELD, FILENAME_FIELD, SOURCE_FIELD, DOC_ID_FIELD, VERSION_FIELD],
        search_params={"metric_type": "COSINE", "params": {}},
    )

    documents = []
    for hit in search_results[0]:
        entity = hit.get("entity", {})
        documents.append(
            Document(
                page_content=entity.get(TEXT_FIELD, ""),
                metadata={
                    DOC_ID_FIELD: entity.get(DOC_ID_FIELD, ""),
                    FILENAME_FIELD: entity.get(FILENAME_FIELD, "未知文件"),
                    SOURCE_FIELD: entity.get(SOURCE_FIELD, ""),
                    VERSION_FIELD: entity.get(VERSION_FIELD, 0),
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
        version = doc.metadata.get(VERSION_FIELD, 0)
        formatted_docs.append(f"[来源: {filename} | 版本: {version}]\n{doc.page_content}")
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
        "在查询到的内容开头标引用的文件名和版本，格式如 [来源: xxx.txt | 版本: 1]\n[答复内容]。\n\n"
        f"上下文：\n{context}\n\n"
        f"问题：{question}"
    )
    response = llm.invoke(prompt_text)
    return response.content if hasattr(response, "content") else str(response)


# =========================
# 8) 主程序：命令行交互
# =========================
def main() -> None:
    print("=" * 60)
    print("RAG 智能问答助手已启动（MilvusClient 增量同步版）")
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
