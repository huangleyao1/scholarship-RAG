#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag.py — 检索增强生成（RAG）问答主流程

流程：问题 → LLM改写为多个检索词组 → 多路向量检索（去重合并）→ 拼上下文 → LLM生成

两个关键设计（详见 notes.md）：
1. 查询改写（expand_query）：长问题直接检索会被"查询稀释"，
   先让LLM改写成2-3个短词组分别检索，多路召回后去重合并。
2. hnsw_ef=500：Chroma 默认的 HNSW 探索因子太小，小向量库上
   近似搜索会漏掉正确答案，这里显式调大（等价于全库精确检索）。
"""

import requests
import chromadb
from sentence_transformers import SentenceTransformer
from config import API_KEY

MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")
client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection(name="scholarship")

QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
TOP_K = 3        # 每个检索词组捞回的块数
MAX_BLOCKS = 8   # 最终喂给 LLM 的最大块数（防上下文爆炸）

NAME_MAP = {
    "定稿-2025年修订-人工智能与计算机学院本科生综合素质考评实施细则":
        "《人工智能与计算机学院本科生综合素质考评实施细则（2025年修订）》",
    "人工智能与计算机学院2025—2026学年本科生综合素质考评表":
        "《人工智能与计算机学院2025—2026学年本科生综合素质考评表》",
}


def chat(prompt: str) -> str:
    """调用 DeepSeek API（评测与改写共用一个入口，便于统一管理）"""
    r = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": "deepseek-chat",
              "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def expand_query(q: str) -> list:
    """让LLM把长问题改写成短检索词组，解决查询稀释（见 notes.md 第3条）"""
    prompt = ("你是检索助手。把用户的问题改写成2-3个简短的检索关键词组，"
              "每个词组一行，只输出词组本身，不要解释。\n\n"
              f"问题：{q}")
    text = chat(prompt)
    queries = [line.strip() for line in text.split("\n") if line.strip()]
    queries.append(q)          # 原始问题保留一份
    return queries[:4]


def _query(emb, k):
    """带 ef 保护的向量库查询：ef 远大于库规模(41)时，近似搜索退化为精确检索"""
    try:
        return collection.query(query_embeddings=emb, n_results=k, hnsw_ef=500)
    except TypeError:   # 旧版 chromadb 不支持 hnsw_ef 参数
        return collection.query(query_embeddings=emb, n_results=k)


def retrieve(q: str, k: int = TOP_K) -> str:
    """检索：多词组分别检索 → 按块文本去重 → 拼成带来源标注的上下文"""
    seen = {}
    for sub in expand_query(q):
        emb = MODEL.encode([QUERY_INSTRUCTION + sub],
                           normalize_embeddings=True).tolist()
        res = _query(emb, k)
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            if doc not in seen:              # 不同词组可能捞到同一块，去重
                seen[doc] = (doc, meta)
    blocks = []
    for doc, meta in list(seen.values())[:MAX_BLOCKS]:
        source = NAME_MAP.get(meta["source"], meta["source"])
        blocks.append(f"【来源：{source}】\n{doc}")
    return "\n\n---\n\n".join(blocks)


def ask(q: str) -> str:
    """生成：检索 + 拼prompt + 调LLM"""
    context = retrieve(q)
    prompt = f"""
    你是陕西师范大学综合素质测评的咨询助手。请严格根据以下资料回答。
    如果答案需要综合多块资料的信息（例如一条资料说明规则、另一条说明表格结构），请将它们组合成完整回答，不要因没有单一板块直接给出答案而拒绝回答。
    如果资料中没有答案，请直接说明"资料中未提及，建议询问辅导员或评议小组。"，不要编造。
    回答末尾请注明依据的条款和来源文件。
资料：
{context}
问题：{q}
    """
    return chat(prompt)


if __name__ == "__main__":
    while True:
        q = input("\n提问（q 退出）：").strip()
        if q == "q":
            break
        print(ask(q))