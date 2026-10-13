#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_from_clean.py — 从干净语料备份一键构建向量库（推荐给新克隆的用户）

跳过"PDF→md→切块"的完整流水线（那条链的中间产物容易损坏，见 notes.md 第2条），
直接使用人工清洗后的 chunk 备份 data/chunks/chunks_clean.jsonl 建库。

用法（项目根目录）：
    python src/build_from_clean.py
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

CLEAN = "data/chunks/chunks_clean.jsonl"
COLLECTION = "scholarship"
CHROMA_DIR = "data/chroma"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EXCLUDE = "附件1"   # 冗余的上位文件，不进入向量库（见 notes.md 第4条）


def main():
    chunks = []
    with open(CLEAN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    kept = [c for c in chunks if EXCLUDE not in json.dumps(c.get("metadata", {}),
                                                           ensure_ascii=False)]
    print(f"读取 {len(chunks)} 块，剔除冗余后 {len(kept)} 块")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"})

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode([c["content"] for c in kept],
                       normalize_embeddings=True).tolist()
    col.add(ids=[f"chunk-{i}" for i in range(len(kept))],
            embeddings=emb,
            documents=[c["content"] for c in kept],
            metadatas=[c["metadata"] for c in kept])
    print(f"已入库 {col.count()} 块（cosine 空间）")
    print("复现评测: python src/eval.py --top-k 3   启动问答: python src/rag.py")


if __name__ == "__main__":
    main()