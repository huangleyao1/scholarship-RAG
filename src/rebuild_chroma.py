#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebuild_chroma.py — 从旧库导出干净语料，剔除学校办法，余弦空间重建

安全顺序：先全量备份 → 再删旧库建新的。任何一步失败旧数据都在 jsonl 里。

用法（项目根目录）：
    python src/rebuild_chroma.py            # 剔除 source 含"附件1"的 chunk
    python src/rebuild_chroma.py --dry-run  # 只备份和统计，不重建
"""

import json
import argparse
import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION = "scholarship"
CHROMA_DIR = "data/chroma"
BACKUP = "data/chunks/chunks_clean.jsonl"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EXCLUDE = "附件1"   # source 含此串的 chunk 被剔除


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 1. 全量备份（无论后面做什么，先把干净数据落盘）
    old = client.get_collection(COLLECTION)
    data = old.get(include=["documents", "metadatas"])
    with open(BACKUP, "w", encoding="utf-8") as f:
        for doc, meta in zip(data["documents"], data["metadatas"]):
            f.write(json.dumps({"content": doc, "metadata": meta},
                               ensure_ascii=False) + "\n")
    print(f"1/4 已备份 {len(data['documents'])} 块 → {BACKUP}")

    # 2. 剔除
    kept = [(d, m) for d, m in zip(data["documents"], data["metadatas"])
            if EXCLUDE not in json.dumps(m, ensure_ascii=False)]
    print(f"2/4 剔除'{EXCLUDE}'后保留 {len(kept)} 块（剔除 "
          f"{len(data['documents']) - len(kept)} 块）")

    if args.dry_run:
        print("dry-run 结束，未重建")
        return

    # 3. 删旧库、建余弦空间新库
    client.delete_collection(COLLECTION)
    col = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"})
    print("3/4 已删除旧库，新建 cosine 空间 collection")

    # 4. 重新编码入库
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode([d for d, _ in kept],
                       normalize_embeddings=True).tolist()
    col.add(
        ids=[f"chunk-{i}" for i in range(len(kept))],
        embeddings=emb,
        documents=[d for d, _ in kept],
        metadatas=[m for _, m in kept],
    )
    print(f"4/4 已入库 {col.count()} 块。现在跑: python src/eval.py --top-k 3")


if __name__ == "__main__":
    main()