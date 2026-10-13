#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_lab.py — 对照实验：排除指定来源后重测检索召回

为什么不用 chroma 的 where 过滤：数据只有 75 块，内存检索秒级完成，
且实现透明，方便你确认评测逻辑（顺便当作"独立参考实现"交叉验证 eval.py）。

用法（项目根目录）：
    python src/eval_lab.py                 # 对比：不排除 vs 排除学校办法
    python src/eval_lab.py --exclude 附件1  # 排除 source 含"附件1"的 chunk
"""

import json
import argparse
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "data/chroma"
EVAL_SET = "data/eval_set.jsonl"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
SOURCE_KEYS = ["source", "file", "filename", "file_name", "document", "doc", "path"]


def load_all():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(client.list_collections()[0].name)
    data = col.get(include=["documents", "metadatas"])
    chunks = []
    for doc, meta in zip(data["documents"], data["metadatas"]):
        src = next((str(meta[k]) for k in SOURCE_KEYS if k in meta and meta[k]), "")
        chunks.append({"text": doc, "meta": meta, "source": src})
    return chunks


def load_qa():
    items = []
    with open(EVAL_SET, encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            raw = it.get("gt", "").strip()
            if raw and raw != "TODO":
                it["gts"] = [g.strip() for g in raw.split(";") if g.strip()]
                items.append(it)
    return items


def hit_level(gts, hits):
    best = "miss"
    for gt in gts:
        if "::" in gt:
            fp, cp = gt.split("::", 1)
        else:
            fp, cp = gt, None
        fp = fp.strip()
        cn = cp.replace(" ", "").replace("　", "") if cp else None
        for h in hits:
            src, blob = h["source"], (h["text"] + " " + json.dumps(h["meta"], ensure_ascii=False))
            blob = blob.replace(" ", "").replace("　", "")
            if not ((fp in src) or src.endswith(fp) or (fp in blob)):
                continue
            if cn is None:
                best = max(best, "file", key=["miss", "file", "clause"].index)
                break
            if cn in blob:
                return "clause"
            best = "file"
    return best


def evaluate(chunks, items, model, topk=3):
    docs_emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    qs = [QUERY_INSTRUCTION + it["q"] for it in items]
    qs_emb = model.encode(qs, normalize_embeddings=True)
    sim = qs_emb @ docs_emb.T
    hit = miss = 0
    for i, it in enumerate(items):
        idx = np.argsort(-sim[i])[:topk]
        lv = hit_level(it["gts"], [chunks[j] for j in idx])
        if lv == "clause":
            hit += 1
        else:
            miss += 1
    return hit, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="附件1")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    chunks = load_all()
    items = load_qa()
    model = SentenceTransformer(MODEL_NAME)

    base_hit, base_miss = evaluate(chunks, items, model, args.top_k)
    filtered = [c for c in chunks if args.exclude not in c["source"]]
    exp_hit, exp_miss = evaluate(filtered, items, model, args.top_k)

    n = len(items)
    print(f"总chunk: {len(chunks)} → 排除'{args.exclude}'后: {len(filtered)}")
    print(f"{'':<14}{'命中':>6}{'条款级命中率(Top-%d)' % args.top_k:>22}")
    print(f"{'不排除':<14}{base_hit:>6}{base_hit / n * 100:>20.1f}%")
    print(f"{'排除后':<14}{exp_hit:>6}{exp_hit / n * 100:>20.1f}%")


if __name__ == "__main__":
    main()