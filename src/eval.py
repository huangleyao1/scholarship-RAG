#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from collections import defaultdict

import chromadb
from sentence_transformers import SentenceTransformer

# ===================== 配置区（按你的项目实际改） =====================
CHROMA_DIR   = "data/chroma"
EVAL_SET     = "data/eval_set.jsonl"
REPORT       = "eval_report.json"

# 必须和 src/embed.py 建库时用的模型完全一致
MODEL_NAME   = "BAAI/bge-small-zh-v1.5"
# BGE 官方推荐：短查询加指令前缀，通常能提升 5~10 个点
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# metadata 里"来源文件"字段的候选名（自动探测，命中第一个就用）
SOURCE_KEYS = ["source", "file", "filename", "file_name", "document", "doc", "path"]
# =====================================================================


def load_eval_set(path):
    items = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            raw = it.get("gt", "").strip()
            if raw in ("", "TODO"):
                skipped += 1
                continue
            # 多候选 gt：分号分隔，命中任一即算对
            it["gts"] = [g.strip() for g in raw.split(";") if g.strip()]
            items.append(it)
    if skipped:
        print(f"[提示] 有 {skipped} 条 gt 还是 TODO，已跳过，记得去 data/eval_set.jsonl 补完\n")
    return items


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    cols = client.list_collections()
    if not cols:
        raise SystemExit(f"错误：{CHROMA_DIR} 里没有 collection，先跑 src/embed.py 建库")
    col = client.get_collection(cols[0].name)
    print(f"使用 collection: {col.name}（共 {col.count()} 个向量）")
    return col


def pick_source(meta: dict) -> str:
    for k in SOURCE_KEYS:
        if k in meta and meta[k]:
            return str(meta[k])
    return ""


def _hit_one(gt: str, hits: list) -> str:
    """clause=条款级命中 / file=仅文件命中（条款没对上）/ miss=完全没命中"""
    if "::" in gt:
        file_part, clause_part = gt.split("::", 1)
    else:
        file_part, clause_part = gt, None
    file_part = file_part.strip()
    clause_norm = (clause_part.replace(" ", "").replace("　", "")
                   if clause_part else None)

    for h in hits:
        src = h["source"]
        blob = (h["text"] + " " + json.dumps(h["meta"], ensure_ascii=False))
        blob = blob.replace(" ", "").replace("　", "")
        file_ok = (file_part in src) or src.endswith(file_part) or (file_part in blob)
        if not file_ok:
            continue
        if clause_norm is None:
            return "file"
        if clause_norm in blob:
            return "clause"
    return "miss"


def hit_level(gts: list, hits: list) -> str:
    """多条候选 gt 取最优结果：clause > file > miss"""
    best = "miss"
    for gt in gts:
        lv = _hit_one(gt, hits)
        if lv == "clause":
            return "clause"
        if lv == "file":
            best = "file"
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-instruction", action="store_true",
                    help="关闭 BGE 查询指令（与 embed.py 保持一致时用）")
    args = ap.parse_args()
    K = max(args.top_k, 5)

    items = load_eval_set(EVAL_SET)
    if not items:
        raise SystemExit("eval_set.jsonl 里没有可用的评测条目，先填 gt")

    col = get_collection()
    model = SentenceTransformer(MODEL_NAME)

    queries = [it["q"] for it in items]
    if args.no_instruction:
        emb = model.encode(queries, normalize_embeddings=True).tolist()
    else:
        emb = model.encode([QUERY_INSTRUCTION + q for q in queries],
                           normalize_embeddings=True).tolist()

    res = col.query(query_embeddings=emb, n_results=K)

    report, stats = [], defaultdict(lambda: defaultdict(int))
    for i, it in enumerate(items):
        hits = [{
            "source": pick_source(res["metadatas"][i][j]),
            "text": res["documents"][i][j],
            "meta": res["metadatas"][i][j],
        } for j in range(len(res["ids"][i]))]

        level = hit_level(it["gts"], hits)
        top1 = hits[0]["source"] if hits else ""

        for k in (1, 3, 5):
            if k <= len(hits):
                lv = hit_level(it["gts"], hits[:k])
                stats["overall"][f"top{k}_{lv}"] += 1
                stats[it["type"]][f"top{k}_{lv}"] += 1

        report.append({
            "id": it["id"], "type": it["type"], "q": it["q"],
            "gt": it["gts"], "top1_source": top1, "level": level,
        })

    n = len(items)
    print(f"\n评测条数: {n}")
    print("=" * 56)
    print(f"{'口径':<10}{'Top-1':>10}{'Top-3':>10}{'Top-5':>10}{'条款级命中(Top-3)':>18}")
    for group in ["overall"] + sorted({it["type"] for it in items}):
        cnt = stats[group]
        tot = n if group == "overall" else sum(1 for it in items if it["type"] == group)
        def rate(k, levels):
            return sum(cnt.get(f"top{k}_{l}", 0) for l in levels) / tot * 100
        loose = [rate(1, ("clause", "file")), rate(3, ("clause", "file")), rate(5, ("clause", "file"))]
        strict3 = rate(3, ("clause",))
        print(f"{group:<10}{loose[0]:>9.1f}%{loose[1]:>9.1f}%{loose[2]:>9.1f}%{strict3:>17.1f}%")

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump({"n": n, "stats": {k: dict(v) for k, v in stats.items()},
                   "details": report}, f, ensure_ascii=False, indent=2)
    print(f"\n明细已写入 {REPORT}")
    print("提示：'file' 命中但条款没中 = 切块问题；'miss' = 检索问题（换 Top-K / 加指令 / 换模型）")


if __name__ == "__main__":
    main()