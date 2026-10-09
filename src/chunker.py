import os
import re
import json

PROCESSED_DIR = "/Users/hly/Desktop/scholarship-RAG/data/processed"
CHUNK_DIR = "/Users/hly/Desktop/scholarship-RAG/data/chunks"

def chunk_by_article(md_text,source):
    """规则文档切块：按'第X条'切分，每条一个块"""
    # 旧：ARTICLE_RE = re.compile(r'第[一二三四五六七八九十百千零]+条')
    ARTICLE_RE = re.compile(r'第[一二三四五六七八九十百零0-9]+条|^\s*（[一二三四五六七八九十]+）', re.M)
    parts = ARTICLE_RE.split(md_text)
    markers = ARTICLE_RE.findall(md_text)

    chunks = []

    # patr[0]: 文件头部第一条前面的部分
    head = parts[0].strip()
    if head:
        chunks.append({
            "content":head,
            "metadata":{
                "source":source,
                "article":"头部"
            }
        })

    for i,body in enumerate(parts[1:]):
        article = markers[i]
        content = (article + body).strip()
        if content:
            chunks.append({
                "content":content,
                "metadata":{
                    "source":source,
                    "article":article
                }
            })     # 切分块的结构
    return chunks


def chunk_form(md_text, source):
    chunks = []
    parts = re.split(r'(?m)^(?=## )', md_text)
    for part in parts:
        part = part.strip()     # strip：去掉末尾\n
        if not part:            
            continue
        m = re.match(r'^##\s+(.+)', part)  
        title = m.group(1) if m else "头部"  
        chunks.append({
            "content": part,
            "metadata": {"source": source, "section": title}
        })                     
    return chunks


def save_chunks(chunks, filename):
    """存成 JSONL:每行一个 JSON"""
    os.makedirs(CHUNK_DIR, exist_ok=True)
    path = os.path.join(CHUNK_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"✅ {filename}: {len(chunks)} 个块")



if __name__ == "__main__":
    jobs = [
        ("定稿-2025年修订-人工智能与计算机学院本科生综合素质考评实施细则.md", "chunk_by_article", "学院细则.jsonl"),
        ("附件1：关于印发《陕西师范大学本科生综合素质考评办法（修订）》的通知.md", "chunk_by_article", "学校办法.jsonl"),
        ("人工智能与计算机学院2025—2026学年本科生综合素质考评表.md", "chunk_form", "测评表.jsonl"),
    ]
    for md_name, func_name, out_name in jobs:
        with open(os.path.join(PROCESSED_DIR, md_name), encoding="utf-8") as f:
            text = f.read()
        chunks = globals()[func_name](text, source=md_name.replace(".md", ""))
        save_chunks(chunks, out_name)









