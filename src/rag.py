import requests 
import chromadb 
from sentence_transformers import SentenceTransformer
from config import API_KEY

MODEL = SentenceTransformer("BAAI/bge-small-zh-v1.5")
client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection(name="scholarship")
NAME_MAP = {
    "定稿-2025年修订-人工智能与计算机学院本科生综合素质考评实施细则":
        "《人工智能与计算机学院本科生综合素质考评实施细则（2025年修订）》",
    "附件1：关于印发《陕西师范大学本科生综合素质考评办法（修订）》的通知":
        "《陕西师范大学本科生综合素质考评办法（修订）》",
    "人工智能与计算机学院2025—2026学年本科生综合素质考评表":
        "《人工智能与计算机学院2025—2026学年本科生综合素质考评表》",
}

def expand_query(q):
    """让LLM把长问题改写成短检索词组"""
    prompt = ("你是检索助手。把用户的问题改写成2-3个简短的检索关键词组，"
              "每个词组一行，只输出词组本身，不要解释。\n\n"
              f"问题：{q}")
    r = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": "deepseek-chat",
              "messages": [{"role": "user", "content": prompt}]},
    )
    text = r.json()["choices"][0]["message"]["content"]
    queries = [line.strip() for line in text.split("\n") if line.strip()]
    queries.append(q)          # 原始问题保留一份
    return queries[:4]

def retrieve(q, k=3):
    """检索：把问题转向量，捞回最相关的k个块"""
    instruction = "为这个句子生成表示以用于检索相关文章：" ###
    seen = {}
    for sub in expand_query(q):           # 每个词组单独检索
        emb = MODEL.encode([instruction + sub], normalize_embeddings=True).tolist()
        res = collection.query(query_embeddings=emb, n_results=k)
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            if doc not in seen:           # 不同词组可能捞到同一块，去重
                seen[doc] = (doc, meta)
    blocks = []
    for doc, meta in list(seen.values())[:8]:   # 最多喂8块，防上下文爆炸
        source = NAME_MAP.get(meta["source"], meta["source"])
        blocks.append(f"【来源：{source}】\n{doc}")
    return "\n\n---\n\n".join(blocks)

def ask(q):
    """生成：检索 + 拼prompt + 调API"""
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
    r = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"model": "deepseek-chat",
              "messages": [{"role": "user", "content": prompt}]},
    )
    return r.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    while True:
        q = input("\n提问（q 退出）：").strip()
        if q == "q":
            break
        print(ask(q))








