'''
1. 装模子   加载 embedding 模型（文本→向量的翻译官）
2. 开仓库   建一个本地向量数据库（ChromaDB，就是个文件夹存放原文和标签）
3. 搬货     读入三个 JSONL 里所有的块
4. 入库     每块算出一个向量，连同原文和标签存进仓库
5. 提问     把问题也变成向量，去仓库找最相似的 3 块
'''

import glob
import chromadb
from sentence_transformers import SentenceTransformer
import json

MODEL_NAME = 'BAAI/bge-small-zh-v1.5'
CHROMA_DIR = '/Users/hly/Desktop/scholarship-RAG/data/chroma'

# 1. 装模子
model = SentenceTransformer(MODEL_NAME)

# 2. 开仓库:创建 ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("scholarship")

# 3. 搬货 
chunks = []
for path in glob.glob("/Users/hly/Desktop/scholarship-RAG/data/chunks/*.jsonl"):
    with open(path,encoding = 'utf-8') as f:
        for line in f :
            chunks.append(json.loads(line))
            print(f"读到 {len(chunks)} 个块")

# 4. 入库
texts = [c['content'] for c in chunks]
embeddings = model.encode(texts,normalize_embeddings=True).tolist()
collection.add(
    ids=[f"chunk-{i}" for i in range(len(chunks))],  
    embeddings=embeddings,
    documents=texts,
    metadatas=[c["metadata"] for c in chunks]
)
print(f"✅ 已入库 {collection.count()} 块")

# 5. 提问测试（小循环，q 退出）
while True:
    q = input("\n提问(q 退出)：").strip()
    if q == 'q':
        break
    q_emb = model.encode([q],normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=3
    )
    for doc,meta in zip(results["documents"][0],results["metadatas"][0]):
        print("-" * 40)
        print("标签：", meta)
        print(doc[:500]) # 只打印前500字，防止刷屏








