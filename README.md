# scholarship-RAG
基于检索增强生成（RAG）的高校综合素质测评细则问答系统。
将 PDF / Word 考评文档解析为结构化知识库，支持语义检索，为后续接入大模型问答做准备。

## 功能
1、多格式文档解析：PDF（PyMuPDF）与 Word（python-docx）→ 结构化 Markdown
2、语义切块：规则文档按"第X条"、测评表按栏目标题切分并添加相应文字描述
3、向量化：BGE-small-zh 中文 Embedding 模型
4、向量存储：ChromaDB 本地持久化
5、检索测试：命令行交互式提问，返回 Top-3 相关条款

## 技术栈
Python 3.11 · PyMuPDF · python-docx · sentence-transformers (BGE) · ChromaDB

## 项目结构
scholarship-RAG/
├── data/
│   ├── raw/          # 原始文档（PDF × 2 + 测评表 docx）
│   ├── processed/    # 解析生成的 Markdown
│   ├── chunks/       # 切块产物（JSONL）
│   └── chroma/       # 向量数据库（自动生成）
├── src/
│   ├── pdf1md.py     # PDF → Markdown
│   ├── docx2md.py    # Word → Markdown
│   ├── chunker.py    # 语义切块
│   └── embed.py      # 向量化入库 + 检索测试
└── .gitignore

## 开发路线
[✅] 文档解析与数据清洗（含人工校对表格）
[✅] 语义切块与元数据标注
[✅] 向量化与检索测试
[ ] 接入 LLM API 生成回答
[ ] 多轮对话支持
[ ] LangChain 重构对比
[ ] 检索质量评测集

## 踩坑记录
检索失败的常见根因是切块问题（如附录粘附成巨型块导致语义稀释），
详见开发笔记 [notes.md](src/notes.md)。

## 快速开始
```bash
# 1. 安装依赖
pip install pymupdf python-docx sentence-transformers chromadb

# 2. 解析文档（在 data/raw/ 放置原始文件后）
python src/pdf1md.py
python src/docx2md.py

# 3. 切块
python src/chunker.py

# 4. 向量化入库 + 提问测试
python src/embed.py


