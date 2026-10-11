# 奖学金-RAG

基于检索增强生成（RAG）的高校综合素质测评细则问答系统。
将 PDF / Word 考评文档解析为结构化知识库，结合语义检索与大模型生成回答，答案附带来源条款，可溯源、可核验。
（说明：data/raw/ 中的文档为陕西师范大学相关测评文件，仅供学习与技术演示使用。）

## 效果示例

> **问**：通过六级可以加多少分？填到综测表的什么位置？
> **答**：通过六级可以加0.5分。填写位置：应填在综测表的 “附加项”部分，具体归入“智育”模块。每行填写三项：加分项（写“英语六级证书”）、加分分数（0.5分）、模块总加分；全部填完后合计“附加项总分”。补充说明：英语六级按425分计，大于或等于425分才加分，否则不加分；同一学年至多加一次分；同一证书四年中仅计算一次。
> **来源**：依据条款：《人工智能与计算机学院本科生综合素质考评实施细则（2025年修订）》“（二）智育附加分加分标准”第1条；《人工智能与计算机学院2025—2026学年本科生综合素质考评表》“附加项”填写说明

## 功能特性

1. **多格式文档解析**：PDF（PyMuPDF）与 Word（python-docx）→ 结构化 Markdown
2. **语义切块**：规则文档按"第X条"切分、测评表按栏目标题切分，并补充相应文字描述，避免表格语义丢失
3. **向量化**：BGE-small-zh-v1.5 中文 Embedding 模型 + ChromaDB 本地持久化
4. **查询改写**：长查询先用 LLM 改写再检索，缓解长问题语义漂移导致的检索失败
5. **可溯源回答**：返回答案的同时标注来源文件与具体条款
6. **检索测试**：命令行交互式提问，返回 Top-3 相关条款及来源文件

## 技术栈

Python 3.11 · PyMuPDF · python-docx · sentence-transformers（BGE-small-zh-v1.5）· ChromaDB · LLM API

## 项目结构

scholarship-RAG/
├── data/
│   ├── raw/          # 原始文档（PDF × 2 + 测评表 docx）
│   ├── processed/    # 解析生成的 Markdown
│   ├── chunks/       # 切块产物（JSONL）
│   └── chroma/       # 向量数据库（自动生成，已 gitignore）
├── src/
│   ├── pdf2md.py     # PDF → Markdown
│   ├── docx2md.py    # Word → Markdown
│   ├── chunker.py    # 语义切块
│   ├── embed.py      # 向量化入库 + 检索测试
│   ├── rag.py        # 检索 + 查询改写 + 调 LLM API + 生成回答
│   ├── config.py     # API Key 等配置（已 gitignore，见下）
│   └── notes.md      # 开发笔记与踩坑记录
├── .gitignore
└── requirements.txt

## 开发路线

1、 文档解析与数据清洗（含人工校对表格）
2、 语义切块与元数据标注
3、 向量化与检索测试
4、 接入 LLM API 生成回答
5、 LLM 查询改写（修复长查询检索失败）
6、 来源映射与答案溯源

## 踩坑纪录

详细问题及修复思路见src/notes.md

## 快速开始

```bash
# 1. 克隆并进入项目
git clone https://github.com/huangleyao1/scholarship-RAG.git
cd scholarship-RAG

# 2. 创建虚拟环境并激活
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key
# 在 src/config.py 中填入你的 LLM API Key（该文件已被 .gitignore 排除）

# 5. 解析文档（data/raw/ 已附带示例文档）
python src/pdfmd.py
python src/docxmd.py

# 6. 语义切块
python src/chunker.py

# 7. 向量化入库 + 检索测试
python src/embed.py

# 8. 启动问答
python src/rag.py
