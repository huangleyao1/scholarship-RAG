import pymupdf4llm
import os
import re

def convert_pdf_to_markdown(pdf_path, output_md_path):
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)

    # 用于存放提取出的图片（公式等）
    image_dir = os.path.join(os.path.dirname(output_md_path), "images")
    os.makedirs(image_dir, exist_ok=True)

    print(f"正在转换 {os.path.basename(pdf_path)}...")

    # write_images=True 会将嵌入的公式等对象导出为图片
    md_text = pymupdf4llm.to_markdown(
        pdf_path,
        write_images=True,
        image_path=image_dir
    )

    # 后处理：修复合并单元格
    md_text = fix_merged_tables(md_text)

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"成功保存到 {output_md_path}\n")


def fix_merged_tables(md_text):
    """修复表格中因合并单元格导致的 <br> 问题"""
    lines = md_text.split("\n")
    fixed_lines = []
    parent_category = None

    for line in lines:
        # 匹配 '|XXX<br>YYY|ZZZ|' 这种模式
        match = re.match(r"\|(.+?)<br>(.+?)\|(.+?)\|", line)
        if match:
            parent = match.group(1).strip()
            child = match.group(2).strip()
            value = match.group(3).strip()
            parent_category = parent
            fixed_lines.append(f"| {parent} - {child} | {value} |")
        elif parent_category and re.match(r"\|(.+?)\|(.+?)\|", line):
            match2 = re.match(r"\|(.+?)\|(.+?)\|", line)
            if match2 and not line.startswith("|---"):
                cell1 = match2.group(1).strip()
                if len(cell1) <= 6 and not cell1.startswith("校") and not cell1.startswith("无"):
                    fixed_lines.append(f"| {parent_category} - {cell1} | {match2.group(2).strip()} |")
                    continue
                else:
                    parent_category = None
            else:
                parent_category = None
        else:
            if not line.startswith("|"):
                parent_category = None

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


# 输入 PDF 路径
path1 = "/Users/hly/Desktop/scholarship-RAG/data/raw/定稿-2025年修订-人工智能与计算机学院本科生综合素质考评实施细则.pdf"
path2 = "/Users/hly/Desktop/scholarship-RAG/data/raw/附件1：关于印发《陕西师范大学本科生综合素质考评办法（修订）》的通知.pdf"

# 输出 Markdown 路径
out_path1 = "/Users/hly/Desktop/scholarship-RAG/data/processed/定稿-2025年修订-人工智能与计算机学院本科生综合素质考评实施细则.md"
out_path2 = "/Users/hly/Desktop/scholarship-RAG/data/processed/附件1：关于印发《陕西师范大学本科生综合素质考评办法（修订）》的通知.md"

convert_pdf_to_markdown(path1, out_path1)
convert_pdf_to_markdown(path2, out_path2)