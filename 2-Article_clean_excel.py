import json
import argparse
import pandas as pd
from pathlib import Path

# 我们刚刚在提取脚本里定义的 AI Food 专属字段顺序
AI_FOOD_COLUMNS = [
    "_pdf",                  # 文件名放在最前面方便对照
    "article_id", "first_author", "title", "journal", "year", "country", "citation",
    "task_type", "application_scenario", "target_object",
    "ai_type", "algorithm", "data_type",
    "tech_depth_level", "food_system_level",  # 这是我们代码自动派生的两个字段
    "ai_role", "domain_integration_level", "remarks",
    "_provider", "_model", "_runtime_s"       # 运行元数据放在最后
]

def main(args):
    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_excel)
    
    if not in_path.exists():
        print(f"❌ 找不到输入文件: {in_path}")
        return

    # 1. 读取 JSONL 文件
    data = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception as e:
                print(f"⚠️ 解析 JSON 失败，跳过一行: {e}")

    if not data:
        print("❌ 没有读取到任何数据，退出。")
        return

    # 2. 转换为 Pandas DataFrame
    df = pd.DataFrame(data)

    # 3. 整理列顺序
    # 确保我们定义的核心列都在，没有的补上空值
    for col in AI_FOOD_COLUMNS:
        if col not in df.columns:
            df[col] = "NR"
            
    # 把不在列表里的其他未知列（如果有）放到最后
    extra_cols = [c for c in df.columns if c not in AI_FOOD_COLUMNS]
    final_cols = AI_FOOD_COLUMNS + extra_cols
    
    df = df[final_cols]

    # 4. 导出为真正的 Excel 文件 (.xlsx)
    # 确保输出目录存在
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 导出时去掉自动生成的索引列
    df.to_excel(out_path, index=False)
    print(f"✅ 转换成功！共处理 {len(df)} 篇文献，Excel 已保存至：{out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", required=True, help="第一步生成的 jsonl 文件路径")
    parser.add_argument("--out_excel", required=True, help="想要保存的 excel 文件路径，以 .xlsx 结尾")
    args = parser.parse_args()
    main(args)