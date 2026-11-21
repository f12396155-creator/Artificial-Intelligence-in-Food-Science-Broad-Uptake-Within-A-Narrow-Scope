import os
import json
import time
import csv
import argparse
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# =======================
# 可调参数
# =======================
COOLDOWN_SEC = 1          
PDF_MAX_CHARS = 120_000    
HTTP_TIMEOUT_DEFAULT = 180
MAX_RETRIES_DEFAULT = 10

# =====================================================
# AI Food Prompt & Rules (内置规则)
# =====================================================
AI_FOOD_PROMPT = """你是一个专业的食品科学与人工智能交叉领域的文献数据提取专家。
请仔细阅读下方提供的文献内容，并严格按照以下极度精确的编码规则，提取对应的 JSON 字段。

【核心原则】
1. **单一主标签与核心贡献**：对于所有分类字段，每篇文献只能选择 1 个最核心的主标签，绝不允许出现多标签并列（如 "A+B"）。编码必须基于文章最主要解决的问题、最核心的模型和最主要的 AI 作用，不要机械罗列文中作为对比或顺带提及的次要内容。若文章涉及多个标签，主标签的确定顺序依次为：标题 -> 摘要研究目标 -> 结论主结果 -> 正文重点。
2. **证据来源定位**：任务与应用场景类优先依据 Title、Abstract、Conclusion 提取；模型与算法类优先依据 Methods、Model、Algorithm 提取；知识融合类优先依据 Discussion、Conclusion 提取。
3. **无明确报告写 NR**：如果原文未明确报告或无法稳定判断，绝对不要主观猜测，统一填写 "NR" (Not Report)。
4. **仅输出 JSON**：不要在 JSON 之外输出任何 markdown 格式、推理过程或纯文本解释。

【基础元数据提取规则】（找不到一律填 NR）：
- article_id：优先提取文章的 DOI 号（如 10.1016/j.xxx），如果没有 DOI，则寻找唯一的文章编号 (Article ID)。
- first_author：第一作者姓名。
- title：文章的完整标题。
- journal：发表该文章的期刊名称。
- year：发表年份（提取4位数字，如 2021）。
- country：第一作者或通讯作者所在的国家名称（通常在作者单位 Affiliation 中）。
- citation：文章的完整引用格式或期刊/卷/期/页码信息。

【字段分类定义及严格选项】（必须且只能从指定选项中选择 1 个）：
- task_type (1选1): classification, regression, detection, generation (未提及填 NR)
- application_scenario (1选1): quality, safety, formulation, nutrition (未提及填 NR)
- target_object (1选1): component, contaminant, sensory, formula (未提及填 NR)
- ai_type (1选1): ML, DL, LLM, hybrid (未提及填 NR)
- algorithm: (字符串)
- data_type (1选1): image, spectral, text, multimodal (未提及填 NR)
- ai_role (1选1): detection, classification, prediction, optimization, generation, recommendation (未提及填 NR)
- domain_integration_level (1选1): AI-only, weak integration, strong integration (未提及填 NR)
"""

# =====================================================
# Schema builder (Hardcoded for AI Food)
# =====================================================
class AIFoodSchema(BaseModel):
    article_id: Optional[str] = Field(default="NR")
    first_author: Optional[str] = Field(default="NR")
    title: Optional[str] = Field(default="NR")
    journal: Optional[str] = Field(default="NR")
    year: Optional[str] = Field(default="NR")
    country: Optional[str] = Field(default="NR")
    citation: Optional[str] = Field(default="NR")
    task_type: Optional[str] = Field(default="NR")
    application_scenario: Optional[str] = Field(default="NR")
    target_object: Optional[str] = Field(default="NR")
    ai_type: Optional[str] = Field(default="NR")
    algorithm: Optional[str] = Field(default="NR")
    data_type: Optional[str] = Field(default="NR")
    ai_role: Optional[str] = Field(default="NR")
    domain_integration_level: Optional[str] = Field(default="NR")
    remarks: Optional[str] = Field(default="NR")

def schema_keys() -> List[str]:
    return list(AIFoodSchema.model_fields.keys())

def clean_and_parse_json(text: str) -> Dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"```(?:json)?(.*?)```", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    text = text.strip()
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            return {}
        return obj
    except Exception:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
        return {}

def coerce_schema(json_text: str) -> Dict[str, Any]:
    obj = clean_and_parse_json(json_text)
    out = {}
    for k in schema_keys():
        v = obj.get(k, "NR")
        if v is None or str(v).strip() == "":
            out[k] = "NR"
        else:
            out[k] = str(v).strip()
    return out

def derive_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    data_type = data.get("data_type", "").lower()
    ai_type = data.get("ai_type", "").upper()
    if ai_type == "NR":
        data["tech_depth_level"] = "NR"
    elif "multimodal" in data_type:
        data["tech_depth_level"] = "Multimodal AI"
    elif ai_type == "LLM":
        data["tech_depth_level"] = "LLM"
    elif ai_type == "DL":
        data["tech_depth_level"] = "DL"
    else:
        data["tech_depth_level"] = "ML"
    app_scenario = data.get("application_scenario", "").lower()
    mapping = {
        "quality": "Quality",
        "safety": "Safety",
        "formulation": "Design",
        "nutrition": "Nutrition"
    }
    data["food_system_level"] = mapping.get(app_scenario, "NR")
    return data

# =====================================================
# GPT-4o API调用
# =====================================================
def call_gpt4o(api_key: str, model: str, pdf_text: str, timeout: int = HTTP_TIMEOUT_DEFAULT, max_retries: int = MAX_RETRIES_DEFAULT) -> Dict[str, Any]:
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY（请写到 .env）")
    
    url = "https://api.openai.com/v1/chat/completions"  # 使用chat/completions端点
    
    # 修改请求格式，采用聊天模型需要的messages格式
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": AI_FOOD_PROMPT},
            {"role": "user", "content": f"PDF内容:\n{pdf_text}"}
        ],
        "max_tokens": 8000,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 200:
                content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                return coerce_schema(content)
            if response.status_code in (408, 429, 500, 502, 503, 504):
                print(f"   🧊 GPT-4o {response.status_code} retry {attempt}/{max_retries} ...")
                time.sleep(min(60, 5 * (2 ** (attempt - 1))))
                continue
            raise RuntimeError(f"HTTP {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"   🧊 连接错误 retry {attempt}/{max_retries} ... {str(e)}")
            time.sleep(min(60, 5 * (2 ** (attempt - 1))))
    raise RuntimeError("GPT-4o API调用失败")

# =====================================================
# PDF处理
# =====================================================
def pdf_to_text(pdf_path: str, max_chars: int = PDF_MAX_CHARS) -> Tuple[str, Dict[str, Any]]:
    if fitz is None:
        raise RuntimeError("缺少 PyMuPDF（fitz）。请 pip install pymupdf")
    doc = fitz.open(pdf_path)
    texts, total, pages_read = [], 0, 0

    for page in doc:
        t = page.get_text("text") or ""
        if not t: continue
        texts.append(t)
        total += len(t)
        pages_read += 1
        if total >= max_chars:
            break

    text = "\n".join(texts)[:max_chars]
    return text, {"pages_read": pages_read, "chars": len(text)}

# =====================================================
# 时间日志记录
# =====================================================
def append_timing_row(csv_path: str, row: Dict[str, Any]) -> None:
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    cols = ["ts", "provider", "model", "run_id", "pdf", "status", "runtime_sec", "pages_read", "chars", "error_type", "error_msg"]
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header: w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})

# =====================================================
# 主执行代码
# =====================================================
def main(args):
    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"没有找到 PDF：{args.pdf_dir}")

    out_dir = Path(args.out_dir) / "AI_Food"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_tag = args.model.replace("/", "_")
    out_jsonl = out_dir / f"{args.provider}__{model_tag}__run{args.run_id}.jsonl"
    timing_csv = str(out_dir / "_extract_timing_log.csv")

    done = set()
    if done: 
        print(f"✅ resume 启用：已跳过 {len(done)} 篇")

    print(f"找到 {len(pdfs)} 个 PDF | provider={args.provider} | model={args.model}")
    print(f"输出 JSONL：{out_jsonl}")

    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}")
        t0 = time.perf_counter()
        status, err_type, err_msg = "FAIL", "", ""
        meta = {"pages_read": None, "chars": None}
        runtime = None

        try:
            pdf_text, meta = pdf_to_text(str(pdf))

            if args.provider == "openai":
                data = call_gpt4o(
                    os.getenv("OPENAI_API_KEY"),
                    args.model,
                    pdf_text,
                    timeout=int(getattr(args, "timeout", HTTP_TIMEOUT_DEFAULT)),
                    max_retries=int(getattr(args, "max_retries", MAX_RETRIES_DEFAULT)),
                )

            data = derive_fields(data)

            runtime = round(time.perf_counter() - t0, 2)
            data["_pdf"] = pdf.name
            data["_provider"] = args.provider
            data["_model"] = args.model
            data["_runtime_s"] = runtime

            with open(out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

            status = "OK"
            print(f"   ✔ 完成 | _runtime_s={runtime}s")

        except Exception as e:
            runtime = round(time.perf_counter() - t0, 2)
            err_type = type(e).__name__
            err_msg = str(e)
            print(f"   ❌ 失败 | {err_msg}")

        append_timing_row(timing_csv, {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "provider": args.provider, "model": args.model, "run_id": args.run_id,
            "pdf": pdf.name, "status": status, "runtime_sec": runtime,
            "pages_read": meta.get("pages_read"), "chars": meta.get("chars"),
            "error_type": err_type, "error_msg": err_msg,
        })

        if args.cooldown_sec > 0:
            time.sleep(args.cooldown_sec)

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", required=True)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--out_dir", default=r"D:\CODE\AI food mission\outputs_raw")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--timeout", type=int, default=HTTP_TIMEOUT_DEFAULT)
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES_DEFAULT)
    parser.add_argument("--cooldown_sec", type=int, default=COOLDOWN_SEC)
    
    args, _ = parser.parse_known_args()
    main(args)