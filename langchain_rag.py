import argparse
import ast
import json
import re
import warnings
from pathlib import Path
from typing import Dict, List

import nltk
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from nltk.tokenize import word_tokenize
from rich.console import Console
from rich.table import Table

from keyword_finder import map_keywords_in_code

warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")
nltk.download("punkt_tab", quiet=True)

console = Console()
BASE_DIR = Path(__file__).resolve().parent

STACK_NAME = "Tax_Calculator"
PROJECT_PATH = str(BASE_DIR / "Tax-Calculator")
CONTEXT_PATH = BASE_DIR / "ai_context.txt"
STOPWORDS = {"what", "how", "when", "where", "which", "with", "from", "into", "about", "under", "that", "this", "does", "have", "your", "their", "there", "is", "are", "for", "the", "and"}


def build_query_keywords(query: str, limit: int = 8) -> List[str]:
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9_]+", query) if len(t) >= 4]
    tokens = [t for t in tokens if t not in STOPWORDS]
    phrases: List[str] = []
    for i in range(len(tokens) - 1):
        phrases.append(f"{tokens[i]} {tokens[i+1]}")
    merged = list(dict.fromkeys(tokens + phrases))
    return merged[:limit]


def build_keyword_map(keywords: List[str]) -> Dict[str, List[str]]:
    return map_keywords_in_code(keywords, PROJECT_PATH)


def build_context(query: str) -> str:
    if not CONTEXT_PATH.exists():
        return "No ai_context.txt found. Run main_rag.py first if needed."

    chunks = CONTEXT_PATH.read_text(encoding="utf-8", errors="ignore").split("\n\n")
    docs = [Document(page_content=c) for c in chunks if c.strip()]
    if not docs:
        return "ai_context.txt exists but is empty."

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 6
    retriever.preprocess_func = word_tokenize
    relevant = retriever.invoke(query)
    snippets = []
    for i, d in enumerate(relevant, 1):
        txt = " ".join(d.page_content.split())
        snippets.append(f"[{i}] {txt[:420]}")
    return "\n".join(snippets)


def build_reference_list(keyword_map: Dict[str, List[str]], limit: int = 80) -> List[str]:
    refs: List[str] = []
    for entries in keyword_map.values():
        refs.extend(entries)
    return list(dict.fromkeys(refs))[:limit]


def show_keyword_table(keyword_map: Dict[str, List[str]], keywords: List[str]) -> None:
    table = Table(title=f"{STACK_NAME} Keyword Hits", show_lines=True, expand=True)
    table.add_column("Keyword", style="cyan", ratio=2)
    table.add_column("Hit Count", style="yellow", ratio=1)
    table.add_column("Top Refs", style="green", overflow="fold", no_wrap=False, ratio=6)
    for keyword in keywords:
        hits = keyword_map.get(keyword, [])
        table.add_row(keyword, str(len(hits)), "\n".join(hits) if hits else "-")
    console.print(table)


def extract_json(text: str) -> Dict:
    raw = text.strip()
    candidates: List[str] = [raw]

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start : end + 1].strip())

    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        try:
            obj = ast.literal_eval(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return {}

def render_structured_output(data: Dict) -> None:
    findings = data.get("findings", []) if isinstance(data, dict) else []
    gaps = data.get("gaps", []) if isinstance(data, dict) else []
    next_steps = data.get("next_steps", []) if isinstance(data, dict) else []

    table = Table(title=f"{STACK_NAME} Findings", show_lines=True, expand=True)
    table.add_column("Finding", style="cyan", overflow="fold", no_wrap=False, ratio=4)
    table.add_column("Impact", style="white", overflow="fold", no_wrap=False, ratio=4)
    table.add_column("Citations", style="green", overflow="fold", no_wrap=False, ratio=5)
    table.add_column("Confidence", style="yellow", overflow="fold", no_wrap=False, ratio=1)

    if findings:
        for f in findings[:8]:
            table.add_row(
                str(f.get("title", "")),
                str(f.get("impact", "")),
                "\n".join(f.get("citations", [])) if isinstance(f.get("citations"), list) else "",
                str(f.get("confidence", "")),
            )
    else:
        table.add_row("No structured findings returned", "", "", "")

    console.print(table)

    if gaps:
        gap_table = Table(title="Evidence Gaps", show_lines=True)
        gap_table.add_column("Gap", style="red")
        for g in gaps[:8]:
            gap_table.add_row(str(g))
        console.print(gap_table)

    if next_steps:
        step_table = Table(title="Next Steps", show_lines=True)
        step_table.add_column("Step", style="magenta")
        for s in next_steps[:8]:
            if isinstance(s, dict):
                step_text = (
                    s.get("title")
                    or s.get("step")
                    or s.get("text")
                    or s.get("description")
                    or json.dumps(s, ensure_ascii=True)
                )
            else:
                step_text = str(s)
            step_table.add_row(step_text)
        console.print(step_table)


def repair_json_with_llm(llm: OllamaLLM, raw_text: str) -> Dict:
    repair_prompt = ChatPromptTemplate.from_template(
        "Convert the following text into valid JSON only. "
        "Do not add commentary. Keep this schema exactly: "
        "{{\"findings\": [{{\"title\": \"...\", \"impact\": \"...\", \"citations\": [\"path:line\"], \"confidence\": \"high|medium|low\"}}], "
        "\"gaps\": [\"...\"], \"next_steps\": [\"...\"]}}.\n\n"
        "Text to convert:\n{raw_text}"
    )
    chain = repair_prompt | llm
    repaired = chain.invoke({"raw_text": raw_text})
    return extract_json(repaired)




def coerce_structured_from_raw(raw_text: str, refs: List[str] | None = None) -> Dict:
    compact = " ".join(str(raw_text).split())
    snippet = compact[:420] if compact else "No model output returned."
    used_refs = list(dict.fromkeys(refs or []))[:8]
    return {
        "findings": [
            {
                "title": "Unstructured model output",
                "impact": snippet,
                "citations": used_refs,
                "confidence": "low",
            }
        ],
        "gaps": [
            "Model did not return valid structured JSON; showing best-effort fallback summary.",
            "Fallback includes top retrieved references for traceability.",
        ],
        "next_steps": [
            "Re-run the same query and verify Ollama model output format.",
            "Use --debug-retrieval to confirm retrieved evidence quality.",
        ],
    }

def run_audit(user_query: str) -> None:
    dynamic_keywords = build_query_keywords(user_query)
    keywords = dynamic_keywords
    if not keywords:
        keywords = [user_query.strip()]

    console.print(f"\n[bold sea_green]🔍 STEP 1: {STACK_NAME} KEYWORD SCAN[/bold sea_green]")
    keyword_map = build_keyword_map(keywords)
    show_keyword_table(keyword_map, keywords)

    references = build_reference_list(keyword_map)
    context = build_context(user_query)

    console.print("\n[bold sea_green]🤖 STEP 2: STRUCTURED LLM AUDIT[/bold sea_green]")
    llm = OllamaLLM(model="llama3", timeout=300)

    prompt = ChatPromptTemplate.from_template(
        "You are auditing Tax_Calculator code for the user question.\n"
        "Use ONLY the evidence provided.\n\n"
        "Question:\n{question}\n\n"
        "Allowed citations (path:line):\n{references}\n\n"
        "Retrieved context snippets:\n{context}\n\n"
        "Return ONLY valid JSON with this schema:\n"
        "{{\n"
        "  \"findings\": [\n"
        "    {{\"title\": \"...\", \"impact\": \"...\", \"citations\": [\"path:line\"], \"confidence\": \"high|medium|low\"}}\n"
        "  ],\n"
        "  \"gaps\": [\"...\"],\n"
        "  \"next_steps\": [\"...\"]\n"
        "}}\n"
        "Rules: 3-7 findings; every finding needs >=1 citation from allowed list; if weak evidence, use low confidence and include a gap."
    )

    chain = prompt | llm
    raw = chain.invoke({
        "question": user_query,
        "references": "\n".join(references) if references else "No references found",
        "context": context,
    })

    data = extract_json(raw)
    if not data:
        console.print("[yellow]Primary JSON parse failed; attempting repair...[/yellow]")
        data = repair_json_with_llm(llm, raw)
    if not data:
        console.print("[yellow]Repair also failed; using best-effort structured fallback.[/yellow]")
        data = coerce_structured_from_raw(raw, refs=references)

    console.print(f"\n[bold cyan]✨ FINAL {STACK_NAME} AUDIT REPORT[/bold cyan]")
    render_structured_output(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Run {STACK_NAME} langchain audit")
    parser.add_argument("query", nargs="?", default=None, help="Optional audit query")
    args = parser.parse_args()

    query = args.query
    if not query:
        query = input("Enter your audit question: ").strip()
    if not query:
        raise SystemExit("A query is required.")

    run_audit(query)
