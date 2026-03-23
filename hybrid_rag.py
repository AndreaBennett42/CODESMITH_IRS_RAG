import argparse
import json
import os
import re
import shutil
import textwrap
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich import box
from rich.console import Console
from rich.table import Table


@dataclass
class Config:
    direct_file_code_dir: Path
    tax_calc_code_dir: Path
    direct_file_extensions: List[str]
    tax_calc_extensions: List[str]
    facts_xml: Path
    cache_dir: Path
    embedding_model: str
    llm_model: str
    vector_top_k: int
    graph_seed_k: int
    graph_expand_hops: int
    graph_max_return: int
    keyword_top_refs: int
    keyword_max_refs_per_keyword: int


def load_config() -> Config:
    load_dotenv()
    direct_file_code_dir = Path(
        os.getenv("DIRECT_FILE_CODE_DIR", os.getenv("CODE_DIR", "./direct-file/direct-file"))
    ).resolve()
    tax_calc_code_dir = Path(
        os.getenv("TAX_CALC_CODE_DIR", "./Tax-Calculator/taxcalc")
    ).resolve()
    direct_file_extensions = [
        ext.strip().lower()
        for ext in os.getenv(
            "DIRECT_FILE_CODE_EXTENSIONS",
            ".xml,.java,.ts,.tsx,.json,.yaml,.yml,.md,.py",
        ).split(",")
        if ext.strip()
    ]
    tax_calc_extensions = [
        ext.strip().lower()
        for ext in os.getenv("TAX_CALC_CODE_EXTENSIONS", ".py").split(",")
        if ext.strip()
    ]
    facts_xml = Path(
        os.getenv("FACTS_XML", "./direct-file/backend/src/main/resources/tax")
    ).resolve()
    cache_dir = Path(os.getenv("RAG_CACHE_DIR", "./.rag_cache")).resolve()
    embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    llm_model = os.getenv("LLM_MODEL", "llama3")
    vector_top_k = int(os.getenv("VECTOR_TOP_K", "6"))
    graph_seed_k = int(os.getenv("GRAPH_SEED_K", "4"))
    graph_expand_hops = int(os.getenv("GRAPH_EXPAND_HOPS", "1"))
    graph_max_return = int(os.getenv("GRAPH_MAX_RETURN", "8"))
    keyword_top_refs = int(os.getenv("KEYWORD_TOP_REFS", "5"))
    keyword_max_refs_per_keyword = int(os.getenv("KEYWORD_MAX_REFS_PER_KEYWORD", "20"))
    return Config(
        direct_file_code_dir=direct_file_code_dir,
        tax_calc_code_dir=tax_calc_code_dir,
        direct_file_extensions=direct_file_extensions,
        tax_calc_extensions=tax_calc_extensions,
        facts_xml=facts_xml,
        cache_dir=cache_dir,
        embedding_model=embedding_model,
        llm_model=llm_model,
        vector_top_k=vector_top_k,
        graph_seed_k=graph_seed_k,
        graph_expand_hops=graph_expand_hops,
        graph_max_return=graph_max_return,
        keyword_top_refs=keyword_top_refs,
        keyword_max_refs_per_keyword=keyword_max_refs_per_keyword,
    )


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def phrase_proximity_match(line_tokens: List[str], phrase_tokens: List[str], max_gap: int = 2) -> bool:
    # Ordered, near-adjacent subsequence match for soft phrase hits.
    if not line_tokens or not phrase_tokens:
        return False
    pos = -1
    for pt in phrase_tokens:
        found = -1
        for i in range(pos + 1, len(line_tokens)):
            if line_tokens[i] == pt:
                found = i
                break
        if found < 0:
            return False
        if pos >= 0 and (found - pos - 1) > max_gap:
            return False
        pos = found
    return True


def window_cooccurrence_count(tokens: List[str], query_terms: Set[str], window_size: int = 12) -> int:
    # Max number of distinct query terms observed in any sliding token window.
    if not tokens or not query_terms:
        return 0
    best = 0
    n = len(tokens)
    for i in range(n):
        window = set(tokens[i : min(n, i + window_size)])
        c = len(window & query_terms)
        if c > best:
            best = c
    return best


def stable_hash(value: str) -> str:
    # Short deterministic key for cache files.
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def collect_code_documents(code_dir: Path, extensions: List[str], source: str) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    docs: List[Dict] = []
    seen = set()
    for ext in extensions:
        pattern = f"*{ext}" if ext.startswith(".") else f"*.{ext}"
        for path in code_dir.rglob(pattern):
            if not path.is_file():
                continue
            seen.add(path)
    for path in sorted(seen):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        cursor = 0
        for idx, chunk in enumerate(splitter.split_text(text)):
            pos = text.find(chunk, cursor)
            if pos < 0:
                # Fallback to cursor-based approximation if exact chunk match is not found.
                pos = min(max(cursor, 0), max(len(text) - 1, 0))
            end_pos = min(len(text), pos + len(chunk))
            start_line = text.count("\n", 0, pos) + 1
            end_line = text.count("\n", 0, end_pos) + 1
            # Advance cursor with overlap to improve matching of subsequent chunks.
            cursor = max(pos + 1, end_pos - 150)
            docs.append(
                {
                    "id": f"{path}:{idx}",
                    "path": str(path),
                    "chunk_id": idx,
                    "line_start": start_line,
                    "line_end": end_line,
                    "source": source,
                    "text": chunk,
                }
            )
    return docs


def build_or_load_vector_index(
    config: Config,
    embeddings: OllamaEmbeddings,
    code_dir: Path,
    extensions: List[str],
    source: str,
) -> Tuple[List[Dict], np.ndarray]:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    signature = f"{source}|{code_dir}|{','.join(sorted(extensions))}|{config.embedding_model}"
    key = stable_hash(signature)
    docs_path = config.cache_dir / f"vector_docs_{key}.json"
    vecs_path = config.cache_dir / f"vector_embeddings_{key}.npy"

    if docs_path.exists() and vecs_path.exists():
        docs = json.loads(docs_path.read_text(encoding="utf-8"))
        vectors = np.load(vecs_path)
        return docs, vectors

    docs = collect_code_documents(code_dir, extensions, source)
    if not docs:
        raise RuntimeError(
            f"No files found for {source} under {code_dir} with extensions: {', '.join(extensions)}"
        )

    embedded = embeddings.embed_documents([d["text"] for d in docs])
    vectors = np.asarray(embedded, dtype=np.float32)
    np.save(vecs_path, vectors)
    docs_path.write_text(json.dumps(docs, ensure_ascii=True), encoding="utf-8")
    return docs, vectors


def vector_retrieve(
    query: str,
    docs: List[Dict],
    vectors: np.ndarray,
    embeddings: OllamaEmbeddings,
    top_k: int,
    query_keywords: Optional[List[str]] = None,
) -> List[Dict]:
    q = np.asarray(embeddings.embed_query(query), dtype=np.float32)
    denom = (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q)) + 1e-12
    scores = (vectors @ q) / denom
    candidate_k = min(len(docs), max(top_k * 8, top_k))
    top_idx = np.argsort(-scores)[:candidate_k]

    keywords = query_keywords or []
    phrase_terms = [k for k in keywords if " " in k and len(k.split()) >= 2]
    phrase_token_lists = [tokenize(p) for p in phrase_terms]
    query_term_set: Set[str] = set()
    for k in keywords:
        query_term_set.update(tokenize(k))

    def path_penalty(path: str, kws: List[str]) -> float:
        p = path.lower()
        qtxt = " ".join(kws).lower()
        legal_tip_focus = bool(re.search(r"(tip|tips|section\s+\d+|§\s*\d+|transition relief|qualified tips|form\s*4137)", qtxt))
        if legal_tip_focus and "df-client/df-client-app/src/locales/" in p:
            return 0.12
        return 0.0

    reranked: List[Tuple[float, int]] = []
    for i in top_idx:
        doc = docs[i]
        low_text = doc["text"].lower()
        text_tokens = tokenize(low_text)
        bonus = 0.0
        for p in phrase_terms:
            if p in low_text:
                bonus += 0.08
        for ptoks in phrase_token_lists:
            if len(ptoks) >= 3 and phrase_proximity_match(text_tokens, ptoks, max_gap=2):
                bonus += 0.05
        wc = window_cooccurrence_count(text_tokens, query_term_set, window_size=12)
        if wc >= 2:
            bonus += 0.02 * min(6, wc - 1)
        pen = path_penalty(doc.get("path", ""), keywords)
        reranked.append((float(scores[i]) + bonus - pen, i))

    reranked.sort(reverse=True)
    final_idx = [i for _, i in reranked[:top_k]]
    return [
        {
            "score": float(scores[i]),
            "source": docs[i].get("source", "unknown"),
            "path": docs[i]["path"],
            "chunk_id": docs[i]["chunk_id"],
            "line_start": docs[i].get("line_start", 1),
            "line_end": docs[i].get("line_end", 1),
            "vector_dim": int(vectors.shape[1]) if vectors.ndim == 2 else int(vectors.shape[0]),
            "vector_norm": float(np.linalg.norm(vectors[i])),
            "text": docs[i]["text"],
        }
        for i in final_idx
    ]


def has_tip_logic_evidence(vector_hits: List[Dict], graph_hits: List[Dict]) -> bool:
    markers = (
        "tip_income_deduction",
        "qualified tips",
        "transition relief",
        "section 224",
        "§ 224",
        "form 4137",
    )
    for h in vector_hits:
        blob = f"{h.get('path', '')} {h.get('text', '')}".lower()
        if any(m in blob for m in markers):
            return True
    for n in graph_hits:
        blob = f"{n.get('path', '')} {n.get('name', '')} {n.get('description', '')}".lower()
        if any(m in blob for m in markers):
            return True
    return False


def build_insufficient_evidence_report(
    query: str,
    direct_hits: List[Dict],
    tax_hits: List[Dict],
    graph_hits: List[Dict],
) -> str:
    refs: List[str] = []
    for h in (tax_hits[:3] + direct_hits[:2]):
        refs.append(f"- {h.get('path','-')}:{h.get('line_start',1)}-{h.get('line_end',1)}")
    for g in graph_hits[:2]:
        refs.append(f"- {g.get('path','-')} ({g.get('source_ref','-')})")
    ref_block = "\n".join(refs) if refs else "- No strong refs retrieved."
    return (
        "**Key Findings**\n"
        "- Retrieved evidence does not show direct, high-confidence §224 tip-deduction calculation logic.\n"
        "- Most matches are reporting/general deduction context, not explicit tip-deduction computation.\n\n"
        "**Potential Mismatches or Missing Logic**\n"
        "- Tip-focused query appears under-supported by direct calculation logic in current retrieved refs.\n\n"
        "**High-Confidence References**\n"
        f"{ref_block}\n\n"
        "**Evidence Gaps**\n"
        "- Missing direct evidence for the requested tip-deduction logic path.\n\n"
        "**Next Steps**\n"
        "1. Add stricter query terms around `tip_income_deduction`, `qualified tips`, and `section 224`.\n"
        "2. Re-run and verify at least one Tax-Calculator computation-level ref is present.\n"
        "3. Treat unsupported specifics as unresolved until directly evidenced.\n"
    )


def build_or_load_graph_index(config: Config) -> Dict:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    xml_sources: List[Path]
    if config.facts_xml.is_dir():
        xml_sources = sorted(p for p in config.facts_xml.rglob("*.xml") if p.is_file())
    else:
        xml_sources = [config.facts_xml]

    # Bump cache schema version when graph node fields change.
    signature = "graph_schema_v2|" + "|".join(str(p) for p in xml_sources)
    key = stable_hash(signature)
    graph_path = config.cache_dir / f"graph_index_{key}.json"
    if graph_path.exists():
        return json.loads(graph_path.read_text(encoding="utf-8"))

    if not xml_sources:
        raise RuntimeError(f"No XML files found under: {config.facts_xml}")

    nodes: Dict[str, Dict] = {}
    edges: Dict[str, List[str]] = {}
    reverse_edges: Dict[str, List[str]] = {}

    for xml_file in xml_sources:
        if not xml_file.exists():
            continue
        line_map: Dict[str, int] = {}
        try:
            for line_no, line in enumerate(xml_file.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                m = re.search(r'<Fact[^>]*path="([^"]+)"', line)
                if m and m.group(1) not in line_map:
                    line_map[m.group(1)] = line_no
        except OSError:
            line_map = {}
        root = ET.parse(xml_file).getroot()
        facts = root.findall(".//Fact")
        for fact in facts:
            path = fact.attrib.get("path")
            if not path:
                continue
            name = (fact.findtext("./Name") or "").strip()
            description = (fact.findtext("./Description") or "").strip()
            dependencies = []
            for dep in fact.findall(".//Dependency"):
                dep_path = dep.attrib.get("path")
                if dep_path:
                    dependencies.append(dep_path.strip())

            prev = nodes.get(path, {})
            merged_deps = sorted(set(prev.get("dependencies", []) + dependencies))
            merged_name = name or prev.get("name", "")
            merged_desc = description or prev.get("description", "")
            source_file = prev.get("source_file") or str(xml_file)
            source_line = prev.get("source_line") or line_map.get(path)
            nodes[path] = {
                "path": path,
                "name": merged_name,
                "description": merged_desc,
                "dependencies": merged_deps,
                "source_file": source_file,
                "source_line": source_line,
                "search_blob": f"{path} {merged_name} {merged_desc}".lower(),
            }

    for src, node in nodes.items():
        for dst in node["dependencies"]:
            if dst not in nodes:
                continue
            edges.setdefault(src, []).append(dst)
            reverse_edges.setdefault(dst, []).append(src)

    index = {"nodes": nodes, "edges": edges, "reverse_edges": reverse_edges}
    graph_path.write_text(json.dumps(index, ensure_ascii=True), encoding="utf-8")
    return index


def score_node(query_tokens: List[str], node_blob: str) -> int:
    if not query_tokens:
        return 0
    toks = tokenize(node_blob)
    hay = set(toks)
    score = sum(1 for t in query_tokens if t in hay)
    wc = window_cooccurrence_count(toks, set(query_tokens), window_size=8)
    if wc >= 2:
        score += min(4, wc - 1)
    return score


def derive_graph_query_tokens(keywords: List[str]) -> List[str]:
    # Derive graph tokens from the same keyword set used by keyword scan and vector retrieval.
    tokens: List[str] = []
    legal_base_numbers: Set[str] = set()
    for kw in _canonicalize_keywords_for_retrieval(keywords):
        ident = _parse_legal_identifier(kw)
        if ident:
            compact = re.sub(r"[\s().-]+", "", ident)
            num = re.match(r"^(\d+)", ident)
            if num:
                n = num.group(1)
                legal_base_numbers.add(n)
                tokens.append(n)
            if compact and compact != (num.group(1) if num else "") and re.search(r"[a-z]", compact):
                tokens.append(compact)
            continue
        tokens.extend(tokenize(kw))

    # Stable de-dup preserving order.
    seen: Set[str] = set()
    out: List[str] = []
    for t in tokens:
        # Never keep generic section aliases as standalone search tokens.
        if t in {"section", "sec"}:
            continue
        # Never keep standalone subsection letters as independent search tokens.
        if len(t) == 1 and t.isalpha():
            continue
        # Never keep standalone numeric subsection fragments (e.g., "2" from 163(2))
        # unless that number is itself a legal base section number in the query.
        if t.isdigit() and len(t) <= 2 and t not in legal_base_numbers:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def graph_retrieve(
    query: str,
    query_keywords: List[str],
    graph_index: Dict,
    seed_k: int,
    hops: int,
    max_return: int,
) -> List[Dict]:
    nodes = graph_index["nodes"]
    edges = graph_index["edges"]
    reverse_edges = graph_index["reverse_edges"]
    q_keywords = _canonicalize_keywords_for_retrieval(query_keywords)
    q_tokens = derive_graph_query_tokens(q_keywords)
    q_phrases = [k for k in q_keywords if " " in k and len(k.split()) >= 3]

    scored = []
    for path, node in nodes.items():
        s = score_node(q_tokens, node["search_blob"])
        node_tokens = tokenize(node["search_blob"])
        for ph in q_phrases:
            ptoks = tokenize(ph)
            if ph in node["search_blob"]:
                s += 2
            elif phrase_proximity_match(node_tokens, ptoks, max_gap=2):
                s += 1
        if s > 0:
            scored.append((s, path))
    scored.sort(reverse=True)
    seed_paths = [p for _, p in scored[:seed_k]]

    visited = set(seed_paths)
    frontier = list(seed_paths)
    for _ in range(max(hops, 0)):
        nxt = []
        for p in frontier:
            neighbors = edges.get(p, []) + reverse_edges.get(p, [])
            for n in neighbors:
                if n not in visited:
                    visited.add(n)
                    nxt.append(n)
        frontier = nxt
        if not frontier:
            break

    def node_rank(path: str) -> Tuple[int, int]:
        seed_score = next((s for s, p in scored if p == path), 0)
        degree = len(edges.get(path, [])) + len(reverse_edges.get(path, []))
        return (seed_score, degree)

    ranked = sorted(visited, key=node_rank, reverse=True)[:max_return]
    score_map = {p: s for s, p in scored}
    out = []
    for p in ranked:
        n = nodes[p]
        source_ref = (
            f"{_shorten_path(n.get('source_file', ''))}:{n.get('source_line')}"
            if n.get("source_file") and n.get("source_line")
            else "-"
        )
        out.append(
            {
                "path": n["path"],
                "name": n["name"],
                "description": n["description"],
                "dependencies": n["dependencies"][:8],
                "score": int(score_map.get(p, 0)),
                "source_ref": source_ref,
            }
        )
    return out


def format_vector_context(chunks: List[Dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            "\n".join(
                [
                    f"SOURCE: {c.get('source', 'unknown')}",
                    f"FILE: {c['path']}",
                    f"REF: {c['path']}:{c.get('line_start', 1)}-{c.get('line_end', 1)}",
                    f"CHUNK: {c['chunk_id']}",
                    f"SCORE: {c['score']:.4f}",
                    c["text"][:1800],
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def format_graph_context(nodes: List[Dict]) -> str:
    parts = []
    for n in nodes:
        parts.append(
            "\n".join(
                [
                    f"FACT: {n['path']}",
                    f"SCORE: {n.get('score', 0)}",
                    f"SOURCE_REF: {n.get('source_ref', '-')}",
                    f"NAME: {n['name']}",
                    f"DESCRIPTION: {n['description']}",
                    f"DEPENDENCIES: {', '.join(n['dependencies']) if n['dependencies'] else 'none'}",
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def is_tip_focus_query(query: str) -> bool:
    q = query.lower()
    return bool(
        re.search(
            r"(tip[s]?\b|tip_income_deduction|form\s*4137|§\s*224\b|section\s+224\b)",
            q,
        )
    )


def _format_source_block(title: str, chunks: List[Dict]) -> str:
    if not chunks:
        return f"{title}:\n- none"
    parts = [title + ":"]
    for c in chunks:
        parts.append(
            "\n".join(
                [
                    f"SOURCE: {c.get('source', 'unknown')}",
                    f"FILE: {c['path']}",
                    f"REF: {c['path']}:{c.get('line_start', 1)}-{c.get('line_end', 1)}",
                    f"CHUNK: {c['chunk_id']}",
                    f"SCORE: {c['score']:.4f}",
                    c["text"][:1200],
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def format_vector_context_split(
    direct_chunks: List[Dict],
    tax_chunks: List[Dict],
    tip_focus: bool,
) -> str:
    blocks = [
        _format_source_block("Direct File Retrieval", direct_chunks),
        _format_source_block("Tax-Calculator Retrieval", tax_chunks),
    ]
    if tip_focus and not tax_chunks:
        blocks.append(
            "Tax-Calculator Tip Logic Coverage:\n"
            "- No direct Tax-Calculator matches found for tip-focused query."
        )
    return "\n\n====\n\n".join(blocks)


def _build_intent_signal_tokens(keywords: List[str]) -> Set[str]:
    # Keep high-signal intent tokens; suppress generic audit language.
    generic = {
        "find", "logic", "related", "including", "regarding", "about",
        "guidance", "under",
        "deduction", "deductions", "qualified", "transition", "relief",
        "section", "sec", "form", "information", "reporting",
    }
    sig: Set[str] = set()
    for kw in _canonicalize_keywords_for_retrieval(keywords):
        for t in tokenize(kw):
            if len(t) < 3 or t in generic:
                continue
            sig.add(t)
    return sig


def _token_overlap_score(text: str, signal_tokens: Set[str]) -> int:
    if not signal_tokens:
        return 0
    return len(set(tokenize(text)) & signal_tokens)


def _filter_vector_hits_by_intent(vector_hits: List[Dict], signal_tokens: Set[str], keep_top: int) -> List[Dict]:
    if not vector_hits:
        return vector_hits
    scored = []
    for h in vector_hits:
        blob = f"{h.get('path', '')} {h.get('text', '')}"
        ov = _token_overlap_score(blob, signal_tokens)
        scored.append((ov, h))
    filtered = [h for ov, h in scored if ov >= 1]
    if not filtered:
        return vector_hits[:keep_top]
    return sorted(filtered, key=lambda x: x["score"], reverse=True)[:keep_top]


def _filter_graph_hits_by_intent(graph_hits: List[Dict], signal_tokens: Set[str], keep_top: int) -> List[Dict]:
    if not graph_hits:
        return graph_hits
    scored = []
    for n in graph_hits:
        blob = f"{n.get('path', '')} {n.get('name', '')} {n.get('description', '')}"
        ov = _token_overlap_score(blob, signal_tokens)
        scored.append((ov, n))
    filtered = [n for ov, n in scored if ov >= 1]
    if not filtered:
        return graph_hits[:keep_top]
    return filtered[:keep_top]


def _has_evidence_phrase(phrase: str, vector_hits: List[Dict], graph_hits: List[Dict]) -> bool:
    p = phrase.lower()
    for h in vector_hits:
        blob = f"{h.get('path', '')} {h.get('text', '')}".lower()
        if p in blob:
            return True
    for n in graph_hits:
        blob = f"{n.get('path', '')} {n.get('name', '')} {n.get('description', '')}".lower()
        if p in blob:
            return True
    return False


def _line_has_ref(line: str) -> bool:
    low = line.lower()
    if "ref:" in low or "fact:" in low or "path=" in low:
        return True
    if re.search(r"\b[\w\-/]+\.(py|xml|java|ts|tsx|json|yaml|yml|md):\d+", low):
        return True
    if re.search(r"/[a-z0-9_\-*/]+", low):
        return True
    return False


def _normalize_heading_line(line: str) -> str:
    # Normalize markdown heading variants like "**Key Findings**:" or "## Key Findings".
    s = line.strip().lower()
    s = re.sub(r"^[#\-\*\s]+", "", s)
    s = re.sub(r"[\*\s:]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_section_block(text: str, heading: str) -> List[str]:
    lines = text.splitlines()
    start = None
    headers = {
        "key findings",
        "potential mismatches or missing logic",
        "high-confidence references",
        "evidence gaps",
        "next steps",
    }
    for i, ln in enumerate(lines):
        if _normalize_heading_line(ln) == heading.lower():
            start = i + 1
            break
    if start is None:
        return []
    out: List[str] = []
    for ln in lines[start:]:
        if _normalize_heading_line(ln) in headers:
            break
        out.append(ln)
    return out


def _replace_section_block(text: str, heading: str, new_block_lines: List[str]) -> str:
    lines = text.splitlines()
    headers = [
        "Key Findings",
        "Potential Mismatches or Missing Logic",
        "High-Confidence References",
        "Evidence Gaps",
        "Next Steps",
    ]
    lower_headers = {h.lower() for h in headers}
    start = None
    for i, ln in enumerate(lines):
        if _normalize_heading_line(ln) == heading.lower():
            start = i
            break
    if start is None:
        # If missing, prepend the requested section.
        return "\n".join([heading, *new_block_lines, "", text]).strip()

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _normalize_heading_line(lines[j]) in lower_headers:
            end = j
            break

    replaced = lines[: start + 1] + new_block_lines + lines[end:]
    return "\n".join(replaced).strip()


def _query_signal_terms_for_findings(query: str) -> Set[str]:
    stop = {
        "find", "logic", "related", "including", "regarding", "about", "the", "and", "for", "with",
        "from", "that", "this", "what", "how", "does", "are", "is", "to", "of", "in", "on", "a", "an", "it",
        "guidance", "under", "section", "sec", "information", "reporting",
    }
    terms = [t for t in tokenize(query) if len(t) >= 3 and t not in stop and t not in {"section", "sec"}]
    return set(terms)


def _build_citation_derived_key_findings(query: str, vector_hits: List[Dict], graph_hits: List[Dict]) -> List[str]:
    q_terms = _query_signal_terms_for_findings(query)
    candidates: List[Tuple[int, float, str, str, str]] = []

    for h in vector_hits:
        ref = f"{h.get('path','-')}:L{h.get('line_start',1)}-L{h.get('line_end',1)}"
        blob = f"{h.get('path','')} {h.get('text','')}".lower()
        overlap = sorted(list(set(tokenize(blob)) & q_terms))
        ov = len(overlap)
        if ov <= 0:
            continue
        src = "Tax-Calculator" if h.get("source") == "tax_calculator" else "Direct File"
        note = f'- [{src}] `{ref}` overlap_terms={", ".join(overlap[:6])}'
        dedup_key = h.get('path', ref)
        candidates.append((ov, float(h.get("score", 0.0)), src, dedup_key, note))

    for n in graph_hits:
        ref = n.get("source_ref", "-")
        blob = f"{n.get('path','')} {n.get('name','')} {n.get('description','')}".lower()
        overlap = sorted(list(set(tokenize(blob)) & q_terms))
        ov = len(overlap)
        if ov <= 0:
            continue
        note = f'- [Fact Graph] `{n.get("path","-")}` ({ref}) overlap_terms={", ".join(overlap[:6])}'
        dedup_key = n.get('path', ref)
        candidates.append((ov, float(n.get("score", 0.0)), "Fact Graph", dedup_key, note))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    if not candidates:
        return ["- No citation-derived findings met relevance threshold."]

    out: List[str] = []
    seen_lines: Set[str] = set()
    seen_keys: Set[str] = set()
    seen_sources: Set[str] = set()

    for _, _, src, dedup_key, line in candidates:
        if dedup_key in seen_keys or line in seen_lines:
            continue
        out.append(line)
        seen_lines.add(line)
        seen_keys.add(dedup_key)
        seen_sources.add(src)
        if len(out) >= 5:
            break

    # If Direct File has relevant evidence but was crowded out by Tax-Calculator,
    # give one strong Direct File finding a place in the final list.
    if "Direct File" not in seen_sources:
        for ov, score, src, dedup_key, line in candidates:
            if src != "Direct File":
                continue
            if dedup_key in seen_keys or line in seen_lines:
                continue
            if ov < 2:
                continue
            if len(out) >= 5:
                out.pop()
            out.append(line)
            seen_lines.add(line)
            seen_keys.add(dedup_key)
            seen_sources.add(src)
            break

    if len(out) < 5:
        for _, _, src, dedup_key, line in candidates:
            if dedup_key in seen_keys or line in seen_lines:
                continue
            if src in seen_sources and src != "Fact Graph":
                continue
            out.append(line)
            seen_lines.add(line)
            seen_keys.add(dedup_key)
            seen_sources.add(src)
            if len(out) >= 5:
                break

    return out[:5]


def _build_deterministic_high_conf_refs(vector_hits: List[Dict], graph_hits: List[Dict]) -> List[str]:
    refs: List[str] = []
    seen: Set[str] = set()
    seen_vector_paths: Set[str] = set()
    for h in sorted(vector_hits, key=lambda x: float(x.get("score", 0.0)), reverse=True):
        path = h.get("path", "-")
        if path in seen_vector_paths:
            continue
        seen_vector_paths.add(path)
        ref = f"{h.get('path','-')}:L{h.get('line_start',1)}-L{h.get('line_end',1)}"
        line = f"- `{ref}`"
        if line not in seen:
            seen.add(line)
            refs.append(line)
        if len(refs) >= 5:
            break
    for n in sorted(graph_hits, key=lambda x: int(x.get("score", 0)), reverse=True):
        line = f"- `{n.get('path','-')}` ({n.get('source_ref','-')})"
        if line not in seen:
            seen.add(line)
            refs.append(line)
        if len(refs) >= 8:
            break
    if not refs:
        return ["- No high-confidence references met threshold."]
    return refs


def _build_deterministic_mismatches(query: str, vector_hits: List[Dict], graph_hits: List[Dict]) -> List[str]:
    q_terms = _query_signal_terms_for_findings(query)
    evidence_tokens: Set[str] = set()
    for h in vector_hits:
        evidence_tokens.update(tokenize(f"{h.get('path','')} {h.get('text','')}"))
    for n in graph_hits:
        evidence_tokens.update(tokenize(f"{n.get('path','')} {n.get('name','')} {n.get('description','')}"))

    missing = sorted([t for t in q_terms if t not in evidence_tokens])
    out: List[str] = []
    if missing:
        out.append(f"- Missing direct evidence overlap for query terms: {', '.join(missing[:8])}.")
    if not any("transition" in t for t in evidence_tokens):
        out.append("- Transition-relief logic is not explicitly evidenced in retrieved refs.")
    if not out:
        out.append("- No high-confidence mismatches detected in current retrieved evidence.")
    return out[:4]


def _build_deterministic_next_steps(query: str, vector_hits: List[Dict], graph_hits: List[Dict]) -> List[str]:
    out: List[str] = []
    out.append("1. Verify each Key Findings ref directly in the referenced file/line before accepting conclusions.")
    out.append("2. Re-run with a narrower query if you need calculation-level logic (for example: `tip_income_deduction`, `section 224`, `transition relief`).")
    if not graph_hits:
        out.append("3. Expand graph retrieval scope and re-run to restore Fact Graph coverage.")
    else:
        out.append("3. Compare vector refs against Fact Graph paths to confirm reporting logic vs deduction-calculation logic.")
    return out


def _canonicalize_report_sections(text: str) -> str:
    order = [
        "Key Findings",
        "Potential Mismatches or Missing Logic",
        "High-Confidence References",
        "Evidence Gaps",
        "Next Steps",
    ]
    norm_map = {h.lower(): h for h in order}
    sections: Dict[str, List[str]] = {h: [] for h in order}
    current: Optional[str] = None
    for raw in text.splitlines():
        norm = _normalize_heading_line(raw)
        if norm in norm_map:
            current = norm_map[norm]
            continue
        if current is None:
            continue
        line = raw.rstrip()
        if not line:
            if sections[current] and sections[current][-1] != "":
                sections[current].append("")
            continue
        sections[current].append(line)

    # De-dup lines within each section while preserving order.
    for h in order:
        seen: Set[str] = set()
        deduped: List[str] = []
        for ln in sections[h]:
            key = ln.strip()
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            deduped.append(ln)
        sections[h] = deduped

    out_lines: List[str] = []
    for h in order:
        out_lines.append(f"**{h}**")
        if sections[h]:
            out_lines.extend(sections[h])
        elif h != "Evidence Gaps":
            out_lines.append("- Not provided.")
        out_lines.append("")
    return "\n".join(out_lines).strip()


def enforce_claim_alignment(
    query: str,
    answer: str,
    vector_hits: List[Dict],
    graph_hits: List[Dict],
    tax_hits_present: bool,
) -> str:
    out = answer
    q = query.lower()
    tip_focus = is_tip_focus_query(query)
    drift_topics = [r"qualified business income deduction", r"\bqbi\b"]
    gap_notes: List[str] = []

    # Remove unsupported drift lines.
    drift_re = re.compile("|".join(drift_topics), flags=re.IGNORECASE)
    if not re.search(r"(business income|qbi)", q):
        if not any(_has_evidence_phrase("qualified business income deduction", vector_hits, graph_hits) for _ in [0]):
            lines = out.splitlines()
            kept = [ln for ln in lines if not drift_re.search(ln)]
            if len(kept) != len(lines):
                out = "\n".join(kept).strip()
                gap_notes.append("Off-topic deduction references were removed because retrieved evidence did not support them.")

    # Hard source-coverage note for tip-focused audits.
    if tip_focus and not tax_hits_present:
        gap_notes.append("Tip-focused query but no Tax-Calculator evidence was retrieved; do not treat generic deduction findings as direct tip logic.")

    # Narrow generic multi-step taxable-income blocks to tip-relevant lines for tip-focused queries.
    if tip_focus:
        lines = out.splitlines()
        filtered_lines: List[str] = []
        drop_markers = (
            "senior_deduction",
            "overtime_income_deduction",
            "auto_loan_interest_deduction",
            "qbi",
            "qbied",
            "pre_qbid_taxinc",
        )
        keep_markers = (
            "tip_income_deduction",
            "tips",
            "§ 224",
            "section 224",
            "transition relief",
            "qualified tips",
        )

        for ln in lines:
            low = ln.lower()
            if any(m in low for m in drop_markers) and not any(k in low for k in keep_markers):
                continue
            filtered_lines.append(ln)
        out = "\n".join(filtered_lines).strip()

    # Remove unsupported specific claims unless directly evidenced.
    unsupported_checks = [
        ("box 12", "box 12"),
        ("code \"tp\"", "code tp"),
        ("$25,000", "25000"),
        ("25000", "25000"),
        ("ttoc", "ttoc"),
        ("phase-in", "phase-in"),
        ("income limits", "income limit"),
        ("schedule 1-a", "schedule 1-a"),
    ]
    removed_any = False
    lines = out.splitlines()
    kept_lines: List[str] = []
    for ln in lines:
        low = ln.lower()
        remove = False
        for marker, evidence_phrase in unsupported_checks:
            if marker in low and not _has_evidence_phrase(evidence_phrase, vector_hits, graph_hits):
                remove = True
                removed_any = True
                break
        if not remove:
            kept_lines.append(ln)
    out = "\n".join(kept_lines).strip()
    if removed_any:
        gap_notes.append("Some specific numeric/form-code claims were removed because supporting evidence was not found in retrieved refs.")

    # Deterministic key findings from retrieved evidence only.
    strict_findings = _build_citation_derived_key_findings(query, vector_hits, graph_hits)
    out = _replace_section_block(out, "Key Findings", strict_findings)
    strict_refs = _build_deterministic_high_conf_refs(vector_hits, graph_hits)
    out = _replace_section_block(out, "High-Confidence References", strict_refs)
    strict_mismatches = _build_deterministic_mismatches(query, vector_hits, graph_hits)
    out = _replace_section_block(out, "Potential Mismatches or Missing Logic", strict_mismatches)
    strict_steps = _build_deterministic_next_steps(query, vector_hits, graph_hits)
    out = _replace_section_block(out, "Next Steps", strict_steps)

    # Enforce Key Findings ref coverage after deterministic replacement.
    kf_lines = _extract_section_block(out, "Key Findings")
    bullet_lines = [ln for ln in kf_lines if re.match(r"^\s*(?:[-*]|\d+\.)\s*", ln)]
    missing_ref_count = sum(1 for ln in bullet_lines if not _line_has_ref(ln))
    if missing_ref_count > 0:
        gap_notes.append(f"{missing_ref_count} Key Findings bullet(s) lacked explicit refs (file:line or fact-path).")

    # Explicit reporting-vs-calculation distinction for tip-focused queries.
    if tip_focus:
        has_tip_calc = (
            _has_evidence_phrase("tip_income_deduction", vector_hits, graph_hits)
            or _has_evidence_phrase("qualified tips", vector_hits, graph_hits)
        )
        if not has_tip_calc:
            gap_notes.append("Direct tip deduction calculation logic was not found; current evidence is reporting/input oriented.")

    # Replace Evidence Gaps section deterministically.
    if gap_notes:
        gap_lines = [f"- {g}" for g in dict.fromkeys(gap_notes)]
    else:
        gap_lines = []
    out = _replace_section_block(out, "Evidence Gaps", gap_lines)

    return _canonicalize_report_sections(out)


def append_source_coverage_gaps(
    text: str,
    has_direct: bool,
    has_tax: bool,
    has_graph: bool,
) -> str:
    out = text
    missing = []
    if not has_direct:
        missing.append("Direct File")
    if not has_tax:
        missing.append("Tax-Calculator")
    if not has_graph:
        missing.append("Fact Graph")
    if not missing:
        return out
    existing = _extract_section_block(out, "Evidence Gaps")
    existing_clean = [ln for ln in existing if ln.strip()]
    existing_clean.append(f"- Missing retrieval coverage from: {', '.join(missing)}.")
    # De-dup while preserving order.
    dedup: List[str] = []
    seen: Set[str] = set()
    for ln in existing_clean:
        k = ln.strip()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(k)
    return _replace_section_block(out, "Evidence Gaps", dedup)


def append_low_coverage_gaps(
    text: str,
    direct_count: int,
    tax_count: int,
    graph_count: int,
    min_direct: int = 3,
    min_tax: int = 3,
    min_graph: int = 3,
) -> str:
    existing = _extract_section_block(text, "Evidence Gaps")
    lines = [ln for ln in existing if ln.strip()]

    if direct_count < min_direct:
        lines.append(
            f"- Limited Direct File coverage after filtering ({direct_count} hit(s)); conclusions may rely more on other sources."
        )
    if tax_count < min_tax:
        lines.append(
            f"- Limited Tax-Calculator coverage after filtering ({tax_count} hit(s)); calculation-level logic may be underrepresented."
        )
    if graph_count < min_graph:
        lines.append(
            f"- Limited Fact Graph coverage after filtering ({graph_count} hit(s)); graph-derived constraints/relationships may be incomplete."
        )

    if not lines:
        return text

    dedup: List[str] = []
    seen: Set[str] = set()
    for ln in lines:
        k = ln.strip()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(k)
    return _replace_section_block(text, "Evidence Gaps", dedup)


def derive_query_keywords(query: str) -> List[str]:
    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "what", "how",
        "does", "are", "is", "to", "of", "in", "on", "a", "an", "it",
        "information", "reporting",
    }
    # Query-instruction filler words that are usually low-signal for code/fact retrieval.
    instruction_stopwords = {
        "find", "show", "list", "give", "tell", "explain", "including", "related",
        "regarding", "about", "logic", "guidance", "look", "search", "check",
        "under", "within", "around", "review", "section", "sec", "information", "reporting",
    }
    all_stopwords = stopwords | instruction_stopwords
    tax_signal_terms = {
        "deduction", "deductions", "qualified", "tip", "tips", "transition", "relief",
        "income", "credit", "credits", "agi", "form", "section",
    }
    # Terms that are too broad as standalone keywords for this audit use case.
    banned_single_keywords = {"qualified", "transition", "relief", "section", "sec"}
    query_lc = query.lower()
    tokens = tokenize(query_lc)
    # Respect punctuation boundaries when forming phrase patterns.
    # Phrases are built only within segments split by punctuation.
    segment_tokens: List[List[str]] = []
    for seg in re.split(r"[,:;.!?]+", query_lc):
        stoks = tokenize(seg)
        if stoks:
            segment_tokens.append(stoks)
    seen: Set[str] = set()
    out: List[str] = []

    # Keep quoted phrases as high-signal keywords.
    for phrase in re.findall(r'"([^"]+)"', query):
        p = " ".join(tokenize(phrase))
        if p and p not in seen:
            seen.add(p)
            out.append(p)

    # Capture legal references and forms directly (generic, not section-number specific).
    # Examples supported: "§ 224", "§224(a)", "section 45a", "sec. 199a", "form 4137".
    legal_ref_pattern = re.compile(
        r"(?:§{1,2}\s*|section\s+|sec\.?\s+)"
        r"([0-9]+[a-z0-9.\-]*(?:\([a-z0-9]+\))*)"
    )
    for m in legal_ref_pattern.finditer(query_lc):
        ident = m.group(1).strip()
        if not ident:
            continue
        variants = [f"§ {ident}", f"section {ident}"]
        for v in variants:
            if v not in seen:
                seen.add(v)
                out.append(v)

    for m in re.findall(r"(form\s+\d{3,5})", query_lc):
        k = re.sub(r"\s+", " ", m.strip())
        if k and k not in seen:
            seen.add(k)
            out.append(k)

    for t in tokens:
        if len(t) < 3 or t in all_stopwords or t in seen:
            continue
        if t in banned_single_keywords:
            continue
        seen.add(t)
        out.append(t)

    # Add phrase terms only when they are adjacent inside the same punctuation segment.
    # Cross-punctuation term grouping is reserved for co-occurrence scoring only.
    for stoks in segment_tokens:
        for i in range(len(stoks) - 1):
            t1, t2 = stoks[i], stoks[i + 1]
            if len(t1) < 3 or len(t2) < 3:
                continue
            if t1 in all_stopwords or t2 in all_stopwords:
                continue
            bg = f"{t1} {t2}"
            if bg not in seen:
                seen.add(bg)
                out.append(bg)

    # Add longer phrases (3..5 terms), also constrained to punctuation segments.
    # Cross-punctuation grouping remains a co-occurrence-only signal.
    max_phrase_len = 5
    for stoks in segment_tokens:
        for span in range(3, max_phrase_len + 1):
            for i in range(len(stoks) - span + 1):
                window = stoks[i : i + span]
                if any(len(t) < 3 or t in all_stopwords for t in window):
                    continue
                phrase = " ".join(window)
                if phrase not in seen:
                    seen.add(phrase)
                    out.append(phrase)

    # Add only conservative, domain-safe singular forms.
    # Avoid noisy pluralization (e.g., "transition" -> "transitions").
    safe_singular_whitelist = {
        "tips", "deductions", "credits", "forms", "sections",
    }
    variants: List[str] = []
    for k in out:
        if " " in k:
            continue
        if k in safe_singular_whitelist and k.endswith("s") and len(k) > 4:
            variants.append(k[:-1])
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)

    def kw_priority(k: str) -> Tuple[int, int, int, int]:
        legal = int(k.startswith("§") or k.startswith("section ") or k.startswith("form "))
        phrase = int(" " in k)
        phrase_words = len(k.split()) if phrase else 0
        numeric_ref = int(k.isdigit() and len(k) >= 3)
        signal = int(any(term in k for term in tax_signal_terms))
        # Higher tuple sorts first.
        return (legal + numeric_ref, phrase_words, phrase + signal, len(k))

    ranked = sorted(out, key=kw_priority, reverse=True)

    # Final cleanup: keep banned terms only when they are part of a phrase.
    cleaned: List[str] = []
    for k in ranked:
        if " " not in k and k in banned_single_keywords:
            continue
        # Drop malformed bigrams like "deduction qualified" while keeping
        # meaningful phrases such as "qualified tips" and "transition relief".
        if k == "deduction qualified":
            continue
        cleaned.append(k)

    # Keep "deduction" only when query context includes a tax anchor.
    has_tax_anchor = bool(re.search(r"(§\s*\d+|section\s+\d+|form\s+\d+|\btip[s]?\b|\bincome\b|\bcredit[s]?\b)", query_lc))
    if not has_tax_anchor:
        cleaned = [k for k in cleaned if k != "deduction"]

    return cleaned[:40]


def compile_keyword_pattern(keyword: str) -> re.Pattern:
    k = keyword.strip().lower()
    tail_guard = r"(?![a-z0-9])"
    # Section references: allow either symbol or word form.
    if k.startswith("§"):
        num = re.escape(k.lstrip("§").strip())
        return re.compile(
            rf"(?:§\s*{num}{tail_guard}|\bsection\s+{num}{tail_guard}|\bsec\.?\s+{num}{tail_guard})",
            flags=re.IGNORECASE,
        )
    if k.startswith("section "):
        num = re.escape(k.split(" ", 1)[1].strip())
        return re.compile(
            rf"(?:\bsection\s+{num}{tail_guard}|§\s*{num}{tail_guard}|\bsec\.?\s+{num}{tail_guard})",
            flags=re.IGNORECASE,
        )
    # Form references.
    if k.startswith("form "):
        num = re.escape(k.split(" ", 1)[1].strip())
        return re.compile(rf"\bform\s*{num}\b", flags=re.IGNORECASE)
    # Phrase matching: allow space/_/- separators.
    if " " in k:
        parts = [re.escape(p) for p in k.split() if p]
        sep = r"(?:\s+|_|-)+"
        return re.compile(r"\b" + sep.join(parts) + r"\b", flags=re.IGNORECASE)
    return re.compile(r"\b" + re.escape(k) + r"\b", flags=re.IGNORECASE)


def scan_keyword_hits(
    code_dirs: List[Path],
    extensions_by_dir: List[List[str]],
    keywords: List[str],
    max_refs_per_keyword: int,
) -> Dict[str, Dict]:
    patterns = {k: compile_keyword_pattern(k) for k in keywords}
    results = {k: {"count": 0, "score": 0, "refs": []} for k in keywords}

    def keyword_weight(k: str) -> int:
        legal_or_numeric = k.startswith("§") or k.startswith("section ") or k.startswith("form ") or (k.isdigit() and len(k) >= 3)
        if legal_or_numeric:
            return 4
        if " " in k:
            # Longer phrases carry stronger intent than single tokens.
            return min(6, 2 + len(k.split()))
        return 2

    def phrase_proximity_match(line_tokens: List[str], phrase_tokens: List[str], max_gap: int = 2) -> bool:
        # Ordered, near-adjacent subsequence match for soft phrase hits.
        if not line_tokens or not phrase_tokens:
            return False
        pos = -1
        for pt in phrase_tokens:
            found = -1
            for i in range(pos + 1, len(line_tokens)):
                if line_tokens[i] == pt:
                    found = i
                    break
            if found < 0:
                return False
            if pos >= 0 and (found - pos - 1) > max_gap:
                return False
            pos = found
        return True

    def cooccurrence_bonus(k: str, matched_keyword_count: int) -> int:
        # Reward lines where multiple query terms co-occur.
        if matched_keyword_count < 2:
            return 0
        legal_or_phrase = (
            k.startswith("§")
            or k.startswith("section ")
            or k.startswith("form ")
            or " " in k
        )
        unit = 2 if legal_or_phrase else 1
        return (matched_keyword_count - 1) * unit

    def chunk_cooccurrence_bonus(k: str, matched_keyword_count: int) -> int:
        # Stronger boost in fallback mode because chunk windows are broader than lines.
        if matched_keyword_count < 2:
            return 0
        legal_or_phrase = (
            k.startswith("§")
            or k.startswith("section ")
            or k.startswith("form ")
            or " " in k
        )
        unit = 3 if legal_or_phrase else 2
        return (matched_keyword_count - 1) * unit

    for code_dir, exts in zip(code_dirs, extensions_by_dir):
        seen_paths: Set[Path] = set()
        for ext in exts:
            pattern = f"*{ext}" if ext.startswith(".") else f"*.{ext}"
            for p in code_dir.rglob(pattern):
                if p.is_file():
                    seen_paths.add(p)
        for path in sorted(seen_paths):
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            short = _shorten_path(str(path))
            for line_no, line in enumerate(lines, start=1):
                low = line.lower()
                line_tokens = tokenize(low)
                matched_in_line: Dict[str, int] = {}
                for k in keywords:
                    matches = patterns[k].findall(low)
                    if matches:
                        hit_count = len(matches)
                        matched_in_line[k] = hit_count
                        results[k]["count"] += hit_count
                        weight = keyword_weight(k)
                        results[k]["score"] += hit_count * weight
                        if len(results[k]["refs"]) < max_refs_per_keyword:
                                ref = f"{short}:{line_no}"
                                if ref not in results[k]["refs"]:
                                    results[k]["refs"].append(ref)
                        continue

                    # Soft phrase fallback: if an exact phrase is absent, still count
                    # near ordered co-occurrence on the same line with lower weight.
                    if " " in k and len(k.split()) >= 3:
                        phrase_tokens = tokenize(k)
                        if phrase_proximity_match(line_tokens, phrase_tokens, max_gap=2):
                            hit_count = 1
                            matched_in_line[k] = hit_count
                            results[k]["count"] += hit_count
                            weight = max(1, keyword_weight(k) - 2)
                            results[k]["score"] += hit_count * weight
                            if len(results[k]["refs"]) < max_refs_per_keyword:
                                ref = f"{short}:{line_no}"
                                if ref not in results[k]["refs"]:
                                    results[k]["refs"].append(ref)
                if matched_in_line:
                    matched_count = len(matched_in_line)
                    for k in matched_in_line:
                        results[k]["score"] += cooccurrence_bonus(k, matched_count)

    # Always run a chunk-window co-occurrence pass.
    # If line-level signal is weak, boost this pass; otherwise keep it lighter.
    nonzero_keywords = sum(1 for k in keywords if results[k]["score"] > 0)
    top_score = max((results[k]["score"] for k in keywords), default=0)
    min_signal_keywords = max(2, min(5, len(keywords) // 3))
    needs_chunk_fallback = nonzero_keywords < min_signal_keywords or top_score < 8

    window_size = 24
    step = 12
    for code_dir, exts in zip(code_dirs, extensions_by_dir):
        seen_paths: Set[Path] = set()
        for ext in exts:
            pattern = f"*{ext}" if ext.startswith(".") else f"*.{ext}"
            for p in code_dir.rglob(pattern):
                if p.is_file():
                    seen_paths.add(p)
        for path in sorted(seen_paths):
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            short = _shorten_path(str(path))
            n = len(lines)
            if n == 0:
                continue
            for start in range(0, max(1, n), step):
                end = min(n, start + window_size)
                chunk_text = "\n".join(lines[start:end]).lower()
                matched_in_chunk: List[str] = []
                for k in keywords:
                    if patterns[k].search(chunk_text):
                        matched_in_chunk.append(k)
                if len(matched_in_chunk) < 2:
                    continue
                mcount = len(matched_in_chunk)
                for k in matched_in_chunk:
                    bonus = chunk_cooccurrence_bonus(k, mcount)
                    if not needs_chunk_fallback:
                        bonus = max(1, bonus // 2)
                    results[k]["score"] += bonus
                    if len(results[k]["refs"]) < max_refs_per_keyword:
                        ref = f"{short}:{start + 1}"
                        if ref not in results[k]["refs"]:
                            results[k]["refs"].append(ref)
                if end >= n:
                    break
    return results


def _shorten_path(path: str) -> str:
    try:
        cwd = str(Path.cwd().resolve())
        p = str(Path(path).resolve())
        if p.startswith(cwd + "/"):
            return p[len(cwd) + 1 :]
        return p
    except OSError:
        return path


def _cap_wrapped_lines(lines: List[str], max_lines: int) -> List[str]:
    if len(lines) <= max_lines:
        return lines
    trimmed = lines[: max_lines - 1]
    tail = lines[max_lines - 1]
    trimmed.append((tail[: max(0, len(tail) - 3)] + "...") if len(tail) >= 3 else "...")
    return trimmed


def _line_only_ref(ref: str) -> str:
    # Convert "path:12-18" -> "L12-L18", "path:42" -> "L42"
    if ":" not in ref:
        return ref
    tail = ref.rsplit(":", 1)[1]
    return f"L{tail}"


def _vec_fields(hit: Dict) -> Tuple[str, str]:
    dim = hit.get("vector_dim", None)
    norm = hit.get("vector_norm", None)
    dim_s = str(dim) if isinstance(dim, (int, np.integer)) else ("N/A" if dim in (None, "", "?") else str(dim))
    if isinstance(norm, (float, int, np.floating, np.integer)):
        norm_s = f"{float(norm):.4f}"
    else:
        norm_s = "N/A"
    return dim_s, norm_s


def _filter_display_keywords(keywords: List[str], keyword_hits: Dict[str, Dict]) -> List[str]:
    # Display-only cleanup: hide phrase terms with zero matches.
    filtered: List[str] = []
    for kw in keywords:
        count = int(keyword_hits.get(kw, {}).get("count", 0) or 0)
        # Keep legal references visible even when no matches are found so users can
        # verify that section-aware search terms were generated.
        if _parse_legal_identifier(kw) is not None:
            filtered.append(kw)
            continue
        if " " in kw and count == 0:
            continue
        filtered.append(kw)
    return filtered


def _parse_legal_identifier(keyword: str) -> Optional[str]:
    k = keyword.strip().lower()
    m = re.match(r"^§{1,2}\s*([0-9][a-z0-9.\-]*(?:\([a-z0-9]+\))*)$", k)
    if m:
        return m.group(1).strip()
    m = re.match(r"^(?:section|sec\.?)\s+([0-9][a-z0-9.\-]*(?:\([a-z0-9]+\))*)$", k)
    if m:
        return m.group(1).strip()
    return None


def _legal_base_number(keyword: str) -> Optional[str]:
    ident = _parse_legal_identifier(keyword)
    if not ident:
        return None
    m = re.match(r"^(\d+)", ident)
    return m.group(1) if m else None


def _aggregate_keyword_display_rows(keywords: List[str], keyword_hits: Dict[str, Dict]) -> List[Dict]:
    # Collapse equivalent legal-reference aliases (e.g., "section 224", "§ 224", "224")
    # into one ranking row so they don't crowd out other terms.
    legal_ids: Set[str] = set()
    legal_bases: Set[str] = set()
    for kw in keywords:
        ident = _parse_legal_identifier(kw)
        if ident:
            legal_ids.add(ident)
        base = _legal_base_number(kw)
        if base:
            legal_bases.add(base)

    groups: Dict[str, Dict] = {}
    for kw in keywords:
        data = keyword_hits.get(kw, {"score": 0, "count": 0, "refs": []})
        ident = _parse_legal_identifier(kw)
        if ident:
            base = _legal_base_number(kw)
            gk = f"sec:{base or ident}"
        elif kw.isdigit() and kw in legal_bases:
            gk = f"sec:{kw}"
        else:
            gk = f"kw:{kw}"

        if gk not in groups:
            groups[gk] = {"terms": [], "score": 0, "count": 0, "refs": []}
        g = groups[gk]

        if kw not in g["terms"]:
            g["terms"].append(kw)
        g["score"] = max(int(g["score"]), int(data.get("score", 0) or 0))
        g["count"] = max(int(g["count"]), int(data.get("count", 0) or 0))
        for ref in data.get("refs", []):
            if ref not in g["refs"]:
                g["refs"].append(ref)

    rows: List[Dict] = []
    for g in groups.values():
        terms = g["terms"]
        primary = next((t for t in terms if t.startswith("section ")), None)
        if not primary:
            primary = next((t for t in terms if t.startswith("§")), None)
        if not primary:
            primary = terms[0] if terms else "-"

        aliases = [t for t in terms if t != primary]
        label = primary if not aliases else f"{primary} (merged: {', '.join(aliases)})"
        rows.append(
            {
                "keyword": label,
                "score": int(g["score"]),
                "count": int(g["count"]),
                "refs": g["refs"],
            }
        )

    rows.sort(key=lambda r: int(r["score"]), reverse=True)
    return rows


def _canonicalize_keywords_for_retrieval(keywords: List[str]) -> List[str]:
    # Collapse equivalent legal-ref aliases for retrieval query composition.
    # Example: ["section 224", "§ 224", "224"] -> ["section 224"]
    legal_ids: Set[str] = set()
    legal_bases: Set[str] = set()
    for kw in keywords:
        ident = _parse_legal_identifier(kw)
        if ident:
            legal_ids.add(ident)
        base = _legal_base_number(kw)
        if base:
            legal_bases.add(base)

    out: List[str] = []
    seen_groups: Set[str] = set()
    seen_terms: Set[str] = set()
    for kw in keywords:
        ident = _parse_legal_identifier(kw)
        if ident:
            base = _legal_base_number(kw)
            g = f"sec:{base or ident}"
        elif kw.isdigit() and kw in legal_bases:
            g = f"sec:{kw}"
            ident = kw
        else:
            g = f"kw:{kw}"

        if g in seen_groups:
            continue
        seen_groups.add(g)

        term = f"section {ident}" if ident else kw
        if term not in seen_terms:
            seen_terms.add(term)
            out.append(term)
    return out


def _summarize_keyword_refs(refs: List[str]) -> Tuple[str, str]:
    if not refs:
        return "-", "-"
    file_to_lines: Dict[str, List[str]] = {}
    order: List[str] = []
    for ref in refs:
        if ":" not in ref:
            continue
        file_part, line_part = ref.rsplit(":", 1)
        if file_part not in file_to_lines:
            file_to_lines[file_part] = []
            order.append(file_part)
        l = f"L{line_part}"
        if l not in file_to_lines[file_part]:
            file_to_lines[file_part].append(l)
    if not order:
        return refs[0], "-"
    top_file = max(order, key=lambda f: len(file_to_lines[f]))
    return top_file, ", ".join(file_to_lines[top_file][:8])


def _render_wrapped_table(
    title: str,
    headers: List[str],
    rows: List[List[str]],
    col_widths: List[int],
    max_cell_lines: int = 8,
    col_no_wrap: Optional[Set[int]] = None,
) -> None:
    width = max(110, shutil.get_terminal_size((160, 24)).columns)
    usable = width - (len(headers) + 1)
    zero_idx = [i for i, w in enumerate(col_widths) if w <= 0]
    fixed = sum(w for w in col_widths if w > 0)
    if zero_idx:
        remaining = max(20 * len(zero_idx), usable - fixed)
        per = max(20, remaining // len(zero_idx))
        col_widths = [per if i in zero_idx else w for i, w in enumerate(col_widths)]
    elif fixed < usable and col_widths:
        col_widths[-1] += usable - fixed

    console = Console(width=width)
    table = Table(title=title, box=box.SQUARE, show_lines=True, expand=False)
    no_wrap_cols = col_no_wrap or set()
    for i, h in enumerate(headers):
        is_numeric = h.lower() in {"score", "hit count", "chunk"}
        table.add_column(
            h,
            width=col_widths[i],
            min_width=max(6 if is_numeric else 10, col_widths[i]),
            justify="right" if is_numeric else "left",
            overflow="fold",
            no_wrap=i in no_wrap_cols,
        )

    for row in rows:
        rendered = []
        for i, cell in enumerate(row):
            text = str(cell)
            if i in no_wrap_cols:
                rendered.append(text)
            else:
                wrapped = textwrap.wrap(
                    text,
                    width=col_widths[i],
                    break_long_words=True,
                    break_on_hyphens=True,
                )
                wrapped = wrapped if wrapped else [""]
                wrapped = _cap_wrapped_lines(wrapped, max_cell_lines)
                rendered.append("\n".join(wrapped))
        table.add_row(*rendered)

    console.print()
    console.print(table)


def print_debug_retrieval(search_term: str, direct_hits: List[Dict], tax_hits: List[Dict], graph_hits: List[Dict]) -> None:
    print("\nField Notes:")
    print("- Retrieval metadata: chunk_id helps trace exactly which indexed chunk was retrieved from a file.")
    print("- Vector metadata: vec_dim and vec_norm provide embedding sanity checks (same size, non-zero norm).")
    print("- Graph metadata: dependency_count quickly indicates graph node connectivity/importance.")

    direct_rows = []
    for h in direct_hits:
        vec_dim, vec_norm = _vec_fields(h)
        direct_rows.append(
            [
                f"{h['score']:.4f}",
                _shorten_path(h["path"]),
                f"L{h.get('line_start', 1)}-L{h.get('line_end', 1)}",
                str(h["chunk_id"]),
                vec_dim,
                vec_norm,
            ]
        )
    print(f'\nSearch Term: "{search_term}"')
    _render_wrapped_table(
        "DEBUG RETRIEVAL: DIRECT FILE VECTOR HITS",
        ["Score", "File", "Line Ref", "Chunk ID", "Vec Dim", "Vec Norm"],
        direct_rows or [["-", "No hits", "-", "-", "-", "-"]],
        col_widths=[8, 0, 12, 10, 8, 10],
        max_cell_lines=1000,
        col_no_wrap={0, 2, 3, 4, 5},
    )

    tax_rows = []
    for h in tax_hits:
        vec_dim, vec_norm = _vec_fields(h)
        tax_rows.append(
            [
                f"{h['score']:.4f}",
                _shorten_path(h["path"]),
                f"L{h.get('line_start', 1)}-L{h.get('line_end', 1)}",
                str(h["chunk_id"]),
                vec_dim,
                vec_norm,
            ]
        )
    print(f'\nSearch Term: "{search_term}"')
    _render_wrapped_table(
        "DEBUG RETRIEVAL: TAX CALCULATOR VECTOR HITS",
        ["Score", "File", "Line Ref", "Chunk ID", "Vec Dim", "Vec Norm"],
        tax_rows or [["-", "No hits", "-", "-", "-", "-"]],
        col_widths=[8, 0, 12, 10, 8, 10],
        max_cell_lines=1000,
        col_no_wrap={0, 2, 3, 4, 5},
    )

    graph_rows = []
    for n in graph_hits:
        source_ref = n.get("source_ref", "-")
        source_file = source_ref.rsplit(":", 1)[0] if ":" in source_ref else source_ref
        line_ref = _line_only_ref(source_ref) if source_ref and source_ref != "-" else "-"
        graph_rows.append(
            [
                str(n.get("score", 0)),
                n["path"],
                _shorten_path(source_file),
                line_ref,
                str(len(n.get("dependencies", []))),
            ]
        )
    print(f'\nSearch Term: "{search_term}"')
    _render_wrapped_table(
        "DEBUG RETRIEVAL: FACT GRAPH HITS",
        ["Score", "Target", "File", "Line Ref", "Dependency Count"],
        graph_rows or [["-", "No hits", "-", "-", "-"]],
        col_widths=[8, 0, 0, 12, 16],
        max_cell_lines=1000,
        col_no_wrap={0, 3, 4},
    )


def print_debug_retrieval_plain(
    search_term: str,
    direct_hits: List[Dict],
    tax_hits: List[Dict],
    graph_hits: List[Dict],
    keywords: List[str],
    keyword_hits: Dict[str, Dict],
    keyword_top_refs: int,
    slow_output: bool = False,
    output_delay_ms: int = 80,
) -> None:
    # Keep a fixed separator width for consistent visual rhythm across sections.
    sep_line = "-" * 100
    RESET = "\033[0m"
    C_NOTE = "\033[95m"      # magenta
    C_SEARCH = "\033[1;91m"  # bright red
    C_RESULT = "\033[96m"    # cyan
    C_DIV = "\033[94m"       # blue
    C_HEAD = "\033[92m"      # green

    delay_s = max(0.0, float(output_delay_ms) / 1000.0)

    def emit(line: str, color: str = "") -> None:
        # Single-line mode: avoid wrapped multi-line records for easier scanning/copying.
        print(f"{color}{line}{RESET if color else ''}")
        if slow_output and delay_s:
            time.sleep(delay_s)

    def sep() -> None:
        print(f"{C_DIV}{sep_line}{RESET}")

    print(f"\n{C_HEAD}Field Notes{RESET}")
    emit("NOTE > Retrieval metadata: chunk_id helps trace exactly which indexed chunk was retrieved from a file.", color=C_NOTE)
    emit("NOTE > Vector metadata: vec_dim and vec_norm provide embedding sanity checks (same size, non-zero norm).", color=C_NOTE)
    emit("NOTE > Graph metadata: dependency_count quickly indicates graph node connectivity/importance.", color=C_NOTE)

    print(f"\n{C_HEAD}=== STEP 1: KEYWORD SCAN (PLAIN) ==={RESET}")
    emit(f'[SEARCH PARAMS] search_term="{search_term}"', color=C_SEARCH)
    sep()
    display_keywords = _filter_display_keywords(keywords, keyword_hits)
    kw_rows = _aggregate_keyword_display_rows(display_keywords, keyword_hits)
    for rank, row in enumerate(kw_rows, start=1):
        refs = row.get("refs", [])[:keyword_top_refs]
        top_file, line_refs = _summarize_keyword_refs(refs)
        emit(
            f"[RESULT {rank}] "
            f'keyword="{row.get("keyword", "-")}" | score={row.get("score", 0)} | hits={row.get("count", 0)} '
            f"| file={_shorten_path(top_file)} | lines={line_refs}",
            color=C_RESULT,
        )
        sep()

    print(f"\n{C_HEAD}=== DIRECT FILE VECTOR HITS (PLAIN) ==={RESET}")
    emit(f'[SEARCH PARAMS] search_term="{search_term}"', color=C_SEARCH)
    sep()
    for i, h in enumerate(direct_hits, start=1):
        vec_dim, vec_norm = _vec_fields(h)
        emit(
            f"[RESULT {i}] "
            f"score={h['score']:.4f} | file={_shorten_path(h['path'])} "
            f"| line_ref=L{h.get('line_start', 1)}-L{h.get('line_end', 1)} | chunk_id={h['chunk_id']} "
            f"| vec_dim={vec_dim} | vec_norm={vec_norm}",
            color=C_RESULT,
        )
        sep()

    print(f"\n{C_HEAD}=== TAX CALCULATOR VECTOR HITS (PLAIN) ==={RESET}")
    emit(f'[SEARCH PARAMS] search_term="{search_term}"', color=C_SEARCH)
    sep()
    for i, h in enumerate(tax_hits, start=1):
        vec_dim, vec_norm = _vec_fields(h)
        emit(
            f"[RESULT {i}] "
            f"score={h['score']:.4f} | file={_shorten_path(h['path'])} "
            f"| line_ref=L{h.get('line_start', 1)}-L{h.get('line_end', 1)} | chunk_id={h['chunk_id']} "
            f"| vec_dim={vec_dim} | vec_norm={vec_norm}",
            color=C_RESULT,
        )
        sep()

    print(f"\n{C_HEAD}=== FACT GRAPH HITS (PLAIN) ==={RESET}")
    emit(f'[SEARCH PARAMS] search_term="{search_term}"', color=C_SEARCH)
    sep()
    for i, n in enumerate(graph_hits, start=1):
        source_ref = n.get("source_ref", "-")
        source_file = source_ref.rsplit(":", 1)[0] if ":" in source_ref else source_ref
        line_ref = _line_only_ref(source_ref) if source_ref and source_ref != "-" else "-"
        emit(
            f"[RESULT {i}] "
            f"score={n.get('score', 0)} | target={n['path']} | file={_shorten_path(source_file)} "
            f"| line_ref={line_ref} | dependency_count={len(n.get('dependencies', []))}",
            color=C_RESULT,
        )
        sep()


def ensure_audit_sections(text: str) -> str:
    out = text.strip()
    low = out.lower()
    if "evidence gaps" not in low:
        out += "\n\nEvidence Gaps:\n- Not explicitly provided by model output. Review refs in retrieval tables."
    if "next steps" not in low:
        out += "\n\nNext Steps:\n1. Validate top cited refs in code and graph.\n2. Add/adjust rules and tests.\n3. Re-run hybrid audit."
    return out


def build_retrieval_query(query: str, keywords: List[str]) -> str:
    # Shared expanded query string for vector + graph searches.
    # Keeps all retrieval paths aligned with keyword/legal normalization logic.
    canonical_keywords = _canonicalize_keywords_for_retrieval(keywords)
    if not canonical_keywords:
        return query
    return f'{query}\n\nExpanded search terms: {", ".join(canonical_keywords)}'


def run(
    query: str,
    debug: bool = False,
    plain_debug: bool = False,
    slow_output: bool = True,
    output_delay_ms: int = 80,
    show_loading: bool = False,
) -> str:
    def loading(msg: str) -> None:
        if show_loading:
            print(f"[LOADING] {msg}", flush=True)

    loading("Loading configuration and models...")
    config = load_config()
    embeddings = OllamaEmbeddings(model=config.embedding_model)
    llm = OllamaLLM(model=config.llm_model, timeout=300)

    loading("Building/loading Direct File vector index...")
    direct_docs, direct_vecs = build_or_load_vector_index(
        config,
        embeddings,
        config.direct_file_code_dir,
        config.direct_file_extensions,
        "direct_file",
    )
    loading("Building/loading Tax-Calculator vector index...")
    tax_docs, tax_vecs = build_or_load_vector_index(
        config,
        embeddings,
        config.tax_calc_code_dir,
        config.tax_calc_extensions,
        "tax_calculator",
    )
    loading("Building/loading Fact Graph index...")
    graph_index = build_or_load_graph_index(config)

    loading("Deriving query keywords and retrieval query...")
    keywords = derive_query_keywords(query)
    retrieval_query = build_retrieval_query(query, keywords)
    retrieval_keywords = _canonicalize_keywords_for_retrieval(keywords)

    loading("Running Direct File vector retrieval...")
    direct_hits = vector_retrieve(
        retrieval_query,
        direct_docs,
        direct_vecs,
        embeddings,
        config.vector_top_k,
        query_keywords=retrieval_keywords,
    )
    loading("Running Tax-Calculator vector retrieval...")
    tax_hits = vector_retrieve(
        retrieval_query,
        tax_docs,
        tax_vecs,
        embeddings,
        config.vector_top_k,
        query_keywords=retrieval_keywords,
    )
    vector_hits = sorted(direct_hits + tax_hits, key=lambda x: x["score"], reverse=True)[: config.vector_top_k * 2]
    loading("Running Fact Graph retrieval...")
    graph_hits = graph_retrieve(
        query=query,
        query_keywords=retrieval_keywords,
        graph_index=graph_index,
        seed_k=config.graph_seed_k,
        hops=config.graph_expand_hops,
        max_return=config.graph_max_return,
    )

    # Intent correlation gate for generation context:
    # avoid drifting to generic deduction topics with weak query overlap.
    signal_tokens = _build_intent_signal_tokens(retrieval_keywords)
    vector_hits = _filter_vector_hits_by_intent(vector_hits, signal_tokens, keep_top=config.vector_top_k * 2)
    graph_hits = _filter_graph_hits_by_intent(graph_hits, signal_tokens, keep_top=config.graph_max_return)

    # Maintain explicit per-source context coverage.
    tip_focus = is_tip_focus_query(query)
    direct_context_hits = [h for h in vector_hits if h.get("source") == "direct_file"][: config.vector_top_k]
    tax_context_hits = [h for h in vector_hits if h.get("source") == "tax_calculator"][: config.vector_top_k]
    if tip_focus and not tax_context_hits and tax_hits:
        tax_context_hits = tax_hits[: min(2, len(tax_hits))]
    if tip_focus and not has_tip_logic_evidence(direct_context_hits + tax_context_hits, graph_hits):
        loading("Insufficient tip evidence detected; preparing fallback report...")
        fallback = build_insufficient_evidence_report(query, direct_context_hits, tax_context_hits, graph_hits)
        fallback = append_source_coverage_gaps(
            text=fallback,
            has_direct=bool(direct_context_hits),
            has_tax=bool(tax_context_hits),
            has_graph=bool(graph_hits),
        )
        return ensure_audit_sections(fallback)
    if debug:
        loading("Scanning keyword hits for debug output...")
        keyword_hits = scan_keyword_hits(
            code_dirs=[config.direct_file_code_dir, config.tax_calc_code_dir],
            extensions_by_dir=[config.direct_file_extensions, config.tax_calc_extensions],
            keywords=keywords,
            max_refs_per_keyword=config.keyword_max_refs_per_keyword,
        )
        display_keywords = _filter_display_keywords(keywords, keyword_hits)
        agg_kw_rows = _aggregate_keyword_display_rows(display_keywords, keyword_hits)
        kw_rows = []
        for row in agg_kw_rows:
            refs = row["refs"][: config.keyword_top_refs]
            top_file, line_refs = _summarize_keyword_refs(refs)
            kw_rows.append(
                [
                    f'"{row["keyword"]}"',
                    str(row["score"]),
                    str(row["count"]),
                    _shorten_path(top_file),
                    line_refs,
                ]
            )
        kw_rows = sorted(kw_rows, key=lambda r: int(r[1]), reverse=True)
        loading("Rendering debug retrieval output...")
        if plain_debug:
            print_debug_retrieval_plain(
                search_term=query,
                direct_hits=direct_hits,
                tax_hits=tax_hits,
                graph_hits=graph_hits,
                keywords=display_keywords,
                keyword_hits=keyword_hits,
                keyword_top_refs=config.keyword_top_refs,
                slow_output=slow_output,
                output_delay_ms=output_delay_ms,
            )
        else:
            _render_wrapped_table(
                "STEP 1: KEYWORD SCAN",
                ["Keyword", "Score", "Hit Count", "File", "Line Refs"],
                kw_rows or [["-", "0", "0", "-", "-"]],
                col_widths=[20, 8, 12, 0, 0],
                max_cell_lines=1000,
                col_no_wrap={0, 1, 2},
            )
            print_debug_retrieval(query, direct_hits, tax_hits, graph_hits)

    loading("Building prompt context and generating audit answer...")
    vector_context = format_vector_context_split(direct_context_hits, tax_context_hits, tip_focus=tip_focus)
    graph_context = format_graph_context(graph_hits)

    prompt = ChatPromptTemplate.from_template(
        "You are auditing tax code behavior against an IRS fact graph.\n\n"
        'Question: "{query}"\n\n'
        "Vector retrieval context from two codebases (Direct File + Tax Calculator):\n{vector_context}\n\n"
        "Graph retrieval context from IRS fact graph:\n{graph_context}\n\n"
        "Reasoning constraints:\n"
        "- Prioritize direct overlap with section/form refs and domain terms in the question.\n"
        "- Do NOT infer relevance from generic words alone (for example: deduction, income, credit).\n"
        "- If evidence is weak or off-topic, explicitly state no direct correlation.\n"
        "- If query intent is tip-focused, keep only tip-relevant logic and citations.\n"
        "- Exclude unrelated taxable-income pipeline steps (senior/overtime/auto-loan/QBI) unless explicitly requested.\n"
        "- When a broad multi-step block is retrieved, extract and report only the tip-relevant step(s).\n"
        "- In Key Findings, append explicit refs in-line for each bullet (file:line or fact-path).\n"
        "- Do not include specific numeric/form-code claims unless directly supported by retrieved refs.\n"
        "- Keep findings grounded to cited refs shown in retrieval context.\n\n"
        "Return with these exact section headings:\n"
        "Key Findings\n"
        "Potential Mismatches or Missing Logic\n"
        "High-Confidence References\n"
        "Evidence Gaps\n"
        "Next Steps\n"
    )
    chain = prompt | llm
    raw = chain.invoke(
        {
            "query": query,
            "vector_context": vector_context,
            "graph_context": graph_context,
        }
    )
    aligned = enforce_claim_alignment(
        query=query,
        answer=raw,
        vector_hits=direct_context_hits + tax_context_hits,
        graph_hits=graph_hits,
        tax_hits_present=bool(tax_context_hits),
    )
    aligned = append_source_coverage_gaps(
        text=aligned,
        has_direct=bool(direct_context_hits),
        has_tax=bool(tax_context_hits),
        has_graph=bool(graph_hits),
    )
    aligned = append_low_coverage_gaps(
        text=aligned,
        direct_count=len(direct_context_hits),
        tax_count=len(tax_context_hits),
        graph_count=len(graph_hits),
    )
    return ensure_audit_sections(aligned)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Vector RAG + Graph Retrieval for tax audit.")
    parser.add_argument("query", nargs="?", type=str, help="Question or topic to audit.")
    parser.add_argument(
        "--debug-retrieval",
        action="store_true",
        help="Print debug retrieval output (plain mode by default).",
    )
    parser.add_argument("--plain-debug", action="store_true", help="Print plain-text debug lines (no tables).")
    parser.add_argument("--rich-debug", action="store_true", help="Print Rich table debug output.")
    args = parser.parse_args()
    query = args.query or input("Enter your audit query: ").strip()
    if not query:
        raise SystemExit("No query provided.")
    debug = args.debug_retrieval or args.plain_debug or args.rich_debug
    # Permanent default: debug output is plain unless rich is explicitly requested.
    plain_debug = (args.debug_retrieval or args.plain_debug) and not args.rich_debug
    result = run(
        query,
        debug=debug,
        plain_debug=plain_debug,
        slow_output=True,
        output_delay_ms=80,
        show_loading=debug,
    )
    print(result)


if __name__ == "__main__":
    main()
