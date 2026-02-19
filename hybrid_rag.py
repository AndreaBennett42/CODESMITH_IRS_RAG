import argparse
import json
import os
import re
import shutil
import textwrap
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
) -> List[Dict]:
    q = np.asarray(embeddings.embed_query(query), dtype=np.float32)
    denom = (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q)) + 1e-12
    scores = (vectors @ q) / denom
    top_idx = np.argsort(-scores)[:top_k]
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
        for i in top_idx
    ]


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
    hay = set(tokenize(node_blob))
    return sum(1 for t in query_tokens if t in hay)


def graph_retrieve(query: str, graph_index: Dict, seed_k: int, hops: int, max_return: int) -> List[Dict]:
    nodes = graph_index["nodes"]
    edges = graph_index["edges"]
    reverse_edges = graph_index["reverse_edges"]
    q_tokens = tokenize(query)

    scored = []
    for path, node in nodes.items():
        s = score_node(q_tokens, node["search_blob"])
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


def derive_query_keywords(query: str) -> List[str]:
    stopwords = {
        "the", "and", "for", "with", "from", "that", "this", "what", "how",
        "does", "are", "is", "to", "of", "in", "on", "a", "an", "it",
    }
    query_lc = query.lower()
    tokens = tokenize(query_lc)
    seen: Set[str] = set()
    out: List[str] = []

    # Keep quoted phrases as high-signal keywords.
    for phrase in re.findall(r'"([^"]+)"', query):
        p = " ".join(tokenize(phrase))
        if p and p not in seen:
            seen.add(p)
            out.append(p)

    # Capture legal references and forms directly.
    for m in re.findall(r"(§\s*\d+[a-z0-9\-]*|section\s+\d+[a-z0-9\-]*|form\s+\d{3,5})", query_lc):
        k = re.sub(r"\s+", " ", m.strip())
        if k and k not in seen:
            seen.add(k)
            out.append(k)

    for t in tokens:
        if len(t) < 3 or t in stopwords or t in seen:
            continue
        seen.add(t)
        out.append(t)

    # Add bigrams from meaningful adjacent tokens.
    terms = [t for t in tokens if len(t) >= 3 and t not in stopwords]
    for i in range(len(terms) - 1):
        bg = f"{terms[i]} {terms[i+1]}"
        if bg not in seen:
            seen.add(bg)
            out.append(bg)

    # Add common normalized variants for better matching.
    variants: List[str] = []
    for k in out:
        if k.endswith("s") and len(k) > 4:
            variants.append(k[:-1])
        elif not k.endswith("s") and " " not in k and len(k) > 4:
            variants.append(f"{k}s")
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)

    return out[:10]


def compile_keyword_pattern(keyword: str) -> re.Pattern:
    k = keyword.strip().lower()
    # Phrase matching: allow space/_/- separators.
    if " " in k:
        parts = [re.escape(p) for p in k.split() if p]
        sep = r"(?:\s+|_|-)+"
        return re.compile(r"\b" + sep.join(parts) + r"\b")
    # Section references.
    if k.startswith("§"):
        num = re.escape(k.lstrip("§").strip())
        return re.compile(rf"§\s*{num}\b")
    # Form references.
    if k.startswith("form "):
        num = re.escape(k.split(" ", 1)[1].strip())
        return re.compile(rf"\bform\s*{num}\b")
    return re.compile(r"\b" + re.escape(k) + r"\b")


def scan_keyword_hits(
    code_dirs: List[Path],
    extensions_by_dir: List[List[str]],
    keywords: List[str],
    max_refs_per_keyword: int,
) -> Dict[str, Dict]:
    patterns = {k: compile_keyword_pattern(k) for k in keywords}
    results = {k: {"count": 0, "score": 0, "refs": []} for k in keywords}
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
                for k in keywords:
                    matches = patterns[k].findall(low)
                    if matches:
                        hit_count = len(matches)
                        results[k]["count"] += hit_count
                        weight = 2 if (" " in k or k.startswith("§") or k.startswith("form ")) else 1
                        results[k]["score"] += hit_count * weight
                        if len(results[k]["refs"]) < max_refs_per_keyword:
                            ref = f"{short}:{line_no}"
                            if ref not in results[k]["refs"]:
                                results[k]["refs"].append(ref)
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
) -> None:
    # Keep a fixed separator width for consistent visual rhythm across sections.
    sep_line = "-" * 100
    RESET = "\033[0m"
    C_NOTE = "\033[95m"      # magenta
    C_SEARCH = "\033[1;91m"  # bright red
    C_RESULT = "\033[96m"    # cyan
    C_DIV = "\033[94m"       # blue
    C_HEAD = "\033[92m"      # green

    def emit(line: str, color: str = "") -> None:
        # Single-line mode: avoid wrapped multi-line records for easier scanning/copying.
        print(f"{color}{line}{RESET if color else ''}")

    def sep() -> None:
        print(f"{C_DIV}{sep_line}{RESET}")

    print(f"\n{C_HEAD}Field Notes{RESET}")
    emit("NOTE > Retrieval metadata: chunk_id helps trace exactly which indexed chunk was retrieved from a file.", color=C_NOTE)
    emit("NOTE > Vector metadata: vec_dim and vec_norm provide embedding sanity checks (same size, non-zero norm).", color=C_NOTE)
    emit("NOTE > Graph metadata: dependency_count quickly indicates graph node connectivity/importance.", color=C_NOTE)

    print(f"\n{C_HEAD}=== STEP 1: KEYWORD SCAN (PLAIN) ==={RESET}")
    emit(f'[SEARCH PARAMS] search_term="{search_term}"', color=C_SEARCH)
    sep()
    for rank, kw in enumerate(sorted(keywords, key=lambda k: int(keyword_hits.get(k, {}).get("score", 0)), reverse=True), start=1):
        data = keyword_hits.get(kw, {"score": 0, "count": 0, "refs": []})
        refs = data.get("refs", [])[:keyword_top_refs]
        top_file, line_refs = _summarize_keyword_refs(refs)
        emit(
            f"[RESULT {rank}] "
            f'keyword="{kw}" | score={data.get("score", 0)} | hits={data.get("count", 0)} '
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


def run(query: str, debug: bool = False, plain_debug: bool = False) -> str:
    config = load_config()
    embeddings = OllamaEmbeddings(model=config.embedding_model)
    llm = OllamaLLM(model=config.llm_model, timeout=300)

    direct_docs, direct_vecs = build_or_load_vector_index(
        config,
        embeddings,
        config.direct_file_code_dir,
        config.direct_file_extensions,
        "direct_file",
    )
    tax_docs, tax_vecs = build_or_load_vector_index(
        config,
        embeddings,
        config.tax_calc_code_dir,
        config.tax_calc_extensions,
        "tax_calculator",
    )
    graph_index = build_or_load_graph_index(config)

    direct_hits = vector_retrieve(query, direct_docs, direct_vecs, embeddings, config.vector_top_k)
    tax_hits = vector_retrieve(query, tax_docs, tax_vecs, embeddings, config.vector_top_k)
    vector_hits = sorted(direct_hits + tax_hits, key=lambda x: x["score"], reverse=True)[: config.vector_top_k * 2]
    graph_hits = graph_retrieve(
        query=query,
        graph_index=graph_index,
        seed_k=config.graph_seed_k,
        hops=config.graph_expand_hops,
        max_return=config.graph_max_return,
    )
    if debug:
        keywords = derive_query_keywords(query)
        keyword_hits = scan_keyword_hits(
            code_dirs=[config.direct_file_code_dir, config.tax_calc_code_dir],
            extensions_by_dir=[config.direct_file_extensions, config.tax_calc_extensions],
            keywords=keywords,
            max_refs_per_keyword=config.keyword_max_refs_per_keyword,
        )
        kw_rows = []
        for kw in keywords:
            refs = keyword_hits[kw]["refs"][: config.keyword_top_refs]
            top_file, line_refs = _summarize_keyword_refs(refs)
            kw_rows.append(
                [
                    f'"{kw}"',
                    str(keyword_hits[kw]["score"]),
                    str(keyword_hits[kw]["count"]),
                    _shorten_path(top_file),
                    line_refs,
                ]
            )
        kw_rows = sorted(kw_rows, key=lambda r: int(r[1]), reverse=True)
        if plain_debug:
            print_debug_retrieval_plain(
                search_term=query,
                direct_hits=direct_hits,
                tax_hits=tax_hits,
                graph_hits=graph_hits,
                keywords=keywords,
                keyword_hits=keyword_hits,
                keyword_top_refs=config.keyword_top_refs,
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

    vector_context = format_vector_context(vector_hits)
    graph_context = format_graph_context(graph_hits)

    prompt = ChatPromptTemplate.from_template(
        "You are auditing tax code behavior against an IRS fact graph.\n\n"
        'Question: "{query}"\n\n'
        "Vector retrieval context from two codebases (Direct File + Tax Calculator):\n{vector_context}\n\n"
        "Graph retrieval context from IRS fact graph:\n{graph_context}\n\n"
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
    return ensure_audit_sections(raw)


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
    result = run(query, debug=debug, plain_debug=plain_debug)
    print(result)


if __name__ == "__main__":
    main()
