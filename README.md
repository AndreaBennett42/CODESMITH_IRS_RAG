# CODESMITH IRS Project (Hybrid RAG)

Hybrid script runs a 3-search flow over your code dataset and fact graph.

## Quick Start
```bash
cd /Users/andreabennett/CS/CODESMITH_IRS_PROJECT
source .venv_ai/bin/activate
./env_check.sh
python hybrid_rag.py --debug-retrieval
```

## Runtime And Environment
- Runtime: local macOS
- Active environment: `.venv_ai`
- Main script: `hybrid_rag.py`

Environment checks:
- `./env_check.sh` validates `.venv_ai` is active
- If wrong env, it prints `WRONG ENV DETECTED` and activation instructions

## 3 Searches
1. Direct File vector search
- Searches chunks from `./direct-file`
- Returns score + file/line refs + chunk/vector metadata

2. Tax-Calculator vector search
- Searches chunks from `./Tax-Calculator/taxcalc`
- Returns score + file/line refs + chunk/vector metadata

3. Fact-graph search
- Searches XML facts/dependencies from `FACTS_XML` directory
- Returns graph score + fact path + source file/line + dependency count

## Prompt And Output
Prompt behavior:
- If query is missing, script prompts: `Enter your audit query:`
- LLM is asked to return:
  - `Key Findings`
  - `Potential Mismatches or Missing Logic`
  - `High-Confidence References`
  - `Evidence Gaps`
  - `Next Steps`

Output controls:
- Enforced structure includes `Evidence Gaps` and `Next Steps`
- Retrieval output includes explicit refs/metadata

Debug modes:
- `--debug-retrieval` (default debug mode = plain)
- `--rich-debug` (Rich table output)

Plain debug markers:
- `NOTE > ...`
- `[SEARCH PARAMS] ...`
- `[RESULT n] ...`

## Config
Configured in `.env`:
- `DIRECT_FILE_CODE_DIR=./direct-file`
- `TAX_CALC_CODE_DIR=./Tax-Calculator/taxcalc`
- `FACTS_XML=./direct-file/backend/src/main/resources/tax`

Notes:
- `FACTS_XML` supports directory indexing (all XML files)
- Text markers are portable across terminals; colors may vary by terminal
