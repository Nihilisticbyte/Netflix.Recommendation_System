# Repository Structure Analysis & Visualisation System

Parses local Git repositories into an interactive dependency graph with
AI-generated plain-English file summaries.

---

## Project Structure

```
repo-visualizer/
├── backend/
│   ├── main.py                  # FastAPI app, CORS, router registration
│   ├── requirements.txt
│   ├── .env.example
│   └── routers/
│       ├── graph.py             # Directory traversal, dependency extraction,
│       │                        # React Flow JSON output
│       └── ai.py                # AI summary endpoint + SHA-256 local cache
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── .env.example
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── index.css
        ├── hooks/
        │   ├── useGraphData.js   # Fetches /api/graph/ → nodes + edges
        │   └── useFileSummary.js # Calls /api/ai/summarise + session cache
        └── components/
            ├── RepoGraph.jsx     # Main canvas (React Flow) + top bar
            ├── FileNode.jsx      # Custom node: colour strip, LoC, complexity
            └── SummaryPanel.jsx  # Side panel: metrics + AI summary
```

---

## Team Workload Split

| File | Responsibility |
|---|---|
| `backend/routers/graph.py` | Directory traversal, import extraction, edge resolution |
| `backend/routers/ai.py` | AI API integration, SHA-256 caching |
| `frontend/src/components/RepoGraph.jsx` | Canvas layout, data fetching, node click logic |
| `frontend/src/components/FileNode.jsx` | Node card visual (functional logic) |
| `frontend/src/components/SummaryPanel.jsx` | Side panel data display |
| `frontend/src/index.css` | + component styling 
---

## Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env → add GEMINI_API_KEY or OPENAI_API_KEY

uvicorn main:app --reload --port 8000
```

Swagger UI available at: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Open `http://localhost:5173`, paste a local repository path, click **Analyse**.

---

## How It Works

### 1. Directory Traversal (`graph.py`)

`os.walk` traverses the repository root. Directories in `SKIP_DIRS`
(`.git`, `node_modules`, `__pycache__`, etc.) are pruned before
recursing so they are never scanned.

For each supported file:
- Lines of Code are counted
- Complexity is assigned: Low (< 50 lines), Medium (50–200), High (200+)
- Imports are extracted using the appropriate parser

### 2. Dependency Extraction

| Language | Method |
|---|---|
| Python | AST parser (`ast.parse`), regex fallback on SyntaxError |
| JavaScript / TypeScript | Regex — ES `import`, CommonJS `require`, dynamic `import()` |
| C / C++ | Regex — `#include "..."` and `#include <...>` |
| Java | Regex — `import com.example.Foo;` |
| Go | Regex — `import "pkg"` |

After extracting import names, a stem-map resolves them to actual node IDs
where possible (e.g. `import utils` → `src/utils.py`), creating edges
without executing any code.

### 3. React Flow Output

The backend returns exactly the JSON structure React Flow expects:

```json
{
  "nodes": [
    {
      "id": "src/utils.py",
      "type": "fileNode",
      "position": { "x": 60, "y": 60 },
      "data": {
        "label": "utils.py",
        "filePath": "/abs/path/src/utils.py",
        "extension": ".py",
        "linesOfCode": 142,
        "complexity": "medium",
        "imports": ["os", "re", "pathlib"],
        "colour": "#3572A5"
      }
    }
  ],
  "edges": [
    {
      "id": "e__main.py__src-utils.py",
      "source": "main.py",
      "target": "src/utils.py",
      "animated": false
    }
  ],
  "meta": {
    "root": "/abs/path",
    "totalFiles": 12,
    "totalEdges": 7,
    "truncated": false
  }
}
```

### 4. AI Summary (`ai.py`)

When a node is clicked, the frontend:
1. Fetches the file's raw content via `GET /api/graph/file-content`
2. Sends it to `POST /api/ai/summarise`

The backend checks the local `cache.json` using a SHA-256 hash of the
file content as the key. If the file hasn't changed since the last
analysis, the cached summary is returned immediately (no AI call made).

The AI prompt asks for a 3-sentence plain-English explanation covering:
- The file's main purpose
- Key functions or classes defined
- How it fits into the larger project

### 5. Caching Strategy

```
┌─────────────────────────────────────────────────────┐
│  cache.json  (persistent, survives server restarts)  │
│                                                      │
│  "/path/to/file.py" : {                              │
│    "hash"      : "sha256 of file content",           │
│    "summary"   : "This file handles...",             │
│    "model_used": "gemini-1.5-flash",                 │
│    "cached_at" : 1718000000.0                        │
│  }                                                   │
└─────────────────────────────────────────────────────┘

File unchanged → hash matches → return cached summary (free)
File changed   → hash differs → call AI, update cache entry
```

The frontend also keeps an in-memory session cache so repeated clicks
on the same node within one browser session skip the HTTP call entirely.

---

## Switching AI Provider

In `backend/.env`:

```
AI_PROVIDER=openai          # default is "gemini"
OPENAI_API_KEY=sk-...
```

No code changes needed.

---

## Adding More Languages

In `backend/routers/graph.py`, add the extension to `SUPPORTED_EXTENSIONS`
and write an extractor function:

```python
def _ruby_imports(source: str) -> list[str]:
    return re.findall(r'^require\s+[\'"]([^\'"]+)[\'"]', source, re.MULTILINE)

# Then in _extract_imports():
if ext == ".rb":
    return _ruby_imports(source)
```
