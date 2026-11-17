# Agent notes

- Python 3.11 venv lives in `.venv` (created with `uv venv --seed --python 3.11 .venv`; the
  machine has no system Python 3.11). Run tools as `.venv/Scripts/python.exe -m ...`.
- Verify changes with: `pytest -q`, `ruff check .`, `ruff format --check .`.
- Tests must stay offline: use the fakes in `tests/conftest.py` (`FakeEmbedder`,
  `FakeReranker`, `FakeLLM`) and `httpx.MockTransport`.
- `langchain-community` must stay `<0.4` while ragas is 0.4.x.
- Do not run `scripts/rebuild_index.py` or `scripts/ingest_pubmed.py` while the API is running
  on the same `CHROMA_PATH`; set `CHROMA_PATH` to a temp directory for experiments.
- The project has its own Git repository (remote `origin`:
  `https://github.com/Arun-2010/Medical-Evidence-RAG-.git`, branch `main`). It is nested inside
  another repository rooted in the user's home directory; always run git from the project directory.
- Never commit `.env`, `data/raw`, `data/processed`, `data/chroma` or evaluation outputs.
- README screenshots live in `docs/images/`.
