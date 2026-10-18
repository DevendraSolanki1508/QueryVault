import os
import re
import subprocess

print("\n" + "=" * 52)
print("   QueryVault - Project Status Check")
print("=" * 52)

checks = []


def record(ok, message):
    checks.append(1 if ok else 0)
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {message}")


# -- Files ----------------------------------------------------
print("\nFiles:")
files = {
    "ingest.py": "Ingestion pipeline",
    "query_engine.py": "Hybrid retrieval + reranking + chat",
    "server.py": "FastAPI backend",
    "index.html": "Chat UI",
    "evaluate.py": "Evaluation script",
    "README.md": "Documentation",
    "requirements.txt": "Dependencies",
}
for name, desc in files.items():
    ok = os.path.exists(name) and os.path.getsize(name) > 100
    record(ok, f"{name:<18} {desc}")

# -- requirements.txt sanity ----------------------------------
print("\nrequirements.txt:")
if os.path.exists("requirements.txt"):
    bad = []
    with open("requirements.txt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith("#") and not re.match(r"^[A-Za-z0-9_.\-\[\]]+\s*(==|>=|<=|~=|@|$)", line):
                bad.append(i)
    record(not bad, "all lines look like packages" if not bad else f"junk on lines {bad[:5]} - regenerate with: pip freeze > requirements.txt")
else:
    record(False, "requirements.txt missing")

# -- UI labels ------------------------------------------------
print("\nUI labels:")
if os.path.exists("index.html"):
    with open("index.html", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    record("NotesMind" not in html, "index.html uses QueryVault name" if "NotesMind" not in html else "index.html still says NotesMind")
    record("Mistral 7B" not in html, "model label updated" if "Mistral 7B" not in html else "index.html still says Mistral 7B")

# -- ChromaDB -------------------------------------------------
print("\nChromaDB:")
try:
    import chromadb
    col = chromadb.PersistentClient(path="./chroma_db").get_collection("notesmind")
    n = col.count()
    record(n > 0, f"{n} chunks indexed" if n > 0 else "empty - run python ingest.py")
except Exception as e:
    record(False, f"{e} - run python ingest.py")

# -- Ollama + model -------------------------------------------
print("\nOllama:")
model_in_code = None
if os.path.exists("query_engine.py"):
    with open("query_engine.py", encoding="utf-8", errors="ignore") as f:
        m = re.search(r'LLM_MODEL\s*=\s*"([^"]+)"', f.read())
        model_in_code = m.group(1) if m else None
try:
    import ollama
    installed = [x.model for x in ollama.list().models]
    record(True, "Ollama is running")
    if model_in_code:
        ok = any(model_in_code == i or i.startswith(model_in_code + ":") for i in installed)
        record(ok, f"query_engine.py uses '{model_in_code}'" + ("" if ok else " (not installed)"))
except Exception:
    record(False, "Ollama not running - open the Ollama app")

# -- Documents ------------------------------------------------
print("\nDocuments:")
if os.path.isdir("docs"):
    docs = [d for d in os.listdir("docs") if not d.startswith(".")]
    record(len(docs) > 0, f"{len(docs)} file(s) in docs/: {', '.join(docs)}" if docs else "docs/ is empty")
else:
    record(False, "docs/ folder missing")

# -- Evaluation results ---------------------------------------
print("\nEvaluation:")
record(os.path.exists("eval_results.json"), "eval_results.json exists" if os.path.exists("eval_results.json") else "run python evaluate.py")

# -- Git ------------------------------------------------------
print("\nGit:")
if os.path.isdir(".git"):
    log = subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    record(True, "repo found, latest commits:")
    for line in log.splitlines():
        print(f"         {line}")
    record(not status, "everything committed" if not status else "uncommitted changes - run git add . && git commit && git push")
else:
    record(False, "not a git repo")

# -- Score ----------------------------------------------------
done, total = sum(checks), len(checks)
pct = round(done / total * 100)
print("\n" + "=" * 52)
print(f"   {done}/{total} checks passed - {pct}%")
print("=" * 52 + "\n")