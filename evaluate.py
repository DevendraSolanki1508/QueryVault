import json
import time
from query_engine import load_query_engine, chat

# Test set: questions whose answers exist in your text PDF
TESTS = [
    {"q": "What is the aim of Experiment 1?",
     "keywords": ["matlab", "waveform"]},
    {"q": "What tools are required for Experiment 1?",
     "keywords": ["matlab ide"]},
    {"q": "What output does the discrete waveform program produce?",
     "keywords": ["impulse", "step", "ramp"]},
    {"q": "What are the lab rules about pen drives?",
     "keywords": ["pen drive"]},
    {"q": "What does the institute mission say about teamwork?",
     "keywords": ["teamwork"]},
]

def score(text, keywords):
    text = text.lower()
    hits = sum(1 for k in keywords if k in text)
    return hits / len(keywords)

def main():
    retriever, reranker, _ = load_query_engine()
    results = []

    for t in TESTS:
        start = time.time()
        answer, nodes = chat(t["q"], retriever, reranker, [])
        latency = round(time.time() - start, 1)

        context = " ".join(n.text for n in nodes)
        retrieval = score(context, t["keywords"])
        accuracy = score(answer, t["keywords"])

        results.append({
            "question": t["q"],
            "retrieval_score": retrieval,
            "answer_score": accuracy,
            "latency_s": latency,
            "answer": answer,
        })
        print(f"\nQ: {t['q']}")
        print(f"  retrieval {retrieval:.0%} | answer {accuracy:.0%} | {latency}s")

    n = len(results)
    summary = {
        "avg_retrieval": round(sum(r["retrieval_score"] for r in results) / n, 2),
        "avg_answer": round(sum(r["answer_score"] for r in results) / n, 2),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / n, 1),
    }
    print("\n=== SUMMARY ===")
    print(summary)

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

if __name__ == "__main__":
    main()