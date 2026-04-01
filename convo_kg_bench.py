#!/usr/bin/env python3
"""
convo_kg_bench.py — Conversational Knowledge Graph PoC benchmark

All-ONNX + llama-server pipeline for PinePhone Pro (RK3399S, 4GB RAM):
  - Embedder:  snowflake-arctic-embed-xs (ONNX int8, 23MB)   ~23ms
  - Reranker:  ms-marco-MiniLM-L2-v2 (ONNX int8 ARM64, 16MB) ~35ms
  - NER:       distilbert-NER (ONNX int8, 66MB)               ~118ms
  - LLM:       llama-server HTTP (gemma3-270M or qwen3.5-0.8B)
  - VecDB:     qdrant-edge-py (in-process Rust)

Usage:
    uv run python convo_kg_bench.py              # bench mode
    uv run python convo_kg_bench.py interactive   # interactive mode
"""

import json
import os
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
from tokenizers import Tokenizer

from qdrant_edge import (
    Distance, EdgeConfig, EdgeShard, EdgeVectorParams,
    Point, Query, QueryRequest, UpdateOperation,
)

# ─── Config ──────────────────────────────────────────────────────
MODELS_DIR = os.path.expanduser("~/models")
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5-0.8b")

EMBED_DIR = os.path.join(MODELS_DIR, "snowflake-arctic-embed-xs")
RERANK_DIR = os.path.join(MODELS_DIR, "ms-marco-MiniLM-L2-v2")
NER_DIR = os.path.join(MODELS_DIR, "distilbert-NER")

SHARD_DIR = "./convo-kg-shard"
VEC_NAME = "content"
VEC_DIM = 384
RERANK_K = 10
RERANK_TOP = 3

MODE = sys.argv[1] if len(sys.argv) > 1 else "bench"


# ─── Timer ───────────────────────────────────────────────────────
class Timer:
    def __init__(self): self.elapsed = 0.0
    def __enter__(self):
        self._s = time.perf_counter(); return self
    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._s


# ─── Embedder (ONNX) ────────────────────────────────────────────
class ONNXEmbedder:
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_dir):
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_padding()
        self.tok.enable_truncation(max_length=512)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(
            os.path.join(model_dir, "model_quantized.onnx"),
            sess_options=opts, providers=["CPUExecutionProvider"])

    def encode(self, texts, query=False):
        if isinstance(texts, str):
            texts = [texts]
        if query:
            texts = [self.QUERY_PREFIX + t for t in texts]
        encoded = self.tok.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)
        outs = self.sess.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })
        embs = outs[0][:, 0, :]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-12)


# ─── Reranker (ONNX cross-encoder) ──────────────────────────────
class ONNXReranker:
    def __init__(self, model_dir):
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_padding()
        self.tok.enable_truncation(max_length=512)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(
            os.path.join(model_dir, "model_qint8_arm64.onnx"),
            sess_options=opts, providers=["CPUExecutionProvider"])
        self._input_names = [i.name for i in self.sess.get_inputs()]

    def rank(self, query, documents, top_k=3):
        pairs = [(query, doc) for doc in documents]
        encoded = self.tok.encode_batch(pairs)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array(
                [e.type_ids for e in encoded], dtype=np.int64)
        outs = self.sess.run(None, feeds)
        scores = outs[0].flatten()
        idxs = np.argsort(scores)[::-1][:top_k]
        return [{"text": documents[i], "score": float(scores[i])} for i in idxs]


# ─── NER (ONNX distilbert) ──────────────────────────────────────
class ONNXNER:
    def __init__(self, model_dir):
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(
            os.path.join(model_dir, "model_quantized.onnx"),
            sess_options=opts, providers=["CPUExecutionProvider"])
        config = json.load(open(os.path.join(model_dir, "config.json")))
        self.labels = config["id2label"]

    def extract(self, text):
        enc = self.tok.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        logits = self.sess.run(None, {
            "input_ids": ids, "attention_mask": mask})[0][0]
        preds = np.argmax(logits, axis=-1)
        offsets = enc.offsets

        raw = []
        current = None
        for pred, (start, end) in zip(preds, offsets):
            if start == 0 and end == 0:
                continue
            label = self.labels[str(pred)]
            tag = label[2:] if label != "O" else None
            if label.startswith("B-"):
                if current:
                    raw.append(current)
                current = {"type": tag, "start": start, "end": end}
            elif label.startswith("I-") and current and tag == current["type"]:
                current["end"] = end
            else:
                if current:
                    raw.append(current)
                current = None
        if current:
            raw.append(current)

        # merge adjacent same-type spans — widen gap to absorb hyphens,
        # spaces, and subword splits in technical names
        merged = []
        for span in raw:
            gap_ok = False
            if merged and merged[-1]["type"] == span["type"]:
                gap = span["start"] - merged[-1]["end"]
                if 0 <= gap <= 3:
                    gap_text = text[merged[-1]["end"]:span["start"]]
                    if not any(c in gap_text for c in ".?!,;:"):
                        gap_ok = True
            if gap_ok:
                merged[-1]["end"] = span["end"]
            else:
                merged.append(dict(span))

        # expand partial tokens to word boundaries
        WB = set(" \t\n,.;:!?()[]{}\x27\x22")
        for s in merged:
            while s["start"] > 0 and text[s["start"] - 1] not in WB:
                s["start"] -= 1
            while s["end"] < len(text) and text[s["end"]] not in WB:
                s["end"] += 1

        entities = [{"name": text[s["start"]:s["end"]].strip(), "type": s["type"]}
                    for s in merged if s["end"] - s["start"] > 1]

        # deduplicate: drop exact name dupes (keep first type seen)
        # and short fragments that are substrings of longer entities
        seen_names = set()
        deduped = []
        for e in entities:
            key = e["name"].lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            if any(key in o["name"].lower()
                   and len(o["name"]) > len(e["name"])
                   for o in entities):
                continue
            deduped.append(e)
        return deduped


# ─── LLM (via llama-server HTTP) ────────────────────────────────
def llm_generate(prompt, max_tokens=128, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        r = requests.post(LLM_URL, json={
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False,
        }, timeout=60)
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return text
    except Exception as e:
        return f"[LLM error: {e}]"


def llm_stream_bench(prompt, max_tokens=64, system=None, runs=3):
    """Benchmark streaming LLM: TTFT, decode tok/s, and total time."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    results = []
    for _ in range(runs):
        t0 = time.perf_counter()
        t_first = None
        tokens = 0
        text = ""

        r = requests.post(LLM_URL, json={
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True,
        }, stream=True, timeout=60)

        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {}).get("content", "")
                if delta:
                    if t_first is None:
                        t_first = time.perf_counter() - t0
                    tokens += 1
                    text += delta
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

        total = time.perf_counter() - t0
        gen_time = total - (t_first or total)
        gen_toks = max(tokens - 1, 1)
        results.append({
            "ttft_ms": round((t_first or total) * 1000, 1),
            "total_ms": round(total * 1000, 1),
            "tokens": tokens,
            "tok_per_s": round(gen_toks / gen_time, 1) if gen_time > 0 else 0,
            "text": text[:80],
        })

    return results


# ─── KG helpers ──────────────────────────────────────────────────
def extract_kg(ner, user_msg, assistant_msg):
    full_text = f"{user_msg} {assistant_msg}"
    entities = ner.extract(full_text)
    seen = set()
    unique = []
    for e in entities:
        key = e["name"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(e)
    relations = []
    for i, e1 in enumerate(unique):
        for e2 in unique[i + 1:]:
            relations.append({
                "from": e1["name"], "to": e2["name"],
                "rel": "discussed_with",
            })
    return {"entities": unique, "relations": relations}


def upsert_kg(shard, extracted, turn_id, embedder):
    texts, payloads = [], []
    pid = turn_id * 100
    for e in extracted.get("entities", []):
        n, t = e.get("name", ""), e.get("type", "unknown")
        if n:
            txt = f"{n} ({t})"
            texts.append(txt)
            payloads.append({"type": "entity", "name": n,
                             "entity_type": t, "text": txt, "turn": turn_id})
    for r in extracted.get("relations", []):
        fr, to, rl = r.get("from", ""), r.get("to", ""), r.get("rel", "related_to")
        if fr and to:
            txt = f"{fr} {rl.replace('_', ' ')} {to}"
            texts.append(txt)
            payloads.append({"type": "relation", "from": fr, "to": to,
                             "relation": rl, "text": txt, "turn": turn_id})
    if not texts:
        return 0
    vecs = embedder.encode(texts)
    pts = [Point(id=pid + i, vector={VEC_NAME: v.tolist()}, payload=p)
           for i, (v, p) in enumerate(zip(vecs, payloads))]
    shard.update(UpdateOperation.upsert_points(pts))
    return len(pts)


def search_kg(shard, query, embedder, limit=RERANK_K):
    qv = embedder.encode(query, query=True)[0].tolist()
    try:
        return shard.query(QueryRequest(
            query=Query.Nearest(qv, using=VEC_NAME),
            limit=limit, with_vector=False, with_payload=True,
        ))
    except Exception:
        return []


def rag(shard, query, embedder, reranker):
    tm = {}
    t0 = time.perf_counter()
    hits = search_kg(shard, query, embedder)
    t1 = time.perf_counter()
    tm["search_ms"] = round((t1 - t0) * 1000, 1)

    ctx = ""
    if hits:
        docs = [h.payload.get("text", "") for h in hits]
        ranked = reranker.rank(query, docs, top_k=RERANK_TOP)
        t2 = time.perf_counter()
        tm["rerank_ms"] = round((t2 - t1) * 1000, 1)
        ctx = "\n".join(r["text"] for r in ranked)
    else:
        t2 = t1
        tm["rerank_ms"] = 0.0

    if ctx:
        prompt = f"Context:\n{ctx}\n\nAnswer: {query}"
    else:
        prompt = query

    resp = llm_generate(prompt, max_tokens=128,
                        system="Answer concisely. /no_think")
    t3 = time.perf_counter()
    tm["generate_ms"] = round((t3 - t2) * 1000, 1)
    tm["total_ms"] = round((t3 - t0) * 1000, 1)
    return resp, tm, ctx


# ─── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Conversational KG Bench — PinePhone Pro")
    print(f"LLM: {LLM_URL} ({LLM_MODEL})")
    print("=" * 60)

    print("\nLoading models...")

    with Timer() as t:
        embedder = ONNXEmbedder(EMBED_DIR)
    print(f"  Embedder (ONNX int8):       {t.elapsed:.2f}s")

    with Timer() as t:
        reranker = ONNXReranker(RERANK_DIR)
    print(f"  Reranker (ONNX ARM64):      {t.elapsed:.2f}s")

    with Timer() as t:
        ner = ONNXNER(NER_DIR)
    print(f"  NER (distilbert ONNX int8): {t.elapsed:.2f}s")

    with Timer() as t:
        test = llm_generate("Say OK.", max_tokens=8,
                            system="Reply with OK only. /no_think")
    print(f"  LLM warmup:                 {t.elapsed:.2f}s  → {test[:20]}")

    # Init qdrant-edge shard
    if Path(SHARD_DIR).exists():
        shutil.rmtree(SHARD_DIR)
    Path(SHARD_DIR).mkdir(parents=True, exist_ok=True)
    shard = EdgeShard.create(SHARD_DIR, EdgeConfig(
        vectors={VEC_NAME: EdgeVectorParams(
            size=VEC_DIM, distance=Distance.Cosine)}
    ))

    seeds = [
        "Qdrant Edge is an in-process vector search engine for embedded devices.",
        "Snowflake arctic-embed-xs is a 22M parameter embedding model with 384 dimensions.",
        "The RK3399S has dual Cortex-A72 cores at 1.8GHz and quad Cortex-A53 cores.",
        "A cross-encoder reranker scores query-document pairs for relevance.",
    ]
    svecs = embedder.encode(seeds)
    shard.update(UpdateOperation.upsert_points([
        Point(id=9000 + i, vector={VEC_NAME: v.tolist()},
              payload={"type": "seed", "text": d, "turn": -1})
        for i, (d, v) in enumerate(zip(seeds, svecs))
    ]))
    print(f"\nSeeded graph with {len(seeds)} points.")

    if MODE == "bench":
        print("\n" + "=" * 60)
        print("BENCHMARK: 7-turn conversation with KG building")
        print("=" * 60)

        conversation = [
            "What is Qdrant Edge?",
            "How does it compare to a regular Qdrant server?",
            "Can it run on an RK3399S with 4GB RAM?",
            "What embedding model should I use for edge devices?",
            "Tell me about snowflake-arctic-embed-xs.",
            "What did we discuss about the RK3399S?",
            "Summarize what we've talked about so far.",
        ]
        results = {"turns": [], "llm_model": LLM_MODEL}
        ts, tr, tg, te, tu = [], [], [], [], []

        for tid, msg in enumerate(conversation):
            print(f"\n{'─' * 50}")
            print(f"Turn {tid + 1}: {msg}")

            resp, tm, ctx = rag(shard, msg, embedder, reranker)
            print(f"  → {resp[:200]}")
            print(f"  Timings: {tm}")
            ts.append(tm["search_ms"])
            tr.append(tm["rerank_ms"])
            tg.append(tm["generate_ms"])

            t0 = time.perf_counter()
            ext = extract_kg(ner, msg, resp)
            ext_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            n = upsert_kg(shard, ext, tid, embedder)
            ups_ms = (time.perf_counter() - t0) * 1000

            te.append(ext_ms)
            tu.append(ups_ms)
            ent_names = [e["name"] for e in ext.get("entities", [])]
            print(f"  KG: {len(ext.get('entities', []))} entities, "
                  f"{len(ext.get('relations', []))} relations → {n} pts "
                  f"[extract={ext_ms:.0f}ms upsert={ups_ms:.0f}ms]")
            if ent_names:
                print(f"  Entities: {', '.join(ent_names[:8])}")

            results["turns"].append({
                "turn": tid + 1, "query": msg,
                "response": resp[:100],
                "context_used": bool(ctx), "points_added": n,
                "entities": ent_names, **tm,
                "extract_ms": round(ext_ms, 1),
                "upsert_ms": round(ups_ms, 1),
            })

        # Micro-benchmarks
        print(f"\n{'─' * 50}")
        print("Micro-benchmarks (20 runs each)")

        embedder.encode("warmup", query=True)
        elats = []
        for _ in range(20):
            t0 = time.perf_counter()
            embedder.encode("The quick brown fox.", query=True)
            elats.append(time.perf_counter() - t0)

        tdocs = ["Doc " + str(i) for i in range(10)]
        reranker.rank("test", tdocs, top_k=3)
        rlats = []
        for _ in range(20):
            t0 = time.perf_counter()
            reranker.rank("test", tdocs, top_k=3)
            rlats.append(time.perf_counter() - t0)

        ner_text = "Qdrant Edge runs on the RK3399S in Berlin."
        ner.extract(ner_text)
        nlats = []
        for _ in range(20):
            t0 = time.perf_counter()
            ner.extract(ner_text)
            nlats.append(time.perf_counter() - t0)

        results["micro"] = {
            "embed_mean_ms": round(statistics.mean(elats) * 1000, 2),
            "embed_p50_ms": round(sorted(elats)[10] * 1000, 2),
            "rerank_10docs_mean_ms": round(statistics.mean(rlats) * 1000, 2),
            "rerank_10docs_p50_ms": round(sorted(rlats)[10] * 1000, 2),
            "ner_mean_ms": round(statistics.mean(nlats) * 1000, 2),
            "ner_p50_ms": round(sorted(nlats)[10] * 1000, 2),
        }
        results["summary"] = {
            "search_mean_ms": round(statistics.mean(ts), 1),
            "rerank_mean_ms": round(statistics.mean(tr), 1),
            "generate_mean_ms": round(statistics.mean(tg), 1),
            "ner_extract_mean_ms": round(statistics.mean(te), 1),
            "upsert_mean_ms": round(statistics.mean(tu), 1),
        }

        print(f"\n  Embed ONNX:    mean={results['micro']['embed_mean_ms']}ms")
        print(f"  Rerank ONNX:   mean={results['micro']['rerank_10docs_mean_ms']}ms")
        print(f"  NER ONNX:      mean={results['micro']['ner_mean_ms']}ms")

        # Streaming LLM benchmark
        print(f"\n{'─' * 50}")
        print("LLM streaming benchmark (3 runs each)")

        prompts = [
            ("Short", "What is a vector database?"),
            ("Medium", "Explain how embeddings work for semantic search in two sentences."),
            ("With context", "Context:\nQdrant Edge is an in-process vector search engine.\n\nAnswer: What is Qdrant Edge?"),
        ]
        stream_results = {}
        for label, p in prompts:
            runs = llm_stream_bench(p, max_tokens=64,
                                    system="Answer concisely. /no_think", runs=3)
            ttfts = [r["ttft_ms"] for r in runs]
            toks = [r["tok_per_s"] for r in runs]
            totals = [r["total_ms"] for r in runs]
            print(f"  {label:12s}: TTFT={statistics.mean(ttfts):.0f}ms  "
                  f"gen={statistics.mean(toks):.1f} tok/s  "
                  f"total={statistics.mean(totals):.0f}ms")
            stream_results[label.lower().replace(" ", "_")] = {
                "ttft_mean_ms": round(statistics.mean(ttfts), 1),
                "tok_per_s_mean": round(statistics.mean(toks), 1),
                "total_mean_ms": round(statistics.mean(totals), 1),
                "sample": runs[0]["text"],
            }
        results["streaming"] = stream_results

        shard.close()
        print(f"\n{'=' * 60}")
        print("FULL RESULTS")
        print('=' * 60)
        print(json.dumps(results, indent=2))
        outfile = f"convo_kg_results_{LLM_MODEL.replace('.', '_').replace(' ', '_')}.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n→ Saved to {outfile}")

    elif MODE == "interactive":
        print("\n" + "=" * 60)
        print("Interactive mode — type 'quit' to exit, 'graph' for stats")
        print("=" * 60 + "\n")
        tid = 0
        try:
            while True:
                msg = input("You: ").strip()
                if not msg:
                    continue
                if msg.lower() == "quit":
                    break
                if msg.lower() == "graph":
                    print(f"  {shard.info()}\n")
                    continue

                resp, tm, ctx = rag(shard, msg, embedder, reranker)
                print(f"\nAssistant: {resp}")
                print(f"  [{tm['total_ms']:.0f}ms | search={tm['search_ms']:.0f} "
                      f"rerank={tm['rerank_ms']:.0f} gen={tm['generate_ms']:.0f}]")

                t0 = time.perf_counter()
                ext = extract_kg(ner, msg, resp)
                n = upsert_kg(shard, ext, tid, embedder)
                ent_names = [e["name"] for e in ext.get("entities", [])]
                print(f"  [+{n} pts in {(time.perf_counter() - t0) * 1000:.0f}ms"
                      f" | entities: {', '.join(ent_names) if ent_names else 'none'}]\n")
                tid += 1
        except (EOFError, KeyboardInterrupt):
            pass
        shard.close()
        print("\nDone.")


if __name__ == "__main__":
    main()
