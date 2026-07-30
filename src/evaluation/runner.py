import os
import json
import time


def call_with_retry(fn, max_retries=3):
    """Generic retry wrapper for rate-limited API calls."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 20 * (attempt + 1)
                print(f"    Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise
    print("    Gave up after max retries")
    return None


def run_evaluation(queries, results_path, baseline_fn, text_rag_fn, mm_rag_fn):
    """
    Runs baseline_fn(query), text_rag_fn(query), mm_rag_fn(query) on each query,
    checkpointing results to results_path after every single query so a
    disconnect only costs the current unfinished query, not the whole batch.
    Resumable: re-running with the same queries and results_path skips
    anything already completed.
    """
    if os.path.exists(results_path):
        with open(results_path) as f:
            eval_results = json.load(f)
        print(f"Resuming: {len(eval_results)} queries already completed")
    else:
        eval_results = []

    completed_queries = {r["query"] for r in eval_results}

    for i, query in enumerate(queries):
        if query in completed_queries:
            print(f"[{i+1}/{len(queries)}] Already done, skipping: {query[:60]}")
            continue

        print(f"[{i+1}/{len(queries)}] Running: {query[:60]}")

        try:
            baseline = call_with_retry(lambda: baseline_fn(query))
            text_rag = call_with_retry(lambda: text_rag_fn(query))
            mm_rag = call_with_retry(lambda: mm_rag_fn(query))

            eval_results.append({
                "query": query,
                "baseline": baseline,
                "text_only_rag": text_rag,
                "mm_rag": mm_rag
            })

            with open(results_path, "w") as f:
                json.dump(eval_results, f, indent=2)

            print(f"    Done and saved. ({len(eval_results)}/{len(queries)} total)")
        except Exception as e:
            print(f"    FAILED on this query: {e}")
            print(f"    Skipping to next query -- re-run this cell later to retry just this one")

    print(f"\\nAll done: {len(eval_results)} of {len(queries)} queries completed")
    return eval_results
