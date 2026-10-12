"""Soak test for a vLLM server: sustained concurrent load, detect engine hangs.

Mimics the workload that hung vLLM 0.24 cu12 (README of models/Qwen3.8-27B): ~500-token
prompts, <=64 output tokens, temperature 1, many requests in flight, some with n>1.
A hang = a request exceeding its timeout, or zero completions for STALL_S seconds.
Usage: python soak.py <port> <minutes> <out.jsonl>
"""
from __future__ import annotations

import asyncio, json, os, random, sys, time
from openai import AsyncOpenAI

PORT, MINUTES, OUT = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
CONC = 48
TIMEOUT_S, STALL_S = int(os.environ.get("TIMEOUT_S", 180)), int(os.environ.get("STALL_S", 300))
c = AsyncOpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="x", timeout=TIMEOUT_S, max_retries=0)
M = ""
done = errors = timeouts = 0
last_done_t = time.time()
words = "alpha beta gamma delta function return value parse token buffer index merge".split()


async def one(rng: random.Random) -> None:
    global done, errors, timeouts, last_done_t
    prompt = " ".join(rng.choice(words) for _ in range(400)) + "\nWrite one Python line that uses these words."
    n = 8 if rng.random() < 0.25 else 1
    try:
        r = await c.chat.completions.create(
            model=M, messages=[{"role": "user", "content": prompt}], max_tokens=64, temperature=1.0, n=n,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        assert len(r.choices) == n and all(ch.message.content for ch in r.choices)
        done += 1
        last_done_t = time.time()
    except Exception as e:  # noqa: BLE001
        if "timed out" in str(e).lower() or "timeout" in type(e).__name__.lower():
            timeouts += 1
        else:
            errors += 1
            print("ERR", type(e).__name__, str(e)[:200], flush=True)


async def worker(i: int, stop: float) -> None:
    rng = random.Random(i)
    while time.time() < stop:
        await one(rng)


async def main() -> None:
    global M
    M = (await c.models.list()).data[0].id
    t0 = time.time(); stop = t0 + MINUTES * 60
    tasks = [asyncio.create_task(worker(i, stop)) for i in range(CONC)]
    verdict = "PASS"
    with open(OUT, "a") as f:
        while any(not t.done() for t in tasks):
            await asyncio.sleep(60)
            rec = {"t_min": round((time.time() - t0) / 60, 1), "done": done, "errors": errors, "timeouts": timeouts}
            f.write(json.dumps(rec) + "\n"); f.flush(); print(rec, flush=True)
            if timeouts or time.time() - last_done_t > STALL_S:
                verdict = "HANG"
                break
    for t in tasks:
        t.cancel()
    print(f"SOAK_{verdict} done={done} errors={errors} timeouts={timeouts} minutes={(time.time()-t0)/60:.1f}", flush=True)


asyncio.run(main())
