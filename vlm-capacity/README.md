# VLM capacity benchmark & infra sourcing report

A measurement harness for **Qwen3-VL-30B-A3B** serving on 2× NVIDIA H200 NVL,
built to answer one question with numbers rather than estimates:

> Can this estate process 24,000 images/day in a 10-hour window, and what
> happens at 10 / 20 / 30 images per second?

It produces a sourcing report with utilization statistics, storage and RAM
figures, capacity verdicts per scenario, compute economics, and trend charts
showing GPU state before, during and after processing.

---

## Three things to fix before you benchmark anything

These came out of reading `gpu_api_server_v6.py` and `llm_proxy_v2.py`. The
first two are blockers; the third is why the answer to "30 images/sec?" is
currently "no" for a reason that has nothing to do with the hardware.

### 1. The proxy could not carry VLM traffic at all

`llm_proxy_v2.InferRequest` has no `images` field, so a VLM request through
`/v1/infer` arrived at the GPU server with its images silently stripped and the
model answered as if blind. Independently, `AVAILABLE_MODELS` listed neither
`qwen3-vl` nor `internvl`, so the pydantic validator rejected the request with
a 422 before any of that mattered.

`proxy/llm_proxy_v3.py` fixes both, plus three related problems: the 120 s
timeout (shorter than a cold 30B VLM load), the synchronous `def infer` (which
put a 40-thread Starlette pool in front of a load test), and the flattening of
the GPU server's 503 into a 502 (which made saturation indistinguishable from
breakage). Text-model requests produce a byte-identical upstream payload to v2.

### 2. `qwen3-vl`'s registry entry contradicts its own documentation

```python
"tensor_parallel_size":   1,
"gpu_memory_utilization": 0.60,   # ~49 GB per GPU (~98 GB total)
```

The comment says ~49 GB per GPU, the file header's VRAM plan says
`TP=2  0.35  ~49 GB / GPU`, and the code says TP=1 at 0.60 — which is ~85 GB on
GPU 0 alone, beside mistral's 21 GB, with **GPU 1 completely unused**. Whichever
you intend, the benchmark measures what is deployed, so settle this first or the
report describes a configuration you did not mean to ship.

### 3. `max_concurrent: 2` is the real capacity ceiling

```python
"max_concurrent":  2,           # asyncio.Semaphore depth
"max_num_seqs":    max_concurrent * 4,   # = 8
```

That semaphore sits **in front of** vLLM, so the continuous batching the engine
exists to do never gets more than 2 sequences to work with. The 10/20/30 img/s
scenarios will fail against this, and they will fail on the semaphore, not on
the H200. Run the benchmark once as-is to establish the baseline, then raise it
and run again — the delta is the most valuable number in the report.

---

## What's here

| Path | Runs on | What it does |
|---|---|---|
| `proxy/llm_proxy_v3.py` | RHEL VM | Patched proxy: images, VLM models, async upstream, metrics passthrough |
| `bench/make_images.py` | anywhere | Generates a synthetic invoice corpus, if you lack real images |
| `bench/mock_gpu_server.py` | anywhere | Fake endpoint for dry-running the harness without GPU time |
| `bench/gpu_monitor.py` | **GPU box** | 1 Hz `nvidia-smi` + host RAM/disk + vLLM `/metrics` sampler |
| `bench/loadgen.py` | RHEL VM | Open-loop arrival-rate load generator |
| `bench/analyze.py` | anywhere | Joins the CSVs, computes statistics, renders charts, writes the report |
| `bench/run_all.sh` | RHEL VM | Orchestrates all of the above over SSH |

---

## Quick start

```bash
pip install -r requirements.txt
```

### Step 1 — deploy the patched proxy

```bash
# on 10.19.71.246
export API_KEYS="falcon:secret-falcon1,...,bench:secret-bench"
export GPU_HOST=10.66.98.137
uvicorn llm_proxy_v3:app --host 0.0.0.0 --port 8071

curl -s localhost:8071/v1/health | jq .          # vision_models should list qwen3-vl
curl -s localhost:8071/v1/gpu-health | jq .vram  # proves the proxy can reach the GPU box
```

### Step 2 — dry-run the harness (no GPU time)

Prove the plumbing works before booking the H200:

```bash
python3 bench/mock_gpu_server.py --port 9099 --max-concurrent 2 &
python3 bench/make_images.py --out ./corpus --count 300
python3 bench/loadgen.py --corpus ./corpus --out ./results-dry \
    --proxy http://127.0.0.1:9099 --steady-duration 30 --burst-duration 10 \
    --idle 5 --cooldown 5
python3 bench/analyze.py --results ./results-dry --out ./report-dry
```

Any report generated this way is stamped **SYNTHETIC** in a banner at the top —
`loadgen` detects the mock's marker and records it in `run_meta.json`, so a
dry-run can never be mistaken for a capacity measurement.

### Step 3 — the real run

```bash
GPU_SSH=root@10.66.98.137 ./bench/run_all.sh
```

Roughly 20 minutes: 60 s idle baseline, warmup (absorbs the lazy model load),
450 s at 0.67/s consuming all 300 images, then 60 s each at 10 / 20 / 30 img/s
with cooldowns, then a 60 s idle baseline. The report lands in `report/REPORT.md`.

---

## Running it manually (no SSH between the hosts)

**On the GPU box**, before anything else:

```bash
python3 gpu_monitor.py --out ./results --interval 1.0
```

**On the VM**, while that runs:

```bash
python3 loadgen.py --corpus ./corpus --out ./results \
    --proxy http://127.0.0.1:8071 --api-key secret-bench --model qwen3-vl
```

Stop the sampler with Ctrl-C when the load generator finishes, copy its three
CSVs next to the VM's `requests.csv`, then:

```bash
python3 analyze.py --results ./results --out ./report \
    --daily-images 24000 --window-hours 10 --clock-offset-s 0
```

> **Check the clocks.** The samplers run on different hosts and `analyze.py`
> joins them on wall-clock time. Run `date +%s` on both; if they differ, pass
> the difference (`VM − GPU`) as `--clock-offset-s`. The analyzer warns when the
> sample window does not cover the load window, but it cannot detect a small
> skew — that just quietly smears each phase's GPU statistics into its
> neighbours.

---

## Knobs that matter

| Flag | Default | Why you'd change it |
|---|---|---|
| `--rates` | `0.67 10 20 30` | The four scenarios. `0.67` is 24,000 ÷ (10 × 3600) |
| `--max-new-tokens` | `768` | Output length dominates decode cost. Set it to your real cap |
| `--arrival` | `poisson` | `fixed` for evenly spaced arrivals; Poisson is more realistic |
| `--image-mode` | `base64` | `path` if the corpus sits on the GPU box — removes upload from the measurement |
| `--max-inflight` | `400` | Client-side safety cap. Rows shed here are excluded from success rates |
| `--sla-p95` | `30` s | Drives the SUSTAINABLE verdict. Set it to your actual SLA |
| `--cost-gpu-hour` | `3.50` | An assumption. Replace with your quote before circulating the report |

---

## How to read the output

**`achieved_rate` is the whole answer.** If you offer 30 images/sec and the node
delivers 4, the shortfall is the finding — not the utilization number sitting
next to it.

**`nvidia-smi` utilization is time-occupancy, not work.** It reports the share
of the sampling window in which at least one kernel was resident, not how much
of the SM array was busy. One decoding sequence at batch size 1 can read 95%
while most of the H200 idles. That is precisely what `max_concurrent: 2`
produces, and it is why the report always prints utilization beside delivered
throughput rather than on its own.

**`saturated` vs `shed_client`.** The first is the GPU server returning 503
after a request waited `QUEUE_TIMEOUT_S` for a slot — a genuine capacity signal.
The second is the load generator refusing to add more load. They are counted
separately and only the first counts against the server.

**Verdicts.** `SUSTAINABLE` needs all three of: ≥99% success, ≥95% of the
offered rate delivered, and p95 within the SLA. `DEGRADED` is ≥80% of offered
rate with one of those broken. `FAILED` is anything less.

---

## Corpus realism

Throughput scales with **vision-token count**, which scales with image
resolution and visual density — not with file size. The synthetic corpus is
1700×2200 invoice pages with real text to OCR, which is a reasonable stand-in
for scanned documents, but it is a stand-in. Two server-side caps apply to
whatever you feed in:

- `MAX_IMAGE_SIDE_PX = 2048` — anything larger is downscaled before the vision tower
- `mm_processor_kwargs.max_pixels = 1280*28*28` — the real cap on vision tokens per image

If your production documents are denser, multi-page, or need finer OCR detail
than `max_pixels` currently allows, re-run with a sample of them. Nothing else
in this harness changes the answer as much as the corpus does.
