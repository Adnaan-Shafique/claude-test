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
| `--gpus-per-node` | `2` | The sourcing unit — an H200 NVL pair |
| `--retention-days` | `30` | How long raw images are kept |
| `--image-kb-low` / `--image-kb-high` | `150` / `800` | The bracket around the measured page size |
| `--replica-factor` | `2` | Copies of pipeline data (primary + backup) |
| `--fs-high-water` | `0.80` | Highest filesystem fill you will plan to |
| `--growth-headroom` | `1.5` | Volume growth over the planning horizon |
| `--cpu-ms-per-image` | `60` | Host CPU per image for decode/resize/normalise — **measure this** |
| `--cpu-headroom` | `1.4` | CPU sizing multiplier |

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

## Proving what the second GPU is worth

The registry currently runs `qwen3-vl` at `tensor_parallel_size: 1`,
`gpu_memory_utilization: 0.60` — one card, with the second idle. The obvious
test is "run it on 1 GPU, show it's not enough, conclude we need 2." That test
will almost certainly return a null result, for a reason worth walking through
before you spend H200 time on it.

**The current setup is not VRAM-bound.** On one card:

| | |
|---|---|
| Budget at `0.60` | `0.60 × 141 GB` ≈ **85 GB** |
| Weights (30.5B @ bf16) | ≈ **61 GB** |
| Left for KV cache | ≈ **21 GB** |
| KV per token (48 layers × 4 KV heads × 128 dim × 2 × 2 B) | ≈ **96 kB** |
| KV capacity | ≈ **220,000 tokens** |
| One image request (~1,350 vision + prompt + output) | ≈ **2,500 tokens** |
| **Concurrent sequences the KV cache could hold** | **≈ 85** |

Verify the per-token figure against your `config.json` — it is estimated from
the Qwen3-30B-A3B base. Even if it is off by 2×, the conclusion holds:
`max_concurrent: 2` uses a **low single-digit percentage** of the KV cache that
one GPU already has. Neither TP=1 nor TP=2 is the binding constraint. The
semaphore is. Compare TP=1 against TP=2 without touching it and both land on
roughly the same number.

So run three configurations, in this order:

| Run | `tensor_parallel_size` | `gpu_memory_utilization` | `max_concurrent` | GPUs |
|---|---|---|---|---|
| **A** baseline | 1 | 0.60 | 2 | 1 |
| **B** semaphore raised | 1 | 0.60 | 32 | 1 |
| **C** second GPU added | 2 | 0.35 | 32 | 2 |

`B − A` is what the config change was worth on one GPU. `C − B` is what the
second GPU was worth. **Only the second is a hardware argument** — run A against
C directly and the config change's gain gets attributed to the silicon.

```bash
for cfg in A B C; do
  # edit MODEL_CONFIGS["qwen3-vl"] per the table, restart the GPU server, then:
  GPU_SSH=root@10.66.98.137 RESULTS=./results-$cfg ./bench/run_all.sh
done

python3 bench/compare.py --out ./report-compare \
    --run "A baseline TP=1 c=2:./results-A" \
    --run "B semaphore TP=1 c=32:./results-B" \
    --run "C two GPUs TP=2 c=32:./results-C"
```

Each run records the server's live `/v1/models` output into `run_meta.json`, so
`compare.py` labels runs from what was **actually deployed** rather than from
the directory name. A run whose config could not be read is flagged as not
safely comparable.

**Where TP=2 genuinely helps**, once the batch is large enough to use it:
`2 × (0.35 × 141 − 30.5 − ~3)` ≈ **32 GB of KV** versus 21 GB on one card, and
**double the aggregate memory bandwidth** — and decode is bandwidth-bound. That
is a real mechanism. It just cannot show up while the batch is capped at 2.

If run B already clears 0.67 img/s with headroom — on the arithmetic above, it
should — then the defensible conclusion is *one GPU covers the committed
workload; the second buys burst headroom, model-swap capacity and failover*.
`compare.py` writes that conclusion itself, in whichever direction the data
points.

## Sizing assumptions

The report sizes the estate as **one node of 2 × H200 NVL**, and sizes storage
and RAM three ways — Low / Expected / High — rather than quoting one number.

**Storage.** 24,000 images/day at 30-day retention means 720,000 images resident
at steady state. Expected uses the measured corpus average; Low and High bracket
it (and are widened automatically if the measurement falls outside them, so the
columns stay monotonic). On top of raw images the model counts derived copies,
extracted JSON sized from **measured** output tokens, and audit logs, then
applies the replica factor, an 80% filesystem high-water mark, and a growth
multiplier. The report recommends provisioning against the **High** case: the
delta is negligible against the GPU spend, and a full image volume stops the
pipeline outright. Images and model weights are sized as **separate volumes** —
weights are a fixed ~260 GB read at load time, images are an unbounded
append-and-expire stream, and sharing one volume lets an ingest backlog prevent
a model from loading.

**RAM.** Most terms are fixed (OS, CUDA contexts per GPU, runtime, the transient
peak while weights stream in). The one that scales is the request path: every
in-flight image is simultaneously a base64 string, a decoded RGB bitmap, and a
preprocessed tensor. At today's `max_concurrent: 2` that is invisible; at the
16–32 the report recommends it becomes the largest variable consumer — so
raising concurrency is a RAM decision as well as a throughput one. The Expected
case also holds the active VLM's 61 GB of weights in page cache, which is what
keeps a model swap off the critical path; the High case keeps both VLMs cached
so the `EVICT_GROUPS` swap re-reads from cache rather than disk.

Every one of these is a flag. Change them and re-run — nothing needs editing.

## The summary table

The report ends (section 11) with one table covering all four scenarios: GPU,
VRAM, nodes, concurrency, RAM, CPU, network, storage, power, and the measured
verdict. The **GPU row is fixed** at `2 × H200 NVL` — that is the sourcing unit,
so what varies across scenarios is how many of that unit are needed, never what
it contains.

Per-node rows (RAM, CPU, NIC) come out constant across scenarios, and that is
correct rather than a bug: scaling is horizontal, so every node carries the same
share and needs the same hardware. What scales is the node count and the
fleet-wide rows that follow from it.

Two rows deserve scrutiny:

- **Concurrency per node** comes from Little's Law (`L = λ × W`), with `W` taken
  from the **lowest** offered rate. Service time under saturation is inflated by
  queueing, so using a saturated scenario's latency would size the fleet off its
  own congestion.
- **CPU is the only row with no measurement behind it.** This harness does not
  profile host CPU per image. The 60 ms/image default is reasonable for JPEG
  decode plus resize and normalise at this page size, but time a decode loop
  over your own corpus and re-run with `--cpu-ms-per-image` before quoting it in
  a purchase order. The report says so in place.

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
