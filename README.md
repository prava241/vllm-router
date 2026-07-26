# Distributed vLLM Scheduler

A distributed LLM serving framework for evaluating request scheduling policies across multiple vLLM workers. The project separates scheduling from model execution, allowing new routing algorithms to be implemented and compared without modifying the inference engine.

This project is also a personal exploration of machine learning systems. The goal is to gain hands-on experience building distributed inference infrastructure, studying scheduling algorithms, and understanding the systems challenges behind serving large language models efficiently.

## Architecture

The system consists of:

* **Controller** – manages worker registration, heartbeats, request routing, and metrics collection.
* **Workers** – independent FastAPI processes running a single `AsyncLLMEngine` on a GPU (e.g., Google Colab). Workers register with the controller, report resource utilization, and execute generation requests.

## Running it

This setup assumes the controller runs on your local machine and one or more workers run elsewhere (e.g., a Google Colab GPU runtime), connected over the internet via Cloudflare Tunnel.

### 1. Start the controller locally

```
pip install -r requirements.txt
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

### 2. Expose the controller publicly

Workers need a public URL to reach your local controller. Download [`cloudflared`](https://github.com/cloudflare/cloudflared/releases/latest) and run:

```
./cloudflared tunnel --url http://localhost:8000
```

Copy the printed `https://*.trycloudflare.com` URL — this is your `CONTROLLER_URL`.

### 3. Start a worker

On the worker machine (e.g. `notebook.ipynb` in Colab): clone the repo, install requirements, log into Hugging Face (the default model, `meta-llama/Llama-3.1-8B-Instruct`, is gated), download `cloudflared` into `src/worker/`, then run:

```
cd src/worker
python worker.py --controller-url <CONTROLLER_URL> --host 0.0.0.0 --port 8000
```

The worker starts its own Cloudflare tunnel, registers itself with the controller, and begins sending heartbeats.

### 4. Send requests

```
curl -X POST "http://localhost:8000/sessions?priority=HIGH"
curl -X POST http://localhost:8000/sessions/<session_id>/request \
  -H 'Content-Type: application/json' \
  -d '{"prompt_token_ids": [1, 2, 3], "max_tokens": 64}'
```

Note: `priority` is passed as a query parameter (`?priority=HIGH`), not a JSON body — FastAPI treats a bare `Enum`/scalar argument that way by default.

## Experiments

Scheduling policies will be evaluated under two synthetic workloads:

* **Uniform** request distribution
* **Zipf** request distribution

The primary baseline is **round-robin** worker selection.

Metrics include:

* Time to First Token (TTFT)
* End-to-end latency
* Throughput (tokens/sec)
* GPU utilization
* GPU memory utilization

## Roadmap

### Phase 1

* Random and round-robin scheduling
* Metrics collection
* Prefix caching disabled

### Phase 2

* Metric-based scheduling using features such as queue length, GPU utilization, memory usage, and observed latency
* Compare heuristic scheduling policies against the round-robin baseline

### Phase 3

* Enable prefix caching
* KV-cache-aware and prefix-aware routing
* Evaluate the impact of cache locality on latency and throughput

### Future Work

* Learned scheduling policies
* User-level rate limiting
* Priority queues and admission control
* Heterogeneous GPU support
* Worker load prediction

## Technologies

* Python
* FastAPI
* vLLM
* Hugging Face Transformers
* HTTPX
* NVIDIA NVML
* Cloudflare Tunnel
