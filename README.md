# Distributed vLLM Scheduler

A distributed LLM serving framework for evaluating request scheduling policies across multiple vLLM workers. The project separates scheduling from model execution, allowing new routing algorithms to be implemented and compared without modifying the inference engine.

This project is also a personal exploration of machine learning systems. The goal is to gain hands-on experience building distributed inference infrastructure, studying scheduling algorithms, and understanding the systems challenges behind serving large language models efficiently.

## Architecture

The system consists of:

* **Controller** – manages worker registration, heartbeats, request routing, and metrics collection.
* **Workers** – independent FastAPI processes running a single `AsyncLLMEngine` on a GPU (e.g., Google Colab). Workers register with the controller, report resource utilization, and execute generation requests.

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
