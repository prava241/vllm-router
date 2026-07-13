vLLM Inference Router

This is a project I am using to teach myself about vLLM, network/routing concepts, and controller-worker architectures. I am designing a router (controller.py) that takes user generation requests and routes them to different workers (Colab GPUs running vLLM). I will test these routing decisions on two different workload types: uniform and zipf. I will compare them to a baseline of round-robin worker selection.

Architecture:
-- Server: handles user requests, including session management and generate requests. also handles communication with workers.
-- Controller: makes routing decisions, tracks worker state (GPU utilization, cache, heartbeats), tracks worker/queue assignments.
-- Worker: connects to Colab GPU, runs vllm, does the actual generation. sends periodic hearbeats with state updates.

To Dos:
-- Set up worker: will need to use cloudflare tunnel to communicate with worker
-- Identify relevant worker stats: what should the worker send on heartbeat?
-- Routing/queue policy:
    -- should tasks be assigned to worker queues or should workers pick up available tasks from a general queue?
    -- should tasks be routed immediately or should there be some delay (what if a big, important task arrives right after a small task?)
    -- how to weight different factors
-- Modeling the cache vs querying vLLM for cache state