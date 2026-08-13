import time

from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.sampling_params import SamplingParams

from models import *


class VLLMModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.engine = None

    async def load(self):
        args = AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=1,
            quantization="awq",
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            dtype="half",
            enforce_eager=True,
        )

        print(f"Loading {self.model_name}...")
        self.engine = AsyncLLMEngine.from_engine_args(args)
        print("vLLM engine loaded.")

    async def generate(
        self,
        request: UserRequest,
        dispatch_latency: float
    ) -> GenerateResponse:

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        start = time.perf_counter()

        first_token_time = None
        final_output = None

        stream = self.engine.generate(
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=sampling_params,
            request_id=request.request_id,
        )

        async for output in stream:

            # first time we observe generated output
            if (
                first_token_time is None
                and output.outputs
                and len(output.outputs[0].text) > 0
            ):
                first_token_time = time.perf_counter()

            final_output = output

        end = time.perf_counter()

        #TODO: error handling here

        completion = final_output.outputs[0]

        prompt_tokens = len(request.prompt_token_ids)
        completion_tokens = len(completion.token_ids)

        total_latency = end - start

        if first_token_time is None:
            ttft = total_latency
        else:
            ttft = first_token_time - start

        decode_time = max(end - (first_token_time or end), 1e-6)

        tps = completion_tokens / decode_time

        return GenerateResponse(
            request_id=request.request_id,
            text=completion.text,
            finish_reason=completion.finish_reason,
            metrics=GenerationMetrics(
                dispatch_latency=dispatch_latency,
                timestamp=time.time(),
                ttft=ttft,
                total_latency=total_latency,
                tps=tps,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                num_cached_tokens=getattr(
                    final_output,
                    "num_cached_tokens",
                    0,
                ),
            ),
        )