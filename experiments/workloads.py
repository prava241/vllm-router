# workloads.py
#
# Prompt generators for experiment workloads:
#   - Uniform: every prompt is fully distinct, no shared prefix.
#   - Zipf: a small pool of shared prefixes (system prompt + context block),
#     picked per-request via a Zipf-skewed distribution, each followed by a
#     unique question.
#
# Both generators return fully materialized list[list[int]] token id lists,
# so tokenization cost never leaks into a driver script's measured timing.

import random

TOKENIZER_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_tokenizer(model_name: str = TOKENIZER_MODEL):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def generate_uniform_prompts(n: int, tokenizer) -> list[list[int]]:
    topics = [
        "the history of the Roman aqueducts",
        "how neural networks learn from data",
        "the plot of a mystery novel set in Venice",
        "the rules of competitive chess openings",
        "the migration patterns of Arctic terns",
        "how to brew a proper cup of pour-over coffee",
        "the economics of supply and demand",
        "the geology of volcanic island formation",
        "the basics of orbital mechanics",
        "the fermentation process in sourdough bread",
    ]

    prompts = []
    for i in range(n):
        topic = random.choice(topics)
        text = f"Write a detailed, unique explanation (request #{i}) about {topic}."
        prompts.append(text)

    return [tokenizer(p)["input_ids"] for p in prompts]


def _build_shared_prefixes(num_prefixes: int) -> list[str]:
    system_prompt = "You are a helpful assistant answering questions accurately and concisely."
    contexts = [
        "Context: the user is a graduate student researching distributed systems.",
        "Context: the user is a hobbyist cook asking about recipes.",
        "Context: the user is preparing for a technical interview.",
        "Context: the user is a novelist looking for historical detail.",
        "Context: the user is debugging a production incident.",
        "Context: the user is a teacher preparing a lesson plan.",
        "Context: the user is planning a trip and wants local tips.",
        "Context: the user is studying for a certification exam.",
        "Context: the user is a hobby astronomer.",
        "Context: the user is learning a new programming language.",
    ]

    prefixes = []
    for i in range(num_prefixes):
        context = contexts[i % len(contexts)]
        prefixes.append(f"{system_prompt}\n{context}\n")

    return prefixes


def generate_zipf_prompts(
    n: int,
    tokenizer,
    num_prefixes: int = 10,
    zipf_param: float = 1.2,
) -> list[list[int]]:
    prefixes = _build_shared_prefixes(num_prefixes)

    weights = [1 / (i + 1) ** zipf_param for i in range(num_prefixes)]
    prefix_indices = random.choices(range(num_prefixes), weights=weights, k=n)

    prompts = []
    for i, prefix_idx in enumerate(prefix_indices):
        question = f"Question #{i}: what should I know before getting started?"
        # Tokenize the combined prefix+question text as ONE string rather than
        # concatenating separately-tokenized ids — tokenizer merges at the seam
        # can shift token boundaries and silently break vLLM's exact-token-id
        # prefix-cache matching.
        text = prefixes[prefix_idx] + question
        prompts.append(text)

    return [tokenizer(p)["input_ids"] for p in prompts]
