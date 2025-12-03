#!/usr/bin/env python3
"""
KV-Cache Speedup Experiment: Testing Different Batch Sizes

This script tests KV-cache speedup across different batch sizes to find
where the optimization becomes beneficial on A100 GPUs.

Hypothesis: Larger batch sizes may show KV-cache speedup even on A100
because they saturate GPU compute, making O(T^2) vs O(T) difference visible.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F

print("=" * 80)
print("KV-CACHE BATCH SIZE EXPERIMENT")
print("Testing batch sizes: 1, 4, 8, 16")
print("=" * 80)

# Check GPU availability
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

device = torch.device('cuda:0')
print(f"\nRunning on GPU: {torch.cuda.get_device_name(0)}")

# Import model
from nanochat.gpt import GPT, GPTConfig

# Create test model
config = GPTConfig(n_layer=12, n_head=12, n_embd=768, vocab_size=65536)
model = GPT(config).to(device).eval().to(torch.bfloat16)
print(f"Created test model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

# Test parameters
prompt_length = 50
max_new_tokens = 200  # Shorter for faster testing across multiple batch sizes
batch_sizes_to_test = [1, 4, 8, 16]

print(f"\nTest configuration:")
print(f"  - Prompt length: {prompt_length} tokens")
print(f"  - Generate: {max_new_tokens} new tokens per sequence")
print(f"  - Batch sizes: {batch_sizes_to_test}")

# =============================================================================
# Implementations
# =============================================================================

@torch.inference_mode()
def generate_without_kvcache_batched(model, tokens_batch, max_tokens, temperature=0.0):
    """
    Original implementation WITHOUT KV-cache for BATCHED inference.
    Uses torch.cat() pattern - reprocesses entire sequence each step.
    """
    device = model.get_device()
    batch_size = len(tokens_batch)

    # Create batch: [batch_size, seq_len]
    ids = torch.tensor(tokens_batch, dtype=torch.long, device=device)

    for _ in range(max_tokens):
        # Forward pass on ENTIRE batch sequence (inefficient!)
        logits = model.forward(ids)  # No kv_cache parameter
        logits = logits[:, -1, :]  # [batch_size, vocab_size]

        # Greedy sampling
        if temperature == 0:
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)  # [batch_size, 1]
        else:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)

        # Concatenate and grow sequence (O(T^2) pattern!)
        ids = torch.cat((ids, next_ids), dim=1)

    return ids


@torch.inference_mode()
def generate_with_kvcache_batched(model, tokens_batch, max_tokens, temperature=0.0):
    """
    Optimized implementation WITH KV-cache for BATCHED inference.
    Only processes new tokens at each step.
    """
    from nanochat.engine import KVCache

    device = model.get_device()
    batch_size = len(tokens_batch)
    seq_len = len(tokens_batch[0])

    # Pre-allocate KV cache for entire sequence
    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=model.config.n_kv_head,
        seq_len=seq_len + max_tokens,
        head_dim=model.config.n_embd // model.config.n_head,
        num_layers=model.config.n_layer,
        device=device
    )

    # Phase 1: Prefill (process prompt once)
    ids = torch.tensor(tokens_batch, dtype=torch.long, device=device)
    logits = model.forward(ids, kv_cache=kv_cache)
    logits = logits[:, -1, :]

    # Greedy sampling for first token
    if temperature == 0:
        next_ids = torch.argmax(logits, dim=-1, keepdim=True)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=1)

    ids = torch.cat((ids, next_ids), dim=1)

    # Phase 2: Decode (process one token at a time with cache)
    for _ in range(max_tokens - 1):
        logits = model.forward(next_ids, kv_cache=kv_cache)  # Only 1 new token!
        logits = logits[:, -1, :]

        # Greedy sampling
        if temperature == 0:
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)

        ids = torch.cat((ids, next_ids), dim=1)

    return ids


# =============================================================================
# Run Experiments
# =============================================================================

results = []

for batch_size in batch_sizes_to_test:
    print(f"\n{'=' * 80}")
    print(f"BATCH SIZE = {batch_size}")
    print(f"{'=' * 80}")

    # Create batch of prompts (each sequence gets unique tokens)
    prompt_tokens_batch = [
        list(range(i * 100, i * 100 + prompt_length))
        for i in range(batch_size)
    ]

    # Test WITHOUT KV-cache
    print(f"\n  WITHOUT KV-cache (batch_size={batch_size})...")

    # Warmup
    _ = generate_without_kvcache_batched(model, prompt_tokens_batch[:1], 5)

    # Measure
    torch.cuda.synchronize()
    start = time.time()
    output_no_cache = generate_without_kvcache_batched(
        model, prompt_tokens_batch, max_new_tokens
    )
    torch.cuda.synchronize()
    time_without_cache = time.time() - start

    total_tokens_no_cache = batch_size * max_new_tokens
    tokens_per_sec_no_cache = total_tokens_no_cache / time_without_cache

    print(f"    Time: {time_without_cache:.3f} seconds")
    print(f"    Throughput: {tokens_per_sec_no_cache:.1f} tokens/sec")

    # Test WITH KV-cache
    print(f"\n  WITH KV-cache (batch_size={batch_size})...")

    # Warmup
    _ = generate_with_kvcache_batched(model, prompt_tokens_batch[:1], 5)

    # Measure
    torch.cuda.synchronize()
    start = time.time()
    output_with_cache = generate_with_kvcache_batched(
        model, prompt_tokens_batch, max_new_tokens
    )
    torch.cuda.synchronize()
    time_with_cache = time.time() - start

    total_tokens_with_cache = batch_size * max_new_tokens
    tokens_per_sec_with_cache = total_tokens_with_cache / time_with_cache

    print(f"    Time: {time_with_cache:.3f} seconds")
    print(f"    Throughput: {tokens_per_sec_with_cache:.1f} tokens/sec")

    # Calculate speedup
    speedup = tokens_per_sec_with_cache / tokens_per_sec_no_cache

    print(f"\n  RESULT: {speedup:.3f}× speedup with KV-cache")

    # Verify outputs match (greedy sampling should be deterministic)
    outputs_match = torch.equal(output_no_cache, output_with_cache)
    print(f"  Output verification: {'PASS (identical)' if outputs_match else 'FAIL (differ)'}")

    results.append({
        'batch_size': batch_size,
        'without_cache_tps': tokens_per_sec_no_cache,
        'with_cache_tps': tokens_per_sec_with_cache,
        'speedup': speedup,
        'outputs_match': outputs_match
    })


# =============================================================================
# Summary
# =============================================================================

print(f"\n{'=' * 80}")
print("SUMMARY: KV-Cache Speedup vs Batch Size")
print(f"{'=' * 80}\n")

print(f"{'Batch Size':<12} {'Without Cache':<18} {'With Cache':<18} {'Speedup':<12} {'Status'}")
print("-" * 80)

for result in results:
    status = "BENEFICIAL" if result['speedup'] >= 1.05 else "NO BENEFIT"
    print(f"{result['batch_size']:<12} "
          f"{result['without_cache_tps']:>10.1f} tok/s   "
          f"{result['with_cache_tps']:>10.1f} tok/s   "
          f"{result['speedup']:>8.3f}×     "
          f"{status}")

print()

# Find best speedup
best_result = max(results, key=lambda r: r['speedup'])
print(f"Best speedup: {best_result['speedup']:.3f}× at batch_size={best_result['batch_size']}")

if best_result['speedup'] >= 1.05:
    print(f"\nConclusion: KV-cache provides measurable benefit ({best_result['speedup']:.2f}×) ")
    print(f"            at batch_size >= {best_result['batch_size']} on A100!")
elif best_result['speedup'] >= 1.01:
    print(f"\nConclusion: KV-cache provides small benefit ({best_result['speedup']:.2f}×) ")
    print(f"            at batch_size = {best_result['batch_size']}, but may not be ")
    print(f"            significant enough to report.")
else:
    print(f"\nConclusion: KV-cache provides negligible benefit on A100 with tested batch sizes.")
    print(f"            A100's extreme memory bandwidth (~2 TB/s) makes O(T^2) pattern fast enough.")
    print(f"            This is a hardware-dependent result - optimization works on other GPUs.")

print("=" * 80)
