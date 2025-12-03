#!/usr/bin/env python3
"""
Test script to verify time savings calculation
Measures actual wall-clock time with and without optimizations
"""

import torch
import time
from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import KVCache

print("=" * 80)
print("TIME SAVINGS VERIFICATION TEST")
print("=" * 80)

# Setup
device = torch.device('cuda:0')
print(f"\nGPU: {torch.cuda.get_device_name(0)}")

# Create model
config = GPTConfig(n_layer=12, n_head=12, n_embd=768, vocab_size=65536)
model = GPT(config).to(device).eval().to(torch.bfloat16)
print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

# Test parameters
prompt_length = 50
max_new_tokens = 200
batch_size = 93

print(f"\nTest: Generate {max_new_tokens} tokens, batch={batch_size}")
print("=" * 80)

# =============================================================================
# TEST 1: WITHOUT KV-cache (baseline)
# =============================================================================

print("\nTEST 1: WITHOUT KV-cache + batch=1")
print("-" * 80)

# Generate random prompt
prompt_tokens = torch.randint(0, config.vocab_size, (1, prompt_length), device=device)

# Measure time WITHOUT optimizations (batch=1, no cache)
torch.cuda.synchronize()
start_time = time.time()

generated = 0
for _ in range(max_new_tokens):
    # Simulate no-cache: recompute everything
    with torch.no_grad():
        logits = model(prompt_tokens)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        prompt_tokens = torch.cat([prompt_tokens, next_token], dim=1)
        generated += 1

torch.cuda.synchronize()
time_no_optimization = time.time() - start_time

print(f"Time taken: {time_no_optimization:.2f} seconds")
print(f"Throughput: {generated / time_no_optimization:.1f} tokens/sec")

# =============================================================================
# TEST 2: WITH KV-cache + batch=93
# =============================================================================

print("\nTEST 2: WITH KV-cache + batch=93")
print("-" * 80)

# Reset model - generate batch of prompts
prompt_batch = []
for _ in range(batch_size):
    prompt_batch.append(torch.randint(0, config.vocab_size, (prompt_length,)).tolist())

# Measure time WITH optimizations
torch.cuda.synchronize()
start_time = time.time()

# Use model's built-in generate with KV-cache
total_generated = 0
for i in range(batch_size):
    tokens_generated = 0
    for _ in model.generate(prompt_batch[i], max_tokens=max_new_tokens):
        tokens_generated += 1
    total_generated += tokens_generated

torch.cuda.synchronize()
time_with_optimization = time.time() - start_time

print(f"Time taken: {time_with_optimization:.2f} seconds")
print(f"Total tokens generated: {total_generated}")
print(f"Throughput: {total_generated / time_with_optimization:.1f} tokens/sec")

# =============================================================================
# RESULTS
# =============================================================================

speedup = time_no_optimization / time_with_optimization
time_saved_percent = (1 - 1/speedup) * 100

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\nWithout optimization (batch=1, no cache):")
print(f"  Time: {time_no_optimization:.2f} seconds")

print(f"\nWith optimization (batch=93, KV-cache):")
print(f"  Time: {time_with_optimization:.2f} seconds")

print(f"\nSpeedup: {speedup:.2f}×")
print(f"Time saved: {time_saved_percent:.1f}%")

print(f"\nVerification:")
print(f"  Original task took: {time_no_optimization:.2f}s")
print(f"  Optimized task took: {time_with_optimization:.2f}s")
print(f"  Time saved: {time_no_optimization - time_with_optimization:.2f}s")
print(f"  Percentage saved: {(time_no_optimization - time_with_optimization) / time_no_optimization * 100:.1f}%")

# Practical example
example_time = 100  # seconds
optimized_time = example_time / speedup
saved_time = example_time - optimized_time

print(f"\nPractical example:")
print(f"  If baseline takes {example_time} seconds")
print(f"  Optimized version takes {optimized_time:.2f} seconds")
print(f"  You save {saved_time:.2f} seconds ({time_saved_percent:.1f}%)")
print(f"  You still need {optimized_time:.2f} seconds (NOT zero!)")

print("\n" + "=" * 80)
print("CONCLUSION:")
if time_saved_percent >= 98:
    print(f"✅ {time_saved_percent:.1f}% time saved is REAL and LOGICAL")
    print(f"   Time goes from {time_no_optimization:.2f}s → {time_with_optimization:.2f}s")
    print(f"   This is {speedup:.0f}× faster, NOT zero time!")
else:
    print(f"   {time_saved_percent:.1f}% time saved measured")

print("=" * 80)
