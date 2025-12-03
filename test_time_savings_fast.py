#!/usr/bin/env python3
"""
Fast test to verify time savings calculation
Compares SAME workload (200 tokens) with and without optimization
"""

import torch
import time
from nanochat.gpt import GPT, GPTConfig

print("=" * 80)
print("TIME SAVINGS VERIFICATION TEST (FAST VERSION)")
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

print(f"\nTest: Generate {max_new_tokens} tokens (same workload for both tests)")
print("=" * 80)

# =============================================================================
# TEST 1: WITHOUT KV-cache
# =============================================================================

print("\nTEST 1: WITHOUT KV-cache (baseline)")
print("-" * 80)

# Generate random prompt
prompt_tokens = torch.randint(0, config.vocab_size, (1, prompt_length), device=device)

# Measure time WITHOUT KV-cache
torch.cuda.synchronize()
start_time = time.time()

generated = 0
current_tokens = prompt_tokens.clone()
for _ in range(max_new_tokens):
    with torch.no_grad():
        logits = model(current_tokens)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_tokens = torch.cat([current_tokens, next_token], dim=1)
        generated += 1

torch.cuda.synchronize()
time_without_kvcache = time.time() - start_time

throughput_without = generated / time_without_kvcache

print(f"Generated: {generated} tokens")
print(f"Time taken: {time_without_kvcache:.2f} seconds")
print(f"Throughput: {throughput_without:.1f} tokens/sec")

# =============================================================================
# TEST 2: WITH KV-cache
# =============================================================================

print("\nTEST 2: WITH KV-cache (optimized)")
print("-" * 80)

# Use same prompt
prompt_list = prompt_tokens[0].tolist()

# Measure time WITH KV-cache
torch.cuda.synchronize()
start_time = time.time()

generated = 0
for _ in model.generate(prompt_list, max_tokens=max_new_tokens):
    generated += 1

torch.cuda.synchronize()
time_with_kvcache = time.time() - start_time

throughput_with = generated / time_with_kvcache

print(f"Generated: {generated} tokens")
print(f"Time taken: {time_with_kvcache:.2f} seconds")
print(f"Throughput: {throughput_with:.1f} tokens/sec")

# =============================================================================
# RESULTS
# =============================================================================

speedup = time_without_kvcache / time_with_kvcache
time_saved_seconds = time_without_kvcache - time_with_kvcache
time_saved_percent = (1 - 1/speedup) * 100

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\nSame workload (200 tokens generation):")
print(f"  WITHOUT KV-cache: {time_without_kvcache:.2f}s ({throughput_without:.1f} tok/s)")
print(f"  WITH KV-cache:    {time_with_kvcache:.2f}s ({throughput_with:.1f} tok/s)")

print(f"\nSpeedup: {speedup:.2f}×")
print(f"Time saved: {time_saved_seconds:.2f} seconds ({time_saved_percent:.1f}%)")

print(f"\nVerification:")
print(f"  Formula: (1 - 1/{speedup:.2f}) × 100% = {time_saved_percent:.1f}%")
print(f"  Check: ({time_without_kvcache:.2f} - {time_with_kvcache:.2f}) / {time_without_kvcache:.2f} × 100% = {time_saved_percent:.1f}%")

# Practical example
example_time = 100  # seconds
optimized_time = example_time / speedup
saved_time = example_time - optimized_time

print(f"\nPractical example:")
print(f"  If baseline takes {example_time} seconds")
print(f"  → Optimized takes {optimized_time:.2f} seconds")
print(f"  → You save {saved_time:.2f} seconds ({time_saved_percent:.1f}%)")
print(f"  → You STILL NEED {optimized_time:.2f} seconds (NOT zero!)")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)

if time_saved_percent >= 80:
    print(f"✅ {time_saved_percent:.1f}% time saved is REAL and LOGICAL")
    print(f"   • Original time: {time_without_kvcache:.2f}s")
    print(f"   • Optimized time: {time_with_kvcache:.2f}s")
    print(f"   • {speedup:.1f}× faster does NOT mean zero time!")
    print(f"   • You complete the work in {100 - time_saved_percent:.1f}% of original time")
else:
    print(f"   {time_saved_percent:.1f}% time saved measured")

print("=" * 80)
