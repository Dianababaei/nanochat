#!/usr/bin/env python3
"""
Quick torch.compile Benchmark
Measures speedup from torch.compile on representative batches
"""

import torch
import time
from nanochat.gpt import GPT, GPTConfig

print("=" * 80)
print("TORCH.COMPILE BENCHMARK")
print("=" * 80)

# Check GPU
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    exit(1)

device = torch.device('cuda:0')
print(f"\nGPU: {torch.cuda.get_device_name(0)}")

# Create model (same as your training config)
config = GPTConfig(n_layer=12, n_head=12, n_embd=768, vocab_size=65536)
model = GPT(config).to(device).train().to(torch.bfloat16)
print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

# Test parameters (matching SFT training)
batch_size = 93  # Auto-discovered optimal
seq_len = 2047   # Fixed-length padding
num_warmup = 5
num_iterations = 50

print(f"\nConfiguration:")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {seq_len}")
print(f"  Iterations: {num_iterations} (after {num_warmup} warmup)")

# Create dummy data
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

# Optimizer (needed for backward pass)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("\n" + "=" * 80)
print("TEST 1: WITHOUT torch.compile (Baseline)")
print("=" * 80)

# Warmup
for _ in range(num_warmup):
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        targets.view(-1)
    )
    loss.backward()
    optimizer.zero_grad()

# Measure
torch.cuda.synchronize()
times_without_compile = []

for i in range(num_iterations):
    start = time.time()

    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        targets.view(-1)
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.synchronize()
    times_without_compile.append(time.time() - start)

    if (i + 1) % 10 == 0:
        print(f"  Iteration {i+1}/{num_iterations}")

avg_time_without = sum(times_without_compile) / len(times_without_compile)
tokens_per_sec_without = (batch_size * seq_len) / avg_time_without

print(f"\nRESULTS WITHOUT COMPILE:")
print(f"  Avg time per step: {avg_time_without*1000:.1f} ms")
print(f"  Throughput: {tokens_per_sec_without:,.0f} tokens/sec")

print("\n" + "=" * 80)
print("TEST 2: WITH torch.compile (dynamic=False)")
print("=" * 80)

# Compile the model
print("\nCompiling model (this takes ~30-60 seconds)...")
model_compiled = torch.compile(model, dynamic=False)
print("Compilation complete!")

# Warmup (includes first compilation)
print("\nWarming up compiled model...")
for i in range(num_warmup):
    logits = model_compiled(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        targets.view(-1)
    )
    loss.backward()
    optimizer.zero_grad()
    if i == 0:
        print("  First pass complete (kernels compiled)")

# Measure
torch.cuda.synchronize()
times_with_compile = []

for i in range(num_iterations):
    start = time.time()

    logits = model_compiled(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        targets.view(-1)
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.synchronize()
    times_with_compile.append(time.time() - start)

    if (i + 1) % 10 == 0:
        print(f"  Iteration {i+1}/{num_iterations}")

avg_time_with = sum(times_with_compile) / len(times_with_compile)
tokens_per_sec_with = (batch_size * seq_len) / avg_time_with

print(f"\nRESULTS WITH COMPILE:")
print(f"  Avg time per step: {avg_time_with*1000:.1f} ms")
print(f"  Throughput: {tokens_per_sec_with:,.0f} tokens/sec")

# Calculate speedup
speedup = tokens_per_sec_with / tokens_per_sec_without

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print(f"\nWithout compile: {tokens_per_sec_without:>10,.0f} tok/s")
print(f"With compile:    {tokens_per_sec_with:>10,.0f} tok/s")
print(f"\nSpeedup:         {speedup:>10.2f}×")

if speedup >= 1.5:
    print(f"\n✅ EXCELLENT: {speedup:.2f}× speedup confirms torch.compile is working well!")
elif speedup >= 1.3:
    print(f"\n✅ GOOD: {speedup:.2f}× speedup is within expected range")
elif speedup >= 1.1:
    print(f"\n⚠️  MODEST: {speedup:.2f}× speedup is lower than expected")
else:
    print(f"\n❌ WARNING: {speedup:.2f}× speedup suggests compilation may not be working")

print("\n" + "=" * 80)
