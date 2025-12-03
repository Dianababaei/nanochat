#!/usr/bin/env python3
"""
Realistic torch.compile Benchmark - Simulates Full Training Loop

Tests torch.compile speedup in a scenario closer to actual SFT training:
- Includes data loading simulation
- Includes gradient accumulation
- Measures wall-clock time over 100 steps
- Compares WITH vs WITHOUT compile
"""

import torch
import time
import random
from nanochat.gpt import GPT, GPTConfig

print("=" * 80)
print("REALISTIC TORCH.COMPILE BENCHMARK")
print("Simulating 100 training steps with full training loop")
print("=" * 80)

# Check GPU
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    exit(1)

device = torch.device('cuda:0')
print(f"\nGPU: {torch.cuda.get_device_name(0)}")

# Configuration matching your actual training
config = GPTConfig(n_layer=12, n_head=12, n_embd=768, vocab_size=65536)
print(f"Model: ~{sum(p.numel() for p in GPT(config).parameters()) / 1e6:.1f}M params")

# Training parameters
batch_size = 32
seq_len = 2047
num_steps = 100
vocab_size = 65536

print(f"\nConfiguration:")
print(f"  Batch size: {batch_size}")
print(f"  Sequence length: {seq_len}")
print(f"  Training steps: {num_steps}")
print(f"  Vocab size: {vocab_size:,}")

# Simulate dataset (random tokens)
print(f"\nGenerating synthetic dataset...")
dataset = []
for _ in range(num_steps * 2):  # Extra data
    input_ids = torch.randint(0, vocab_size, (seq_len,))
    dataset.append(input_ids)
print(f"Dataset ready: {len(dataset)} samples")

def run_training(model, optimizer, num_steps, batch_size, dataset, desc):
    """Run training loop and measure time"""

    model.train()
    total_tokens = 0

    # Warmup (1 step)
    batch_data = random.sample(dataset, batch_size)
    input_ids = torch.stack(batch_data).to(device)
    targets = input_ids.clone()

    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        targets.view(-1)
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Actual measurement
    torch.cuda.synchronize()
    start_time = time.time()

    for step in range(num_steps):
        # Simulate data loading (random sampling like real training)
        batch_data = random.sample(dataset, batch_size)
        input_ids = torch.stack(batch_data).to(device)
        targets = input_ids.clone()

        # Forward pass
        logits = model(input_ids)

        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1)
        )

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        total_tokens += batch_size * seq_len

        # Progress
        if (step + 1) % 20 == 0:
            print(f"  {desc}: Step {step+1}/{num_steps}")

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    throughput = total_tokens / elapsed_time
    avg_step_time = elapsed_time / num_steps * 1000  # ms

    return elapsed_time, throughput, avg_step_time

# =============================================================================
# TEST 1: WITHOUT torch.compile
# =============================================================================

print("\n" + "=" * 80)
print("TEST 1: WITHOUT torch.compile (Baseline)")
print("=" * 80)

model_baseline = GPT(config).to(device).to(torch.bfloat16)
optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=1e-4)

time_no_compile, throughput_no_compile, step_time_no_compile = run_training(
    model_baseline,
    optimizer_baseline,
    num_steps,
    batch_size,
    dataset,
    "WITHOUT compile"
)

print(f"\nRESULTS WITHOUT COMPILE:")
print(f"  Total time: {time_no_compile:.1f} seconds")
print(f"  Avg step time: {step_time_no_compile:.1f} ms")
print(f"  Throughput: {throughput_no_compile:,.0f} tokens/sec")

# Clean up
del model_baseline, optimizer_baseline
torch.cuda.empty_cache()

# =============================================================================
# TEST 2: WITH torch.compile
# =============================================================================

print("\n" + "=" * 80)
print("TEST 2: WITH torch.compile (dynamic=False)")
print("=" * 80)

model_compiled = GPT(config).to(device).to(torch.bfloat16)
print("\nCompiling model (first compilation takes ~30-60 seconds)...")
model_compiled = torch.compile(model_compiled, dynamic=False)
print("Model compiled!")

optimizer_compiled = torch.optim.AdamW(model_compiled.parameters(), lr=1e-4)

time_with_compile, throughput_with_compile, step_time_with_compile = run_training(
    model_compiled,
    optimizer_compiled,
    num_steps,
    batch_size,
    dataset,
    "WITH compile"
)

print(f"\nRESULTS WITH COMPILE:")
print(f"  Total time: {time_with_compile:.1f} seconds")
print(f"  Avg step time: {step_time_with_compile:.1f} ms")
print(f"  Throughput: {throughput_with_compile:,.0f} tokens/sec")

# =============================================================================
# COMPARISON
# =============================================================================

speedup = throughput_with_compile / throughput_no_compile
time_saved = time_no_compile - time_with_compile
time_saved_pct = (time_saved / time_no_compile) * 100

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

print(f"\nWithout compile:")
print(f"  Total time:     {time_no_compile:.1f} seconds")
print(f"  Step time:      {step_time_no_compile:.1f} ms")
print(f"  Throughput:     {throughput_no_compile:,.0f} tokens/sec")

print(f"\nWith compile:")
print(f"  Total time:     {time_with_compile:.1f} seconds")
print(f"  Step time:      {step_time_with_compile:.1f} ms")
print(f"  Throughput:     {throughput_with_compile:,.0f} tokens/sec")

print(f"\nSpeedup:          {speedup:.2f}×")
print(f"Time saved:       {time_saved:.1f} seconds ({time_saved_pct:.1f}% faster)")

if speedup >= 1.5:
    print(f"\n✅ EXCELLENT: {speedup:.2f}× speedup confirms torch.compile works!")
elif speedup >= 1.3:
    print(f"\n✅ GOOD: {speedup:.2f}× speedup is within expected range")
elif speedup >= 1.1:
    print(f"\n⚠️  MODEST: {speedup:.2f}× speedup is lower than expected")
else:
    print(f"\n❌ WARNING: {speedup:.2f}× speedup suggests compilation issues")

print("\n" + "=" * 80)
print("\nNOTE: This benchmark simulates realistic training including:")
print("  - Random data sampling (like DataLoader)")
print("  - Full forward + backward + optimizer steps")
print("  - 100 training steps (not just isolated forward/backward)")
print("  - Measures wall-clock time (real-world scenario)")
print("=" * 80)
