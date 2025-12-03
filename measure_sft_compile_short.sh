#!/bin/bash
# Quick SFT benchmark: WITH vs WITHOUT torch.compile

echo "=========================================="
echo "SFT torch.compile Benchmark"
echo "=========================================="

# Test 1: WITHOUT compile (100 steps)
echo ""
echo "TEST 1: Running SFT WITHOUT torch.compile (100 steps)..."
echo ""

# Temporarily disable torch.compile in scripts/chat_sft.py
# (You'll need to comment out line 108-110 manually, or use sed)

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/chat_sft.py \
    --max_steps=100 \
    --out_dir=/tmp/sft_no_compile \
    2>&1 | tee /tmp/sft_no_compile.log

# Extract throughput
NO_COMPILE_TPS=$(grep "tokens/sec" /tmp/sft_no_compile.log | tail -50 | awk '{sum+=$NF; count+=1} END {print sum/count}')

echo ""
echo "=========================================="
echo "Without compile: $NO_COMPILE_TPS tok/s (avg of last 50 steps)"
echo "=========================================="

# Test 2: WITH compile (100 steps)
echo ""
echo "TEST 2: Running SFT WITH torch.compile (100 steps)..."
echo ""

# Re-enable torch.compile in scripts/chat_sft.py

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/chat_sft.py \
    --max_steps=100 \
    --out_dir=/tmp/sft_with_compile \
    2>&1 | tee /tmp/sft_with_compile.log

# Extract throughput
WITH_COMPILE_TPS=$(grep "tokens/sec" /tmp/sft_with_compile.log | tail -50 | awk '{sum+=$NF; count+=1} END {print sum/count}')

echo ""
echo "=========================================="
echo "With compile: $WITH_COMPILE_TPS tok/s (avg of last 50 steps)"
echo "=========================================="

# Calculate speedup
SPEEDUP=$(echo "scale=2; $WITH_COMPILE_TPS / $NO_COMPILE_TPS" | bc)

echo ""
echo "=========================================="
echo "FINAL RESULT"
echo "=========================================="
echo "Without compile: $NO_COMPILE_TPS tok/s"
echo "With compile:    $WITH_COMPILE_TPS tok/s"
echo "Speedup:         ${SPEEDUP}×"
echo "=========================================="
