#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  const size_t d = 512;
  
  // Move all keys and values to SRAM at the beginning
  for (size_t j = 0; j < keys.size(); ++j) {
    gpu_sim.MoveMatrixToSharedMem(keys[j]);
    gpu_sim.MoveMatrixToSharedMem(values[j]);
  }

  Matrix *K = nullptr;
  Matrix *V = nullptr;

  for (size_t round = 0; round < keys.size(); ++round) {
    auto current_query = rater.GetNextQuery();
    const size_t num_keys = round + 1;

    // Issue IO for current query early
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Build K incrementally
    if (K == nullptr) {
      K = matrix_memory_allocator.Allocate("K");
      gpu_sim.Copy(keys[0], K, kInSharedMemory);
    } else {
      Matrix *new_K = matrix_memory_allocator.Allocate("K_" + std::to_string(round));
      gpu_sim.Concat(K, keys[round], new_K, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(K);
      K = new_K;
    }

    // Build V incrementally
    if (V == nullptr) {
      V = matrix_memory_allocator.Allocate("V");
      gpu_sim.Copy(values[0], V, kInSharedMemory);
    } else {
      Matrix *new_V = matrix_memory_allocator.Allocate("V_" + std::to_string(round));
      gpu_sim.Concat(V, values[round], new_V, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(V);
      V = new_V;
    }

    // Compute K^T once per round
    Matrix *Kt = matrix_memory_allocator.Allocate("Kt_" + std::to_string(round));
    gpu_sim.Copy(K, Kt, kInSharedMemory);
    gpu_sim.Transpose(Kt, kInSharedMemory);

    std::vector<Matrix *> attention_rows;

    // Process each query row
    for (size_t q_row = 0; q_row < num_keys; ++q_row) {
      // Get query row
      Matrix *q = matrix_memory_allocator.Allocate("q_" + std::to_string(round) + "_" + std::to_string(q_row));
      gpu_sim.GetRow(current_query, q_row, q, kInSharedMemory);

      // Compute q * K^T
      Matrix *qKt = matrix_memory_allocator.Allocate("qKt_" + std::to_string(round) + "_" + std::to_string(q_row));
      gpu_sim.MatMul(q, Kt, qKt);
      gpu_sim.ReleaseMatrix(q);

      // Compute softmax
      Matrix *exp = matrix_memory_allocator.Allocate("exp_" + std::to_string(round) + "_" + std::to_string(q_row));
      gpu_sim.MatExp(qKt, exp);
      gpu_sim.ReleaseMatrix(qKt);

      Matrix *sum_exp = matrix_memory_allocator.Allocate("sum_exp_" + std::to_string(round) + "_" + std::to_string(q_row));
      gpu_sim.Sum(exp, sum_exp);

      Matrix *softmax = matrix_memory_allocator.Allocate("softmax_" + std::to_string(round) + "_" + std::to_string(q_row));
      gpu_sim.MatDiv(exp, sum_exp, softmax);
      gpu_sim.ReleaseMatrix(exp);
      gpu_sim.ReleaseMatrix(sum_exp);

      // Compute attention row
      Matrix *attn_row = matrix_memory_allocator.Allocate("attn_row_" + std::to_string(round) + "_" + std::to_string(q_row));
      gpu_sim.MatMul(softmax, V, attn_row);
      gpu_sim.ReleaseMatrix(softmax);

      attention_rows.push_back(attn_row);
    }

    // Release Kt early
    gpu_sim.ReleaseMatrix(Kt);

    // Concatenate attention rows
    Matrix *attention = attention_rows[0];
    for (size_t j = 1; j < num_keys; ++j) {
      Matrix *new_attn = matrix_memory_allocator.Allocate("attention_" + std::to_string(round));
      gpu_sim.Concat(attention, attention_rows[j], new_attn, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(attention);
      gpu_sim.ReleaseMatrix(attention_rows[j]);
      attention = new_attn;
    }

    // Move to HBM early to overlap IO with any remaining calculations
    gpu_sim.MoveMatrixToGpuHbm(attention);

    // Run the simulator
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Commit the answer
    rater.CommitAnswer(*attention);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
