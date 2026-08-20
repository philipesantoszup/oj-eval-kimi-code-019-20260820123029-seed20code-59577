#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  const size_t d = 512;
  
  Matrix *K = nullptr;
  Matrix *V = nullptr;

  for (size_t round = 0; round < keys.size(); ++round) {
    auto current_query = rater.GetNextQuery();
    const size_t num_keys = round + 1;

    // Issue IO instructions early to maximize parallelism
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(keys[round]);
    gpu_sim.MoveMatrixToSharedMem(values[round]);

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

    // Compute K^T
    Matrix *Kt = matrix_memory_allocator.Allocate("Kt_" + std::to_string(round));
    gpu_sim.Copy(K, Kt, kInSharedMemory);
    gpu_sim.Transpose(Kt, kInSharedMemory);

    // Compute Q * K^T
    Matrix *QKt = matrix_memory_allocator.Allocate("QKt_" + std::to_string(round));
    gpu_sim.MatMul(current_query, Kt, QKt);
    gpu_sim.ReleaseMatrix(Kt); // Release early

    // Compute softmax for each row
    std::vector<Matrix *> softmax_rows;
    for (size_t row = 0; row < num_keys; ++row) {
      Matrix *row_mat = matrix_memory_allocator.Allocate("row_" + std::to_string(round) + "_" + std::to_string(row));
      gpu_sim.GetRow(QKt, row, row_mat, kInSharedMemory);

      Matrix *exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(round) + "_" + std::to_string(row));
      gpu_sim.MatExp(row_mat, exp_row);
      gpu_sim.ReleaseMatrix(row_mat);

      Matrix *sum_exp = matrix_memory_allocator.Allocate("sum_exp_" + std::to_string(round) + "_" + std::to_string(row));
      gpu_sim.Sum(exp_row, sum_exp);

      Matrix *softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(round) + "_" + std::to_string(row));
      gpu_sim.MatDiv(exp_row, sum_exp, softmax_row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(sum_exp);

      softmax_rows.push_back(softmax_row);
    }

    gpu_sim.ReleaseMatrix(QKt); // Release early

    // Build softmax matrix
    Matrix *softmax_mat = softmax_rows[0];
    for (size_t j = 1; j < num_keys; ++j) {
      Matrix *new_softmax = matrix_memory_allocator.Allocate("softmax_" + std::to_string(round) + "_" + std::to_string(j));
      gpu_sim.Concat(softmax_mat, softmax_rows[j], new_softmax, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(softmax_mat);
      gpu_sim.ReleaseMatrix(softmax_rows[j]);
      softmax_mat = new_softmax;
    }
    if (num_keys == 1) {
      // softmax_rows[0] is softmax_mat, no need to release yet
    } else {
      // softmax_rows[0] was released in the loop
    }

    // Compute attention = softmax_mat * V
    Matrix *attention = matrix_memory_allocator.Allocate("attention_" + std::to_string(round));
    gpu_sim.MatMul(softmax_mat, V, attention);
    gpu_sim.ReleaseMatrix(softmax_mat);

    // Move to HBM early to overlap IO with calculations
    gpu_sim.MoveMatrixToGpuHbm(attention);

    // Run the simulator
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Commit answer
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
