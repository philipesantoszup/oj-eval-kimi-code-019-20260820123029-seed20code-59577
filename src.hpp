#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  Matrix *K = nullptr;
  Matrix *V = nullptr;

  for (size_t round = 0; round < keys.size(); ++round) {
    auto current_query = rater.GetNextQuery();
    size_t num_keys = round + 1;
    size_t d = 512;

    // Move current query to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Move current key and value to SRAM
    gpu_sim.MoveMatrixToSharedMem(keys[round]);
    gpu_sim.MoveMatrixToSharedMem(values[round]);

    // Build K matrix incrementally
    if (K == nullptr) {
      // First round
      K = matrix_memory_allocator.Allocate("K_0");
      gpu_sim.Copy(keys[0], K, kInSharedMemory);
    } else {
      // Subsequent rounds: concatenate existing K with new key
      Matrix *new_K = matrix_memory_allocator.Allocate("K_" + std::to_string(round));
      gpu_sim.Concat(K, keys[round], new_K, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(K);
      K = new_K;
    }

    // Build V matrix incrementally
    if (V == nullptr) {
      // First round
      V = matrix_memory_allocator.Allocate("V_0");
      gpu_sim.Copy(values[0], V, kInSharedMemory);
    } else {
      // Subsequent rounds: concatenate existing V with new value
      Matrix *new_V = matrix_memory_allocator.Allocate("V_" + std::to_string(round));
      gpu_sim.Concat(V, values[round], new_V, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(V);
      V = new_V;
    }

    // Transpose K to get K^T (d x num_keys)
    Matrix *K_transpose = matrix_memory_allocator.Allocate("K_transpose");
    gpu_sim.Copy(K, K_transpose, kInSharedMemory);
    gpu_sim.Transpose(K_transpose, kInSharedMemory);

    // Compute Q * K^T (num_keys x num_keys)
    Matrix *QKt = matrix_memory_allocator.Allocate("QKt");
    gpu_sim.MatMul(current_query, K_transpose, QKt);

    // Compute softmax for each row of QKt
    std::vector<Matrix *> softmax_rows;
    for (size_t row = 0; row < num_keys; ++row) {
      // Get the row
      Matrix *row_mat = matrix_memory_allocator.Allocate("row_" + std::to_string(round) + "_" + std::to_string(row));
      gpu_sim.GetRow(QKt, row, row_mat, kInSharedMemory);

      // Compute exponent
      Matrix *exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(round) + "_" + std::to_string(row));
      gpu_sim.MatExp(row_mat, exp_row);
      gpu_sim.ReleaseMatrix(row_mat);

      // Compute sum of exp_row
      Matrix *sum_exp = matrix_memory_allocator.Allocate("sum_exp_" + std::to_string(round) + "_" + std::to_string(row));
      gpu_sim.Sum(exp_row, sum_exp);

      // Compute softmax row = exp_row / sum_exp
      Matrix *softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(round) + "_" + std::to_string(row));
      gpu_sim.MatDiv(exp_row, sum_exp, softmax_row);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(sum_exp);

      softmax_rows.push_back(softmax_row);
    }

    // Concatenate softmax rows to form softmax matrix (num_keys x num_keys)
    Matrix *softmax_mat = nullptr;
    if (num_keys == 1) {
      softmax_mat = softmax_rows[0];
    } else {
      softmax_mat = softmax_rows[0];
      for (size_t j = 1; j < num_keys; ++j) {
        Matrix *new_softmax = matrix_memory_allocator.Allocate("softmax_" + std::to_string(round) + "_" + std::to_string(j));
        gpu_sim.Concat(softmax_mat, softmax_rows[j], new_softmax, 0, kInSharedMemory);
        if (j > 1) {
          gpu_sim.ReleaseMatrix(softmax_mat);
        }
        gpu_sim.ReleaseMatrix(softmax_rows[j]);
        softmax_mat = new_softmax;
      }
      // Release the first softmax row
      gpu_sim.ReleaseMatrix(softmax_rows[0]);
    }

    // Compute softmax_mat * V (num_keys x d)
    Matrix *attention = matrix_memory_allocator.Allocate("attention_" + std::to_string(round));
    gpu_sim.MatMul(softmax_mat, V, attention);
    gpu_sim.ReleaseMatrix(softmax_mat);

    // Release intermediate matrices
    gpu_sim.ReleaseMatrix(K_transpose);
    gpu_sim.ReleaseMatrix(QKt);

    // Move the result to HBM
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
