#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t round = 0; round < keys.size(); ++round) {
    auto current_query = rater.GetNextQuery();
    size_t num_keys = round + 1;
    size_t d = 512;

    // Move all necessary matrices to SRAM
    gpu_sim.MoveMatrixToSharedMem(current_query);
    for (size_t j = 0; j < num_keys; ++j) {
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      gpu_sim.MoveMatrixToSharedMem(values[j]);
    }

    // Concatenate keys to form K matrix (num_keys x d)
    Matrix *K = nullptr;
    if (num_keys == 1) {
      K = matrix_memory_allocator.Allocate("K_single");
      gpu_sim.Copy(keys[0], K, kInSharedMemory);
    } else {
      K = matrix_memory_allocator.Allocate("K_0");
      gpu_sim.Copy(keys[0], K, kInSharedMemory);
      for (size_t j = 1; j < num_keys; ++j) {
        Matrix *new_K = matrix_memory_allocator.Allocate("K_" + std::to_string(j));
        gpu_sim.Concat(K, keys[j], new_K, 0, kInSharedMemory);
        if (j > 1) {
          gpu_sim.ReleaseMatrix(K);
        }
        K = new_K;
      }
    }

    // Concatenate values to form V matrix (num_keys x d)
    Matrix *V = nullptr;
    if (num_keys == 1) {
      V = matrix_memory_allocator.Allocate("V_single");
      gpu_sim.Copy(values[0], V, kInSharedMemory);
    } else {
      V = matrix_memory_allocator.Allocate("V_0");
      gpu_sim.Copy(values[0], V, kInSharedMemory);
      for (size_t j = 1; j < num_keys; ++j) {
        Matrix *new_V = matrix_memory_allocator.Allocate("V_" + std::to_string(j));
        gpu_sim.Concat(V, values[j], new_V, 0, kInSharedMemory);
        if (j > 1) {
          gpu_sim.ReleaseMatrix(V);
        }
        V = new_V;
      }
    }

    // Transpose K to get K^T (d x num_keys)
    Matrix *K_transpose = matrix_memory_allocator.Allocate("K_transpose");
    // Wait, Transpose is in-place. So we need to copy K first then transpose.
    gpu_sim.Copy(K, K_transpose, kInSharedMemory);
    gpu_sim.Transpose(K_transpose, kInSharedMemory);

    // Compute Q * K^T (num_keys x num_keys)
    Matrix *QKt = matrix_memory_allocator.Allocate("QKt");
    gpu_sim.MatMul(current_query, K_transpose, QKt);

    // Compute softmax for each row of QKt
    // We need to process each row individually
    std::vector<Matrix *> softmax_rows;
    for (size_t row = 0; row < num_keys; ++row) {
      // Get the row
      Matrix *row_mat = matrix_memory_allocator.Allocate("row_" + std::to_string(row));
      gpu_sim.GetRow(QKt, row, row_mat, kInSharedMemory);

      // Compute exponent
      Matrix *exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(row));
      gpu_sim.MatExp(row_mat, exp_row);

      // Compute sum of exp_row
      Matrix *sum_exp = matrix_memory_allocator.Allocate("sum_exp_" + std::to_string(row));
      gpu_sim.Sum(exp_row, sum_exp);

      // Compute softmax row = exp_row / sum_exp
      Matrix *softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(row));
      gpu_sim.MatDiv(exp_row, sum_exp, softmax_row);

      softmax_rows.push_back(softmax_row);

      // Release intermediate matrices
      gpu_sim.ReleaseMatrix(row_mat);
      gpu_sim.ReleaseMatrix(exp_row);
      gpu_sim.ReleaseMatrix(sum_exp);
    }

    // Concatenate softmax rows to form softmax matrix (num_keys x num_keys)
    Matrix *softmax_mat = nullptr;
    if (num_keys == 1) {
      softmax_mat = softmax_rows[0];
    } else {
      softmax_mat = softmax_rows[0];
      for (size_t j = 1; j < num_keys; ++j) {
        Matrix *new_softmax = matrix_memory_allocator.Allocate("softmax_" + std::to_string(j));
        gpu_sim.Concat(softmax_mat, softmax_rows[j], new_softmax, 0, kInSharedMemory);
        if (j > 1) {
          gpu_sim.ReleaseMatrix(softmax_mat);
        }
        softmax_mat = new_softmax;
      }
    }

    // Compute softmax_mat * V (num_keys x d)
    Matrix *attention = matrix_memory_allocator.Allocate("attention");
    gpu_sim.MatMul(softmax_mat, V, attention);

    // Move the result to HBM
    gpu_sim.MoveMatrixToGpuHbm(attention);

    // Run the simulator
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Commit the answer
    rater.CommitAnswer(*attention);

    // Note: We don't need to release attention as rater.CommitAnswer releases it
    // Release other intermediate matrices to save memory
    if (num_keys > 1) {
      gpu_sim.ReleaseMatrix(K);
      gpu_sim.ReleaseMatrix(V);
    }
    gpu_sim.ReleaseMatrix(K_transpose);
    gpu_sim.ReleaseMatrix(QKt);
    if (num_keys > 1) {
      gpu_sim.ReleaseMatrix(softmax_mat);
    }
    // Release softmax_rows if num_keys == 1, else they were released during concatenation
    if (num_keys == 1) {
      gpu_sim.ReleaseMatrix(softmax_rows[0]);
    }

    // Move keys and values back to HBM? Actually, no, we need them in future rounds.
    // Wait, no, in future rounds we'll need to move them again anyway.
    // Actually, let's think: in round i, we use keys[0..i]. In round i+1, we use keys[0..i+1].
    // So we could keep keys[0..i] in SRAM for the next round. But for now, let's just
    // implement a simple solution first.
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
