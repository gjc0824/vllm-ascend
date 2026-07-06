#include <torch/extension.h>
#include <limits>
#include <optional>
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ATen/Parallel.h>
#include <torch/script.h>

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <omp.h>
#include <assert.h>
#include <torch/torch.h>
#include <pthread.h>

#if defined(__GNUC__) || defined(__clang__)
#define HOT_FUNCTION __attribute__((hot))
#define FORCE_INLINE inline __attribute__((always_inline))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define RESTRICT __restrict__
#else
#define HOT_FUNCTION
#define FORCE_INLINE inline
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define RESTRICT
#endif

constexpr int32_t EPOCH_RESET_THRESHOLD = 1 << 30;


void bindToCore(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
}

void bindToCore(std::vector<int> core_ids) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for(int core_id : core_ids) {
        CPU_SET(core_id, &cpuset);
    }

    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
}

void print_current_thread_affinity() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    // Get affinity of the current calling thread
    pthread_t current_thread = pthread_self();
    int result = pthread_getaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

    if (result != 0) {
        std::cerr << "Error getting affinity mask" << std::endl;
        return;
    }

    // Print out which CPUs are available to this thread
    std::cout << "Thread is allowed to run on CPUs: ";
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &cpuset)) {
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;
}

at::Tensor restore_tensor(uintptr_t ptr_val, const std::vector<int64_t>& shape, torch::ScalarType dtype = torch::kBFloat16) {
    if (ptr_val == 0) return at::Tensor(); // 处理空指针情况
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    return torch::from_blob(reinterpret_cast<void*>(ptr_val), shape, options);
}

FORCE_INLINE int choose_lru_resident_threads(
    const int num_reqs,
    const int workspace_threads,
    const int requested_threads
) {
    if (num_reqs <= 1) {
        return 1;
    }
    int active_threads = num_reqs;
    if (workspace_threads > 0) {
        active_threads = std::min(active_threads, workspace_threads);
    }
    if (requested_threads > 0) {
        active_threads = std::min(active_threads, requested_threads);
    }
    return std::max(active_threads, 1);
}

FORCE_INLINE bool is_valid_lru_resident_token(const int32_t token, const int32_t max_token) {
    return token >= 0 && token < max_token;
}

FORCE_INLINE void reset_lru_resident_row(
    int32_t* RESTRICT slot_to_token_row,
    int32_t* RESTRICT lru_slots_row,
    const int32_t capacity
) {
    std::fill(slot_to_token_row, slot_to_token_row + capacity, -1);
    for (int32_t slot = 0; slot < capacity; ++slot) {
        lru_slots_row[slot] = slot;
    }
}

FORCE_INLINE int32_t next_lru_resident_epoch(
    int32_t* RESTRICT token_mark,
    int32_t* RESTRICT token_pos,
    int32_t* RESTRICT epoch,
    const int32_t max_token
) {
    int32_t base = epoch[0] + 1;
    if (UNLIKELY(base >= EPOCH_RESET_THRESHOLD)) {
        std::fill(token_mark, token_mark + max_token, 0);
        std::fill(token_pos, token_pos + max_token, -1);
        base = 1;
    }
    epoch[0] = base;
    return base;
}

FORCE_INLINE void process_one_lru_resident_row(
    const int row,
    const int32_t topk,
    const int32_t capacity,
    const int32_t max_token,
    const int64_t* RESTRICT req_ids,
    int64_t* RESTRICT last_req_ids,
    const int32_t* RESTRICT topk_indices,
    int32_t* RESTRICT slot_to_token,
    int32_t* RESTRICT lru_slots,
    int32_t* RESTRICT current_slots,
    int32_t* RESTRICT miss_count,
    int32_t* RESTRICT miss_tokens_out,
    int32_t* RESTRICT miss_slots_out,
    int32_t* RESTRICT token_mark,
    int32_t* RESTRICT token_pos,
    int32_t* RESTRICT slot_workspace,
    int32_t* RESTRICT miss_positions,
    int32_t* RESTRICT epoch
) {
    int32_t* RESTRICT slot_to_token_row = slot_to_token + static_cast<int64_t>(row) * capacity;
    int32_t* RESTRICT lru_slots_row = lru_slots + static_cast<int64_t>(row) * capacity;
    int32_t* RESTRICT current_slots_row = current_slots + static_cast<int64_t>(row) * topk;
    int32_t* RESTRICT miss_tokens_row = miss_tokens_out + static_cast<int64_t>(row) * topk;
    int32_t* RESTRICT miss_slots_row = miss_slots_out + static_cast<int64_t>(row) * topk;
    const int32_t* RESTRICT topk_row = topk_indices + static_cast<int64_t>(row) * topk;
    int32_t* RESTRICT hit_slots = slot_workspace;
    int32_t* RESTRICT evictable_slots = slot_workspace + capacity;
    int32_t* RESTRICT assigned_miss_slots = slot_workspace + static_cast<int64_t>(capacity) * 2;

    std::fill(current_slots_row, current_slots_row + topk, -1);
    std::fill(miss_tokens_row, miss_tokens_row + topk, -1);
    std::fill(miss_slots_row, miss_slots_row + topk, -1);
    miss_count[row] = 0;

    const int64_t req_id = req_ids[row];
    if (UNLIKELY(last_req_ids[row] != req_id)) {
        reset_lru_resident_row(slot_to_token_row, lru_slots_row, capacity);
        last_req_ids[row] = req_id;
    }

    const int32_t base = next_lru_resident_epoch(token_mark, token_pos, epoch, max_token);
    for (int32_t pos = 0; pos < topk; ++pos) {
        const int32_t token = topk_row[pos];
        if (LIKELY(is_valid_lru_resident_token(token, max_token)) && token_mark[token] != base) {
            token_mark[token] = base;
            token_pos[token] = pos;
        }
    }

    int32_t hit_count = 0;
    int32_t evictable_count = 0;
    for (int32_t order = 0; order < capacity; ++order) {
        const int32_t slot = lru_slots_row[order];
        if (UNLIKELY(slot < 0 || slot >= capacity)) {
            continue;
        }
        const int32_t token = slot_to_token_row[slot];
        if (LIKELY(is_valid_lru_resident_token(token, max_token)) && token_mark[token] == base) {
            const int32_t pos = token_pos[token];
            current_slots_row[pos] = slot;
            hit_slots[hit_count] = slot;
            ++hit_count;
        } else {
            evictable_slots[evictable_count] = slot;
            ++evictable_count;
        }
    }

    int32_t local_miss_count = 0;
    for (int32_t pos = 0; pos < topk; ++pos) {
        const int32_t token = topk_row[pos];
        if (LIKELY(is_valid_lru_resident_token(token, max_token)) && current_slots_row[pos] < 0) {
            miss_tokens_row[local_miss_count] = token;
            miss_positions[local_miss_count] = pos;
            ++local_miss_count;
        }
    }

    const int32_t assign_count = std::min(local_miss_count, evictable_count);
    for (int32_t miss_idx = 0; miss_idx < assign_count; ++miss_idx) {
        const int32_t slot = evictable_slots[miss_idx];
        const int32_t token = miss_tokens_row[miss_idx];
        const int32_t pos = miss_positions[miss_idx];
        slot_to_token_row[slot] = token;
        current_slots_row[pos] = slot;
        miss_slots_row[miss_idx] = slot;
        assigned_miss_slots[miss_idx] = slot;
        miss_count[row] = miss_idx + 1;
    }

    int32_t write_pos = 0;
    for (int32_t idx = assign_count; idx < evictable_count; ++idx) {
        lru_slots_row[write_pos] = evictable_slots[idx];
        ++write_pos;
    }
    for (int32_t idx = 0; idx < assign_count; ++idx) {
        lru_slots_row[write_pos] = assigned_miss_slots[idx];
        ++write_pos;
    }
    for (int32_t idx = 0; idx < hit_count; ++idx) {
        lru_slots_row[write_pos] = hit_slots[idx];
        ++write_pos;
    }
}

HOT_FUNCTION void lru_resident_compact(
    uintptr_t req_ids_ptr,
    uintptr_t last_req_ids_ptr,
    uintptr_t topk_indices_ptr,
    uintptr_t slot_to_token_ptr,
    uintptr_t lru_slots_ptr,
    uintptr_t current_slots_ptr,
    uintptr_t miss_count_ptr,
    uintptr_t miss_tokens_ptr,
    uintptr_t miss_slots_ptr,
    uintptr_t token_mark_workspace_ptr,
    uintptr_t token_pos_workspace_ptr,
    uintptr_t slot_workspace_ptr,
    uintptr_t miss_position_workspace_ptr,
    uintptr_t epochs_ptr,
    int64_t num_reqs,
    int64_t topk,
    int64_t capacity,
    int64_t max_token,
    int64_t workspace_threads,
    int64_t requested_threads
) {
    if (num_reqs <= 0 || topk <= 0 || capacity <= 0 || max_token <= 0) {
        return;
    }

    auto* RESTRICT req_ids = reinterpret_cast<int64_t*>(req_ids_ptr);
    auto* RESTRICT last_req_ids = reinterpret_cast<int64_t*>(last_req_ids_ptr);
    auto* RESTRICT topk_indices = reinterpret_cast<int32_t*>(topk_indices_ptr);
    auto* RESTRICT slot_to_token = reinterpret_cast<int32_t*>(slot_to_token_ptr);
    auto* RESTRICT lru_slots = reinterpret_cast<int32_t*>(lru_slots_ptr);
    auto* RESTRICT current_slots = reinterpret_cast<int32_t*>(current_slots_ptr);
    auto* RESTRICT miss_count = reinterpret_cast<int32_t*>(miss_count_ptr);
    auto* RESTRICT miss_tokens = reinterpret_cast<int32_t*>(miss_tokens_ptr);
    auto* RESTRICT miss_slots = reinterpret_cast<int32_t*>(miss_slots_ptr);
    auto* RESTRICT token_mark_workspace = reinterpret_cast<int32_t*>(token_mark_workspace_ptr);
    auto* RESTRICT token_pos_workspace = reinterpret_cast<int32_t*>(token_pos_workspace_ptr);
    auto* RESTRICT slot_workspace = reinterpret_cast<int32_t*>(slot_workspace_ptr);
    auto* RESTRICT miss_position_workspace = reinterpret_cast<int32_t*>(miss_position_workspace_ptr);
    auto* RESTRICT epochs = reinterpret_cast<int32_t*>(epochs_ptr);

    const int num_reqs_int = static_cast<int>(num_reqs);
    const int topk_int = static_cast<int>(topk);
    const int capacity_int = static_cast<int>(capacity);
    const int max_token_int = static_cast<int>(max_token);
    const int active_threads = choose_lru_resident_threads(
        num_reqs_int,
        static_cast<int>(workspace_threads),
        static_cast<int>(requested_threads)
    );

    if (active_threads == 1) {
        for (int row = 0; row < num_reqs_int; ++row) {
            process_one_lru_resident_row(
                row,
                topk_int,
                capacity_int,
                max_token_int,
                req_ids,
                last_req_ids,
                topk_indices,
                slot_to_token,
                lru_slots,
                current_slots,
                miss_count,
                miss_tokens,
                miss_slots,
                token_mark_workspace,
                token_pos_workspace,
                slot_workspace,
                miss_position_workspace,
                epochs
            );
        }
        return;
    }

    #pragma omp parallel num_threads(active_threads)
    {
        const int thread_id = omp_get_thread_num();
        int32_t* RESTRICT token_mark = token_mark_workspace + static_cast<int64_t>(thread_id) * max_token_int;
        int32_t* RESTRICT token_pos = token_pos_workspace + static_cast<int64_t>(thread_id) * max_token_int;
        int32_t* RESTRICT slots = slot_workspace + static_cast<int64_t>(thread_id) * capacity_int * 3;
        int32_t* RESTRICT miss_positions = miss_position_workspace + static_cast<int64_t>(thread_id) * topk_int;
        int32_t* RESTRICT epoch = epochs + thread_id;
        for (int row = thread_id; row < num_reqs_int; row += active_threads) {
            process_one_lru_resident_row(
                row,
                topk_int,
                capacity_int,
                max_token_int,
                req_ids,
                last_req_ids,
                topk_indices,
                slot_to_token,
                lru_slots,
                current_slots,
                miss_count,
                miss_tokens,
                miss_slots,
                token_mark,
                token_pos,
                slots,
                miss_positions,
                epoch
            );
        }
    }
}

void
get_kv_topk_kernel(
    const at::Tensor& k_nope_h, // [block_num, block_size, head_kv, nope_dim]
    const at::Tensor& k_pe_h, // [block_num, block_size, head_kv, rope_dim]
    const at::Tensor& topk_indices, // [num_token, 1, topk]
    const at::Tensor& actual_seq_lengths_query, // [num_reqs], == num_token if without spec decode
    const at::Tensor& block_table, // [num_reqs, max_block_num]
    const int& thread_num,
    at::Tensor& k_nope_topk,
    at::Tensor& k_pe_topk
) {
    auto k_nope = k_nope_h.contiguous();
    auto k_pe = k_pe_h.contiguous();

    TORCH_CHECK(topk_indices.size(1) == 1, "topk_indices must have shape [N, 1, S]");
    auto topk_indices_squeezed = topk_indices.squeeze(1); // [N, S]

    int32_t token_num = topk_indices.size(0);  // Batch size
    int32_t pe_dim = k_pe.size(3);    // Embedding dimension of pe
    int32_t num_page = k_nope.size(0);    // Number of pages
    int32_t page_size = k_nope.size(1);   // Page size
    int32_t head_kv = k_nope.size(2);      // Number of heads of key/value
    int32_t value_dim = k_nope.size(3); // head dim of k_nope & value
    int32_t sqen_len = topk_indices_squeezed.size(1);  // Source sequence length (TopK)
    int32_t max_num_pages = block_table.size(1);  // Max number of pages per request
    int32_t num_requests = actual_seq_lengths_query.size(0);

    // auto cpu_pinned_opt = k_nope_h.options()
    //     .device(at::kCPU)         // 限定CPU设备
    //     .pinned_memory(true);     // 开启页锁定内存
    // at::Tensor k_nope_cpu_pinned = at::empty({token_num, sqen_len, head_kv, value_dim}, cpu_pinned_opt);
    // at::Tensor k_pe_cpu_pinned = at::empty({token_num, sqen_len, head_kv, pe_dim}, cpu_pinned_opt);

    bfloat16_t* k_nope_ptr = static_cast<bfloat16_t*>(k_nope.data_ptr());
    bfloat16_t* k_pe_ptr = static_cast<bfloat16_t*>(k_pe.data_ptr());
    int32_t* topk_indices_ptr = static_cast<int32_t*>(topk_indices_squeezed.data_ptr());
    int32_t* actual_seq_lengths_query_ptr = static_cast<int32_t*>(actual_seq_lengths_query.data_ptr());
    int32_t* block_table_ptr = static_cast<int32_t*>(block_table.data_ptr());

    // bfloat16_t* k_nope_pinned_ptr = static_cast<bfloat16_t*>(k_nope_cpu_pinned.data_ptr());
    // bfloat16_t* k_pe_pinned_ptr = static_cast<bfloat16_t*>(k_pe_cpu_pinned.data_ptr());
    bfloat16_t* k_nope_pinned_ptr = static_cast<bfloat16_t*>(k_nope_topk.data_ptr());
    bfloat16_t* k_pe_pinned_ptr = static_cast<bfloat16_t*>(k_pe_topk.data_ptr());

    auto req_ids = at::empty({token_num}, at::kInt);   // at::empty，分配一块指定形状和数据类型的内存空间，创建一维张量，int32类型
    auto page_indices = at::empty({token_num, sqen_len}, at::kInt);   //创建二维张量
    auto pos_in_page = at::empty({token_num, sqen_len}, at::kInt);

    int32_t* req_ids_ptr = req_ids.data_ptr<int32_t>();
    int32_t* page_indices_ptr = page_indices.data_ptr<int32_t>();
    int32_t* pos_in_page_ptr = pos_in_page.data_ptr<int32_t>();

    for (int32_t r = 0; r < num_requests; ++r) {
        int32_t start = (r == 0) ? 0 : actual_seq_lengths_query_ptr[r - 1];
        int32_t end = actual_seq_lengths_query_ptr[r];

        std::fill(req_ids_ptr + start, req_ids_ptr + end, r);
    }

    // multi thread related. Currently parallel in seq_len(topk) dim.
    // int thread_num = 16;
    int size_per_thread = sqen_len / thread_num;
    std::vector<int> thread_size(thread_num, size_per_thread);
    thread_size[thread_num - 1] += sqen_len - size_per_thread * thread_num;
    std::vector<int> thread_start(thread_num);
    int cumsum = 0;
    for(int i = 0; i < thread_num; ++i) {
        thread_start[i] = cumsum;
        cumsum += thread_size[i];
    }
    // std::cout << "thread start: "; for(auto i : thread_start) std::cout << i << " "; std::cout << std::endl;
    // std::cout << "thread size: "; for(auto i : thread_size) std::cout << i << " "; std::cout << std::endl;

    at::set_num_threads(thread_num);
    omp_set_num_threads(thread_num);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int t_start = thread_start[tid];
        int t_end = t_start + thread_size[tid];
        for (int32_t i = 0; i < token_num; ++i) {
            int32_t req_id = req_ids_ptr[i];
            // for (int32_t s = 0; s < sqen_len; ++s) {
            for (int32_t s = t_start; s < t_end; ++s) {
                int32_t global_token_pos = topk_indices_ptr[i * sqen_len + s];

                if (global_token_pos < 0) {
                    continue;
                    // page_indices_ptr[i * sqen_len + s] = 0;
                    // pos_in_page_ptr[i * sqen_len + s] = 0;
                    // for (int32_t h = 0; h < head_kv; ++h) {
                    //     for (int32_t d = 0; d < value_dim; ++d) {
                    //         k_nope_pinned_ptr[i * sqen_len * head_kv * value_dim +
                    //                          s * head_kv * value_dim +
                    //                          h * value_dim + d] = static_cast<bfloat16_t>(0.0f);
                    //     }
                    //     for (int32_t d = 0; d < pe_dim; ++d) {
                    //         k_pe_pinned_ptr[i * sqen_len * head_kv * pe_dim +
                    //                        s * head_kv * pe_dim +
                    //                        h * pe_dim + d] = static_cast<bfloat16_t>(0.0f);
                    //     }
                    // }
                    // continue;
                }

                int32_t rel_page_idx = global_token_pos / page_size;
                int32_t offset_in_page = global_token_pos % page_size;

                int32_t physical_page_id = block_table_ptr[req_id * max_num_pages + rel_page_idx];

                page_indices_ptr[i * sqen_len + s] = physical_page_id;
                pos_in_page_ptr[i * sqen_len + s] = offset_in_page;

                for (int32_t h = 0; h < head_kv; ++h) {
                    for (int32_t d = 0; d < value_dim; ++d) {
                        k_nope_pinned_ptr[i * sqen_len * head_kv * value_dim +
                                    s * head_kv * value_dim +
                                    h * value_dim + d] =
                            k_nope_ptr[physical_page_id * page_size * head_kv * value_dim +
                                    offset_in_page * head_kv * value_dim +
                                    h * value_dim + d];
                    }
                    for (int32_t d = 0; d < pe_dim; ++d) {
                        k_pe_pinned_ptr[i * sqen_len * head_kv * pe_dim +
                                    s * head_kv * pe_dim +
                                    h * pe_dim + d] =
                            k_pe_ptr[physical_page_id * page_size * head_kv * pe_dim +
                                    offset_in_page * head_kv * pe_dim +
                                    h * pe_dim + d];
                    }
                }
            }
        }
    }

    return;
}

void
get_kv_topk(
    uintptr_t k_nope_h_ptr,
    uintptr_t k_pe_h_ptr,
    uintptr_t topk_indices_ptr,
    uintptr_t actual_seq_lengths_query_ptr,
    uintptr_t block_table_ptr,
    uintptr_t k_nope_topk_ptr,
    uintptr_t k_pe_topk_ptr,
    std::vector<int64_t> k_nope_h_shape, // [block_num, block_size, head_kv, nope_dim]
    std::vector<int64_t> k_pe_h_shape, // [block_num, block_size, head_kv, rope_dim]
    std::vector<int64_t> topk_indices_shape, // [num_token, 1, topk]
    std::vector<int64_t> actual_seq_lengths_query_shape, // [num_reqs], == num_token if without spec decode
    std::vector<int64_t> block_table_shape, // [num_reqs, max_block_num]
    std::vector<int64_t> k_nope_topk_shape, // output, [num_reqs, topk, head_kv, nope_dim]
    std::vector<int64_t> k_pe_topk_shape, // output, [num_reqs, topk, head_kv, rope_dim]
    const int& thread_num
) {
    at::Tensor k_nope_h = restore_tensor(k_nope_h_ptr, k_nope_h_shape);
    at::Tensor k_pe_h = restore_tensor(k_pe_h_ptr, k_pe_h_shape);
    at::Tensor topk_indices = restore_tensor(topk_indices_ptr, topk_indices_shape, torch::kInt);
    at::Tensor actual_seq_lengths_query = restore_tensor(actual_seq_lengths_query_ptr, actual_seq_lengths_query_shape, torch::kInt);
    at::Tensor block_table = restore_tensor(block_table_ptr, block_table_shape, torch::kInt);
    at::Tensor k_nope_topk = restore_tensor(k_nope_topk_ptr, k_nope_topk_shape);
    at::Tensor k_pe_topk = restore_tensor(k_pe_topk_ptr, k_pe_topk_shape);
    get_kv_topk_kernel(
        k_nope_h,
        k_pe_h,
        topk_indices,
        actual_seq_lengths_query,
        block_table,
        thread_num,
        k_nope_topk,
        k_pe_topk
    );
    return;
}

int32_t
compute_lru_resident_addrs(
    const at::Tensor& miss_count,
    const at::Tensor& miss_tokens,
    const at::Tensor& miss_slots,
    const at::Tensor& block_table,
    const int32_t block_size,
    const int32_t token_size_bytes_k,
    const int32_t token_size_bytes_v,
    const int64_t gvas_k_base,
    const int64_t gvas_v_base,
    const int64_t addr_k_base,
    const int64_t addr_v_base,
    const int32_t resident_capacity,
    const int32_t max_num_threads,
    at::Tensor& gvas_buffer,
    at::Tensor& addr_buffer,
    at::Tensor& size_buffer,
    at::Tensor& num_tokens_buffer
) {
    const int32_t num_reqs = miss_tokens.size(0);
    const int32_t topk = miss_tokens.size(1);
    const int32_t max_num_tokens_to_load = num_reqs * topk;
    const int32_t max_num_blocks = block_table.size(1);
    const int32_t block_size_bytes_k = block_size * token_size_bytes_k;
    const int32_t block_size_bytes_v = block_size * token_size_bytes_v;
    TORCH_CHECK(miss_count.size(0) >= num_reqs,
        "miss_count size not enough for loading.");
    TORCH_CHECK(miss_slots.size(0) == num_reqs && miss_slots.size(1) == topk,
        "miss_slots shape should match miss_tokens.");
    TORCH_CHECK(gvas_buffer.size(0) >= max_num_tokens_to_load * 2,
        "gvas_buffer size not enough for loading.");
    TORCH_CHECK(addr_buffer.size(0) >= max_num_tokens_to_load * 2,
        "addr_buffer size not enough for loading.");
    TORCH_CHECK(size_buffer.size(0) >= max_num_tokens_to_load * 2,
        "size_buffer size not enough for loading.");
    TORCH_CHECK(miss_count.scalar_type() == at::kInt,
        "miss_count wrong dtype, should be int32.");
    TORCH_CHECK(miss_tokens.scalar_type() == at::kInt,
        "miss_tokens wrong dtype, should be int32.");
    TORCH_CHECK(miss_slots.scalar_type() == at::kInt,
        "miss_slots wrong dtype, should be int32.");
    TORCH_CHECK(block_table.scalar_type() == at::kInt,
        "block_table wrong dtype, should be int32.");
    TORCH_CHECK(gvas_buffer.scalar_type() == at::kLong,
        "gvas_buffer wrong dtype, should be int64.");
    TORCH_CHECK(addr_buffer.scalar_type() == at::kLong,
        "addr_buffer wrong dtype, should be int64.");
    TORCH_CHECK(size_buffer.scalar_type() == at::kInt,
        "size_buffer wrong dtype, should be int32.");

    const int32_t* miss_count_ptr = static_cast<int32_t*>(miss_count.data_ptr());
    const int32_t* miss_tokens_ptr = static_cast<int32_t*>(miss_tokens.data_ptr());
    const int32_t* miss_slots_ptr = static_cast<int32_t*>(miss_slots.data_ptr());
    const int32_t* block_table_ptr = static_cast<int32_t*>(block_table.data_ptr());
    int64_t* gvas_buffer_ptr = static_cast<int64_t*>(gvas_buffer.data_ptr());
    int64_t* addr_buffer_ptr = static_cast<int64_t*>(addr_buffer.data_ptr());
    int32_t* size_buffer_ptr = static_cast<int32_t*>(size_buffer.data_ptr());
    int32_t* num_tokens_buffer_ptr = static_cast<int32_t*>(num_tokens_buffer.data_ptr());

    int n_threads = std::min(num_reqs, max_num_threads);
    n_threads = std::max(n_threads, 1);

    std::vector<int32_t> num_tokens_to_load_req(num_reqs);
    #pragma omp parallel for num_threads(n_threads)
    for (size_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
        int32_t count = miss_count_ptr[req_idx];
        count = std::max(count, 0);
        count = std::min(count, topk);
        num_tokens_to_load_req[req_idx] = count;
    }
    int32_t num_tokens_to_load_sum = std::accumulate(num_tokens_to_load_req.begin(), num_tokens_to_load_req.end(), 0);
    std::vector<int32_t> req_start_locs(num_reqs);
    int32_t cumsum = 0;
    for (size_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
        req_start_locs[req_idx] = cumsum;
        cumsum += num_tokens_to_load_req[req_idx];
    }

    #pragma omp parallel for num_threads(n_threads)
    for (size_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
        int32_t req_start_loc_k = req_start_locs[req_idx];
        int32_t req_start_loc_v = num_tokens_to_load_sum + req_start_loc_k;
        for (int32_t idx = 0; idx < num_tokens_to_load_req[req_idx]; ++idx) {
            const int32_t token = miss_tokens_ptr[req_idx * topk + idx];
            const int32_t slot = miss_slots_ptr[req_idx * topk + idx];
            if (token < 0 || slot < 0 || slot >= resident_capacity) {
                continue;
            }
            const int32_t block_id = token / block_size;
            if (block_id < 0 || block_id >= max_num_blocks) {
                continue;
            }
            const int32_t offset_in_block = token % block_size;
            const int32_t block_indice = block_table_ptr[req_idx * max_num_blocks + block_id];
            const int64_t gvas_k =
                gvas_k_base + block_indice * block_size_bytes_k + offset_in_block * token_size_bytes_k;
            const int64_t gvas_v =
                gvas_v_base + block_indice * block_size_bytes_v + offset_in_block * token_size_bytes_v;
            const int64_t addr_k = addr_k_base + (req_idx * resident_capacity + slot) * token_size_bytes_k;
            const int64_t addr_v = addr_v_base + (req_idx * resident_capacity + slot) * token_size_bytes_v;
            gvas_buffer_ptr[req_start_loc_k + idx] = gvas_k;
            gvas_buffer_ptr[req_start_loc_v + idx] = gvas_v;
            addr_buffer_ptr[req_start_loc_k + idx] = addr_k;
            addr_buffer_ptr[req_start_loc_v + idx] = addr_v;
        }
    }

    std::fill_n(size_buffer_ptr, num_tokens_to_load_sum, token_size_bytes_k);
    std::fill_n(&(size_buffer_ptr[num_tokens_to_load_sum]), num_tokens_to_load_sum, token_size_bytes_v);

    num_tokens_buffer_ptr[0] = num_tokens_to_load_sum * 2;
    return num_tokens_to_load_sum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    namespace py = pybind11;
    m.def("get_kv_topk", &get_kv_topk, "High performance topk combine");
    m.def("lru_resident_compact", &lru_resident_compact,
          "CPU LRU resident compact miss prepare with OpenMP row-level parallelism");
    m.def("compute_lru_resident_addrs", &compute_lru_resident_addrs,
          "Compute sparse H2D metadata for compact LRU resident miss loads");
}
