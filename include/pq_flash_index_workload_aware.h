// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include "pq_flash_index.h"

namespace diskann {

  // PQFlashIndexWorkloadAware: Reader for disk indexes built with workload-aware layout
  //
  // This class extends PQFlashIndex to handle disk indexes where nodes are reordered
  // for better I/O locality. The key difference is in how disk offsets are computed:
  //
  // Standard:           offset = f(node_id)
  // Workload-aware:     offset = f(logical_to_physical[node_id])
  //
  // All node IDs in the public API remain as logical IDs for compatibility.
  // Neighbor IDs stored on disk are also logical IDs.
  // Only the physical placement of nodes in the disk file is reordered.
  //
  template<typename T>
  class PQFlashIndexWorkloadAware : public PQFlashIndex<T> {
   public:
    DISKANN_DLLEXPORT PQFlashIndexWorkloadAware(
        std::shared_ptr<AlignedFileReader> &fileReader,
        diskann::Metric                     metric = diskann::Metric::L2);
    DISKANN_DLLEXPORT ~PQFlashIndexWorkloadAware();

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load(diskann::MemoryMappedFiles &files,
                               uint32_t num_threads, const char *index_prefix);
#else
    // Load disk index and permutation file
    // Expects: <index_prefix>_disk.index and <index_prefix>_disk.index_disk_perm.bin
    DISKANN_DLLEXPORT int load(uint32_t num_threads, const char *index_prefix);
#endif

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load_from_separate_paths(
        diskann::MemoryMappedFiles &files, uint32_t num_threads,
        const char *index_filepath, const char *pivots_filepath,
        const char *compressed_filepath, const char *permutation_filepath);
#else
    // Load from separate paths including permutation file
    DISKANN_DLLEXPORT int load_from_separate_paths(
        uint32_t num_threads, const char *index_filepath,
        const char *pivots_filepath, const char *compressed_filepath,
        const char *permutation_filepath);
#endif

    // Override search methods to apply permutation
    DISKANN_DLLEXPORT void cached_beam_search(
        const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width,
        const bool use_reorder_data = false, QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT void cached_beam_search(
        const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const _u32 io_limit,
        const bool use_reorder_data = false, QueryStats *stats = nullptr);

    // Block-aware search: utilizes all nodes in each read sector
    void cached_beam_search_block_utilized(
      const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
      float *res_dists, const _u64 beam_width,
      const bool use_reorder_data = false, QueryStats *stats = nullptr);

    void cached_beam_search_block_utilized(
        const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const _u32 io_limit,
        const bool use_reorder_data = false, QueryStats *stats = nullptr);

    // Override cache loading to apply permutation
    DISKANN_DLLEXPORT void load_cache_list(std::vector<uint32_t> &node_list);

   protected:
    // Load permutation from <path>_disk_perm.bin
    // Format: <npts:u64><logical_to_physical[npts]:u32>
    DISKANN_DLLEXPORT void load_permutation(const std::string &permutation_file);

    // Map logical node ID to physical disk position
    // This is the key override: all disk offset computations must use this mapping
    inline _u64 get_physical_idx(_u32 logical_id) const {
      if (logical_id >= logical_to_physical_.size()) {
        return logical_id;  // Fallback for out-of-range
      }
      return logical_to_physical_[logical_id];
    }

    // Map physical position back to logical ID (rarely needed)
    inline _u32 get_logical_id(_u64 physical_idx) const {
      if (physical_idx >= physical_to_logical_.size()) {
        return (_u32)physical_idx;  // Fallback
      }
      return physical_to_logical_[physical_idx];
    }

   private:
    // Permutation mappings
    std::vector<_u32> logical_to_physical_;  // logical_id -> physical_idx
    std::vector<_u32> physical_to_logical_;  // physical_idx -> logical_id
    
    bool permutation_loaded_ = false;
  };

}  // namespace diskann

