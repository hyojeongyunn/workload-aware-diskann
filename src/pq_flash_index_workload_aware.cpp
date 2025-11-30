// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"

#include "timer.h"
#include "pq_flash_index_workload_aware.h"
#include <fstream>

namespace diskann {

  template<typename T>
  PQFlashIndexWorkloadAware<T>::PQFlashIndexWorkloadAware(
      std::shared_ptr<AlignedFileReader> &fileReader,
      diskann::Metric metric)
      : PQFlashIndex<T>(fileReader, metric) {
    diskann::cout << "Initializing PQFlashIndexWorkloadAware" << std::endl;
  }

  template<typename T>
  PQFlashIndexWorkloadAware<T>::~PQFlashIndexWorkloadAware() {
  }

#ifdef EXEC_ENV_OLS
  template<typename T>
  int PQFlashIndexWorkloadAware<T>::load(diskann::MemoryMappedFiles &files,
                                         uint32_t num_threads,
                                         const char *index_prefix) {
    // TODO: Implement OLS version
    diskann::cerr << "OLS version not yet implemented for workload-aware index"
                  << std::endl;
    return -1;
  }
#else
  template<typename T>
  int PQFlashIndexWorkloadAware<T>::load(uint32_t num_threads,
                                         const char *index_prefix) {
    // Step 1: Load the standard disk index using parent class
    int ret = PQFlashIndex<T>::load(num_threads, index_prefix);
    if (ret != 0) {
      diskann::cerr << "Failed to load base disk index" << std::endl;
      return ret;
    }

    // Step 2: Load the permutation file
    std::string permutation_file = std::string(index_prefix) + "_disk.index_disk_perm.bin";
    try {
      load_permutation(permutation_file);
      diskann::cout << "Loaded workload-aware permutation from " 
                    << permutation_file << std::endl;
    } catch (const std::exception &e) {
      diskann::cerr << "Warning: Could not load permutation file: " << e.what()
                    << std::endl;
      diskann::cerr << "Falling back to identity permutation" << std::endl;
      
      // Fallback: identity permutation
      _u64 npts = this->num_points;
      logical_to_physical_.resize(npts);
      physical_to_logical_.resize(npts);
      for (_u64 i = 0; i < npts; i++) {
        logical_to_physical_[i] = (_u32)i;
        physical_to_logical_[i] = (_u32)i;
      }
      permutation_loaded_ = true;
    }

    return 0;
  }
#endif

#ifdef EXEC_ENV_OLS
  template<typename T>
  int PQFlashIndexWorkloadAware<T>::load_from_separate_paths(
      diskann::MemoryMappedFiles &files, uint32_t num_threads,
      const char *index_filepath, const char *pivots_filepath,
      const char *compressed_filepath, const char *permutation_filepath) {
    // TODO: Implement OLS version
    diskann::cerr << "OLS version not yet implemented for workload-aware index"
                  << std::endl;
    return -1;
  }
#else
  template<typename T>
  int PQFlashIndexWorkloadAware<T>::load_from_separate_paths(
      uint32_t num_threads, const char *index_filepath,
      const char *pivots_filepath, const char *compressed_filepath,
      const char *permutation_filepath) {
    // Step 1: Load base index components
    int ret = PQFlashIndex<T>::load_from_separate_paths(
        num_threads, index_filepath, pivots_filepath, compressed_filepath);
    if (ret != 0) {
      diskann::cerr << "Failed to load base disk index components" << std::endl;
      return ret;
    }

    // Step 2: Load permutation
    try {
      load_permutation(permutation_filepath);
      diskann::cout << "Loaded workload-aware permutation from " 
                    << permutation_filepath << std::endl;
    } catch (const std::exception &e) {
      diskann::cerr << "Warning: Could not load permutation file: " << e.what()
                    << std::endl;
      diskann::cerr << "Falling back to identity permutation" << std::endl;
      
      // Fallback: identity permutation
      _u64 npts = this->num_points;
      logical_to_physical_.resize(npts);
      physical_to_logical_.resize(npts);
      for (_u64 i = 0; i < npts; i++) {
        logical_to_physical_[i] = (_u32)i;
        physical_to_logical_[i] = (_u32)i;
      }
      permutation_loaded_ = true;
    }

    return 0;
  }
#endif

  template<typename T>
  void PQFlashIndexWorkloadAware<T>::load_permutation(const std::string &permutation_file) {
    std::ifstream reader(permutation_file, std::ios::binary);
    if (!reader.is_open()) {
      throw diskann::ANNException("Cannot open permutation file: " + permutation_file, -1);
    }

    // Read format: <npts:u64><logical_to_physical[npts]:u32>
    _u64 npts;
    reader.read((char *)&npts, sizeof(_u64));
    
    if (npts != this->num_points) {
      throw diskann::ANNException(
          "Permutation file npts mismatch: file=" + std::to_string(npts) +
          " index=" + std::to_string(this->num_points), -1);
    }

    // Load logical_to_physical mapping
    logical_to_physical_.resize(npts);
    reader.read((char *)logical_to_physical_.data(), npts * sizeof(_u32));
    reader.close();

    // Build inverse mapping: physical_to_logical
    physical_to_logical_.resize(npts);
    for (_u64 logical_id = 0; logical_id < npts; logical_id++) {
      _u32 physical_idx = logical_to_physical_[logical_id];
      if (physical_idx >= npts) {
        throw diskann::ANNException(
            "Invalid physical_idx in permutation: " + std::to_string(physical_idx), -1);
      }
      physical_to_logical_[physical_idx] = (_u32)logical_id;
    }

    permutation_loaded_ = true;
    diskann::cout << "Permutation loaded: " << npts << " mappings" << std::endl;
  }

  // Override cached_beam_search to apply permutation when reading nodes from disk
  template<typename T>
  void PQFlashIndexWorkloadAware<T>::cached_beam_search(
      const T *query1, const _u64 k_search, const _u64 l_search, _u64 *indices,
      float *distances, const _u64 beam_width, const bool use_reorder_data,
      QueryStats *stats) {
    cached_beam_search(query1, k_search, l_search, indices, distances,
                       beam_width, std::numeric_limits<_u32>::max(),
                       use_reorder_data, stats);
  }

  // Main search method with permutation applied for disk reads
  template<typename T>
  void PQFlashIndexWorkloadAware<T>::cached_beam_search(
      const T *query1, const _u64 k_search, const _u64 l_search, _u64 *indices,
      float *distances, const _u64 beam_width, const _u32 io_limit,
      const bool use_reorder_data, QueryStats *stats) {
    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);

    ScratchStoreManager<SSDThreadData<T>> manager(this->thread_data);
    auto                                  data = manager.scratch_space();
    IOContext                            &ctx = data->ctx;
    auto                                  query_scratch = &(data->scratch);
    auto pq_query_scratch = query_scratch->_pq_scratch;

    // reset query scratch
    query_scratch->reset();

    // copy query to thread specific aligned and allocated memory
    float  query_norm = 0;
    T     *aligned_query_T = query_scratch->aligned_query_T;
    float *query_float = pq_query_scratch->aligned_query_float;
    float *query_rotated = pq_query_scratch->rotated_query;

    // if inner product, normalize the query
    if (this->metric == diskann::Metric::INNER_PRODUCT) {
      for (size_t i = 0; i < this->data_dim - 1; i++) {
        aligned_query_T[i] = query1[i];
        query_norm += query1[i] * query1[i];
      }
      aligned_query_T[this->data_dim - 1] = 0;

      query_norm = std::sqrt(query_norm);

      for (size_t i = 0; i < this->data_dim - 1; i++) {
        aligned_query_T[i] /= query_norm;
      }
      pq_query_scratch->set(this->data_dim, aligned_query_T);
    } else {
      for (size_t i = 0; i < this->data_dim; i++) {
        aligned_query_T[i] = query1[i];
      }
      pq_query_scratch->set(this->data_dim, aligned_query_T);
    }

    // pointers to buffers for data
    T    *data_buf = query_scratch->coord_scratch;
    _u64 &data_buf_idx = query_scratch->coord_idx;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    // query <-> PQ chunk centers distances
    this->pq_table.preprocess_query(query_rotated);
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
    this->pq_table.populate_chunk_distances(query_rotated, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
    _u8   *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
                                                            const _u64 n_ids,
                                                            float *dists_out) {
      diskann::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                                pq_coord_scratch);
      diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                              dists_out);
    };
    Timer query_timer, io_timer, cpu_timer;

    tsl::robin_set<_u64>  &visited = query_scratch->visited;
    NeighborPriorityQueue &retset = query_scratch->retset;
    retset.reserve(l_search);
    std::vector<Neighbor> &full_retset = query_scratch->full_retset;

    _u32  best_medoid = 0;
    float best_dist = (std::numeric_limits<float>::max)();
    for (_u64 cur_m = 0; cur_m < this->num_medoids; cur_m++) {
      float cur_expanded_dist = this->dist_cmp_float->compare(
          query_float, this->centroid_data + this->aligned_dim * cur_m,
          (unsigned) this->aligned_dim);
      if (cur_expanded_dist < best_dist) {
        best_medoid = this->medoids[cur_m];
        best_dist = cur_expanded_dist;
      }
    }

    compute_dists(&best_medoid, 1, dist_scratch);
    retset.insert(Neighbor(best_medoid, dist_scratch[0]));
    visited.insert(best_medoid);

    unsigned cmps = 0;
    unsigned hops = 0;
    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
        cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);

    while (retset.has_unexpanded_node() && num_ios < io_limit) {
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      cached_nhoods.clear();
      sector_scratch_idx = 0;
      // find new beam
      _u32 num_seen = 0;
      while (retset.has_unexpanded_node() && frontier.size() < beam_width &&
             num_seen < beam_width) {
        auto nbr = retset.closest_unexpanded();
        num_seen++;
        auto iter = this->nhood_cache.find(nbr.id);
        if (iter != this->nhood_cache.end()) {
          cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));
          if (stats != nullptr) {
            stats->n_cache_hits++;
          }
        } else {
          frontier.push_back(nbr.id);
        }
        if (this->count_visited_nodes) {
          reinterpret_cast<std::atomic<_u32> &>(
              this->node_visit_counter[nbr.id].second)
              .fetch_add(1);
        }
      }

      // read nhoods of frontier ids
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;
        
        // Track unique sectors for accurate I/O counting
        std::unordered_set<_u64> frontier_sectors;
        
        for (_u64 i = 0; i < frontier.size(); i++) {
          auto id = frontier[i];  // logical ID
          std::pair<_u32, char *> fnhood;
          fnhood.first = id;
          fnhood.second = sector_scratch + sector_scratch_idx * SECTOR_LEN;
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          
          // WORKLOAD-AWARE: Apply permutation for disk read
          _u64 physical_id = get_physical_idx(id);
          _u64 sector_no = physical_id / this->nnodes_per_sector + 1;
          frontier_read_reqs.emplace_back(
              sector_no * SECTOR_LEN, 
              SECTOR_LEN,
              fnhood.second);
          
          frontier_sectors.insert(sector_no);
        }
        
        // Count unique sectors, not nodes
        if (stats != nullptr) {
          stats->n_4k += frontier_sectors.size();
          stats->n_ios += frontier_sectors.size();
        }
        num_ios += frontier_sectors.size();
        
        io_timer.reset();
#ifdef USE_BING_INFRA
        this->reader->read(frontier_read_reqs, ctx, true);
#else
        this->reader->read(frontier_read_reqs, ctx);
#endif
        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
        }
      }

      // process cached nhoods
      for (auto &cached_nhood : cached_nhoods) {
        auto  global_cache_iter = this->coord_cache.find(cached_nhood.first);
        T    *node_fp_coords_copy = global_cache_iter->second;
        float cur_expanded_dist;
        if (!this->use_disk_index_pq) {
          cur_expanded_dist = this->dist_cmp->compare(
              aligned_query_T, node_fp_coords_copy, (unsigned) this->aligned_dim);
        } else {
          if (this->metric == diskann::Metric::INNER_PRODUCT)
            cur_expanded_dist = this->disk_pq_table.inner_product(
                query_float, (_u8 *) node_fp_coords_copy);
          else
            cur_expanded_dist =
                this->disk_pq_table.l2_distance(
                    query_float, (_u8 *) node_fp_coords_copy);
        }
        full_retset.push_back(
            Neighbor((unsigned) cached_nhood.first, cur_expanded_dist));

        _u64      nnbrs = cached_nhood.second.first;
        unsigned *node_nbrs = cached_nhood.second.second;

        // compute node_nbrs <-> query dists in PQ space
        cpu_timer.reset();
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += (double) nnbrs;
          stats->cpu_us += (double) cpu_timer.elapsed();
        }

        // process prefetched nhood
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];  // logical ID from disk
          if (visited.insert(id).second) {
            cmps++;
            float    dist = dist_scratch[m];
            Neighbor nn(id, dist);
            retset.insert(nn);
          }
        }
      }
#ifdef USE_BING_INFRA
      int  completedIndex = -1;
      long requestCount = static_cast<long>(frontier_read_reqs.size());
      while (requestCount > 0 &&
             getNextCompletedRequest(ctx, requestCount, completedIndex)) {
        assert(completedIndex >= 0);
        auto &frontier_nhood = frontier_nhoods[completedIndex];
        (*ctx.m_pRequestsStatus)[completedIndex] = IOContext::PROCESS_COMPLETE;
#else
      for (auto &frontier_nhood : frontier_nhoods) {
#endif
        // WORKLOAD-AWARE: Apply permutation for offset calculation
        _u32 logical_id = frontier_nhood.first;
        _u64 physical_id = get_physical_idx(logical_id);
        
        char *node_disk_buf =
            (char *)frontier_nhood.second + 
            (physical_id % this->nnodes_per_sector) * this->max_node_len;
        
        unsigned *node_buf = (unsigned *)((char *)node_disk_buf + this->disk_bytes_per_point);
        _u64      nnbrs = (_u64) (*node_buf);
        T        *node_fp_coords = (T *)node_disk_buf;
        
        if (data_buf_idx == MAX_N_CMPS)
          data_buf_idx = 0;

        T *node_fp_coords_copy = data_buf + (data_buf_idx * this->aligned_dim);
        data_buf_idx++;
        memcpy(node_fp_coords_copy, node_fp_coords, this->disk_bytes_per_point);
        float cur_expanded_dist;
        if (!this->use_disk_index_pq) {
          cur_expanded_dist = this->dist_cmp->compare(
              aligned_query_T, node_fp_coords_copy, (unsigned) this->aligned_dim);
        } else {
          if (this->metric == diskann::Metric::INNER_PRODUCT)
            cur_expanded_dist = this->disk_pq_table.inner_product(
                query_float, (_u8 *) node_fp_coords_copy);
          else
            cur_expanded_dist = this->disk_pq_table.l2_distance(
                query_float, (_u8 *) node_fp_coords_copy);
        }
        full_retset.push_back(
            Neighbor(frontier_nhood.first, cur_expanded_dist));
        unsigned *node_nbrs = (node_buf + 1);
        
        // compute node_nbrs <-> query dist in PQ space
        cpu_timer.reset();
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += (double) nnbrs;
          stats->cpu_us += (double) cpu_timer.elapsed();
        }

        cpu_timer.reset();
        // process prefetch-ed nhood
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];  // logical ID from disk
          if (visited.insert(id).second) {
            cmps++;
            float dist = dist_scratch[m];
            if (stats != nullptr) {
              stats->n_cmps++;
            }

            Neighbor nn(id, dist);
            retset.insert(nn);
          }
        }

        if (stats != nullptr) {
          stats->cpu_us += (double) cpu_timer.elapsed();
        }
      }

      hops++;
    }

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end());

    if (use_reorder_data) {
      if (!(this->reorder_data_exists)) {
        throw ANNException(
            "Requested use of reordering data which does not exist in index file",
            -1, __FUNCSIG__, __FILE__, __LINE__);
      }

      std::vector<AlignedRead> vec_read_reqs;

      if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
        full_retset.erase(
            full_retset.begin() + k_search * FULL_PRECISION_REORDER_MULTIPLIER,
            full_retset.end());

      // Track unique sectors for accurate I/O counting
      std::unordered_set<_u64> reorder_sectors;
      
      for (size_t i = 0; i < full_retset.size(); ++i) {
        // Reorder data uses logical IDs (no permutation)
        auto id = full_retset[i].id;
        _u64 sector = ((_u64) id) / this->nvecs_per_sector + this->reorder_data_start_sector;
        vec_read_reqs.emplace_back(
            sector * SECTOR_LEN,
            SECTOR_LEN, sector_scratch + i * SECTOR_LEN);

        reorder_sectors.insert(sector);
      }

      // Count unique sectors, not vectors
      if (stats != nullptr) {
        stats->n_4k += reorder_sectors.size();
        stats->n_ios += reorder_sectors.size();
      }

      io_timer.reset();
#ifdef USE_BING_INFRA
      this->reader->read(vec_read_reqs, ctx, false);
#else
      this->reader->read(vec_read_reqs, ctx);
#endif
      if (stats != nullptr) {
        stats->io_us += io_timer.elapsed();
      }

      for (size_t i = 0; i < full_retset.size(); ++i) {
        auto id = full_retset[i].id;
        auto location =
            (sector_scratch + i * SECTOR_LEN) + 
            ((((_u64) id) % this->nvecs_per_sector) * this->data_dim * sizeof(float));
        full_retset[i].distance =
            this->dist_cmp->compare(aligned_query_T, (T *) location, this->data_dim);
      }

      std::sort(full_retset.begin(), full_retset.end());
    }

    // copy k_search values
    for (_u64 i = 0; i < k_search; i++) {
      indices[i] = full_retset[i].id;
      if (distances != nullptr) {
        distances[i] = full_retset[i].distance;
        if (this->metric == diskann::Metric::INNER_PRODUCT) {
          distances[i] = (-distances[i]);
          if (this->max_base_norm != 0)
            distances[i] *= (this->max_base_norm * query_norm);
        }
      }
    }

#ifdef USE_BING_INFRA
    ctx.m_completeCount = 0;
#endif

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }


  // template<typename T>
  // void PQFlashIndexWorkloadAware<T>::cached_beam_search_block_utilized(
  //     const T *query1, const _u64 k_search, const _u64 l_search, _u64 *indices,
  //     float *distances, const _u64 beam_width,
  //     const bool use_reorder_data, QueryStats *stats) {
  //   cached_beam_search_block_utilized(
  //       query1, k_search, l_search, indices, distances,
  //       beam_width, std::numeric_limits<_u32>::max(),
  //       use_reorder_data, stats);
  // }
//   // Block-aware search: utilizes all nodes in each read sector
//   template<typename T>
//   void PQFlashIndexWorkloadAware<T>::cached_beam_search_block_utilized(
//       const T *query1, const _u64 k_search, const _u64 l_search, _u64 *indices,
//       float *distances, const _u64 beam_width, const _u32 io_limit,
//       const bool use_reorder_data, QueryStats *stats) {
//     if (beam_width > MAX_N_SECTOR_READS)
//       throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
//                          -1, __FUNCSIG__, __FILE__, __LINE__);

//     ScratchStoreManager<SSDThreadData<T>> manager(this->thread_data);
//     auto                                  data = manager.scratch_space();
//     IOContext                            &ctx = data->ctx;
//     auto                                  query_scratch = &(data->scratch);
//     auto pq_query_scratch = query_scratch->_pq_scratch;

//     // reset query scratch
//     query_scratch->reset();

//     // copy query to thread specific aligned and allocated memory
//     float  query_norm = 0;
//     T     *aligned_query_T = query_scratch->aligned_query_T;
//     float *query_float = pq_query_scratch->aligned_query_float;
//     float *query_rotated = pq_query_scratch->rotated_query;

//     // if inner product, normalize the query
//     if (this->metric == diskann::Metric::INNER_PRODUCT) {
//       for (size_t i = 0; i < this->data_dim - 1; i++) {
//         aligned_query_T[i] = query1[i];
//         query_norm += query1[i] * query1[i];
//       }
//       aligned_query_T[this->data_dim - 1] = 0;

//       query_norm = std::sqrt(query_norm);

//       for (size_t i = 0; i < this->data_dim - 1; i++) {
//         aligned_query_T[i] /= query_norm;
//       }
//       pq_query_scratch->set(this->data_dim, aligned_query_T);
//     } else {
//       for (size_t i = 0; i < this->data_dim; i++) {
//         aligned_query_T[i] = query1[i];
//       }
//       pq_query_scratch->set(this->data_dim, aligned_query_T);
//     }

//     // pointers to buffers for data
//     T    *data_buf = query_scratch->coord_scratch;
//     _u64 &data_buf_idx = query_scratch->coord_idx;
//     _mm_prefetch((char *) data_buf, _MM_HINT_T1);

//     // sector scratch
//     char *sector_scratch = query_scratch->sector_scratch;
//     _u64 &sector_scratch_idx = query_scratch->sector_idx;

//     // query <-> PQ chunk centers distances
//     this->pq_table.preprocess_query(query_rotated);
//     float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
//     this->pq_table.populate_chunk_distances(query_rotated, pq_dists);

//     // query <-> neighbor list
//     float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
//     _u8   *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

//     // lambda to batch compute query<-> node distances in PQ space
//     auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
//                                                             const _u64 n_ids,
//                                                             float *dists_out) {
//       diskann::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
//                                 pq_coord_scratch);
//       diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
//                               dists_out);
//     };
//     Timer query_timer, io_timer, cpu_timer;

//     tsl::robin_set<_u64>  &visited = query_scratch->visited;
//     NeighborPriorityQueue &retset = query_scratch->retset;
//     retset.reserve(l_search);
//     std::vector<Neighbor> &full_retset = query_scratch->full_retset;

//     _u32  best_medoid = 0;
//     float best_dist = (std::numeric_limits<float>::max)();
//     for (_u64 cur_m = 0; cur_m < this->num_medoids; cur_m++) {
//       float cur_expanded_dist = this->dist_cmp_float->compare(
//           query_float, this->centroid_data + this->aligned_dim * cur_m,
//           (unsigned) this->aligned_dim);
//       if (cur_expanded_dist < best_dist) {
//         best_medoid = this->medoids[cur_m];
//         best_dist = cur_expanded_dist;
//       }
//     }

//     compute_dists(&best_medoid, 1, dist_scratch);
//     retset.insert(Neighbor(best_medoid, dist_scratch[0]));
//     visited.insert(best_medoid);

//     unsigned cmps = 0;
//     unsigned hops = 0;
//     unsigned num_ios = 0;
//     unsigned k = 0;

//     // Threshold-based full_retset filtering to avoid unbounded growth
//     _u64 full_retset_capacity = k_search * FULL_PRECISION_REORDER_MULTIPLIER;
//     float full_retset_threshold = std::numeric_limits<float>::max();
//     bool threshold_initialized = false;

//     // Helper: check if candidate should be added to full_retset
//     auto should_add_to_full_retset = [&](float distance) -> bool {
//       if (full_retset.size() < full_retset_capacity) {
//         return true;
//       }
//       if (!threshold_initialized) {
//         // First time reaching capacity: sort and set threshold
//         std::sort(full_retset.begin(), full_retset.end());
//         full_retset_threshold = full_retset[full_retset_capacity - 1].distance;
//         threshold_initialized = true;
//       }
//       return distance < full_retset_threshold;
//     };

//     // cleared every iteration
//     std::vector<unsigned> frontier;
//     frontier.reserve(2 * beam_width);
//     std::vector<AlignedRead> frontier_read_reqs;
//     frontier_read_reqs.reserve(2 * beam_width);
//     std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
//         cached_nhoods;
//     cached_nhoods.reserve(2 * beam_width);

//     while (retset.has_unexpanded_node() && num_ios < io_limit) {
//       // clear iteration state
//       frontier.clear();
//       frontier_read_reqs.clear();
//       cached_nhoods.clear();
//       sector_scratch_idx = 0;
      
//       // find new beam
//       _u32 num_seen = 0;
//       while (retset.has_unexpanded_node() && frontier.size() < beam_width &&
//              num_seen < beam_width) {
//         auto nbr = retset.closest_unexpanded();
//         num_seen++;
//         auto iter = this->nhood_cache.find(nbr.id);
//         if (iter != this->nhood_cache.end()) {
//           cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));
//           if (stats != nullptr) {
//             stats->n_cache_hits++;
//           }
//         } else {
//           frontier.push_back(nbr.id);
//         }
//         if (this->count_visited_nodes) {
//           reinterpret_cast<std::atomic<_u32> &>(
//               this->node_visit_counter[nbr.id].second)
//               .fetch_add(1);
//         }
//       }

//       // process cached nhoods (same as existing cached_beam_search)
//       for (auto &cached_nhood : cached_nhoods) {
//         auto  global_cache_iter = this->coord_cache.find(cached_nhood.first);
//         T    *node_fp_coords_copy = global_cache_iter->second;
//         float cur_expanded_dist;
//         if (!this->use_disk_index_pq) {
//           cur_expanded_dist = this->dist_cmp->compare(
//               aligned_query_T, node_fp_coords_copy, (unsigned) this->aligned_dim);
//         } else {
//           if (this->metric == diskann::Metric::INNER_PRODUCT)
//             cur_expanded_dist = this->disk_pq_table.inner_product(
//                 query_float, (_u8 *) node_fp_coords_copy);
//           else
//             cur_expanded_dist =
//                 this->disk_pq_table.l2_distance(
//                     query_float, (_u8 *) node_fp_coords_copy);
//         }
        
//         // Cached nodes are always added (like baseline cached_beam_search)
//         full_retset.push_back(
//             Neighbor((unsigned) cached_nhood.first, cur_expanded_dist));

//         _u64      nnbrs = cached_nhood.second.first;
//         unsigned *node_nbrs = cached_nhood.second.second;

//         // compute node_nbrs <-> query dists in PQ space
//         cpu_timer.reset();
//         compute_dists(node_nbrs, nnbrs, dist_scratch);
//         if (stats != nullptr) {
//           stats->n_cmps += (double) nnbrs;
//           stats->cpu_us += (double) cpu_timer.elapsed();
//         }

//         // process prefetched nhood
//         for (_u64 m = 0; m < nnbrs; ++m) {
//           unsigned id = node_nbrs[m];  // logical ID from disk
//           if (visited.insert(id).second) {
//             cmps++;
//             float    dist = dist_scratch[m];
//             Neighbor nn(id, dist);
//             retset.insert(nn);
//           }
//         }
//       }

//       // read nhoods of frontier ids - GROUP BY SECTOR
//       if (!frontier.empty()) {
//         if (stats != nullptr)
//           stats->n_hops++;
        
//         // Map sector_no -> list of frontier logical IDs whose neighborhoods need expansion
//         std::unordered_map<_u64, std::vector<_u32>> sector_to_expand_ids;
//         std::unordered_map<_u64, char*> sector_buffers;
        
//         for (_u64 i = 0; i < frontier.size(); i++) {
//           auto id = frontier[i];  // logical ID
//           _u64 physical_id = get_physical_idx(id);
//           _u64 sector_no = physical_id / this->nnodes_per_sector + 1;
          
//           sector_to_expand_ids[sector_no].push_back(id);
          
//           // Allocate buffer for this sector if not already done
//           if (sector_buffers.find(sector_no) == sector_buffers.end()) {
//             char* buf = sector_scratch + sector_scratch_idx * SECTOR_LEN;
//             sector_scratch_idx++;
//             sector_buffers[sector_no] = buf;
            
//             frontier_read_reqs.emplace_back(
//                 sector_no * SECTOR_LEN, 
//                 SECTOR_LEN,
//                 buf);
//           }
//         }
        
//         // Count unique sectors for I/O stats
//         _u64 unique_sectors = sector_buffers.size();
//         if (stats != nullptr) {
//           stats->n_4k += unique_sectors;
//           stats->n_ios += unique_sectors;
//         }
//         num_ios += unique_sectors;
        
//         io_timer.reset();
// #ifdef USE_BING_INFRA
//         this->reader->read(frontier_read_reqs, ctx, true);
// #else
//         this->reader->read(frontier_read_reqs, ctx);
// #endif
//         if (stats != nullptr) {
//           stats->io_us += (double) io_timer.elapsed();
//         }

//         // Process each sector: utilize ALL nodes in each sector
//         for (auto &sector_entry : sector_buffers) {
//           _u64 sector_no = sector_entry.first;
//           char *sector_buf = sector_entry.second;
//           auto &expand_ids = sector_to_expand_ids[sector_no];
          
//           // Create a set for quick lookup of frontier IDs in this sector
//           tsl::robin_set<_u32> expand_set(expand_ids.begin(), expand_ids.end());
          
//           _u64 sector_base_phys = (sector_no - 1) * this->nnodes_per_sector;
          
//           // Iterate over all slots in this sector
//           for (_u64 slot = 0; slot < this->nnodes_per_sector; slot++) {
//             _u64 physical_id = sector_base_phys + slot;
//             if (physical_id >= this->num_points) break;  // out-of-range
            
//             _u32 logical_id = get_logical_id(physical_id);
//             char *node_buf = sector_buf + slot * this->max_node_len;
            
//             // Read coordinates
//             T *node_fp_coords = (T *)node_buf;
            
//             // Copy to data buffer and compute distance
//             if (data_buf_idx == MAX_N_CMPS)
//               data_buf_idx = 0;
            
//             T *node_fp_coords_copy = data_buf + (data_buf_idx * this->aligned_dim);
//             data_buf_idx++;
//             memcpy(node_fp_coords_copy, node_fp_coords, this->disk_bytes_per_point);
            
//             float cur_expanded_dist;
//             if (!this->use_disk_index_pq) {
//               cur_expanded_dist = this->dist_cmp->compare(
//                   aligned_query_T, node_fp_coords_copy, (unsigned) this->aligned_dim);
//             } else {
//               if (this->metric == diskann::Metric::INNER_PRODUCT)
//                 cur_expanded_dist = this->disk_pq_table.inner_product(
//                     query_float, (_u8 *) node_fp_coords_copy);
//               else
//                 cur_expanded_dist = this->disk_pq_table.l2_distance(
//                     query_float, (_u8 *) node_fp_coords_copy);
//             }
            
//             // Track block utilization: we scored this vertex
//             if (stats != nullptr) {
//               stats->n_block_vertices_scored += 1.0;
//             }
            
//             // Check if this node is in the frontier (needs expansion)
//             bool is_frontier = expand_set.find(logical_id) != expand_set.end();
            
//             if (is_frontier) {
//               // Frontier nodes are ALWAYS added to full_retset (like baseline)
//               full_retset.push_back(Neighbor(logical_id, cur_expanded_dist));
              
//               // Mark frontier node as visited and add to retset
//               if (visited.insert(logical_id).second) {
//                 retset.insert(Neighbor(logical_id, cur_expanded_dist));
//               }
              
//               // Expand its neighbors
//               unsigned *node_header = (unsigned *)(node_buf + this->disk_bytes_per_point);
//               _u64 nnbrs = *node_header;
//               unsigned *node_nbrs = node_header + 1;
              
//               // compute node_nbrs <-> query dist in PQ space
//               cpu_timer.reset();
//               compute_dists(node_nbrs, nnbrs, dist_scratch);
//               if (stats != nullptr) {
//                 stats->n_cmps += (double) nnbrs;
//                 stats->cpu_us += (double) cpu_timer.elapsed();
//               }
              
//               // process neighbors
//               for (_u64 m = 0; m < nnbrs; ++m) {
//                 unsigned id = node_nbrs[m];  // logical ID from disk
//                 if (visited.insert(id).second) {
//                   cmps++;
//                   float dist = dist_scratch[m];
//                   if (stats != nullptr) {
//                     stats->n_cmps++;
//                   }
                  
//                   Neighbor nn(id, dist);
//                   retset.insert(nn);
//                 }
//               }
//             } else {
//               // Bonus node (co-located but not frontier): apply threshold filter
//               if (should_add_to_full_retset(cur_expanded_dist)) {
//                 full_retset.push_back(Neighbor(logical_id, cur_expanded_dist));
//               }
//             }
//           }
//         }
//       }

//       hops++;
//     }

//     // re-sort by distance
//     std::sort(full_retset.begin(), full_retset.end());

//     if (use_reorder_data) {
//       if (!(this->reorder_data_exists)) {
//         throw ANNException(
//             "Requested use of reordering data which does not exist in index file",
//             -1, __FUNCSIG__, __FILE__, __LINE__);
//       }

//       std::vector<AlignedRead> vec_read_reqs;

//       if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
//         full_retset.erase(
//             full_retset.begin() + k_search * FULL_PRECISION_REORDER_MULTIPLIER,
//             full_retset.end());

//       // Track unique sectors for accurate I/O counting
//       std::unordered_set<_u64> reorder_sectors;
      
//       for (size_t i = 0; i < full_retset.size(); ++i) {
//         // Reorder data uses logical IDs (no permutation)
//         auto id = full_retset[i].id;
//         _u64 sector = ((_u64) id) / this->nvecs_per_sector + this->reorder_data_start_sector;
//         vec_read_reqs.emplace_back(
//             sector * SECTOR_LEN,
//             SECTOR_LEN, sector_scratch + i * SECTOR_LEN);

//         reorder_sectors.insert(sector);
//       }

//       // Count unique sectors, not vectors
//       if (stats != nullptr) {
//         stats->n_4k += reorder_sectors.size();
//         stats->n_ios += reorder_sectors.size();
//       }

//       io_timer.reset();
// #ifdef USE_BING_INFRA
//       this->reader->read(vec_read_reqs, ctx, false);
// #else
//       this->reader->read(vec_read_reqs, ctx);
// #endif
//       if (stats != nullptr) {
//         stats->io_us += io_timer.elapsed();
//       }

//       for (size_t i = 0; i < full_retset.size(); ++i) {
//         auto id = full_retset[i].id;
//         auto location =
//             (sector_scratch + i * SECTOR_LEN) + 
//             ((((_u64) id) % this->nvecs_per_sector) * this->data_dim * sizeof(float));
//         full_retset[i].distance =
//             this->dist_cmp->compare(aligned_query_T, (T *) location, this->data_dim);
//       }

//       std::sort(full_retset.begin(), full_retset.end());
//     }

//     // copy k_search values
//     for (_u64 i = 0; i < k_search; i++) {
//       indices[i] = full_retset[i].id;
//       if (distances != nullptr) {
//         distances[i] = full_retset[i].distance;
//         if (this->metric == diskann::Metric::INNER_PRODUCT) {
//           distances[i] = (-distances[i]);
//           if (this->max_base_norm != 0)
//             distances[i] *= (this->max_base_norm * query_norm);
//         }
//       }
//     }

// #ifdef USE_BING_INFRA
//     ctx.m_completeCount = 0;
// #endif

//     if (stats != nullptr) {
//       stats->total_us = (double) query_timer.elapsed();
//       if (stats->n_ios > 0) {
//         stats->avg_vertices_per_block = 
//             stats->n_block_vertices_scored / (double)stats->n_ios;
//       }
//     }
//   }
  // template<typename T>
  // void PQFlashIndexWorkloadAware<T>::cached_beam_search_block_utilized(
  //     const T *query1, const _u64 k_search, const _u64 l_search, _u64 *indices,
  //     float *distances, const _u64 beam_width, const bool use_reorder_data,
  //     QueryStats *stats) {
  //   cached_beam_search_block_utilized(
  //       query1, k_search, l_search, indices, distances,
  //       beam_width, std::numeric_limits<_u32>::max(),
  //       use_reorder_data, stats);
  // }

  // // Block-aware search: read sectors once and utilize all nodes in each sector.
  // template<typename T>
  // void PQFlashIndexWorkloadAware<T>::cached_beam_search_block_utilized(
  //     const T *query1, const _u64 k_search, const _u64 l_search, _u64 *indices,
  //     float *distances, const _u64 beam_width, const _u32 io_limit,
  //     const bool use_reorder_data, QueryStats *stats) {
  //   if (beam_width > MAX_N_SECTOR_READS)
  //     throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
  //                       -1, __FUNCSIG__, __FILE__, __LINE__);

  //   ScratchStoreManager<SSDThreadData<T>> manager(this->thread_data);
  //   auto                                  data = manager.scratch_space();
  //   IOContext                            &ctx = data->ctx;
  //   auto                                  query_scratch = &(data->scratch);
  //   auto pq_query_scratch = query_scratch->_pq_scratch;

  //   // reset query scratch
  //   query_scratch->reset();

  //   // copy query to thread specific aligned and allocated memory
  //   float  query_norm = 0;
  //   T     *aligned_query_T = query_scratch->aligned_query_T;
  //   float *query_float = pq_query_scratch->aligned_query_float;
  //   float *query_rotated = pq_query_scratch->rotated_query;

  //   // if inner product, normalize the query
  //   if (this->metric == diskann::Metric::INNER_PRODUCT) {
  //     for (size_t i = 0; i < this->data_dim - 1; i++) {
  //       aligned_query_T[i] = query1[i];
  //       query_norm += query1[i] * query1[i];
  //     }
  //     aligned_query_T[this->data_dim - 1] = 0;

  //     query_norm = std::sqrt(query_norm);

  //     for (size_t i = 0; i < this->data_dim - 1; i++) {
  //       aligned_query_T[i] /= query_norm;
  //     }
  //     pq_query_scratch->set(this->data_dim, aligned_query_T);
  //   } else {
  //     for (size_t i = 0; i < this->data_dim; i++) {
  //       aligned_query_T[i] = query1[i];
  //     }
  //     pq_query_scratch->set(this->data_dim, aligned_query_T);
  //   }

  //   // pointers to buffers for data
  //   T    *data_buf = query_scratch->coord_scratch;
  //   _u64 &data_buf_idx = query_scratch->coord_idx;
  //   _mm_prefetch((char *) data_buf, _MM_HINT_T1);

  //   // sector scratch
  //   char *sector_scratch = query_scratch->sector_scratch;
  //   _u64 &sector_scratch_idx = query_scratch->sector_idx;

  //   // query <-> PQ chunk centers distances
  //   this->pq_table.preprocess_query(query_rotated);
  //   float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
  //   this->pq_table.populate_chunk_distances(query_rotated, pq_dists);

  //   // query <-> neighbor list
  //   float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
  //   _u8   *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

  //   // lambda to batch compute query<-> node distances in PQ space
  //   auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
  //                                                           const _u64 n_ids,
  //                                                           float *dists_out) {
  //     diskann::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
  //                               pq_coord_scratch);
  //     diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
  //                             dists_out);
  //   };

  //   Timer query_timer, io_timer, cpu_timer;

  //   tsl::robin_set<_u64>  &visited = query_scratch->visited;
  //   NeighborPriorityQueue &retset = query_scratch->retset;
  //   retset.reserve(l_search);
  //   std::vector<Neighbor> &full_retset = query_scratch->full_retset;

  //   // pick best medoid
  //   _u32  best_medoid = 0;
  //   float best_dist = (std::numeric_limits<float>::max)();
  //   for (_u64 cur_m = 0; cur_m < this->num_medoids; cur_m++) {
  //     float cur_expanded_dist = this->dist_cmp_float->compare(
  //         query_float, this->centroid_data + this->aligned_dim * cur_m,
  //         (unsigned) this->aligned_dim);
  //     if (cur_expanded_dist < best_dist) {
  //       best_medoid = this->medoids[cur_m];
  //       best_dist = cur_expanded_dist;
  //     }
  //   }

  //   compute_dists(&best_medoid, 1, dist_scratch);
  //   retset.insert(Neighbor(best_medoid, dist_scratch[0]));
  //   visited.insert(best_medoid);

  //   unsigned cmps = 0;
  //   unsigned hops = 0;
  //   unsigned num_ios = 0;

  //   // per-iteration working buffers
  //   std::vector<unsigned> frontier;
  //   frontier.reserve(2 * beam_width);
  //   std::vector<AlignedRead> frontier_read_reqs;
  //   frontier_read_reqs.reserve(2 * beam_width);
  //   std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>>
  //       cached_nhoods;
  //   cached_nhoods.reserve(2 * beam_width);

  //   if (stats != nullptr) {
  //     stats->n_block_vertices_scored = 0.0;
  //     stats->avg_vertices_per_block = 0.0;
  //   }

  //   while (retset.has_unexpanded_node() && num_ios < io_limit) {
  //     // clear iteration state
  //     frontier.clear();
  //     frontier_read_reqs.clear();
  //     cached_nhoods.clear();
  //     sector_scratch_idx = 0;

  //     // find new beam
  //     _u32 num_seen = 0;
  //     while (retset.has_unexpanded_node() && frontier.size() < beam_width &&
  //           num_seen < beam_width) {
  //       auto nbr = retset.closest_unexpanded();
  //       num_seen++;

  //       auto iter = this->nhood_cache.find(nbr.id);
  //       if (iter != this->nhood_cache.end()) {
  //         cached_nhoods.push_back(std::make_pair(nbr.id, iter->second));
  //         if (stats != nullptr) {
  //           stats->n_cache_hits++;
  //         }
  //       } else {
  //         frontier.push_back(nbr.id);
  //       }

  //       if (this->count_visited_nodes) {
  //         reinterpret_cast<std::atomic<_u32> &>(
  //             this->node_visit_counter[nbr.id].second)
  //             .fetch_add(1);
  //       }
  //     }

  //     // process cached nhoods (same as cached_beam_search)
  //     for (auto &cached_nhood : cached_nhoods) {
  //       auto  global_cache_iter = this->coord_cache.find(cached_nhood.first);
  //       T    *node_fp_coords_copy = global_cache_iter->second;
  //       float cur_expanded_dist;
  //       if (!this->use_disk_index_pq) {
  //         cur_expanded_dist = this->dist_cmp->compare(
  //             aligned_query_T, node_fp_coords_copy,
  //             (unsigned) this->aligned_dim);
  //       } else {
  //         if (this->metric == diskann::Metric::INNER_PRODUCT)
  //           cur_expanded_dist = this->disk_pq_table.inner_product(
  //               query_float, (_u8 *) node_fp_coords_copy);
  //         else
  //           cur_expanded_dist =
  //               this->disk_pq_table.l2_distance(
  //                   query_float, (_u8 *) node_fp_coords_copy);
  //       }

  //       full_retset.push_back(
  //           Neighbor((unsigned) cached_nhood.first, cur_expanded_dist));

  //       _u64      nnbrs = cached_nhood.second.first;
  //       unsigned *node_nbrs = cached_nhood.second.second;

  //       cpu_timer.reset();
  //       compute_dists(node_nbrs, nnbrs, dist_scratch);
  //       if (stats != nullptr) {
  //         stats->n_cmps += (double) nnbrs;
  //         stats->cpu_us += (double) cpu_timer.elapsed();
  //       }

  //       for (_u64 m = 0; m < nnbrs; ++m) {
  //         unsigned id = node_nbrs[m];  // logical ID
  //         if (visited.insert(id).second) {
  //           cmps++;
  //           float    dist = dist_scratch[m];
  //           Neighbor nn(id, dist);
  //           retset.insert(nn);
  //         }
  //       }
  //     }

  //     // read nhoods of frontier ids - GROUPED BY SECTOR
  //     if (!frontier.empty()) {
  //       if (stats != nullptr)
  //         stats->n_hops++;

  //       // small arrays: beam_width <= 64, so linear search is fine
  //       std::vector<_u64> sector_ids;
  //       std::vector<char *> sector_bufs;
  //       std::vector<std::vector<_u32>> sector_frontier_ids;

  //       sector_ids.reserve(frontier.size());
  //       sector_bufs.reserve(frontier.size());
  //       sector_frontier_ids.reserve(frontier.size());

  //       for (_u64 i = 0; i < frontier.size(); i++) {
  //         _u32 logical_id = frontier[i];
  //         _u64 physical_id = get_physical_idx(logical_id);
  //         _u64 sector_no = physical_id / this->nnodes_per_sector + 1;

  //         // locate sector index
  //         size_t idx = 0;
  //         for (; idx < sector_ids.size(); ++idx) {
  //           if (sector_ids[idx] == sector_no)
  //             break;
  //         }
  //         if (idx == sector_ids.size()) {
  //           // allocate new buffer for this sector
  //           char *buf = sector_scratch + sector_scratch_idx * SECTOR_LEN;
  //           sector_scratch_idx++;

  //           sector_ids.push_back(sector_no);
  //           sector_bufs.push_back(buf);
  //           sector_frontier_ids.emplace_back();
  //           frontier_read_reqs.emplace_back(
  //               sector_no * SECTOR_LEN, SECTOR_LEN, buf);
  //         }
  //         sector_frontier_ids[idx].push_back(logical_id);
  //       }

  //       // I/O stats: count unique sectors
  //       _u64 unique_sectors = sector_ids.size();
  //       if (stats != nullptr) {
  //         stats->n_4k += unique_sectors;
  //         stats->n_ios += unique_sectors;
  //       }
  //       num_ios += unique_sectors;

  //       io_timer.reset();
  // #ifdef USE_BING_INFRA
  //       this->reader->read(frontier_read_reqs, ctx, true);
  // #else
  //       this->reader->read(frontier_read_reqs, ctx);
  // #endif
  //       if (stats != nullptr) {
  //         stats->io_us += (double) io_timer.elapsed();
  //       }

  //       // Process each sector: utilize ALL nodes within that sector
  //       for (size_t si = 0; si < sector_ids.size(); ++si) {
  //         _u64  sector_no = sector_ids[si];
  //         char *sector_buf = sector_bufs[si];

  //         // quick membership check for frontier nodes in this sector
  //         tsl::robin_set<_u32> frontier_set(
  //             sector_frontier_ids[si].begin(), sector_frontier_ids[si].end());

  //         _u64 sector_base_phys =
  //             (sector_no - 1) * this->nnodes_per_sector;

  //         for (_u64 slot = 0; slot < this->nnodes_per_sector; ++slot) {
  //           _u64 physical_id = sector_base_phys + slot;
  //           if (physical_id >= this->num_points)
  //             break;

  //           _u32 logical_id = get_logical_id(physical_id);

  //           char *node_disk_buf =
  //               sector_buf + slot * this->max_node_len;

  //           // coords
  //           if (data_buf_idx == MAX_N_CMPS)
  //             data_buf_idx = 0;

  //           T *node_fp_coords_copy =
  //               data_buf + (data_buf_idx * this->aligned_dim);
  //           data_buf_idx++;

  //           memcpy(node_fp_coords_copy, node_disk_buf,
  //                 this->disk_bytes_per_point);

  //           float cur_expanded_dist;
  //           if (!this->use_disk_index_pq) {
  //             cur_expanded_dist = this->dist_cmp->compare(
  //                 aligned_query_T, node_fp_coords_copy,
  //                 (unsigned) this->aligned_dim);
  //           } else {
  //             if (this->metric == diskann::Metric::INNER_PRODUCT)
  //               cur_expanded_dist = this->disk_pq_table.inner_product(
  //                   query_float, (_u8 *) node_fp_coords_copy);
  //             else
  //               cur_expanded_dist = this->disk_pq_table.l2_distance(
  //                   query_float, (_u8 *) node_fp_coords_copy);
  //           }

  //           if (stats != nullptr) {
  //             stats->n_block_vertices_scored += 1.0;
  //           }

  //           bool is_frontier =
  //               frontier_set.find(logical_id) != frontier_set.end();

  //           if (is_frontier) {
  //             // frontier 노드는 기존 cached_beam_search 처럼 full_retset 에 넣고
  //             // 이웃을 확장한다.
  //             full_retset.push_back(
  //                 Neighbor((unsigned) logical_id, cur_expanded_dist));

  //             unsigned *node_buf_u = (unsigned *) (node_disk_buf +
  //                                                 this->disk_bytes_per_point);
  //             _u64      nnbrs = (_u64) (*node_buf_u);
  //             unsigned *node_nbrs = node_buf_u + 1;

  //             cpu_timer.reset();
  //             compute_dists(node_nbrs, nnbrs, dist_scratch);
  //             if (stats != nullptr) {
  //               stats->n_cmps += (double) nnbrs;
  //               stats->cpu_us += (double) cpu_timer.elapsed();
  //             }

  //             cpu_timer.reset();
  //             for (_u64 m = 0; m < nnbrs; ++m) {
  //               unsigned id = node_nbrs[m];  // logical neighbor id
  //               if (visited.insert(id).second) {
  //                 cmps++;
  //                 float dist = dist_scratch[m];
  //                 if (stats != nullptr) {
  //                   stats->n_cmps++;
  //                 }
  //                 Neighbor nn(id, dist);
  //                 retset.insert(nn);
  //               }
  //             }
  //             if (stats != nullptr) {
  //               stats->cpu_us += (double) cpu_timer.elapsed();
  //             }
  //           } else {
  //             // block 안에 같이 들어 있는 bonus vertex:
  //             // 그래프 확장은 하지 않고 결과 후보(full_retset)에만 추가.
  //             full_retset.push_back(
  //                 Neighbor((unsigned) logical_id, cur_expanded_dist));
  //           }
  //         }
  //       }
  //     }

  //     hops++;
  //   }

  //   // re-sort by distance
  //   std::sort(full_retset.begin(), full_retset.end());

  //   if (use_reorder_data) {
  //     if (!(this->reorder_data_exists)) {
  //       throw ANNException(
  //           "Requested use of reordering data which does not exist in index file",
  //           -1, __FUNCSIG__, __FILE__, __LINE__);
  //     }

  //     std::vector<AlignedRead> vec_read_reqs;

  //     if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
  //       full_retset.erase(
  //           full_retset.begin() + k_search * FULL_PRECISION_REORDER_MULTIPLIER,
  //           full_retset.end());

  //     // Track unique sectors for accurate I/O counting
  //     std::unordered_set<_u64> reorder_sectors;

  //     for (size_t i = 0; i < full_retset.size(); ++i) {
  //       // Reorder data uses logical IDs (no permutation)
  //       auto id = full_retset[i].id;
  //       _u64 sector = ((_u64) id) / this->nvecs_per_sector +
  //                     this->reorder_data_start_sector;
  //       vec_read_reqs.emplace_back(
  //           sector * SECTOR_LEN,
  //           SECTOR_LEN, sector_scratch + i * SECTOR_LEN);

  //       reorder_sectors.insert(sector);
  //     }

  //     if (stats != nullptr) {
  //       stats->n_4k += reorder_sectors.size();
  //       stats->n_ios += reorder_sectors.size();
  //     }

  //     io_timer.reset();
  // #ifdef USE_BING_INFRA
  //     this->reader->read(vec_read_reqs, ctx, false);
  // #else
  //     this->reader->read(vec_read_reqs, ctx);
  // #endif
  //     if (stats != nullptr) {
  //       stats->io_us += io_timer.elapsed();
  //     }

  //     for (size_t i = 0; i < full_retset.size(); ++i) {
  //       auto id = full_retset[i].id;
  //       auto location =
  //           (sector_scratch + i * SECTOR_LEN) +
  //           ((((_u64) id) % this->nvecs_per_sector) *
  //           this->data_dim * sizeof(float));
  //       full_retset[i].distance =
  //           this->dist_cmp->compare(aligned_query_T, (T *) location,
  //                                   this->data_dim);
  //     }

  //     std::sort(full_retset.begin(), full_retset.end());
  //   }

  //   // copy k_search values
  //   for (_u64 i = 0; i < k_search; i++) {
  //     indices[i] = full_retset[i].id;
  //     if (distances != nullptr) {
  //       distances[i] = full_retset[i].distance;
  //       if (this->metric == diskann::Metric::INNER_PRODUCT) {
  //         distances[i] = (-distances[i]);
  //         if (this->max_base_norm != 0)
  //           distances[i] *= (this->max_base_norm * query_norm);
  //       }
  //     }
  //   }

  // #ifdef USE_BING_INFRA
  //   ctx.m_completeCount = 0;
  // #endif

  //   if (stats != nullptr) {
  //     stats->total_us = (double) query_timer.elapsed();
  //     if (stats->n_ios > 0) {
  //       stats->avg_vertices_per_block =
  //           stats->n_block_vertices_scored /
  //           (double) stats->n_ios;
  //     }
  //   }
  // }

  template<typename T>
  void PQFlashIndexWorkloadAware<T>::cached_beam_search_block_utilized(
      const T *query1, const _u64 k_search, const _u64 l_search, _u64 *indices,
      float *distances, const _u64 beam_width,
      const bool use_reorder_data, QueryStats *stats) {
    cached_beam_search_block_utilized(
        query1, k_search, l_search, indices, distances,
        beam_width, std::numeric_limits<_u32>::max(),
        use_reorder_data, stats);
  }

  // Block-aware search: same search semantics as cached_beam_search,
  // but when we read a sector we cache *all* nodes in that sector
  // into per-query local caches to avoid future disk I/O.
  template<typename T>
  void PQFlashIndexWorkloadAware<T>::cached_beam_search_block_utilized(
      const T *query1, const _u64 k_search, const _u64 l_search, _u64 *indices,
      float *distances, const _u64 beam_width, const _u32 io_limit,
      const bool use_reorder_data, QueryStats *stats) {
    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                        -1, __FUNCSIG__, __FILE__, __LINE__);

    ScratchStoreManager<SSDThreadData<T>> manager(this->thread_data);
    auto                                  data = manager.scratch_space();
    IOContext                            &ctx = data->ctx;
    auto                                  query_scratch = &(data->scratch);
    auto pq_query_scratch = query_scratch->_pq_scratch;

    // reset query scratch
    query_scratch->reset();

    // copy query to thread specific aligned and allocated memory
    float  query_norm = 0;
    T     *aligned_query_T = query_scratch->aligned_query_T;
    float *query_float = pq_query_scratch->aligned_query_float;
    float *query_rotated = pq_query_scratch->rotated_query;

    // if inner product, normalize the query
    if (this->metric == diskann::Metric::INNER_PRODUCT) {
      for (size_t i = 0; i < this->data_dim - 1; i++) {
        aligned_query_T[i] = query1[i];
        query_norm += query1[i] * query1[i];
      }
      aligned_query_T[this->data_dim - 1] = 0;

      query_norm = std::sqrt(query_norm);

      for (size_t i = 0; i < this->data_dim - 1; i++) {
        aligned_query_T[i] /= query_norm;
      }
      pq_query_scratch->set(this->data_dim, aligned_query_T);
    } else {
      for (size_t i = 0; i < this->data_dim; i++) {
        aligned_query_T[i] = query1[i];
      }
      pq_query_scratch->set(this->data_dim, aligned_query_T);
    }

    // pointers to buffers for data
    T    *data_buf = query_scratch->coord_scratch;
    _u64 &data_buf_idx = query_scratch->coord_idx;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    _u64 &sector_scratch_idx = query_scratch->sector_idx;

    // query <-> PQ chunk centers distances
    this->pq_table.preprocess_query(query_rotated);
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
    this->pq_table.populate_chunk_distances(query_rotated, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
    _u8   *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
                                                            const _u64 n_ids,
                                                            float *dists_out) {
      diskann::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                                pq_coord_scratch);
      diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                              dists_out);
    };

    Timer query_timer, io_timer, cpu_timer;

    tsl::robin_set<_u64>  &visited = query_scratch->visited;
    NeighborPriorityQueue &retset = query_scratch->retset;
    retset.reserve(l_search);
    std::vector<Neighbor> &full_retset = query_scratch->full_retset;

    // per-query local caches (thread-safe, lifetime = this search call)
    // logical_id -> (nnbrs, vector<nbrs>)
    std::unordered_map<_u32, std::pair<_u32, std::vector<unsigned>>> local_nhood_cache;
    // logical_id -> coords (aligned_dim, disk_bytes_per_point valid)
    std::unordered_map<_u32, std::unique_ptr<T[]>> local_coord_cache;

    if (stats != nullptr) {
      stats->n_block_vertices_scored = 0.0;
      stats->avg_vertices_per_block = 0.0;
    }

    // pick best medoid
    _u32  best_medoid = 0;
    float best_dist = (std::numeric_limits<float>::max)();
    for (_u64 cur_m = 0; cur_m < this->num_medoids; cur_m++) {
      float cur_expanded_dist = this->dist_cmp_float->compare(
          query_float, this->centroid_data + this->aligned_dim * cur_m,
          (unsigned) this->aligned_dim);
      if (cur_expanded_dist < best_dist) {
        best_medoid = this->medoids[cur_m];
        best_dist = cur_expanded_dist;
      }
    }

    // seed from best medoid
    compute_dists(&best_medoid, 1, dist_scratch);
    retset.insert(Neighbor(best_medoid, dist_scratch[0]));
    visited.insert(best_medoid);

    unsigned cmps = 0;
    unsigned hops = 0;
    unsigned num_ios = 0;

    // per-iteration working buffers
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, std::pair<unsigned, unsigned *>>> cached_nhoods;
    cached_nhoods.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);
    std::vector<std::pair<unsigned, char *>> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);

    while (retset.has_unexpanded_node() && num_ios < io_limit) {
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      cached_nhoods.clear();
      sector_scratch_idx = 0;

      // ------------------------------------------------------------
      // 1) Find new beam (same semantics as cached_beam_search)
      //    but also check local_nhood_cache as "cached"
      // ------------------------------------------------------------
      _u32 num_seen = 0;
      while (retset.has_unexpanded_node() && frontier.size() < beam_width &&
            num_seen < beam_width) {
        auto nbr = retset.closest_unexpanded();
        num_seen++;

        bool found_in_cache = false;

        // global cache first
        auto g_it = this->nhood_cache.find(nbr.id);
        if (g_it != this->nhood_cache.end()) {
          cached_nhoods.push_back(std::make_pair(nbr.id, g_it->second));
          found_in_cache = true;
        } else {
          // then local cache
          auto l_it = local_nhood_cache.find(nbr.id);
          if (l_it != local_nhood_cache.end()) {
            cached_nhoods.emplace_back(
                nbr.id,
                std::make_pair(l_it->second.first,
                              l_it->second.second.data()));
            found_in_cache = true;
          }
        }

        if (found_in_cache) {
          if (stats != nullptr) {
            stats->n_cache_hits++;
          }
        } else {
          frontier.push_back(nbr.id);
        }

        if (this->count_visited_nodes) {
          reinterpret_cast<std::atomic<_u32> &>(
              this->node_visit_counter[nbr.id].second)
              .fetch_add(1);
        }
      }

      // ------------------------------------------------------------
      // 2) Read nhoods of frontier ids, grouped by sector
      //    BUT search semantics = baseline:
      //      - we only expand frontier nodes
      //      - extra nodes in each sector are *just cached*, not expanded
      // ------------------------------------------------------------
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;

        // sector_no -> sector buffer
        std::vector<_u64>           sector_ids;
        std::vector<char *>         sector_bufs;
        sector_ids.reserve(frontier.size());
        sector_bufs.reserve(frontier.size());

        for (_u64 i = 0; i < frontier.size(); i++) {
          _u32 logical_id = frontier[i];
          _u64 physical_id = get_physical_idx(logical_id);
          _u64 sector_no   = physical_id / this->nnodes_per_sector + 1;

          // find or add sector
          size_t idx = 0;
          for (; idx < sector_ids.size(); ++idx) {
            if (sector_ids[idx] == sector_no)
              break;
          }
          char *buf = nullptr;
          if (idx == sector_ids.size()) {
            // new sector
            buf = sector_scratch + sector_scratch_idx * SECTOR_LEN;
            sector_scratch_idx++;

            sector_ids.push_back(sector_no);
            sector_bufs.push_back(buf);

            frontier_read_reqs.emplace_back(
                sector_no * SECTOR_LEN,
                SECTOR_LEN,
                buf);
          } else {
            buf = sector_bufs[idx];
          }

          // each frontier node remembers which buffer holds its sector
          frontier_nhoods.emplace_back(logical_id, buf);
        }

        // I/O stats by unique sectors
        _u64 unique_sectors = sector_ids.size();
        if (stats != nullptr) {
          stats->n_4k += unique_sectors;
          stats->n_ios += unique_sectors;
        }
        num_ios += unique_sectors;

        io_timer.reset();
  #ifdef USE_BING_INFRA
        this->reader->read(frontier_read_reqs, ctx, true);
  #else
        this->reader->read(frontier_read_reqs, ctx);
  #endif
        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
        }

        // --------------------------------------------------------
        // 2-1) Block utilization: for each sector we just read,
        //      parse *all* nodes in that sector and put them into
        //      per-query local caches (coords + nhood).
        //      -> No graph expansion here.
        // --------------------------------------------------------
        for (size_t si = 0; si < sector_ids.size(); ++si) {
          _u64  sector_no = sector_ids[si];
          char *sector_buf = sector_bufs[si];

          _u64 sector_base_phys =
              (sector_no - 1) * this->nnodes_per_sector;

          for (_u64 slot = 0; slot < this->nnodes_per_sector; ++slot) {
            _u64 physical_id = sector_base_phys + slot;
            if (physical_id >= this->num_points)
              break;

            _u32 logical_id = get_logical_id(physical_id);
            char *node_disk_buf =
                sector_buf + slot * this->max_node_len;

            // already globally cached?
            if (this->nhood_cache.find(logical_id) != this->nhood_cache.end())
              continue;
            if (local_nhood_cache.find(logical_id) != local_nhood_cache.end())
              continue;

            // coords for this node
            std::unique_ptr<T[]> coords(new T[this->aligned_dim]);
            std::memset(coords.get(), 0, this->aligned_dim * sizeof(T));
            std::memcpy(coords.get(), node_disk_buf,
                        this->disk_bytes_per_point);
            local_coord_cache.emplace(logical_id, std::move(coords));

            // nhood
            unsigned *node_buf_u =
                (unsigned *) (node_disk_buf + this->disk_bytes_per_point);
            _u64      nnbrs = (_u64) (*node_buf_u);
            unsigned *node_nbrs = node_buf_u + 1;

            std::vector<unsigned> nbrs_vec((size_t) nnbrs);
            std::memcpy(nbrs_vec.data(), node_nbrs,
                        nnbrs * sizeof(unsigned));
            local_nhood_cache.emplace(
                logical_id,
                std::make_pair((_u32) nnbrs, std::move(nbrs_vec)));

            if (stats != nullptr) {
              stats->n_block_vertices_scored += 1.0;
            }
          }
        }
      }

      // ------------------------------------------------------------
      // 3) Process cached nhoods (same as cached_beam_search),
      //    but coords/nhood은 global + local cache 모두에서 찾음.
      // ------------------------------------------------------------
      for (auto &cached_nhood : cached_nhoods) {
        _u32 cached_id = cached_nhood.first;

        // coords: global first, then local
        T *node_fp_coords_copy = nullptr;
        auto g_it = this->coord_cache.find(cached_id);
        if (g_it != this->coord_cache.end()) {
          node_fp_coords_copy = g_it->second;
        } else {
          auto l_it = local_coord_cache.find(cached_id);
          if (l_it != local_coord_cache.end()) {
            node_fp_coords_copy = l_it->second.get();
          } else {
            // should not happen; skip this cached node
            continue;
          }
        }

        float cur_expanded_dist;
        if (!this->use_disk_index_pq) {
          cur_expanded_dist = this->dist_cmp->compare(
              aligned_query_T, node_fp_coords_copy,
              (unsigned) this->aligned_dim);
        } else {
          if (this->metric == diskann::Metric::INNER_PRODUCT)
            cur_expanded_dist = this->disk_pq_table.inner_product(
                query_float, (_u8 *) node_fp_coords_copy);
          else
            cur_expanded_dist =
                this->disk_pq_table.l2_distance(
                    query_float, (_u8 *) node_fp_coords_copy);
        }

        full_retset.push_back(
            Neighbor((unsigned) cached_id, cur_expanded_dist));

        _u64      nnbrs     = cached_nhood.second.first;
        unsigned *node_nbrs = cached_nhood.second.second;

        cpu_timer.reset();
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += (double) nnbrs;
          stats->cpu_us += (double) cpu_timer.elapsed();
        }

        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];  // logical ID
          if (visited.insert(id).second) {
            cmps++;
            float    dist = dist_scratch[m];
            Neighbor nn(id, dist);
            retset.insert(nn);
          }
        }
      }

      // ------------------------------------------------------------
      // 4) Process frontier nodes (exactly same as cached_beam_search)
      //    -> 여기서만 그래프 확장 발생
      // ------------------------------------------------------------
      for (auto &frontier_nhood : frontier_nhoods) {
        _u32 logical_id = frontier_nhood.first;
        _u64 physical_id = get_physical_idx(logical_id);

        char *node_disk_buf =
            (char *) frontier_nhood.second +
            (physical_id % this->nnodes_per_sector) * this->max_node_len;

        unsigned *node_buf =
            (unsigned *) ((char *) node_disk_buf + this->disk_bytes_per_point);
        _u64      nnbrs = (_u64) (*node_buf);
        T        *node_fp_coords = (T *) node_disk_buf;

        if (data_buf_idx == MAX_N_CMPS)
          data_buf_idx = 0;

        T *node_fp_coords_copy =
            data_buf + (data_buf_idx * this->aligned_dim);
        data_buf_idx++;
        std::memcpy(node_fp_coords_copy, node_fp_coords,
                    this->disk_bytes_per_point);

        float cur_expanded_dist;
        if (!this->use_disk_index_pq) {
          cur_expanded_dist = this->dist_cmp->compare(
              aligned_query_T, node_fp_coords_copy,
              (unsigned) this->aligned_dim);
        } else {
          if (this->metric == diskann::Metric::INNER_PRODUCT)
            cur_expanded_dist = this->disk_pq_table.inner_product(
                query_float, (_u8 *) node_fp_coords_copy);
          else
            cur_expanded_dist =
                this->disk_pq_table.l2_distance(
                    query_float, (_u8 *) node_fp_coords_copy);
        }

        full_retset.push_back(
            Neighbor((unsigned) logical_id, cur_expanded_dist));

        unsigned *node_nbrs = (node_buf + 1);

        cpu_timer.reset();
        compute_dists(node_nbrs, nnbrs, dist_scratch);
        if (stats != nullptr) {
          stats->n_cmps += (double) nnbrs;
          stats->cpu_us += (double) cpu_timer.elapsed();
        }

        cpu_timer.reset();
        for (_u64 m = 0; m < nnbrs; ++m) {
          unsigned id = node_nbrs[m];  // logical ID
          if (visited.insert(id).second) {
            cmps++;
            float dist = dist_scratch[m];
            if (stats != nullptr) {
              stats->n_cmps++;
            }

            Neighbor nn(id, dist);
            retset.insert(nn);
          }
        }

        if (stats != nullptr) {
          stats->cpu_us += (double) cpu_timer.elapsed();
        }
      }

      hops++;
    }

    // ------------------------------------------------------------
    // 5) 그대로: reorder, top-k copy, stats 마무리
    // ------------------------------------------------------------

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end());

    if (use_reorder_data) {
      if (!(this->reorder_data_exists)) {
        throw ANNException(
            "Requested use of reordering data which does not exist in index file",
            -1, __FUNCSIG__, __FILE__, __LINE__);
      }

      std::vector<AlignedRead> vec_read_reqs;

      if (full_retset.size() > k_search * FULL_PRECISION_REORDER_MULTIPLIER)
        full_retset.erase(
            full_retset.begin() + k_search * FULL_PRECISION_REORDER_MULTIPLIER,
            full_retset.end());

      std::unordered_set<_u64> reorder_sectors;

      for (size_t i = 0; i < full_retset.size(); ++i) {
        auto id = full_retset[i].id;
        _u64 sector =
            ((_u64) id) / this->nvecs_per_sector +
            this->reorder_data_start_sector;
        vec_read_reqs.emplace_back(
            sector * SECTOR_LEN,
            SECTOR_LEN, sector_scratch + i * SECTOR_LEN);
        reorder_sectors.insert(sector);
      }

      if (stats != nullptr) {
        stats->n_4k += reorder_sectors.size();
        stats->n_ios += reorder_sectors.size();
      }

      io_timer.reset();
  #ifdef USE_BING_INFRA
      this->reader->read(vec_read_reqs, ctx, false);
  #else
      this->reader->read(vec_read_reqs, ctx);
  #endif
      if (stats != nullptr) {
        stats->io_us += io_timer.elapsed();
      }

      for (size_t i = 0; i < full_retset.size(); ++i) {
        auto id = full_retset[i].id;
        auto location =
            (sector_scratch + i * SECTOR_LEN) +
            ((((_u64) id) % this->nvecs_per_sector) *
            this->data_dim * sizeof(float));
        full_retset[i].distance =
            this->dist_cmp->compare(aligned_query_T,
                                    (T *) location,
                                    this->data_dim);
      }

      std::sort(full_retset.begin(), full_retset.end());
    }

    // copy k_search values
    for (_u64 i = 0; i < k_search; i++) {
      indices[i] = full_retset[i].id;
      if (distances != nullptr) {
        distances[i] = full_retset[i].distance;
        if (this->metric == diskann::Metric::INNER_PRODUCT) {
          distances[i] = (-distances[i]);
          if (this->max_base_norm != 0)
            distances[i] *= (this->max_base_norm * query_norm);
        }
      }
    }

  #ifdef USE_BING_INFRA
    ctx.m_completeCount = 0;
  #endif

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
      if (stats->n_ios > 0) {
        stats->avg_vertices_per_block =
            stats->n_block_vertices_scored /
            (double) stats->n_ios;
      }
    }
  }

  // Override load_cache_list to apply permutation when loading cached nodes
  template<typename T>
  void PQFlashIndexWorkloadAware<T>::load_cache_list(std::vector<uint32_t> &node_list) {
    diskann::cout << "Loading the cache list into memory (workload-aware).." << std::flush;
    _u64 num_cached_nodes = node_list.size();

    // borrow thread data
    ScratchStoreManager<SSDThreadData<T>> manager(this->thread_data);
    auto       this_thread_data = manager.scratch_space();
    IOContext &ctx = this_thread_data->ctx;

    this->nhood_cache_buf = new unsigned[num_cached_nodes * (this->max_degree + 1)];
    memset(this->nhood_cache_buf, 0, num_cached_nodes * (this->max_degree + 1));

    _u64 coord_cache_buf_len = num_cached_nodes * this->aligned_dim;
    diskann::alloc_aligned((void **) &this->coord_cache_buf,
                           coord_cache_buf_len * sizeof(T), 8 * sizeof(T));
    memset(this->coord_cache_buf, 0, coord_cache_buf_len * sizeof(T));

    size_t BLOCK_SIZE = 8;
    size_t num_blocks = DIV_ROUND_UP(num_cached_nodes, BLOCK_SIZE);

    for (_u64 block = 0; block < num_blocks; block++) {
      _u64 start_idx = block * BLOCK_SIZE;
      _u64 end_idx = (std::min)(num_cached_nodes, (block + 1) * BLOCK_SIZE);
      std::vector<AlignedRead>             read_reqs;
      std::vector<std::pair<_u32, char *>> nhoods;
      
      for (_u64 node_idx = start_idx; node_idx < end_idx; node_idx++) {
        AlignedRead read;
        char       *buf = nullptr;
        alloc_aligned((void **) &buf, SECTOR_LEN, SECTOR_LEN);
        
        _u32 logical_id = node_list[node_idx];
        nhoods.push_back(std::make_pair(logical_id, buf));
        
        // APPLY PERMUTATION: Convert logical ID to physical ID for disk read
        _u64 physical_id = get_physical_idx(logical_id);
        
        read.len = SECTOR_LEN;
        read.buf = buf;
        // Use physical_id to compute disk offset
        read.offset = (physical_id / this->nnodes_per_sector + 1) * SECTOR_LEN;
        read_reqs.push_back(read);
      }

      this->reader->read(read_reqs, ctx);

      _u64 node_idx = start_idx;
      for (_u32 i = 0; i < read_reqs.size(); i++) {
#if defined(_WINDOWS) && defined(USE_BING_INFRA)
        if ((*ctx.m_pRequestsStatus)[i] != IOContext::READ_SUCCESS) {
          continue;
        }
#endif
        auto &nhood = nhoods[i];
        _u32 logical_id = nhood.first;
        
        // APPLY PERMUTATION: Convert logical ID to physical ID for offset calculation
        _u64 physical_id = get_physical_idx(logical_id);
        
        // Use physical_id to compute offset within sector
        char *node_buf = (char *)nhood.second + 
                        (physical_id % this->nnodes_per_sector) * this->max_node_len;
        
        T *node_coords = (T *)node_buf;  // OFFSET_TO_NODE_COORDS
        T *cached_coords = this->coord_cache_buf + node_idx * this->aligned_dim;
        memcpy(cached_coords, node_coords, this->disk_bytes_per_point);
        
        // Insert into cache with logical_id as key
        this->coord_cache.insert(std::make_pair(logical_id, cached_coords));

        // insert node nhood into nhood_cache
        unsigned *node_nhood = (unsigned *)((char *)node_buf + this->disk_bytes_per_point);
        
        auto nnbrs = *node_nhood;
        unsigned *nbrs = node_nhood + 1;
        
        // Neighbors are already logical IDs, no conversion needed
        std::pair<_u32, unsigned *> cnhood;
        cnhood.first = nnbrs;
        cnhood.second = this->nhood_cache_buf + node_idx * (this->max_degree + 1);
        memcpy(cnhood.second, nbrs, nnbrs * sizeof(unsigned));
        
        // Insert into cache with logical_id as key
        this->nhood_cache.insert(std::make_pair(logical_id, cnhood));
        
        aligned_free(nhood.second);
        node_idx++;
      }
    }
    diskann::cout << "..done." << std::endl;
  }

  // Template instantiations
  template class PQFlashIndexWorkloadAware<float>;
  template class PQFlashIndexWorkloadAware<uint8_t>;
  template class PQFlashIndexWorkloadAware<int8_t>;

}  // namespace diskann

