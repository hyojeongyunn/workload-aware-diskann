# Workload-Aware DiskANN

This repository implements a workload-aware extension of DiskANN for disk-based approximate nearest neighbor (ANN) search under memory constraints and out-of-distribution (OOD) queries.

Instead of assuming that queries follow the same distribution as the indexed data, we explicitly use real query workloads to drive **sharding**, **disk layout**, and **page-level search**.

---

## Key Ideas

### 1. Workload-Aware Sharding & Partial Graph Build

- Standard DiskANN shards only on data vectors, so under OOD workloads true neighbors can be scattered across shards and memory usage is hard to control.
- We use sampled queries (from the same distribution as the offline workload) to perform **workload-aware k-means clustering**:
  - Queries and data are used together to partition the dataset into shards that better align with query behavior.
  - Each shard is loaded into memory independently and a graph is built per shard.
- Graph indexes are stored **per-shard** on disk, so at build time and query time we only need to load a subset of shards, keeping peak memory usage bounded regardless of total dataset size.

---

### 2. Co-Visitation–Based Disk Layout

- In the original DiskANN layout, nodes are written to disk in logical ID (or build) order, so nodes that are frequently visited together by real queries often end up on different disk pages.
- We run beam search for sampled queries on the in-memory graph and log full visitation paths.
- From these paths we build a **co-visitation graph**:
  - For each query path, nodes within a small window (±3 hops) are connected with a weighted edge.
  - Node “strength” is computed from accumulated co-visitation weights.
- We then compute a **logical → physical permutation**:
  - Start BFS from high-strength nodes in the co-visitation graph.
  - Assign consecutive physical IDs so that frequently co-visited nodes are placed close together on disk pages.
- When writing the disk index:
  - Vectors and node records are laid out according to the physical ID permutation.
  - Adjacency lists still use **logical IDs**, so the search logic stays simple and only the logical↔physical mapping layer knows about the disk layout.

---

### 3. Page-Based Adaptive Graph Search

- Vanilla DiskANN-style search manages candidates per node and does not fully exploit the fact that disk I/O happens per **page/sector**, not per node.
- Our search is **page-aware and adaptive**:
  - Beam candidates are tracked in logical ID space.
  - Just before issuing I/O, we map logical → physical IDs, group nodes by sector, and **batch** reads per sector.
- When a sector is read:
  - We parse **all nodes** in that sector and store their coordinates + neighborhoods in a **per-query local cache**.
  - However, we only **expand neighbors** for frontier nodes selected by the beam search.
- This means:
  - **Caching is aggressive** (we keep as much page-level information as possible once paid for).
  - **Graph expansion is adaptive**: nodes are only expanded when they later become frontier candidates.
- A two-level cache is used:
  - A **global cache** for globally hot nodes.
  - A **per-query local page cache** for nodes in sectors touched by the current query.
- This design lets us:
  - Reuse already-loaded pages without additional disk I/O.
  - Control the trade-off between I/O budget and search quality via policy (beam width, I/O limits, cache size), while keeping search semantics compatible with the baseline.