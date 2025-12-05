# HNSW Deep Analysis - From Reference Implementation

**Source**: https://github.com/nmslib/hnswlib

## Key Algorithm Insights

### 1. Neighbor Selection: `getNeighborsByHeuristic2`

```cpp
// From hnswalg.h lines 443-483
void getNeighborsByHeuristic2(priority_queue& top_candidates, size_t M) {
    if (top_candidates.size() < M) return;  // Keep all if fewer than M
    
    priority_queue queue_closest;  // Min-heap by distance
    vector<pair> return_list;
    
    // Convert to min-heap
    while (top_candidates.size() > 0) {
        queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
        top_candidates.pop();
    }
    
    // Greedy selection
    while (queue_closest.size()) {
        if (return_list.size() >= M) break;
        
        auto current = queue_closest.top();
        dist_t dist_to_query = -current.first;
        queue_closest.pop();
        
        bool good = true;
        for (auto& selected : return_list) {
            dist_t dist_to_selected = distance(current, selected);
            if (dist_to_selected < dist_to_query) {  // STRICT <, no threshold!
                good = false;
                break;  // Early exit!
            }
        }
        
        if (good) {
            return_list.push_back(current);
        }
    }
    // ... return results
}
```

**Key differences from Vamana RNG:**
1. **O(M × C) not O(C²)** - only compare to selected, not all candidates
2. **No distance threshold** - uses strict `<` comparison
3. **Early exit on failure** - stops checking once one selected neighbor is closer

### 2. Backedge Updates: Mostly O(1)!

```cpp
// From mutuallyConnectNewElement, lines 586-603
if (sz_link_list_other < Mcurmax) {
    // Room available - just append! O(1)
    data[sz_link_list_other] = cur_c;
    setListCount(ll_other, sz_link_list_other + 1);
} else {
    // Full - need to prune
    // But this only happens when neighbor is at max capacity!
    priority_queue candidates;
    candidates.emplace(distance(cur_c, neighbor), cur_c);
    for (j in existing_neighbors) {
        candidates.emplace(distance(j, neighbor), j);
    }
    getNeighborsByHeuristic2(candidates, Mcurmax);  // Same O(M×C) heuristic
}
```

**Vamana always prunes** when add_edges overflows. HNSW only prunes when neighbor is truly full.

### 3. Search with Priority Queue + Lower Bound

```cpp
// From searchBaseLayerST, lines 309-399
priority_queue top_candidates;  // MAX-heap: worst at top
priority_queue candidate_set;   // MIN-heap: best at top (-distance)

dist_t lowerBound = initial_dist;  // Worst distance in results
top_candidates.emplace(dist, ep_id);
candidate_set.emplace(-dist, ep_id);

while (!candidate_set.empty()) {
    auto current = candidate_set.top();
    dist_t candidate_dist = -current.first;  // Best unexplored
    
    // KEY EARLY STOP: if best unexplored > worst result, done!
    if (candidate_dist > lowerBound && top_candidates.size() == ef) {
        break;
    }
    candidate_set.pop();
    
    // Expand current node...
    for (neighbor : current.neighbors) {
        dist_t dist = distance(query, neighbor);
        
        // Only add if could improve results
        if (top_candidates.size() < ef || lowerBound > dist) {
            candidate_set.emplace(-dist, neighbor);
            top_candidates.emplace(dist, neighbor);
            
            if (top_candidates.size() > ef)
                top_candidates.pop();  // Remove worst
            
            if (!top_candidates.empty())
                lowerBound = top_candidates.top().first;  // Update bound
        }
    }
}
```

**Key insight**: The search maintains `lowerBound` (worst result distance) and stops when the best unexplored candidate is worse than that. This is more aggressive than our "stale iterations" heuristic.

### 4. Prefetching in Search Loop

```cpp
// Aggressive prefetching of next neighbor's data
#ifdef USE_SSE
_mm_prefetch((char*)(visited_array + *(data + j + 1)), _MM_HINT_T0);
_mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
#endif
```

## IMPORTANT: Vamana vs HNSW Differences

**After reading the DiskANN paper carefully, we found that Vamana's RobustPrune
is NOT identical to HNSW's heuristic - and the difference matters!**

| Aspect | Vamana RobustPrune | HNSW Heuristic |
|--------|-------------------|----------------|
| Condition | `α · d(p*, p') ≤ d(p, p')` | `d(p*, p') < d(q, p')` |
| α parameter | Yes (typically 1.2) | No |
| Guarantee | O(log n) search path with α > 1 | No formal guarantee |

The α parameter ensures each search step makes **multiplicative progress**:
> "we would like to ensure that the distance to the query decreases by 
> a multiplicative factor of α > 1 at every node along the search path"

**Recommendation**: Keep Vamana's RobustPrune as default. See `VAMANA_RNG_ANALYSIS.md`.

## Implementation Recommendations for CoreNN

### Priority 1: Keep Vamana RobustPrune (Done ✓)

The original α-RNG pruning is correct and has theoretical guarantees.
Don't replace with HNSW heuristic.

### Priority 2: lowerBound Early Stopping (Done ✓)

This is safe to adopt from HNSW - it's just a search optimization.

### Priority 3: Lazy Backedge Updates

Only prune backedges when neighbor is truly full:
```rust
if neighbor.edges.len() < max_edges {
    neighbor.edges.push(new_node);  // O(1)!
} else {
    prune_with_heuristic(neighbor);  // Only when full
}
```

### Priority 4: Priority Queue Search with lowerBound

Replace sorted Vec with BinaryHeap:
- Track `lower_bound` (worst result distance)
- Stop when best unexplored > lower_bound
- More aggressive than "stale iterations"

### Priority 4: Visited Array Pool

HNSW uses a pool of visited arrays with generation counters:
- Avoids allocating HashSet per search
- Just increment counter to "clear"

## Performance Impact Estimates

| Optimization | Current | With Fix | Improvement |
|--------------|---------|----------|-------------|
| Neighbor selection | O(C²) | O(M×C) | 5-10x faster |
| Backedge updates | Always prune | Usually O(1) | 3-5x faster |
| Search stopping | Stale iterations | lowerBound | 10-30% faster |
| Visited tracking | HashSet alloc | Pool + counter | 5-10% faster |

**Combined insert improvement**: 5-10x faster
**Query improvement**: 10-30% faster

## Visited List Pool (from visited_list_pool.h)

HNSW uses a clever optimization to avoid HashSet allocation per search:

```cpp
class VisitedList {
    vl_type curV;        // Generation counter
    vl_type *mass;       // Array of size max_elements
    
    void reset() {
        curV++;          // Just increment counter to "clear"!
        if (curV == 0) { // Handle wraparound (every 65535 searches)
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    }
};

// Usage in search:
visited_array[candidate_id] = visited_array_tag;  // Mark visited
if (visited_array[candidate_id] == visited_array_tag) continue; // Skip if visited
```

**Key insight**: Instead of clearing or reallocating a HashSet, just increment a counter.
- O(1) "clear" instead of O(n) or allocation
- Cache-friendly: sequential array access
- No allocator overhead during search

This is particularly beneficial for:
- High QPS workloads
- Large datasets (where HashSet allocation is expensive)
- Repeated searches

## Implementation Status in CoreNN

### ✅ Implemented
1. HNSW-style neighbor selection (O(M×C) with early exit)
2. lowerBound-based early stopping in search
3. Lazy pruning via max_add_edges

### 🔜 TODO
1. Visited list pool (avoid DashSet allocation per search)
2. Lazy backedge updates (only prune when neighbor is truly full, not just add_edges)
3. More aggressive prefetching in search loop
