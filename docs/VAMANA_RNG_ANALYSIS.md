# Vamana RobustPrune Algorithm - Deep Analysis

**Source**: DiskANN paper (Subramanya et al., NeurIPS 2019) and FreshDiskANN (Singh et al., 2021)

## Algorithm 2: RobustPrune(p, V, α, R)

```
Input: Graph G, point p, candidate set V, distance threshold α ≥ 1, degree bound R
Output: G is modified by setting at most R new out-neighbors for p

begin
  V ← (V ∪ Nout(p)) \ {p}    // Merge with existing neighbors
  Nout(p) ← ∅                 // Clear p's neighbors
  
  while V ≠ ∅ do
    p* ← argmin_{p' ∈ V} d(p, p')    // Pick closest remaining to p
    Nout(p) ← Nout(p) ∪ {p*}          // Add to neighbors
    
    if |Nout(p)| = R then break       // Stop at max degree
    
    for p' ∈ V do
      if α · d(p*, p') ≤ d(p, p') then   // α-RNG condition
        remove p' from V                  // Prune "covered" points
```

## The α Parameter is CRUCIAL

From the DiskANN paper:

> "To overcome [large diameter], we would like to ensure that the distance to the query 
> decreases by a multiplicative factor of α > 1 at every node along the search path, 
> instead of merely decreasing as in the SNG property."

### What α controls:

| α value | Effect |
|---------|--------|
| α = 1.0 | Standard RNG - more aggressive pruning, sparser graph, potentially larger diameter |
| α > 1.0 | Relaxed pruning - denser graph, **guarantees O(log n) diameter** |
| α = 1.2 | Recommended value in DiskANN paper for disk-based systems |

### Why this matters for search:

With α > 1, each step in GreedySearch makes **multiplicative progress** toward the query:
- `d(query, next_node) ≤ d(query, current_node) / α`
- This bounds search path length to O(log n)
- Critical for disk-based systems where each hop = disk read

## α-RNG Condition Explained

The condition `α · d(p*, p') ≤ d(p, p')` means:

**Remove p' if**: `α × distance(selected, p') ≤ distance(node, p')`

Rearranging: **Keep p' if**: `distance(node, p') < α × distance(selected, p')`

Intuition:
- If p' is far from node (large `d(p, p')`) but close to already-selected p* (small `d(p*, p')`)
- Then p* already "covers" that direction
- We don't need p' as a neighbor

When α > 1:
- The condition is relaxed
- More neighbors are kept (less aggressive pruning)
- Graph is denser but has shorter diameter

## Comparison with HNSW Heuristic

| Aspect | Vamana RobustPrune | HNSW getNeighborsByHeuristic2 |
|--------|-------------------|------------------------------|
| Condition | `α · d(p*, p') ≤ d(p, p')` | `d(p*, p') < d(q, p')` |
| α parameter | Yes (controls density/diameter tradeoff) | No |
| Comparison | Uses actual distance to node p | Uses distance to query q |
| Theoretical guarantee | O(log n) diameter with α > 1 | No formal diameter bound |

### HNSW Heuristic (for reference):
```cpp
for (auto& selected : return_list) {
    dist_t dist_to_selected = distance(current, selected);
    if (dist_to_selected < dist_to_query) {  // Strict <, no α
        good = false;
        break;
    }
}
```

This is simpler but doesn't provide the same theoretical guarantees.

## Complexity Analysis

Both algorithms are O(R × |V|) where R = max_edges, |V| = candidates:
- While loop runs at most R times (we select at most R neighbors)
- Each iteration scans remaining candidates in V

This is NOT O(|V|²) because:
1. We only compare to already-selected neighbors
2. Candidates are progressively removed from V

## Implementation in CoreNN

```rust
fn prune_candidates(&self, node: &VecData, candidate_ids: &[Id]) -> Vec<Id> {
    let max_edges = self.cfg.max_edges;
    let alpha = self.cfg.distance_threshold;  // α parameter

    // ... get sorted candidates ...

    let mut selected: Vec<Id> = Vec::with_capacity(max_edges);
    let mut remaining: VecDeque<Point> = candidates.into();
    
    while let Some(p_star) = remaining.pop_front() {
        selected.push(p_star.id);
        
        if selected.len() >= max_edges {
            break;
        }
        
        // α-RNG condition: keep if d(node, p') < α · d(p*, p')
        remaining.retain(|p_prime| {
            let dist_node_to_candidate = p_prime.dist.0;
            let dist_selected_to_candidate = p_star.dist(p_prime);
            dist_node_to_candidate < alpha * dist_selected_to_candidate
        });
    }
    
    selected
}
```

## Recommendations

1. **Default α = 1.2** as recommended in DiskANN paper
2. **Don't blindly replace with HNSW heuristic** - different theoretical properties
3. **For in-memory only**: α closer to 1.0 may be fine (smaller graph)
4. **For disk-based**: α ≥ 1.2 is important for bounded latency

## References

1. DiskANN: "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node" (NeurIPS 2019)
2. FreshDiskANN: "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search" (2021)
3. HNSW: "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (2016)
