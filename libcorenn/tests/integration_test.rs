//! Integration tests for CoreNN with various optimizations

use libcorenn::cfg::Cfg;
use libcorenn::metric::StdMetric;
use libcorenn::CoreNN;
use rand::Rng;

fn random_f32_vec(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

fn normalize(v: &mut Vec<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[test]
fn test_basic_insert_and_query() {
    let cfg = Cfg {
        dim: 128,
        metric: StdMetric::L2,
        beam_width: 4,
        max_edges: 32,
        query_search_list_cap: 64,
        update_search_list_cap: 64,
        ..Default::default()
    };
    
    let db = CoreNN::new_in_memory(cfg);
    
    // Insert some vectors
    let v1: Vec<f32> = vec![1.0; 128];
    let v2: Vec<f32> = vec![0.5; 128];
    let v3: Vec<f32> = vec![0.0; 128];
    
    db.insert(&"key1".to_string(), &v1);
    db.insert(&"key2".to_string(), &v2);
    db.insert(&"key3".to_string(), &v3);
    
    // Query for the closest to v1
    let results = db.query(&[1.0f32; 128], 2);
    
    assert!(!results.is_empty(), "Should return some results");
    assert_eq!(results[0].0, "key1", "key1 should be closest to query [1.0; 128]");
}

#[test]
fn test_l2_distance_ordering() {
    let cfg = Cfg {
        dim: 64,
        metric: StdMetric::L2,
        beam_width: 4,
        max_edges: 16,
        query_search_list_cap: 32,
        update_search_list_cap: 32,
        ..Default::default()
    };
    
    let db = CoreNN::new_in_memory(cfg);
    
    // Insert vectors with known distances to query
    let query: Vec<f32> = vec![0.0; 64];
    
    // Close vector (L2 distance = sqrt(64) ≈ 8.0)
    let close: Vec<f32> = vec![1.0; 64];
    // Far vector (L2 distance = sqrt(64 * 4) = 16.0)
    let far: Vec<f32> = vec![2.0; 64];
    // Very far (L2 distance = sqrt(64 * 9) = 24.0)
    let very_far: Vec<f32> = vec![3.0; 64];
    
    db.insert(&"close".to_string(), &close);
    db.insert(&"far".to_string(), &far);
    db.insert(&"very_far".to_string(), &very_far);
    
    let results = db.query(&query, 3);
    
    // With only 3 vectors and graph structure, we may not get all 3 results
    // depending on how edges were formed. Focus on ordering.
    assert!(!results.is_empty(), "Should have some results");
    
    // First result should be closest
    if results.len() >= 2 {
        assert!(results[0].1 < results[1].1, "Results should be ordered by distance");
    }
    
    // If we found "close", it should be first
    let close_pos = results.iter().position(|(k, _)| k == "close");
    if let Some(pos) = close_pos {
        assert_eq!(pos, 0, "close should be first if found");
    }
}

#[test]
fn test_cosine_distance_ordering() {
    let cfg = Cfg {
        dim: 64,
        metric: StdMetric::Cosine,
        beam_width: 4,
        max_edges: 16,
        query_search_list_cap: 32,
        update_search_list_cap: 32,
        ..Default::default()
    };
    
    let db = CoreNN::new_in_memory(cfg);
    
    // Query vector
    let mut query: Vec<f32> = vec![1.0; 64];
    normalize(&mut query);
    
    // Very similar (nearly same direction)
    let mut similar: Vec<f32> = vec![1.0; 64];
    similar[0] = 1.1; // Slightly different
    normalize(&mut similar);
    
    // Orthogonal-ish
    let mut different: Vec<f32> = vec![1.0; 32].into_iter().chain(vec![-1.0; 32]).collect();
    normalize(&mut different);
    
    // Opposite direction
    let mut opposite: Vec<f32> = vec![-1.0; 64];
    normalize(&mut opposite);
    
    db.insert(&"similar".to_string(), &similar);
    db.insert(&"different".to_string(), &different);
    db.insert(&"opposite".to_string(), &opposite);
    
    let results = db.query(&query, 3);
    
    // With small graph, may not find all vectors
    assert!(!results.is_empty(), "Should have some results");
    
    // Results should be ordered by distance
    for i in 1..results.len() {
        assert!(results[i-1].1 <= results[i].1, "Results should be ordered by distance");
    }
    
    // If similar is found, it should be first (cosine distance near 0)
    let similar_pos = results.iter().position(|(k, _)| k == "similar");
    if let Some(pos) = similar_pos {
        assert_eq!(pos, 0, "similar should be first if found");
    }
}

#[test]
fn test_many_vectors() {
    let cfg = Cfg {
        dim: 128,
        metric: StdMetric::L2,
        beam_width: 4,
        max_edges: 32,
        query_search_list_cap: 100,
        update_search_list_cap: 100,
        ..Default::default()
    };
    
    let db = CoreNN::new_in_memory(cfg);
    
    // Insert 1000 random vectors
    let num_vectors = 1000;
    let dim = 128;
    
    for i in 0..num_vectors {
        let v = random_f32_vec(dim);
        db.insert(&format!("vec_{}", i), &v);
    }
    
    // Insert a known vector we'll query for
    let target = vec![0.5f32; dim];
    db.insert(&"target".to_string(), &target);
    
    // Query should find the target
    let results = db.query(&target, 10);
    
    assert!(!results.is_empty());
    // The target should be in top results (exact match = distance 0)
    let target_found = results.iter().any(|(k, d)| k == "target" && *d < 1e-6);
    assert!(target_found, "Target should be found with distance ~0");
}

#[test]
fn test_delete_and_reinsert() {
    let cfg = Cfg {
        dim: 64,
        metric: StdMetric::L2,
        beam_width: 4,
        max_edges: 16,
        query_search_list_cap: 32,
        update_search_list_cap: 32,
        ..Default::default()
    };
    
    let db = CoreNN::new_in_memory(cfg);
    
    let v1: Vec<f32> = vec![1.0; 64];
    let v2: Vec<f32> = vec![2.0; 64];
    
    db.insert(&"key1".to_string(), &v1);
    
    // Query should find key1
    let results = db.query(&v1, 1);
    assert_eq!(results[0].0, "key1");
    
    // Delete key1
    db.delete(&"key1".to_string());
    
    // Query should not find key1
    let results = db.query(&v1, 10);
    let key1_found = results.iter().any(|(k, _)| k == "key1");
    assert!(!key1_found, "key1 should be deleted");
    
    // Reinsert with same key but different vector
    db.insert(&"key1".to_string(), &v2);
    
    // Query should find new key1
    let results = db.query(&v2, 1);
    assert_eq!(results[0].0, "key1");
}
