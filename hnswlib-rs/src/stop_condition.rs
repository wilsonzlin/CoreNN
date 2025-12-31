use crate::LabelType;
use ahash::HashMap;
use ahash::HashMapExt;
use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;
use std::hash::Hash;

pub trait SearchStopCondition {
  fn add_point_to_result(&mut self, label: LabelType, datapoint: &[f32], dist: f32);
  fn remove_point_from_result(&mut self, label: LabelType, datapoint: &[f32], dist: f32);

  fn should_stop_search(&self, candidate_dist: f32, lower_bound: f32) -> bool;
  fn should_consider_candidate(&self, candidate_dist: f32, lower_bound: f32) -> bool;
  fn should_remove_extra(&self) -> bool;

  fn filter_results(&mut self, candidates: &mut Vec<(LabelType, f32)>);
}

#[derive(Debug)]
pub struct MultiVectorSearchStopCondition<DOCIDTYPE, F>
where
  DOCIDTYPE: Copy + Eq + Hash + Ord,
  F: Fn(LabelType, &[f32]) -> DOCIDTYPE,
{
  curr_num_docs: usize,
  num_docs_to_search: usize,
  ef_collection: usize,
  doc_counter: HashMap<DOCIDTYPE, usize>,
  search_results: BinaryHeap<(OrderedFloat<f32>, DOCIDTYPE)>,
  doc_id_for: F,
}

impl<DOCIDTYPE, F> MultiVectorSearchStopCondition<DOCIDTYPE, F>
where
  DOCIDTYPE: Copy + Eq + Hash + Ord,
  F: Fn(LabelType, &[f32]) -> DOCIDTYPE,
{
  pub fn new(doc_id_for: F, num_docs_to_search: usize, ef_collection: usize) -> Self {
    let ef_collection = ef_collection.max(num_docs_to_search);
    Self {
      curr_num_docs: 0,
      num_docs_to_search,
      ef_collection,
      doc_counter: HashMap::new(),
      search_results: BinaryHeap::new(),
      doc_id_for,
    }
  }
}

impl<DOCIDTYPE, F> SearchStopCondition for MultiVectorSearchStopCondition<DOCIDTYPE, F>
where
  DOCIDTYPE: Copy + Eq + Hash + Ord,
  F: Fn(LabelType, &[f32]) -> DOCIDTYPE,
{
  fn add_point_to_result(&mut self, label: LabelType, datapoint: &[f32], dist: f32) {
    let doc_id = (self.doc_id_for)(label, datapoint);
    let entry = self.doc_counter.entry(doc_id).or_insert(0);
    if *entry == 0 {
      self.curr_num_docs += 1;
    }
    *entry += 1;
    self.search_results.push((OrderedFloat(dist), doc_id));
  }

  fn remove_point_from_result(&mut self, label: LabelType, datapoint: &[f32], _dist: f32) {
    let doc_id = (self.doc_id_for)(label, datapoint);
    let Some(entry) = self.doc_counter.get_mut(&doc_id) else {
      return;
    };
    *entry -= 1;
    if *entry == 0 {
      self.curr_num_docs -= 1;
    }
    self.search_results.pop();
  }

  fn should_stop_search(&self, candidate_dist: f32, lower_bound: f32) -> bool {
    candidate_dist > lower_bound && self.curr_num_docs == self.ef_collection
  }

  fn should_consider_candidate(&self, candidate_dist: f32, lower_bound: f32) -> bool {
    self.curr_num_docs < self.ef_collection || lower_bound > candidate_dist
  }

  fn should_remove_extra(&self) -> bool {
    self.curr_num_docs > self.ef_collection
  }

  fn filter_results(&mut self, candidates: &mut Vec<(LabelType, f32)>) {
    while self.curr_num_docs > self.num_docs_to_search {
      let dist_cand = candidates.last().unwrap().1;
      let dist_res = self.search_results.peek().unwrap().0 .0;
      debug_assert_eq!(dist_cand, dist_res);
      let doc_id = self.search_results.peek().unwrap().1;
      let entry = self.doc_counter.get_mut(&doc_id).unwrap();
      *entry -= 1;
      if *entry == 0 {
        self.curr_num_docs -= 1;
      }
      self.search_results.pop();
      candidates.pop();
    }
  }
}

#[derive(Debug, Clone)]
pub struct EpsilonSearchStopCondition {
  epsilon: f32,
  min_num_candidates: usize,
  max_num_candidates: usize,
  curr_num_items: usize,
}

impl EpsilonSearchStopCondition {
  pub fn new(epsilon: f32, min_num_candidates: usize, max_num_candidates: usize) -> Self {
    assert!(min_num_candidates <= max_num_candidates);
    Self {
      epsilon,
      min_num_candidates,
      max_num_candidates,
      curr_num_items: 0,
    }
  }
}

impl SearchStopCondition for EpsilonSearchStopCondition {
  fn add_point_to_result(&mut self, _label: LabelType, _datapoint: &[f32], _dist: f32) {
    self.curr_num_items += 1;
  }

  fn remove_point_from_result(&mut self, _label: LabelType, _datapoint: &[f32], _dist: f32) {
    self.curr_num_items -= 1;
  }

  fn should_stop_search(&self, candidate_dist: f32, lower_bound: f32) -> bool {
    if candidate_dist > lower_bound && self.curr_num_items == self.max_num_candidates {
      return true;
    }
    if candidate_dist > self.epsilon && self.curr_num_items >= self.min_num_candidates {
      return true;
    }
    false
  }

  fn should_consider_candidate(&self, candidate_dist: f32, lower_bound: f32) -> bool {
    self.curr_num_items < self.max_num_candidates || lower_bound > candidate_dist
  }

  fn should_remove_extra(&self) -> bool {
    self.curr_num_items > self.max_num_candidates
  }

  fn filter_results(&mut self, candidates: &mut Vec<(LabelType, f32)>) {
    while candidates
      .last()
      .is_some_and(|(_, dist)| *dist > self.epsilon)
    {
      candidates.pop();
    }
    while candidates.len() > self.max_num_candidates {
      candidates.pop();
    }
  }
}
