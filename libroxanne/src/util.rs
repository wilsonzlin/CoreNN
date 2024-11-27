use std::{collections::VecDeque, convert::identity};

pub fn insert_into_ordered_vecdeque<T: Clone, K: Ord>(
  dest: &mut VecDeque<T>,
  src: &[T],
  key: impl Fn(&T) -> K,
) {
  for v in src.into_iter() {
    // WARNING: We can't collect all positions and then insert by index descending, as that doesn't account for when multiple source values are inserted at the same position but between themselves are not sorted. This costs the same anyway because we need to do N insertions.
    let pos = dest
      .binary_search_by(|s| key(s).cmp(&key(v)))
      .map_or_else(identity, identity);
    dest.insert(pos, v.clone());
  }
}

#[cfg(test)]
mod tests {
  #[test]
  fn test_insert_into_ordered_vecdeque() {
    // Empty destination.
    let mut dest = VecDeque::new();
    insert_into_ordered_vecdeque(&mut dest, &[3, 1, 4], |&x| x);
    assert_eq!(dest.into_iter().collect::<Vec<_>>(), vec![1, 3, 4]);

    // Empty source.
    let mut dest = VecDeque::from(vec![1, 2, 3]);
    insert_into_ordered_vecdeque(&mut dest, &[], |&x| x);
    assert_eq!(dest.into_iter().collect::<Vec<_>>(), vec![1, 2, 3]);

    // Both empty.
    let mut dest = VecDeque::<usize>::new();
    insert_into_ordered_vecdeque(&mut dest, &[], |&x| x);
    assert!(dest.is_empty());

    // Interleaved values.
    let mut dest = VecDeque::from(vec![2, 4, 6]);
    insert_into_ordered_vecdeque(&mut dest, &[1, 3, 5], |&x| x);
    assert_eq!(dest.into_iter().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5, 6]);

    // All duplicates.
    let mut dest = VecDeque::from(vec![1, 1, 1]);
    insert_into_ordered_vecdeque(&mut dest, &[1, 1], |&x| x);
    assert_eq!(dest.into_iter().collect::<Vec<_>>(), vec![1, 1, 1, 1, 1]);

    // All source elements larger.
    let mut dest = VecDeque::from(vec![1, 2, 3]);
    insert_into_ordered_vecdeque(&mut dest, &[4, 5, 6], |&x| x);
    assert_eq!(dest.into_iter().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5, 6]);

    // All source elements smaller.
    let mut dest = VecDeque::from(vec![4, 5, 6]);
    insert_into_ordered_vecdeque(&mut dest, &[1, 2, 3], |&x| x);
    assert_eq!(dest.into_iter().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5, 6]);

    // Custom key function.
    let mut dest = VecDeque::from(vec!["aa", "cccc"]);
    insert_into_ordered_vecdeque(&mut dest, &["bbb", "d"], |s| s.len());
    assert_eq!(dest.into_iter().collect::<Vec<_>>(), vec![
      "d", "aa", "bbb", "cccc"
    ]);

    // Single element source and dest.
    let mut dest = VecDeque::from(vec![2]);
    insert_into_ordered_vecdeque(&mut dest, &[1], |&x| x);
    assert_eq!(dest.into_iter().collect::<Vec<_>>(), vec![1, 2]);
  }
}
