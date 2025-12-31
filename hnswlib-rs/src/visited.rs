use parking_lot::Mutex;

pub type VisitTag = u16;

#[derive(Debug)]
pub struct VisitedList {
  cur_v: VisitTag,
  pub mass: Vec<VisitTag>,
}

impl VisitedList {
  pub fn new(num_elements: usize) -> Self {
    Self {
      cur_v: VisitTag::MAX,
      mass: vec![0; num_elements],
    }
  }

  pub fn reset(&mut self) -> VisitTag {
    self.cur_v = self.cur_v.wrapping_add(1);
    if self.cur_v == 0 {
      self.mass.fill(0);
      self.cur_v = 1;
    }
    self.cur_v
  }
}

#[derive(Debug)]
pub struct VisitedListPool {
  pool: Mutex<Vec<VisitedList>>,
  num_elements: usize,
}

impl VisitedListPool {
  pub fn new(initial_pool_size: usize, num_elements: usize) -> Self {
    let mut pool = Vec::with_capacity(initial_pool_size);
    for _ in 0..initial_pool_size {
      pool.push(VisitedList::new(num_elements));
    }
    Self {
      pool: Mutex::new(pool),
      num_elements,
    }
  }

  pub fn resize(&mut self, initial_pool_size: usize, num_elements: usize) {
    *self = Self::new(initial_pool_size, num_elements);
  }

  pub fn get(&self) -> VisitedListHandle<'_> {
    let mut pool = self.pool.lock();
    let mut list = pool
      .pop()
      .unwrap_or_else(|| VisitedList::new(self.num_elements));
    let tag = list.reset();
    VisitedListHandle {
      pool: &self.pool,
      list: Some(list),
      tag,
    }
  }
}

pub struct VisitedListHandle<'a> {
  pool: &'a Mutex<Vec<VisitedList>>,
  list: Option<VisitedList>,
  pub tag: VisitTag,
}

impl<'a> VisitedListHandle<'a> {
  pub fn mass_mut(&mut self) -> &mut [VisitTag] {
    &mut self.list.as_mut().unwrap().mass
  }
}

impl Drop for VisitedListHandle<'_> {
  fn drop(&mut self) {
    if let Some(list) = self.list.take() {
      self.pool.lock().push(list);
    }
  }
}
