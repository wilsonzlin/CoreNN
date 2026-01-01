#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(u32);

impl NodeId {
  #[inline]
  pub fn as_u32(self) -> u32 {
    self.0
  }

  #[inline]
  pub fn as_usize(self) -> usize {
    self.0 as usize
  }

  #[inline]
  pub(crate) fn new(raw: u32) -> Self {
    Self(raw)
  }
}

