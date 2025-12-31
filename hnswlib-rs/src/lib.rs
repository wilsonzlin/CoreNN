pub mod bruteforce;
pub mod error;
pub mod hnsw;
pub mod space;
pub mod stop_condition;
pub mod view;
pub mod visited;

pub use bruteforce::BruteforceIndex;
pub use error::Error;
pub use error::Result;
pub use hnsw::HnswIndex;
pub use space::InnerProductSpace;
pub use space::L2Space;
pub use space::Space;
pub use stop_condition::EpsilonSearchStopCondition;
pub use stop_condition::MultiVectorSearchStopCondition;
pub use stop_condition::SearchStopCondition;
pub use view::HnswIndexView;

pub type TableInt = u32; // Internal ID.
pub type LinkListSizeInt = u32;
pub type LabelType = usize; // External ID.
