//! Public workspace, caller-owned scratch, and transactional admission contracts.

extern crate alloc;

use mp_ntt::{Limb, Ntt, NttPlan, NttWorkspace, SequentialExecutor};

mod multiplication;
#[path = "../../src/tests/schoolbook.rs"]
mod schoolbook;
mod squaring;
