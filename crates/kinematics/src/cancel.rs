//! Cooperative cancellation for long computations.
//!
//! Inverse kinematics with random restarts, a trajectory sweep or a dense
//! collision scan can all run long enough that a user changes their mind. The
//! engine therefore never runs such a job to completion on principle: it checks
//! a token between iterations and returns what it has. The token is a shared
//! flag rather than a channel because the only thing the worker needs to learn
//! is whether to stop, and the only thing the caller needs is to be able to say
//! so from another thread.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A flag a caller can set to ask a running computation to stop.
///
/// Cloning gives another handle to the *same* flag, so a worker and its caller
/// can each hold one.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

impl CancellationToken {
    /// A token that has not been cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// A token that is already cancelled, for tests and for a job abandoned
    /// before it started.
    pub fn cancelled() -> Self {
        let token = Self::new();
        token.cancel();
        token
    }

    /// Asks any computation holding this token to stop at its next check.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }

    /// `true` once [`CancellationToken::cancel`] has been called.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

/// Why a computation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Completion {
    /// It ran to its own conclusion.
    Finished,
    /// A caller cancelled it; any result is the best found so far.
    Cancelled,
}

impl Completion {
    /// `true` when the computation was not interrupted.
    pub fn is_finished(self) -> bool {
        matches!(self, Self::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clone_shares_the_same_flag() {
        let token = CancellationToken::new();
        let worker = token.clone();
        assert!(!worker.is_cancelled());
        token.cancel();
        assert!(
            worker.is_cancelled(),
            "cancelling one handle must cancel them all"
        );
    }

    #[test]
    fn cancelling_is_visible_across_threads() {
        let token = CancellationToken::new();
        let worker = token.clone();
        let handle = std::thread::spawn(move || {
            let mut spins = 0_u64;
            while !worker.is_cancelled() && spins < 100_000_000 {
                spins += 1;
            }
            worker.is_cancelled()
        });
        token.cancel();
        assert!(
            handle.join().expect("worker thread"),
            "the worker must observe the cancellation"
        );
    }
}
