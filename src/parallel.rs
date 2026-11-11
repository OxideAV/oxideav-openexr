//! Worker budget for the chunk-parallel readers and writers.
//!
//! Chunks (scanline blocks, tiles) are independent, so the single-part
//! readers and every writer fan them out over a caller-chosen number of
//! workers. The budget is `1` (serial) unless the caller raises it:
//! [`crate::DecodeOptions::threads`] / [`crate::EncodeOptions::threads`]
//! for the contract API, the framework's execution context for the
//! registry decoder / encoder, or [`with_threads`] around any call of
//! the depth API. The result — decoded samples, written bytes, and the
//! error on malformed input — is the same at every budget.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

thread_local! {
    static BUDGET: Cell<usize> = const { Cell::new(1) };
}

/// Run `f` with a budget of `threads` workers (at least 1) for the
/// chunk-parallel readers and writers called inside it on this thread.
/// The previous budget is restored afterwards; worker threads
/// themselves run with a budget of 1.
pub fn with_threads<R>(threads: usize, f: impl FnOnce() -> R) -> R {
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            BUDGET.with(|b| b.set(self.0));
        }
    }
    let _restore = Restore(BUDGET.with(|b| b.replace(threads.max(1))));
    f()
}

/// The budget in force on this thread.
pub(crate) fn budget() -> usize {
    BUDGET.with(Cell::get)
}

/// `(0..n).map(f)` over up to [`budget`] workers, results in index order.
pub(crate) fn map_indexed<T, F>(n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    let workers = budget().min(n);
    if workers <= 1 {
        return (0..n).map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let mut slots: Vec<Option<T>> = (0..n).map(|_| None).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break done;
                        }
                        done.push((i, f(i)));
                    }
                })
            })
            .collect();
        for handle in handles {
            match handle.join() {
                Ok(done) => {
                    for (i, value) in done {
                        slots[i] = Some(value);
                    }
                }
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
    });
    slots
        .into_iter()
        .map(|slot| slot.expect("every index is claimed by exactly one worker"))
        .collect()
}

/// [`map_indexed`] for fallible work: the error of the lowest failing
/// index, as a serial loop would report it.
pub(crate) fn try_map_indexed<T, E, F>(n: usize, f: F) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
    F: Fn(usize) -> Result<T, E> + Sync,
{
    if budget().min(n) <= 1 {
        return (0..n).map(f).collect();
    }
    map_indexed(n, f).into_iter().collect()
}

/// Call `f(index, chunk)` for every `chunk_len`-sized piece of `data`
/// (the last may be shorter) over up to [`budget`] workers. Each piece
/// is visited once, so the outcome does not depend on the budget.
pub(crate) fn for_each_chunk_mut<T, F>(data: &mut [T], chunk_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let chunk_len = chunk_len.max(1);
    let workers = budget().min(data.len().div_ceil(chunk_len));
    if workers <= 1 {
        for (i, chunk) in data.chunks_mut(chunk_len).enumerate() {
            f(i, chunk);
        }
        return;
    }
    let queue = std::sync::Mutex::new(data.chunks_mut(chunk_len).enumerate());
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    loop {
                        let next = queue.lock().unwrap_or_else(|e| e.into_inner()).next();
                        match next {
                            Some((i, chunk)) => f(i, chunk),
                            None => break,
                        }
                    }
                })
            })
            .collect();
        for handle in handles {
            if let Err(panic) = handle.join() {
                std::panic::resume_unwind(panic);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_is_scoped_and_serial_by_default() {
        assert_eq!(budget(), 1);
        with_threads(6, || {
            assert_eq!(budget(), 6);
            with_threads(0, || assert_eq!(budget(), 1));
            assert_eq!(budget(), 6);
            // Workers do not fan out again.
            assert!(map_indexed(4, |_| budget()).iter().all(|&b| b == 1));
        });
        assert_eq!(budget(), 1);
    }

    #[test]
    fn results_keep_index_order_at_every_budget() {
        let want: Vec<usize> = (0..257).map(|i| i * i).collect();
        for threads in [1, 2, 3, 8, 64] {
            assert_eq!(with_threads(threads, || map_indexed(257, |i| i * i)), want);
        }
        assert!(with_threads(4, || map_indexed(0, |i| i)).is_empty());
    }

    #[test]
    fn chunks_are_each_visited_once() {
        for threads in [1, 3, 8] {
            let mut data = vec![0u32; 1000];
            with_threads(threads, || {
                for_each_chunk_mut(&mut data, 37, |i, chunk| {
                    for (k, v) in chunk.iter_mut().enumerate() {
                        *v += (i * 37 + k) as u32 + 1;
                    }
                })
            });
            assert!(data.iter().enumerate().all(|(i, &v)| v == i as u32 + 1));
        }
    }

    #[test]
    fn the_lowest_failing_index_is_reported() {
        for threads in [1, 4] {
            let r: Result<Vec<usize>, usize> = with_threads(threads, || {
                try_map_indexed(100, |i| if i % 7 == 5 { Err(i) } else { Ok(i) })
            });
            assert_eq!(r, Err(5));
        }
    }
}
