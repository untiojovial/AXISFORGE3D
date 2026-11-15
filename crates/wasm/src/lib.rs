//! The engine as a WebAssembly module.
//!
//! The interface is three functions over linear memory and nothing else — no
//! imports, no generated glue — so the module loads with a bare
//! `WebAssembly.instantiate` and can be embedded in a single HTML file:
//!
//! - `af_alloc(len) -> ptr` reserves `len` bytes for the caller to fill;
//! - `af_call(ptr, len) -> packed` consumes a UTF-8 JSON request written
//!   there and returns the reply's location as `(ptr << 32) | len`;
//! - `af_free(ptr, len)` releases a reply once the caller has read it.
//!
//! Requests and replies are those of [`axisforge_service::Session::handle_json`].
//! The module holds one session, created on first use with the demo cell.

// Exporting unmangled symbols and exchanging raw buffers with the host is
// inherently unsafe; it is confined to the three functions below, each of
// which states the contract it relies on.
#![deny(unsafe_code)]
#![warn(missing_docs)]

use std::cell::RefCell;

use axisforge_service::Session;

thread_local! {
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
}

fn handle(request: &str) -> String {
    SESSION.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            match Session::demo() {
                Ok(session) => *slot = Some(session),
                Err(error) => {
                    return format!(
                        "{{\"ok\":false,\"error\":{:?}}}",
                        format!("could not start the engine: {error}")
                    )
                }
            }
        }
        match slot.as_mut() {
            Some(session) => session.handle_json(request),
            None => "{\"ok\":false,\"error\":\"engine unavailable\"}".to_owned(),
        }
    })
}

/// Reserves `len` bytes and returns their address.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "C" fn af_alloc(len: usize) -> *mut u8 {
    let mut buffer = vec![0_u8; len].into_boxed_slice();
    let pointer = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    pointer
}

/// Releases a buffer obtained from [`af_alloc`] or returned by [`af_call`].
///
/// # Safety
/// `ptr` and `len` must be exactly those of one such buffer, not yet freed.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn af_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        // SAFETY: per the contract, this is a live boxed slice of `len` bytes.
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) });
    }
}

/// Handles the JSON request at `ptr`/`len` and returns the reply's location.
///
/// The request buffer is consumed. The reply is `(ptr << 32) | len`; the
/// caller reads it and releases it with [`af_free`].
///
/// # Safety
/// `ptr` and `len` must be those of a buffer from [`af_alloc`], not yet freed.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn af_call(ptr: *mut u8, len: usize) -> u64 {
    // SAFETY: per the contract, this is a live boxed slice of `len` bytes,
    // and taking ownership here is what "consumed" means.
    let request = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) };
    let reply = match std::str::from_utf8(&request) {
        Ok(text) => handle(text),
        Err(_) => "{\"ok\":false,\"error\":\"request is not UTF-8\"}".to_owned(),
    };
    let mut bytes = reply.into_bytes().into_boxed_slice();
    let (reply_ptr, reply_len) = (bytes.as_mut_ptr() as u64, bytes.len() as u64);
    std::mem::forget(bytes);
    (reply_ptr << 32) | reply_len
}
