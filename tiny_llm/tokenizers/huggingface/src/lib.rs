use tokenizers::tokenizer::Tokenizer;
use std::os::raw::c_char;
use std::panic::catch_unwind;
use std::ffi::{CStr, CString};

pub struct TokenizerHandle {
    pub tokenizer: Tokenizer,
}

#[repr(C)]
pub struct TokenizerEncodeResult {
    pub token_ids: *mut u32,
    pub len: usize,
}

#[repr(C)]
pub struct TokenizerDecodeResult {
    pub token: *mut c_char,
    pub len: usize,
}

#[no_mangle]
pub unsafe extern "C" fn tokenizer_init(path: *const c_char) -> *mut TokenizerHandle {
    let result = catch_unwind(|| -> Option<*mut TokenizerHandle> {
        if path.is_null() {
            return None;
        }

        let path_str = CStr::from_ptr(path).to_str().ok()?;
        let t = Tokenizer::from_file(path_str).ok()?;

        Some(Box::into_raw(Box::new(TokenizerHandle {tokenizer: t})))
    });

    match result {
        Ok(handle) => handle.unwrap_or(std::ptr::null_mut()),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn tokenizer_free(handle: *mut TokenizerHandle) {
    if handle.is_null() {
        return;
    }

    let _ = catch_unwind(|| {
        let _handle = Box::from_raw(handle);
    });
}

#[no_mangle]
pub unsafe extern "C" fn tokenizer_encode(
    handle: *mut TokenizerHandle,
    text: *const c_char,
    add_special_tokens: bool
) -> TokenizerEncodeResult {
    let fallback = TokenizerEncodeResult {
        token_ids: std::ptr::null_mut(),
        len: 0,
    };

    let result = catch_unwind(|| -> Option<TokenizerEncodeResult> {
        if handle.is_null() || text.is_null() {
            return None;
        }

        let input = CStr::from_ptr(text).to_str().ok()?;
        let encode_res = (*handle).tokenizer.encode(input, add_special_tokens).ok()?;
        let ids = encode_res.get_ids().to_vec().into_boxed_slice();
        let len = ids.len();

        Some(TokenizerEncodeResult {
            token_ids: Box::into_raw(ids) as *mut u32,
            len: len,
        })
    });

    match result {
        Ok(res) => res.unwrap_or(fallback),
        Err(_) => fallback,
    }
}

#[no_mangle]
pub unsafe extern "C" fn tokenizer_free_encode_result(res: TokenizerEncodeResult) {
    if res.token_ids.is_null() {
        return;
    }

    let _ = catch_unwind(|| {
        let raw_slice_ptr = std::ptr::slice_from_raw_parts_mut(res.token_ids, res.len);
        let _ = Box::from_raw(raw_slice_ptr);
    });
}

#[no_mangle]
pub unsafe extern "C" fn tokenizer_decode(
    handle: *mut TokenizerHandle,
    ids: *const u32,
    len: usize,
    skip_special_tokens: bool
) -> TokenizerDecodeResult {
    let fallback = TokenizerDecodeResult {
        token: std::ptr::null_mut(),
        len: 0,
    };

    let result = catch_unwind(|| -> Option<TokenizerDecodeResult> {
        if handle.is_null() || ids.is_null() {
            return None;
        }

        let raw_slice_ptr = std::ptr::slice_from_raw_parts(ids, len);
        let decode_res = (*handle).tokenizer.decode(&*raw_slice_ptr, skip_special_tokens).ok()?;
        let c_string = CString::new(decode_res).ok()?;
        let len = c_string.count_bytes();

        Some(TokenizerDecodeResult{
            token: c_string.into_raw(),
            len: len,
        })
    });

    match result {
        Ok(res) => res.unwrap_or(fallback),
        Err(_) => fallback,
    }
}

#[no_mangle]
pub unsafe extern "C" fn tokenizer_free_decode_result(res: TokenizerDecodeResult) {
    if res.token.is_null() {
        return;
    }

    let _ = catch_unwind(||{
        let _ = CString::from_raw(res.token);
    });
}

#[no_mangle]
pub unsafe extern "C" fn tokenizer_vocab_size(handle: *mut TokenizerHandle) -> usize {
    (*handle).tokenizer.get_vocab_size(true)
}
