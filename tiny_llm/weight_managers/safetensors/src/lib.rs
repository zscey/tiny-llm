use memmap2::Mmap;
use safetensors::SafeTensors;
use std::fs::File;
use std::ffi::CStr;
use std::panic::catch_unwind;
use std::os::raw::c_char;

#[repr(C)]
pub struct SliceView {
    pub data: *const u8,
    pub len: usize,
}

pub struct SafeTensorContext {
    _file_handle: File,
    _mmap: Mmap,
    pub tensors: SafeTensors<'static>,
}

#[no_mangle]
pub unsafe extern "C" fn safetensor_init(path: *const c_char) -> *mut SafeTensorContext {
    let init_internal = || -> Option<*mut SafeTensorContext> {
        if path.is_null() {
            return None;
        }

        let path_str = unsafe { CStr::from_ptr(path) }.to_str().ok()?;
        let file = File::open(path_str).ok()?;
        let mmap = Mmap::map(&file).ok()?;

        let tensors = unsafe {
            let t = SafeTensors::deserialize(&mmap).ok()?;
            std::mem::transmute::<SafeTensors<'_>, SafeTensors<'static>>(t)
        };

        let ctx = Box::new(SafeTensorContext {
            _file_handle: file,
            _mmap: mmap,
            tensors,
        });

        Some(Box::into_raw(ctx))
    };

    let result = catch_unwind(|| {
        init_internal()
    });

    match result {
        Ok(maybe_ctx) => maybe_ctx.unwrap_or(std::ptr::null_mut()),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn safetensor_get_tensor(
    ctx: *mut SafeTensorContext,
    name: *const c_char,
) -> SliceView {
    let get_internal = || -> Option<SliceView> {
        if ctx.is_null() || name.is_null() {
            return None;
        }

        let name_str = unsafe { CStr::from_ptr(name) }.to_str().ok()?;
        let t = unsafe { (*ctx).tensors.tensor(name_str) }.ok()?;

        Some(SliceView {
            data: t.data().as_ptr(),
            len: t.data().len(),
        })
    };

    let result = catch_unwind(|| {
        get_internal()
    });

    match result {
        Ok(maybe_slice_view) => maybe_slice_view.unwrap_or(SliceView{
            data: std::ptr::null(),
            len: 0,
        }),
        Err(_) => SliceView {
            data: std::ptr::null(),
            len: 0,
        },
    }
}

#[no_mangle]
pub unsafe extern "C" fn safetensor_free(ctx: *mut SafeTensorContext) {
    if ctx.is_null() {
        return;
    }

    let _ = catch_unwind(|| {
        let _ctx = unsafe { Box::from_raw(ctx) };
    });
}
