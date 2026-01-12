use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::Dtype;
use std::fs::File;
use std::ffi::CStr;
use std::panic::catch_unwind;
use std::os::raw::c_char;

#[repr(C)]
pub struct SliceView {
    pub dtype: u32,
    pub shape: [usize; 8],
    pub shape_dim: usize,
    pub data: *const u8,
    pub data_len: usize,
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

fn dtype_to_u32(dtype: Dtype) -> u32 {
    match dtype {
        Dtype::BOOL => 0,
        Dtype::F4 => 1,
        Dtype::F6_E2M3 => 2,
        Dtype::F6_E3M2 => 3,
        Dtype::U8 => 4,
        Dtype::I8 => 5,
        Dtype::F8_E5M2 => 6,
        Dtype::F8_E4M3 => 7,
        Dtype::F8_E8M0 => 8,
        Dtype::I16 => 9,
        Dtype::U16 => 10,
        Dtype::F16 => 11,
        Dtype::BF16 => 12,
        Dtype::I32 => 13,
        Dtype::U32 => 14,
        Dtype::F32 => 15,
        Dtype::C64 => 16,
        Dtype::F64 => 17,
        Dtype::I64 => 18,
        Dtype::U64 => 19,
        _ => 999, // Unknown
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

        let t_shape = t.shape();
        let shape_dim = t_shape.len();
        if shape_dim > 8 {
            return None;
        }

        let mut shape = [0usize; 8];
        shape[..shape_dim].copy_from_slice(t_shape);

        Some(SliceView {
            dtype: dtype_to_u32(t.dtype()),
            shape,
            shape_dim,
            data: t.data().as_ptr(),
            data_len: t.data().len(),
        })
    };

    let result = catch_unwind(|| {
        get_internal()
    });

    match result {
        Ok(maybe_slice_view) => maybe_slice_view.unwrap_or(SliceView{
            dtype: 999,
            shape: [0usize; 8],
            shape_dim: 0,
            data: std::ptr::null(),
            data_len: 0,
        }),
        Err(_) => SliceView {
            dtype: 999,
            shape: [0usize; 8],
            shape_dim: 0,
            data: std::ptr::null(),
            data_len: 0,
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
