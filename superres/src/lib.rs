//! svgit-superres — ONNX super-resolution preprocessor (Level 3 ML layer).
//!
//! Runs **realesr-general-x4v3** (SRVGGNetCompact, ×4) over an RGBA raster to
//! upscale it 4× before tracing. The point is *low-res* inputs: a small or
//! pixelated logo/clipart traces into clean curves once it's been super-resolved
//! instead of vectorizing the blocky original.
//!
//! The model is fully convolutional with dynamic H/W, so we feed the image at
//! its native size and read back a 4× result. RGB is normalized the RealESRGAN
//! way (plain `/255`, no mean/std) and the network's final clip keeps the output
//! in `[0, 1]`. The alpha channel is upscaled separately (the model is RGB-only)
//! so any existing transparency is preserved.
//!
//! The heavy `ort`/onnxruntime dependency lives only here, keeping the
//! `svgit-pipeline` crate dependency-free.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

const MODEL_FILE: &str = "realesr-general-x4v3.onnx";

/// The model's native (fixed) upscale factor.
pub const SCALE: usize = 4;

/// Don't super-resolve inputs larger than this many pixels. SR only meaningfully
/// helps small/low-res inputs, and the 16× output of a large image would be both
/// slow and memory-heavy. 1024×1024 in → 4096×4096 (≈16.7 MP) out, which stays
/// under the service's decoded-pixel guard. Larger inputs pass through untouched.
const MAX_INPUT_PIXELS: usize = 1024 * 1024;

/// Where the service looks for the model: `$SVGIT_MODEL_DIR/<file>`, defaulting
/// to `./models/<file>`.
pub fn default_model_path() -> PathBuf {
    let dir = std::env::var_os("SVGIT_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"));
    dir.join(MODEL_FILE)
}

/// An upscaled raster: RGBA bytes plus its new dimensions.
#[derive(Debug, Clone)]
pub struct SrImage {
    pub rgba: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

/// Process-wide model session. onnxruntime's Rust `run` needs `&mut`, so it's
/// serialized behind a mutex; the converter pool already caps concurrency.
static SESSION: OnceLock<Mutex<Session>> = OnceLock::new();

fn session(model_path: &Path) -> Result<&'static Mutex<Session>, String> {
    if let Some(s) = SESSION.get() {
        return Ok(s);
    }
    if !model_path.exists() {
        return Err(format!(
            "super-resolution model not found at {} — run scripts/fetch-models.sh \
             (or set SVGIT_MODEL_DIR)",
            model_path.display()
        ));
    }
    let built = Session::builder()
        .map_err(|e| format!("ort session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("ort optimization level: {e}"))?
        .with_intra_threads(2)
        .map_err(|e| format!("ort thread config: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| format!("loading {}: {e}", model_path.display()))?;
    // First writer wins; a racing loser's session is simply dropped.
    let _ = SESSION.set(Mutex::new(built));
    Ok(SESSION.get().expect("session just set"))
}

/// Super-resolve an RGBA buffer 4×, returning the upscaled raster and its new
/// dimensions. Inputs above [`MAX_INPUT_PIXELS`] are returned unchanged (a copy)
/// — SR is for small inputs, and upscaling a large image would blow up cost.
pub fn super_resolve(
    rgba: &[u8],
    width: usize,
    height: usize,
    model_path: &Path,
) -> Result<SrImage, String> {
    let n = width.checked_mul(height).ok_or("image dimensions overflow")?;
    let n4 = n.checked_mul(4).ok_or("image dimensions overflow")?;
    if n == 0 || rgba.len() < n4 {
        return Err("empty or truncated RGBA buffer".to_string());
    }

    // Too large to be worth (or safe to) upscale — hand the original straight
    // back so the caller can carry on with the unmodified raster.
    if n > MAX_INPUT_PIXELS {
        return Ok(SrImage {
            rgba: rgba.to_vec(),
            width,
            height,
        });
    }

    let input = to_input_tensor(rgba, width, height);

    // --- run the model --- (scoped so the lock releases before compositing)
    let (rgb, ow, oh) = {
        let lock = session(model_path)?;
        let mut sess = lock
            .lock()
            .map_err(|_| "model session poisoned".to_string())?;
        let in_name = sess
            .inputs
            .first()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "input".to_string());
        let tensor = Tensor::from_array((vec![1i64, 3, height as i64, width as i64], input))
            .map_err(|e| format!("building input tensor: {e}"))?;
        let outputs = sess
            .run(ort::inputs![in_name => tensor])
            .map_err(|e| format!("super-resolution inference: {e}"))?;
        let (oshape, odata) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("reading upscaled output: {e}"))?;
        // Expect NCHW [1, 3, OH, OW].
        let dims = oshape.as_ref();
        if dims.len() != 4 || dims[1] != 3 {
            return Err(format!("unexpected upscaled output shape {oshape:?}"));
        }
        let oh = dims[2] as usize;
        let ow = dims[3] as usize;
        let ohw = oh.checked_mul(ow).ok_or("upscaled dimensions overflow")?;
        if oh == 0 || ow == 0 || odata.len() < ohw * 3 {
            return Err(format!("truncated upscaled output shape {oshape:?}"));
        }
        (rgb_u8_from_chw(odata, ohw), ow, oh)
    };

    // --- upscale the alpha channel separately to match (model is RGB-only) ---
    let alpha = upscale_alpha(rgba, width, height, ow, oh)?;

    let ohw = ow * oh;
    let mut out = vec![0u8; ohw * 4];
    for p in 0..ohw {
        out[p * 4] = rgb[p * 3];
        out[p * 4 + 1] = rgb[p * 3 + 1];
        out[p * 4 + 2] = rgb[p * 3 + 2];
        out[p * 4 + 3] = alpha[p];
    }

    Ok(SrImage {
        rgba: out,
        width: ow,
        height: oh,
    })
}

/// Normalize an RGBA byte buffer into the model's NCHW f32 tensor: RGB scaled by
/// a flat `/255` (RealESRGAN convention — no mean/std), alpha dropped.
fn to_input_tensor(rgba: &[u8], width: usize, height: usize) -> Vec<f32> {
    let hw = width * height;
    let mut t = vec![0f32; 3 * hw];
    for p in 0..hw {
        for c in 0..3 {
            t[c * hw + p] = rgba[p * 4 + c] as f32 / 255.0;
        }
    }
    t
}

/// Convert a CHW f32 output plane (3 channels, already `[0,1]`) into interleaved
/// RGB bytes. Clamped defensively in case the model emits a hair past the clip.
fn rgb_u8_from_chw(odata: &[f32], ohw: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; ohw * 3];
    for p in 0..ohw {
        for c in 0..3 {
            let v = odata[c * ohw + p].clamp(0.0, 1.0);
            rgb[p * 3 + c] = (v * 255.0).round() as u8;
        }
    }
    rgb
}

/// Resample the source alpha channel to the upscaled size with a high-quality
/// filter, so existing transparency survives the upscale.
fn upscale_alpha(
    rgba: &[u8],
    width: usize,
    height: usize,
    ow: usize,
    oh: usize,
) -> Result<Vec<u8>, String> {
    let hw = width * height;
    let mut a = vec![0u8; hw];
    for (p, slot) in a.iter_mut().enumerate() {
        *slot = rgba[p * 4 + 3];
    }
    let buf = image::ImageBuffer::<image::Luma<u8>, _>::from_raw(width as u32, height as u32, a)
        .ok_or("could not wrap alpha buffer")?;
    let up = image::imageops::resize(
        &buf,
        ow as u32,
        oh as u32,
        image::imageops::FilterType::Lanczos3,
    );
    Ok(up.into_raw())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_tensor_is_plain_scaled_nchw() {
        // 2×1 image: white then black pixel.
        let rgba = [255u8, 255, 255, 255, 0, 0, 0, 255];
        let t = to_input_tensor(&rgba, 2, 1);
        assert_eq!(t.len(), 3 * 2);
        // White pixel (p=0) → 1.0 in every channel plane; black (p=1) → 0.0.
        let hw = 2;
        for c in 0..3 {
            assert!((t[c * hw] - 1.0).abs() < 1e-6, "white ch{c}");
            assert!(t[c * hw + 1].abs() < 1e-6, "black ch{c}");
        }
    }

    #[test]
    fn rgb_from_chw_clamps_and_scales() {
        // 1 pixel, channels: 0.0, 1.0, and an out-of-range 1.5 (must clamp).
        let odata = [0.0f32, 1.0, 1.5];
        let rgb = rgb_u8_from_chw(&odata, 1);
        assert_eq!(rgb, vec![0, 255, 255]);
    }

    #[test]
    fn alpha_upscale_preserves_opaque() {
        // A fully-opaque 2×2 stays opaque at 8×8.
        let rgba = vec![10u8; 2 * 2 * 4]; // alpha == 10 everywhere... make it 255
        let mut rgba = rgba;
        for p in 0..4 {
            rgba[p * 4 + 3] = 255;
        }
        let a = upscale_alpha(&rgba, 2, 2, 8, 8).unwrap();
        assert_eq!(a.len(), 64);
        assert!(a.iter().all(|&v| v == 255));
    }

    #[test]
    fn oversized_input_passes_through_unchanged() {
        // Above the pixel cap: returns a copy with identical dims, never touching
        // the model (so a bogus path is fine — it must not be read).
        let side = 1025; // 1025² > MAX_INPUT_PIXELS
        let rgba = vec![7u8; side * side * 4];
        let out = super_resolve(&rgba, side, side, Path::new("/nonexistent.onnx")).unwrap();
        assert_eq!((out.width, out.height), (side, side));
        assert_eq!(out.rgba, rgba);
    }

    #[test]
    fn super_resolve_rejects_overflowing_dims() {
        // Product (or product*4) overflow must error cleanly, never panic.
        let model = Path::new("/nonexistent/realesr.onnx");
        assert!(super_resolve(&[], usize::MAX, 2, model).is_err());
        assert!(super_resolve(&[], usize::MAX / 3, 1, model).is_err());
    }

    #[test]
    fn default_model_path_honors_env() {
        std::env::set_var("SVGIT_MODEL_DIR", "/tmp/svgit-sr-models");
        assert_eq!(
            default_model_path(),
            PathBuf::from("/tmp/svgit-sr-models").join(MODEL_FILE)
        );
        std::env::remove_var("SVGIT_MODEL_DIR");
    }
}
