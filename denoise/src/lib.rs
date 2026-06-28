//! svgit-denoise — ONNX denoise / JPEG de-block preprocessor (Level 3 ML layer).
//!
//! Runs **SCUNet** (Swin-Conv-UNet, PSNR-optimized) over an RGBA raster to strip
//! JPEG compression blocking and sensor/film noise *before* the tracer quantizes
//! and segments it. Compression artifacts otherwise show up as 8×8 block seams
//! and mosquito speckle, which the color quantizer turns into a haze of spurious
//! tiny regions and bloated paths; cleaning them first yields fewer, smoother
//! shapes. The PSNR model (not the GAN variant) is chosen deliberately — it
//! restores faithfully without hallucinating texture that would itself trace
//! into junk.
//!
//! SCUNet preserves the image size (1×). The model is RGB-only, normalized the
//! KAIR way (plain `/255`, no mean/std); the alpha channel is carried through
//! untouched. Its UNet downsamples ×8 with 8-px attention windows, so inputs are
//! reflect-padded up to a multiple of 64 and cropped back.
//!
//! The heavy `ort`/onnxruntime dependency lives only here, keeping the
//! `svgit-pipeline` crate dependency-free.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

const MODEL_FILE: &str = "scunet_color_psnr.onnx";

/// SCUNet's UNet downsamples ×8 and uses 8-px attention windows, so each input
/// dimension is padded up to a multiple of this (8×8) to keep the skip
/// connections aligned. Padding is reflected and cropped off the result.
const PAD_MULTIPLE: usize = 64;

/// SCUNet is a heavy Swin-Conv-UNet — CPU inference runs ≈13 µs/pixel, so cost
/// is roughly linear in pixels (≈3.4 s at 512×512, ≈8 s at 800×800). Cap input
/// so an opt-in denoise can't block a request for too long; larger inputs pass
/// through unchanged (denoising is most valuable on small/medium artifacted art,
/// and the tracer still handles big clean images fine).
const MAX_INPUT_PIXELS: usize = 640_000; // ~800×800

/// Where the service looks for the model: `$SVGIT_MODEL_DIR/<file>`, defaulting
/// to `./models/<file>`.
pub fn default_model_path() -> PathBuf {
    let dir = std::env::var_os("SVGIT_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"));
    dir.join(MODEL_FILE)
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
            "denoise model not found at {} — run scripts/fetch-models.sh \
             (or set SVGIT_MODEL_DIR)",
            model_path.display()
        ));
    }
    // SCUNet is compute-heavy, so give it as many intra-op threads as the box
    // has (capped). It's serialized behind the mutex anyway, so it won't fight
    // other conversions for cores.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8);
    let built = Session::builder()
        .map_err(|e| format!("ort session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("ort optimization level: {e}"))?
        .with_intra_threads(threads)
        .map_err(|e| format!("ort thread config: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| format!("loading {}: {e}", model_path.display()))?;
    // First writer wins; a racing loser's session is simply dropped.
    let _ = SESSION.set(Mutex::new(built));
    Ok(SESSION.get().expect("session just set"))
}

/// Denoise / de-block an RGBA buffer, returning a same-sized RGBA buffer with the
/// RGB channels cleaned and the alpha channel preserved. Inputs above
/// [`MAX_INPUT_PIXELS`] are returned unchanged (a copy).
pub fn denoise(
    rgba: &[u8],
    width: usize,
    height: usize,
    model_path: &Path,
) -> Result<Vec<u8>, String> {
    let n = width.checked_mul(height).ok_or("image dimensions overflow")?;
    let n4 = n.checked_mul(4).ok_or("image dimensions overflow")?;
    if n == 0 || rgba.len() < n4 {
        return Err("empty or truncated RGBA buffer".to_string());
    }

    // Too large to denoise affordably — hand the original straight back.
    if n > MAX_INPUT_PIXELS {
        return Ok(rgba.to_vec());
    }

    let (pw, ph) = padded_dims(width, height);
    let input = to_input_tensor(rgba, width, height, pw, ph);

    // --- run the model --- (scoped so the lock releases before compositing)
    let clean = {
        let lock = session(model_path)?;
        let mut sess = lock
            .lock()
            .map_err(|_| "model session poisoned".to_string())?;
        let in_name = sess
            .inputs
            .first()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "input".to_string());
        let tensor = Tensor::from_array((vec![1i64, 3, ph as i64, pw as i64], input))
            .map_err(|e| format!("building input tensor: {e}"))?;
        let outputs = sess
            .run(ort::inputs![in_name => tensor])
            .map_err(|e| format!("denoise inference: {e}"))?;
        let (oshape, odata) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("reading denoised output: {e}"))?;
        let dims = oshape.as_ref();
        // Output is same-size NCHW [1,3,ph,pw] (model is 1×).
        if dims.len() != 4 || dims[1] != 3 || (dims[2] as usize) < height || (dims[3] as usize) < width
        {
            return Err(format!("unexpected denoised output shape {oshape:?}"));
        }
        let ow = dims[3] as usize;
        let oh = dims[2] as usize;
        if odata.len() < ow * oh * 3 {
            return Err(format!("truncated denoised output shape {oshape:?}"));
        }
        crop_rgb_from_chw(odata, ow, oh, width, height)
    };

    // Copy the original (keeps alpha) and overwrite RGB with the cleaned values.
    let mut out = rgba.to_vec();
    for p in 0..n {
        out[p * 4] = clean[p * 3];
        out[p * 4 + 1] = clean[p * 3 + 1];
        out[p * 4 + 2] = clean[p * 3 + 2];
    }
    Ok(out)
}

/// Round each dimension up to the next multiple of [`PAD_MULTIPLE`].
fn padded_dims(width: usize, height: usize) -> (usize, usize) {
    let up = |v: usize| v.div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE;
    (up(width), up(height))
}

/// Reflect index `i` back into `[0, n)` using OpenCV/PyTorch reflect-101
/// semantics, valid for any `i` (handles pad larger than the image).
fn reflect(i: usize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let m = i % period;
    if m < n {
        m
    } else {
        period - m
    }
}

/// Build the model's NCHW f32 tensor for a `pw×ph` reflect-padded copy of the
/// source RGB, normalized by a flat `/255` (KAIR/SCUNet convention).
fn to_input_tensor(rgba: &[u8], width: usize, height: usize, pw: usize, ph: usize) -> Vec<f32> {
    let phw = pw * ph;
    let mut t = vec![0f32; 3 * phw];
    for py in 0..ph {
        let sy = reflect(py, height);
        for px in 0..pw {
            let sx = reflect(px, width);
            let src = (sy * width + sx) * 4;
            let dst = py * pw + px;
            for c in 0..3 {
                t[c * phw + dst] = rgba[src + c] as f32 / 255.0;
            }
        }
    }
    t
}

/// Crop the top-left `width×height` region out of a padded CHW f32 output plane
/// (`[0,1]` range) into interleaved RGB bytes.
fn crop_rgb_from_chw(odata: &[f32], pw: usize, ph: usize, width: usize, height: usize) -> Vec<u8> {
    let phw = pw * ph;
    let mut rgb = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let src = y * pw + x;
            let dst = (y * width + x) * 3;
            for c in 0..3 {
                let v = odata[c * phw + src].clamp(0.0, 1.0);
                rgb[dst + c] = (v * 255.0).round() as u8;
            }
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_dims_round_up_to_multiple() {
        assert_eq!(padded_dims(64, 64), (64, 64));
        assert_eq!(padded_dims(65, 1), (128, 64));
        assert_eq!(padded_dims(100, 70), (128, 128));
        assert_eq!(padded_dims(1, 1), (64, 64));
    }

    #[test]
    fn reflect_is_in_range_and_mirrors() {
        // Within range: identity.
        assert_eq!(reflect(0, 10), 0);
        assert_eq!(reflect(9, 10), 9);
        // Just past the edge mirrors back (reflect-101: n-2, not n-1).
        assert_eq!(reflect(10, 10), 8);
        assert_eq!(reflect(11, 10), 7);
        // Degenerate width.
        assert_eq!(reflect(5, 1), 0);
        // Large pad relative to width stays in range (triangle wave).
        for i in 0..500 {
            assert!(reflect(i, 7) < 7);
        }
    }

    #[test]
    fn input_tensor_is_padded_scaled_nchw() {
        // 2×1 image, white then black; pad to 64×64.
        let rgba = [255u8, 255, 255, 255, 0, 0, 0, 255];
        let (pw, ph) = padded_dims(2, 1);
        let t = to_input_tensor(&rgba, 2, 1, pw, ph);
        assert_eq!(t.len(), 3 * pw * ph);
        // (0,0) is the white source pixel → 1.0 in each channel plane.
        let phw = pw * ph;
        for c in 0..3 {
            assert!((t[c * phw] - 1.0).abs() < 1e-6, "ch{c} at origin");
            // (0,1) is the black source pixel → 0.0.
            assert!(t[c * phw + 1].abs() < 1e-6, "ch{c} at x=1");
        }
    }

    #[test]
    fn crop_extracts_top_left_and_clamps() {
        // Padded plane 64×64, 3 channels; fill with a ramp we can check, plus an
        // out-of-range value that must clamp.
        let (pw, ph) = (64usize, 64usize);
        let phw = pw * ph;
        let mut odata = vec![0f32; 3 * phw];
        odata[0] = 0.0; // R at (0,0)
        odata[phw] = 1.0; // G at (0,0)
        odata[2 * phw] = 1.5; // B at (0,0) → clamp to 255
        let rgb = crop_rgb_from_chw(&odata, pw, ph, 1, 1);
        assert_eq!(rgb, vec![0, 255, 255]);
    }

    #[test]
    fn oversized_input_passes_through_unchanged() {
        // Above the pixel cap: returns an identical copy, never touching the
        // model (so a bogus path must not be read).
        let side = 1300; // 1300² > MAX_INPUT_PIXELS
        let rgba = vec![9u8; side * side * 4];
        let out = denoise(&rgba, side, side, Path::new("/nonexistent.onnx")).unwrap();
        assert_eq!(out, rgba);
    }

    #[test]
    fn denoise_rejects_overflowing_dims() {
        let model = Path::new("/nonexistent/scunet.onnx");
        assert!(denoise(&[], usize::MAX, 2, model).is_err());
        assert!(denoise(&[], usize::MAX / 3, 1, model).is_err());
    }

    #[test]
    fn default_model_path_honors_env() {
        std::env::set_var("SVGIT_MODEL_DIR", "/tmp/svgit-dn-models");
        assert_eq!(
            default_model_path(),
            PathBuf::from("/tmp/svgit-dn-models").join(MODEL_FILE)
        );
        std::env::remove_var("SVGIT_MODEL_DIR");
    }
}
