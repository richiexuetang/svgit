//! svgit-bgremove — ONNX background removal (Level 3 ML layer).
//!
//! Runs a salient-object model over an RGBA raster to produce a single-channel
//! matte, then returns a copy of the raster whose alpha channel carries that
//! matte (so the existing alpha-aware tracer drops the background for free).
//!
//! Two models are supported (see [`Model`]): **u2netp** (lightweight, fast) and
//! **isnet-general-use** (heavier, much sharper on fine detail). They differ
//! only in input size and normalization; the pre/post-processing is otherwise
//! shared.
//!
//! The heavy `ort`/onnxruntime dependency lives only here, keeping the
//! `svgit-pipeline` crate dependency-free.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

/// Which salient-object model to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// u2netp — lightweight (~4.6 MB), 320×320, ImageNet normalization. Fast.
    U2netp,
    /// isnet-general-use — high accuracy (~178 MB), 1024×1024, ±0.5
    /// normalization. Sharper edges/fine detail, slower.
    Isnet,
}

impl Model {
    /// Map a form value to a model; anything unrecognized falls back to the
    /// fast default so a bad param can never 500.
    pub fn parse(s: &str) -> Model {
        match s {
            "isnet" | "high" => Model::Isnet,
            _ => Model::U2netp,
        }
    }

    pub fn filename(self) -> &'static str {
        match self {
            Model::U2netp => "u2netp.onnx",
            Model::Isnet => "isnet-general-use.onnx",
        }
    }

    /// Square input side the model expects.
    fn side(self) -> usize {
        match self {
            Model::U2netp => 320,
            Model::Isnet => 1024,
        }
    }

    /// Per-channel normalization mean (applied after scaling by the image max).
    fn mean(self) -> [f32; 3] {
        match self {
            Model::U2netp => [0.485, 0.456, 0.406], // ImageNet (u2net ToTensorLab)
            Model::Isnet => [0.5, 0.5, 0.5],        // DIS GOSNormalize
        }
    }

    fn std(self) -> [f32; 3] {
        match self {
            Model::U2netp => [0.229, 0.224, 0.225],
            Model::Isnet => [1.0, 1.0, 1.0],
        }
    }
}

/// Where the service looks for the chosen model's weights:
/// `$SVGIT_MODEL_DIR/<filename>`, defaulting to `./models/<filename>`.
pub fn default_model_path(model: Model) -> PathBuf {
    let dir = std::env::var_os("SVGIT_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"));
    dir.join(model.filename())
}

/// Tuning for the matte → alpha conversion.
#[derive(Debug, Clone)]
pub struct BgConfig {
    /// Matte cutoff in `[0, 1]`. Pixels whose normalized saliency is at or above
    /// this become fully opaque, below it fully transparent. `0` keeps the raw
    /// soft matte (every saliency level passes through as a partial alpha).
    pub threshold: f32,
}

impl Default for BgConfig {
    fn default() -> Self {
        // A clean foreground/background split at the matte midpoint. The
        // downstream tracer thresholds alpha anyway, so a soft matte would just
        // read as "any non-zero saliency is foreground" — too greedy.
        Self { threshold: 0.5 }
    }
}

/// Process-wide model sessions, one cache per model so switching between them
/// doesn't reload (or clobber) the other. onnxruntime's Rust `run` needs `&mut`,
/// so each is serialized behind a mutex; the converter pool already caps how
/// many requests reach this at once.
static SESSION_U2NETP: OnceLock<Mutex<Session>> = OnceLock::new();
static SESSION_ISNET: OnceLock<Mutex<Session>> = OnceLock::new();

fn session(model: Model, model_path: &Path) -> Result<&'static Mutex<Session>, String> {
    let cell = match model {
        Model::U2netp => &SESSION_U2NETP,
        Model::Isnet => &SESSION_ISNET,
    };
    if let Some(s) = cell.get() {
        return Ok(s);
    }
    if !model_path.exists() {
        return Err(format!(
            "background-removal model not found at {} — run scripts/fetch-models.sh \
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
    let _ = cell.set(Mutex::new(built));
    Ok(cell.get().expect("session just set"))
}

/// Remove the background from an RGBA buffer using `model`, returning a new RGBA
/// buffer whose alpha is the (thresholded) saliency matte multiplied into any
/// pre-existing alpha. RGB is untouched.
pub fn remove_background(
    rgba: &[u8],
    width: usize,
    height: usize,
    model: Model,
    model_path: &Path,
    cfg: &BgConfig,
) -> Result<Vec<u8>, String> {
    let n = width.checked_mul(height).ok_or("image dimensions overflow")?;
    // Guard the channel multiplies too — `n` alone fitting doesn't mean `n*4`
    // does, and this is a public entry point without the service's MP cap.
    let n4 = n.checked_mul(4).ok_or("image dimensions overflow")?;
    let n3 = n.checked_mul(3).ok_or("image dimensions overflow")?;
    if n == 0 || rgba.len() < n4 {
        return Err("empty or truncated RGBA buffer".to_string());
    }

    let side = model.side();

    // --- resize source RGB → side×side (Lanczos, matching rembg) ---
    let mut rgb = vec![0u8; n3];
    for i in 0..n {
        rgb[i * 3] = rgba[i * 4];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    let src = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(width as u32, height as u32, rgb)
        .ok_or("could not wrap RGB buffer")?;
    let small = image::imageops::resize(
        &src,
        side as u32,
        side as u32,
        image::imageops::FilterType::Lanczos3,
    );

    // --- normalize → NCHW [1,3,side,side] ---
    let input = to_input_tensor(small.as_raw(), model);

    // --- run the model --- (scoped so the lock releases before compositing)
    let matte = {
        let lock = session(model, model_path)?;
        let mut sess = lock
            .lock()
            .map_err(|_| "model session poisoned".to_string())?;
        let in_name = sess
            .inputs
            .first()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "input.1".to_string());
        let tensor = Tensor::from_array((vec![1i64, 3, side as i64, side as i64], input))
            .map_err(|e| format!("building input tensor: {e}"))?;
        let outputs = sess
            .run(ort::inputs![in_name => tensor])
            .map_err(|e| format!("matting inference: {e}"))?;
        let (oshape, odata) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("reading saliency output: {e}"))?;
        if odata.len() < side * side {
            return Err(format!("unexpected saliency output shape {oshape:?}"));
        }
        matte_from_output(&odata[..side * side])
    };

    // --- resize matte side×side → original, then composite into alpha ---
    let mbuf = image::ImageBuffer::<image::Luma<u8>, _>::from_raw(side as u32, side as u32, matte)
        .ok_or("could not wrap matte buffer")?;
    let full = image::imageops::resize(
        &mbuf,
        width as u32,
        height as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let mfull = full.as_raw();

    let mut out = rgba.to_vec();
    let cut = (cfg.threshold.clamp(0.0, 1.0) * 255.0).round() as u16;
    for i in 0..n {
        let m = mfull[i] as u16;
        let a: u16 = if cut > 0 {
            if m >= cut {
                255
            } else {
                0
            }
        } else {
            m // soft matte
        };
        // Respect pixels that were already (partially) transparent.
        let existing = rgba[i * 4 + 3] as u16;
        out[i * 4 + 3] = ((a * existing) / 255) as u8;
    }
    Ok(out)
}

/// Normalize a `side×side` RGB byte buffer into the model's NCHW f32 tensor.
fn to_input_tensor(rgb: &[u8], model: Model) -> Vec<f32> {
    let side = model.side();
    let hw = side * side;
    let (mean, std) = (model.mean(), model.std());
    // Both nets divide by the image's own max, not a fixed 255.
    let max = rgb.iter().copied().max().unwrap_or(1).max(1) as f32;
    let mut t = vec![0f32; 3 * hw];
    for p in 0..hw {
        for c in 0..3 {
            let v = rgb[p * 3 + c] as f32 / max;
            t[c * hw + p] = (v - mean[c]) / std[c];
        }
    }
    t
}

/// Min-max normalize the raw saliency output into a 0..255 matte.
fn matte_from_output(odata: &[f32]) -> Vec<u8> {
    let mut mi = f32::INFINITY;
    let mut ma = f32::NEG_INFINITY;
    for &v in odata {
        if v < mi {
            mi = v;
        }
        if v > ma {
            ma = v;
        }
    }
    let range = (ma - mi).max(1e-6);
    odata
        .iter()
        .map(|&v| (((v - mi) / range) * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_tensor_is_normalized_nchw() {
        // A solid mid-gray u2netp-sized image.
        let side = Model::U2netp.side();
        let rgb = vec![128u8; side * side * 3];
        let t = to_input_tensor(&rgb, Model::U2netp);
        assert_eq!(t.len(), 3 * side * side);
        // max == 128, so v == 1.0 everywhere; channel 0 == (1-0.485)/0.229.
        let expect_r = (1.0 - Model::U2netp.mean()[0]) / Model::U2netp.std()[0];
        assert!((t[0] - expect_r).abs() < 1e-5, "got {}", t[0]);
        // Plane boundary: first G-plane element uses the green mean/std.
        let expect_g = (1.0 - Model::U2netp.mean()[1]) / Model::U2netp.std()[1];
        assert!((t[side * side] - expect_g).abs() < 1e-5);
    }

    #[test]
    fn isnet_tensor_uses_pm_half_normalization() {
        // ISNet: (v - 0.5) / 1.0. A white (max) pixel → 0.5.
        let side = Model::Isnet.side();
        let rgb = vec![255u8; side * side * 3];
        let t = to_input_tensor(&rgb, Model::Isnet);
        assert_eq!(t.len(), 3 * side * side);
        assert!((t[0] - 0.5).abs() < 1e-5, "got {}", t[0]);
    }

    #[test]
    fn input_tensor_handles_all_black() {
        // max clamps to 1 (not 0) — no NaN/inf.
        let side = Model::U2netp.side();
        let rgb = vec![0u8; side * side * 3];
        let t = to_input_tensor(&rgb, Model::U2netp);
        assert!(t.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn matte_min_max_stretches_to_full_range() {
        let raw = [0.1f32, 0.2, 0.3, 0.5, 0.9];
        let m = matte_from_output(&raw);
        assert_eq!(*m.first().unwrap(), 0); // the min maps to 0
        assert_eq!(*m.last().unwrap(), 255); // the max maps to 255
    }

    #[test]
    fn matte_flat_input_does_not_divide_by_zero() {
        // min == max → the 1e-6 floor keeps it finite; everything maps to 0.
        let raw = [0.42f32; 16];
        let m = matte_from_output(&raw);
        assert!(m.iter().all(|&v| v == 0));
    }

    #[test]
    fn model_parse_defaults_to_fast() {
        assert_eq!(Model::parse("isnet"), Model::Isnet);
        assert_eq!(Model::parse("high"), Model::Isnet);
        assert_eq!(Model::parse("u2netp"), Model::U2netp);
        assert_eq!(Model::parse("nonsense"), Model::U2netp);
    }

    #[test]
    fn remove_background_rejects_overflowing_dims() {
        // Dimensions whose product (or product*4) overflows usize must error out
        // cleanly, before any allocation or model access — never panic.
        let model = Path::new("/nonexistent/u2netp.onnx");
        let cfg = BgConfig::default();
        assert!(remove_background(&[], usize::MAX, 2, Model::U2netp, model, &cfg).is_err());
        assert!(remove_background(&[], usize::MAX / 3, 1, Model::U2netp, model, &cfg).is_err());
    }

    #[test]
    fn default_model_path_honors_env_and_model() {
        std::env::set_var("SVGIT_MODEL_DIR", "/tmp/svgit-models");
        assert_eq!(
            default_model_path(Model::U2netp),
            PathBuf::from("/tmp/svgit-models").join("u2netp.onnx")
        );
        assert_eq!(
            default_model_path(Model::Isnet),
            PathBuf::from("/tmp/svgit-models").join("isnet-general-use.onnx")
        );
        std::env::remove_var("SVGIT_MODEL_DIR");
    }
}
