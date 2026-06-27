//! svgit-objseg — class-agnostic "segment everything" via FastSAM (a YOLOv8-seg
//! network). Given an RGBA raster it returns a set of instance masks (one per
//! detected object), which the tracer turns into per-object SVG layers.
//!
//! The heavy `ort`/onnxruntime dependency lives here and in `svgit-bgremove`,
//! never in the dependency-free `svgit-pipeline`.
//!
//! Pipeline: letterbox→[1,3,S,S] → FastSAM → (`output0` detections + `output1`
//! mask prototypes) → confidence filter → NMS → per-detection mask =
//! sigmoid(coeffs · prototypes), cropped to its box and mapped back to the
//! original resolution. Shapes are read at runtime so a 640- or 1024-input
//! export, and either `output0` channel order, both work.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

/// Fallback input side when the model declares a dynamic input.
const DEFAULT_SIDE: usize = 1024;
/// Gray letterbox padding (YOLO convention), pre-normalized.
const PAD: f32 = 114.0 / 255.0;
/// Upper bound on any single declared tensor dimension. Output shapes come from
/// the model at runtime; this rejects negative/absurd dims before they reach the
/// index math or allocations below. Real exports are far under this (anchors
/// ~21k, prototype side ~256).
const MAX_DIM: usize = 1_000_000;

pub const MODEL_FILENAME: &str = "FastSAM-x.onnx";

/// `$SVGIT_MODEL_DIR/FastSAM-x.onnx`, defaulting to `./models/FastSAM-x.onnx`.
pub fn default_model_path() -> PathBuf {
    let dir = std::env::var_os("SVGIT_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"));
    dir.join(MODEL_FILENAME)
}

#[derive(Debug, Clone)]
pub struct SegConfig {
    /// Minimum detection confidence to keep (0..1).
    pub conf: f32,
    /// IoU threshold for non-max suppression.
    pub iou: f32,
    /// Hard cap on returned objects (after NMS, largest first).
    pub max_objects: usize,
    /// Sigmoid cutoff for turning a soft prototype mask into a binary mask.
    pub mask_threshold: f32,
    /// Drop masks smaller than this fraction of the image (speckle objects).
    pub min_area_frac: f32,
}

impl Default for SegConfig {
    fn default() -> Self {
        Self {
            conf: 0.4,
            iou: 0.7,
            max_objects: 48,
            mask_threshold: 0.5,
            min_area_frac: 0.0008,
        }
    }
}

/// One detected object.
pub struct Instance {
    /// `width*height` binary mask (1 = object) at the original resolution.
    pub mask: Vec<u8>,
    /// Bounding box in original pixels: (x1, y1, x2, y2), inclusive-exclusive.
    pub bbox: (u32, u32, u32, u32),
    pub score: f32,
    /// Number of set pixels in `mask`.
    pub area: u32,
}

static SESSION: OnceLock<Mutex<Session>> = OnceLock::new();

fn session(model_path: &Path) -> Result<&'static Mutex<Session>, String> {
    if let Some(s) = SESSION.get() {
        return Ok(s);
    }
    if !model_path.exists() {
        return Err(format!(
            "segmentation model not found at {} — run scripts/fetch-models.sh \
             (or set SVGIT_MODEL_DIR)",
            model_path.display()
        ));
    }
    let built = Session::builder()
        .map_err(|e| format!("ort session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("ort optimization level: {e}"))?
        .with_intra_threads(4)
        .map_err(|e| format!("ort thread config: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| format!("loading {}: {e}", model_path.display()))?;
    let _ = SESSION.set(Mutex::new(built));
    Ok(SESSION.get().expect("session just set"))
}

/// A surviving detection in letterboxed input space.
struct Det {
    // xyxy in input pixels.
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    coeffs: Vec<f32>,
}

/// Run FastSAM and return one [`Instance`] per detected object, largest first.
pub fn segment_everything(
    rgba: &[u8],
    width: usize,
    height: usize,
    model_path: &Path,
    cfg: &SegConfig,
) -> Result<Vec<Instance>, String> {
    let n = width.checked_mul(height).ok_or("image dimensions overflow")?;
    let n4 = n.checked_mul(4).ok_or("image dimensions overflow")?;
    if n == 0 || rgba.len() < n4 {
        return Err("empty or truncated RGBA buffer".to_string());
    }

    let lock = session(model_path)?;
    let mut sess = lock.lock().map_err(|_| "model session poisoned".to_string())?;

    // Input side: honor a static model shape, else fall back to the default.
    let side = sess
        .inputs
        .first()
        .and_then(|i| i.input_type.tensor_shape())
        .and_then(|s| s.last().copied())
        .filter(|&d| d > 0)
        .map(|d| d as usize)
        .unwrap_or(DEFAULT_SIDE);

    // --- letterbox preprocess ---
    let (input, scale, padx, pady) = letterbox(rgba, width, height, side)?;
    let in_name = sess
        .inputs
        .first()
        .map(|i| i.name.clone())
        .unwrap_or_else(|| "images".to_string());
    let tensor = Tensor::from_array((vec![1i64, 3, side as i64, side as i64], input))
        .map_err(|e| format!("building input tensor: {e}"))?;

    // This export emits several tensors (decoded detections, raw head feature
    // maps, a redundant coeff tensor, and the mask prototypes). Collect them all,
    // then pick by shape: the prototypes are the rank-4 tensor with the largest
    // spatial area; the detections are the rank-3 tensor whose channel dim equals
    // 4 (box) + 1 (score) + proto channels. Scoped so `outputs` (which borrows
    // the session) drops before we release the lock for mask assembly.
    let all: Vec<(Vec<i64>, Vec<f32>)> = {
        let outputs = sess
            .run(ort::inputs![in_name => tensor])
            .map_err(|e| format!("FastSAM inference: {e}"))?;
        let mut all = Vec::with_capacity(outputs.len());
        for i in 0..outputs.len() {
            let (shape, data) = outputs[i]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("reading model output {i}: {e}"))?;
            all.push((shape.to_vec(), data.to_vec()));
        }
        all
    };
    drop(sess); // release the model before the (CPU-only) mask assembly

    // A declared dimension must be a sane positive size.
    let dim = |d: i64| -> Result<usize, String> {
        if d > 0 && (d as u64) <= MAX_DIM as u64 {
            Ok(d as usize)
        } else {
            Err(format!("implausible model tensor dimension: {d}"))
        }
    };

    // Prototypes: rank-4, largest H*W. We assume the canonical ONNX NCHW layout
    // [1, C, H, W]; the length check below catches a size mismatch (truncated or
    // malformed output) but not a transposed export, which ONNX doesn't produce.
    let (pshape, pdata) = all
        .iter()
        .filter(|(d, _)| d.len() == 4)
        .max_by_key(|(d, _)| d[2].saturating_mul(d[3]))
        .ok_or("model produced no prototype output (rank-4)")?;
    let proto_c = dim(pshape[1])?;
    let ph = dim(pshape[2])?;
    let pw = dim(pshape[3])?;
    let proto_hw = ph.checked_mul(pw).ok_or("prototype dimensions overflow")?;
    if pdata.len() != proto_c.checked_mul(proto_hw).ok_or("prototype size overflow")? {
        return Err(format!(
            "prototype tensor {pshape:?} doesn't match its {} provided elements",
            pdata.len()
        ));
    }

    // Detections: rank-3 whose channel dim (either layout) == 5 + proto_c
    // (4 box + 1 score + proto_c mask coefficients). proto_c <= MAX_DIM, so the
    // addition can't overflow.
    let expected_c = proto_c + 5;
    let (dshape, ddata) = all
        .iter()
        .find(|(d, _)| {
            d.len() == 3 && (d[1] as usize == expected_c || d[2] as usize == expected_c)
        })
        .ok_or_else(|| format!("no detection output with {expected_c} channels found"))?;

    let (num_anchors, stride_anchor, stride_chan) = if dshape[1] as usize == expected_c {
        // [1, C, A]: channel-major. element(ch, a) = ch*A + a.
        let a = dim(dshape[2])?;
        (a, 1usize, a)
    } else {
        // [1, A, C]: anchor-major. element(ch, a) = a*C + ch.
        (dim(dshape[1])?, expected_c, 1usize)
    };
    // Every at(ch, a) lands below expected_c*num_anchors in either layout, so
    // this bound makes the indexing panic-free.
    if ddata.len() < expected_c.checked_mul(num_anchors).ok_or("detection size overflow")? {
        return Err(format!(
            "detection tensor {dshape:?} smaller than its {expected_c}×{num_anchors} shape"
        ));
    }
    let at = |ch: usize, a: usize| -> f32 { ddata[ch * stride_chan + a * stride_anchor] };

    // --- collect detections above the confidence threshold ---
    let mut dets: Vec<Det> = Vec::new();
    for a in 0..num_anchors {
        let score = at(4, a);
        if score < cfg.conf {
            continue;
        }
        let (cx, cy, bw, bh) = (at(0, a), at(1, a), at(2, a), at(3, a));
        let mut coeffs = Vec::with_capacity(proto_c);
        for k in 0..proto_c {
            coeffs.push(at(5 + k, a));
        }
        dets.push(Det {
            x1: cx - bw / 2.0,
            y1: cy - bh / 2.0,
            x2: cx + bw / 2.0,
            y2: cy + bh / 2.0,
            score,
            coeffs,
        });
    }

    let keep = nms(&dets, cfg.iou, cfg.max_objects);

    // --- assemble a full-resolution mask per surviving detection ---
    let min_area = (cfg.min_area_frac * n as f32) as u32;
    let mut instances: Vec<Instance> = Vec::new();
    let mut lowres = vec![0f32; proto_hw]; // reused per detection
    for &di in &keep {
        let d = &dets[di];
        // lowres[i] = sigmoid(sum_k coeffs[k] * proto[k, i])
        for (i, lr) in lowres.iter_mut().enumerate() {
            let mut acc = 0f32;
            for (k, &c) in d.coeffs.iter().enumerate() {
                acc += c * pdata[k * proto_hw + i];
            }
            *lr = sigmoid(acc);
        }

        // Original-space bbox (un-letterbox), clamped.
        let ox1 = (((d.x1 - padx) / scale).floor().max(0.0)) as usize;
        let oy1 = (((d.y1 - pady) / scale).floor().max(0.0)) as usize;
        let ox2 = (((d.x2 - padx) / scale).ceil() as i64).clamp(0, width as i64) as usize;
        let oy2 = (((d.y2 - pady) / scale).ceil() as i64).clamp(0, height as i64) as usize;
        if ox2 <= ox1 || oy2 <= oy1 {
            continue;
        }

        let mut mask = vec![0u8; n];
        let mut area = 0u32;
        let sx = pw as f32 / side as f32;
        let sy = ph as f32 / side as f32;
        for oy in oy1..oy2.min(height) {
            let iy = oy as f32 * scale + pady;
            let py = (iy * sy) as usize;
            if py >= ph {
                continue;
            }
            for ox in ox1..ox2.min(width) {
                let ix = ox as f32 * scale + padx;
                let px = (ix * sx) as usize;
                if px >= pw {
                    continue;
                }
                if lowres[py * pw + px] >= cfg.mask_threshold {
                    mask[oy * width + ox] = 1;
                    area += 1;
                }
            }
        }
        if area < min_area.max(1) {
            continue;
        }
        instances.push(Instance {
            mask,
            bbox: (ox1 as u32, oy1 as u32, ox2 as u32, oy2 as u32),
            score: d.score,
            area,
        });
    }

    // Largest first: a useful default z-order (big objects underneath).
    instances.sort_by_key(|i| std::cmp::Reverse(i.area));
    Ok(instances)
}

/// Resize+pad an RGBA image into a square `side` RGB tensor (NCHW, /255), and
/// return the (scale, pad_x, pad_y) needed to map model coords back to original.
fn letterbox(
    rgba: &[u8],
    w: usize,
    h: usize,
    side: usize,
) -> Result<(Vec<f32>, f32, f32, f32), String> {
    let n3 = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(3))
        .ok_or("image dimensions overflow")?;
    let scale = (side as f32 / w as f32).min(side as f32 / h as f32);
    let nw = ((w as f32 * scale).round() as usize).clamp(1, side);
    let nh = ((h as f32 * scale).round() as usize).clamp(1, side);
    let padx = (side - nw) / 2;
    let pady = (side - nh) / 2;

    let mut rgb = vec![0u8; n3];
    for i in 0..w * h {
        rgb[i * 3] = rgba[i * 4];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    let src = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(w as u32, h as u32, rgb)
        .ok_or("could not wrap RGB buffer")?;
    let small =
        image::imageops::resize(&src, nw as u32, nh as u32, image::imageops::FilterType::Triangle);
    let sp = small.as_raw();

    let hw = side * side;
    let mut t = vec![PAD; 3 * hw];
    for y in 0..nh {
        for x in 0..nw {
            let di = (pady + y) * side + (padx + x);
            let si = (y * nw + x) * 3;
            t[di] = sp[si] as f32 / 255.0;
            t[hw + di] = sp[si + 1] as f32 / 255.0;
            t[2 * hw + di] = sp[si + 2] as f32 / 255.0;
        }
    }
    Ok((t, scale, padx as f32, pady as f32))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn iou(a: &Det, b: &Det) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Greedy class-agnostic NMS. Returns kept indices into `dets`, highest score
/// first, capped at `max`.
fn nms(dets: &[Det], iou_thresh: f32, max: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..dets.len()).collect();
    order.sort_by(|&a, &b| dets[b].score.partial_cmp(&dets[a].score).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep = Vec::new();
    let mut removed = vec![false; dets.len()];
    for i in 0..order.len() {
        let a = order[i];
        if removed[a] {
            continue;
        }
        keep.push(a);
        if keep.len() >= max {
            break;
        }
        for &b in order.iter().skip(i + 1) {
            if !removed[b] && iou(&dets[a], &dets[b]) >= iou_thresh {
                removed[b] = true;
            }
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> Det {
        Det { x1, y1, x2, y2, score, coeffs: vec![] }
    }

    #[test]
    fn iou_identical_is_one() {
        assert!((iou(&det(0., 0., 10., 10., 1.), &det(0., 0., 10., 10., 1.)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_disjoint_is_zero() {
        assert_eq!(iou(&det(0., 0., 10., 10., 1.), &det(20., 20., 30., 30., 1.)), 0.0);
    }

    #[test]
    fn iou_half_overlap() {
        // Two 10x10 boxes overlapping in a 5x10 strip: inter=50, union=150.
        let v = iou(&det(0., 0., 10., 10., 1.), &det(5., 0., 15., 10., 1.));
        assert!((v - (50.0 / 150.0)).abs() < 1e-6);
    }

    #[test]
    fn nms_suppresses_overlap_keeps_best() {
        let dets = vec![
            det(0., 0., 10., 10., 0.9),
            det(1., 1., 11., 11., 0.8), // heavy overlap with #0 -> suppressed
            det(50., 50., 60., 60., 0.7), // disjoint -> kept
        ];
        let keep = nms(&dets, 0.5, 10);
        assert_eq!(keep, vec![0, 2]);
    }

    #[test]
    fn nms_respects_max() {
        let dets = vec![
            det(0., 0., 10., 10., 0.9),
            det(50., 50., 60., 60., 0.8),
            det(100., 100., 110., 110., 0.7),
        ];
        assert_eq!(nms(&dets, 0.5, 2).len(), 2);
    }

    #[test]
    fn letterbox_centers_and_normalizes() {
        // 2x1 image (wide) into side=4: scale=2, nw=4, nh=2, pad top/bottom=1.
        let rgba = vec![255, 0, 0, 255, 0, 0, 255, 255];
        let (t, scale, padx, pady) = letterbox(&rgba, 2, 1, 4).unwrap();
        assert_eq!(t.len(), 3 * 16);
        assert!((scale - 2.0).abs() < 1e-6);
        assert_eq!((padx, pady), (0.0, 1.0));
        // Top row is padding (gray).
        assert!((t[0] - PAD).abs() < 1e-6);
        // Row 1 (padded region) has real content in the R plane.
        assert!(t[4] > 0.5);
    }

    #[test]
    fn segment_everything_rejects_overflow() {
        let m = Path::new("/nonexistent/FastSAM-x.onnx");
        assert!(segment_everything(&[], usize::MAX, 2, m, &SegConfig::default()).is_err());
    }
}
