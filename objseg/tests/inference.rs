//! End-to-end FastSAM test. Ignored by default (needs the ~289 MB FastSAM-x
//! weights + a real image). Writes `bg-objseg.png`: the original with each
//! detected instance tinted a distinct color, for visual inspection.
//!
//!   SVGIT_TEST_IMAGE=/path/to/photo.jpg SVGIT_MODEL_DIR=$PWD/models \
//!     cargo test -p svgit-objseg --test inference -- --ignored --nocapture

use svgit_objseg::{default_model_path, segment_everything, SegConfig};

const PALETTE: [[u8; 3]; 8] = [
    [255, 64, 64],
    [64, 200, 64],
    [80, 120, 255],
    [240, 200, 40],
    [220, 90, 220],
    [40, 210, 210],
    [255, 140, 40],
    [150, 150, 150],
];

#[test]
#[ignore = "needs FastSAM-x weights + SVGIT_TEST_IMAGE"]
fn segments_objects_from_photo() {
    let img_path = match std::env::var("SVGIT_TEST_IMAGE") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SVGIT_TEST_IMAGE unset — skipping");
            return;
        }
    };
    let model = default_model_path();
    assert!(model.exists(), "model missing at {}", model.display());

    let decoded = image::open(&img_path).expect("decode").to_rgba8();
    let (w, h) = (decoded.width() as usize, decoded.height() as usize);
    let rgba = decoded.into_raw();

    let instances = segment_everything(&rgba, w, h, &model, &SegConfig::default())
        .expect("segment_everything should succeed");
    eprintln!("detected {} instances", instances.len());
    for (i, inst) in instances.iter().enumerate() {
        eprintln!(
            "  #{i}: score={:.3} area={} ({:.1}%) bbox={:?}",
            inst.score,
            inst.area,
            100.0 * inst.area as f64 / (w * h) as f64,
            inst.bbox
        );
    }
    assert!(!instances.is_empty(), "no objects segmented");

    // Overlay: dim the original, then tint each instance (largest first, so
    // smaller objects paint visibly on top).
    let mut out = rgba.clone();
    for p in 0..w * h {
        for c in 0..3 {
            out[p * 4 + c] = (out[p * 4 + c] as u16 * 2 / 5) as u8; // 40% brightness
        }
    }
    for (i, inst) in instances.iter().enumerate() {
        let col = PALETTE[i % PALETTE.len()];
        for p in 0..w * h {
            if inst.mask[p] != 0 {
                for c in 0..3 {
                    out[p * 4 + c] = ((out[p * 4 + c] as u16 + col[c] as u16 * 3) / 4) as u8;
                }
            }
        }
    }

    let out_dir = std::env::var("SVGIT_TEST_OUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    let path = out_dir.join("bg-objseg.png");
    image::save_buffer(&path, &out, w as u32, h as u32, image::ColorType::Rgba8)
        .expect("write overlay");
    eprintln!("wrote {}", path.display());
}
