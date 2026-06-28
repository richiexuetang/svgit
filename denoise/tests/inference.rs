//! End-to-end inference test. Ignored by default because it needs the SCUNet
//! weights (`scripts/fetch-models.sh`); CI without them simply skips it.
//!
//! Run with:
//!   cargo test -p svgit-denoise --test inference -- --ignored --nocapture
//!
//! Uses a synthetic image at a non-multiple-of-64 size (exercises the reflect
//! padding) to verify the output is the same size and channel order is preserved.
//! With SVGIT_TEST_IMAGE set it additionally denoises a real image, times it, and
//! writes `denoise-out.png` into SVGIT_TEST_OUT for eyeballing.

use std::time::Instant;
use svgit_denoise::{default_model_path, denoise};

#[test]
#[ignore = "needs SCUNet weights"]
fn denoises_same_size_and_preserves_color() {
    let path = default_model_path();
    if !path.exists() {
        eprintln!("model missing at {} — skipping", path.display());
        return;
    }

    // 100×70 (not a multiple of 64): left half red, right half blue, opaque,
    // with a little high-frequency checker noise the denoiser should calm.
    let (w, h) = (100usize, 70usize);
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let p = (y * w + x) * 4;
            let noise = if (x + y) % 2 == 0 { 25 } else { 0 };
            if x < w / 2 {
                rgba[p] = 200 + noise as u8; // R
                rgba[p + 1] = noise as u8;
                rgba[p + 2] = noise as u8;
            } else {
                rgba[p] = noise as u8;
                rgba[p + 1] = noise as u8;
                rgba[p + 2] = (200 + noise) as u8; // B
            }
            rgba[p + 3] = 255;
        }
    }

    let t0 = Instant::now();
    let out = denoise(&rgba, w, h, &path).expect("denoise should succeed");
    eprintln!("denoise {w}×{h} took {} ms", t0.elapsed().as_millis());
    assert_eq!(out.len(), rgba.len(), "same-size RGBA");

    let at = |x: usize, y: usize| {
        let p = (y * w + x) * 4;
        (out[p], out[p + 1], out[p + 2], out[p + 3])
    };
    let (lr, _lg, lb, la) = at(w / 4, h / 2);
    let (rr, _rg, rb, ra) = at(3 * w / 4, h / 2);
    eprintln!("left=({lr},_,{lb}) right=({rr},_,{rb})");
    assert!(lr > lb + 40, "left half should stay RED, got R={lr} B={lb}");
    assert!(rb > rr + 40, "right half should stay BLUE, got R={rr} B={rb}");
    assert_eq!((la, ra), (255, 255), "alpha preserved");

    // Optionally denoise a real image and time it (useful for tuning the cap).
    if let Ok(img_path) = std::env::var("SVGIT_TEST_IMAGE") {
        let decoded = image::open(&img_path).expect("decode test image").to_rgba8();
        let (iw, ih) = (decoded.width() as usize, decoded.height() as usize);
        let raw = decoded.into_raw();
        let t1 = Instant::now();
        let cleaned = denoise(&raw, iw, ih, &path).expect("denoise real image");
        eprintln!("denoise real {iw}×{ih} took {} ms", t1.elapsed().as_millis());
        let out_dir = std::env::var("SVGIT_TEST_OUT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir());
        let out_path = out_dir.join("denoise-out.png");
        image::save_buffer(
            &out_path,
            &cleaned,
            iw as u32,
            ih as u32,
            image::ColorType::Rgba8,
        )
        .expect("write denoise png");
        eprintln!("wrote {}", out_path.display());
    }
}
