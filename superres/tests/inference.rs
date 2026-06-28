//! End-to-end inference test. Ignored by default because it needs the
//! realesr-general-x4v3 weights (`scripts/fetch-models.sh`); CI without them
//! simply skips it.
//!
//! Run with:
//!   cargo test -p svgit-superres --test inference -- --ignored --nocapture
//!
//! Uses a synthetic red/blue split image (no external asset needed) to verify
//! both the 4× upscale and — critically — that channel order is preserved
//! (RealESRGAN's reference pipeline is BGR via cv2; a wrong export would swap
//! red↔blue). With SVGIT_TEST_IMAGE set it additionally upscales a real image
//! and writes `sr-out.png` into SVGIT_TEST_OUT for eyeballing.

use svgit_superres::{default_model_path, super_resolve, SCALE};

#[test]
#[ignore = "needs realesr-general-x4v3 weights"]
fn upscales_4x_and_preserves_color() {
    let path = default_model_path();
    if !path.exists() {
        eprintln!("model missing at {} — skipping", path.display());
        return;
    }

    // 32×16: left half pure red, right half pure blue, fully opaque.
    let (w, h) = (32usize, 16usize);
    let mut rgba = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let p = (y * w + x) * 4;
            if x < w / 2 {
                rgba[p] = 230; // R
            } else {
                rgba[p + 2] = 230; // B
            }
            rgba[p + 3] = 255;
        }
    }

    let out = super_resolve(&rgba, w, h, &path).expect("super_resolve should succeed");
    assert_eq!((out.width, out.height), (w * SCALE, h * SCALE), "4× dims");
    assert_eq!(out.rgba.len(), out.width * out.height * 4);

    // Sample a pixel well inside each half (avoid the seam) at the new size.
    let at = |x: usize, y: usize| {
        let p = (y * out.width + x) * 4;
        (out.rgba[p], out.rgba[p + 1], out.rgba[p + 2])
    };
    let cy = out.height / 2;
    let (lr, _lg, lb) = at(out.width / 4, cy); // left → should be red
    let (rr, _rg, rb) = at(3 * out.width / 4, cy); // right → should be blue
    eprintln!("left=({lr},_,{lb}) right=({rr},_,{rb})");
    assert!(lr > lb + 40, "left half should stay RED, got R={lr} B={lb}");
    assert!(rb > rr + 40, "right half should stay BLUE, got R={rr} B={rb}");

    // Optionally upscale a real image for visual inspection.
    if let Ok(img_path) = std::env::var("SVGIT_TEST_IMAGE") {
        let decoded = image::open(&img_path).expect("decode test image").to_rgba8();
        let (iw, ih) = (decoded.width() as usize, decoded.height() as usize);
        let up = super_resolve(&decoded.into_raw(), iw, ih, &path).expect("upscale real image");
        let out_dir = std::env::var("SVGIT_TEST_OUT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir());
        let out_path = out_dir.join("sr-out.png");
        image::save_buffer(
            &out_path,
            &up.rgba,
            up.width as u32,
            up.height as u32,
            image::ColorType::Rgba8,
        )
        .expect("write sr png");
        eprintln!("{iw}x{ih} → {}x{} written to {}", up.width, up.height, out_path.display());
    }
}
