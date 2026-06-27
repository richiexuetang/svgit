//! End-to-end inference test. Ignored by default because it needs both the
//! u2netp weights (`scripts/fetch-models.sh`) and a real image; CI without
//! either simply skips it.
//!
//! Run with (SVGIT_TEST_MODEL = u2netp | isnet):
//!   SVGIT_TEST_IMAGE=/path/to/photo.jpg SVGIT_TEST_MODEL=isnet \
//!     cargo test -p svgit-bgremove --test inference -- --ignored --nocapture
//!
//! It writes `bg-matte-<model>.png` (the alpha matte) and `bg-cutout-<model>.png`
//! (the RGBA result) into SVGIT_TEST_OUT for eyeballing.

use svgit_bgremove::{default_model_path, remove_background, BgConfig, Model};

#[test]
#[ignore = "needs weights + SVGIT_TEST_IMAGE"]
fn cutout_splits_foreground_from_background() {
    let img_path = match std::env::var("SVGIT_TEST_IMAGE") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("SVGIT_TEST_IMAGE unset — skipping");
            return;
        }
    };
    let model_name = std::env::var("SVGIT_TEST_MODEL").unwrap_or_else(|_| "u2netp".into());
    let model = Model::parse(&model_name);
    let path = default_model_path(model);
    assert!(
        path.exists(),
        "model missing at {} — run scripts/fetch-models.sh",
        path.display()
    );

    let decoded = image::open(&img_path).expect("decode test image").to_rgba8();
    let (w, h) = (decoded.width() as usize, decoded.height() as usize);
    let rgba = decoded.into_raw();

    let out = remove_background(&rgba, w, h, model, &path, &BgConfig::default())
        .expect("remove_background should succeed");
    assert_eq!(out.len(), rgba.len(), "output is same-sized RGBA");

    // The matte must actually separate something: both transparent and opaque
    // pixels should exist (a uniform result means the net/threshold did nothing).
    let mut transparent = 0usize;
    let mut opaque = 0usize;
    for i in 0..w * h {
        match out[i * 4 + 3] {
            0 => transparent += 1,
            255 => opaque += 1,
            _ => {}
        }
    }
    let pct_fg = 100.0 * opaque as f64 / (w * h) as f64;
    eprintln!(
        "{w}x{h}: {opaque} opaque ({pct_fg:.1}% foreground), {transparent} transparent"
    );
    assert!(opaque > 0, "no foreground detected");
    assert!(transparent > 0, "no background removed");
    // A salient-object cutout should keep a meaningful but not total foreground.
    assert!(
        (1.0..99.0).contains(&pct_fg),
        "implausible foreground fraction {pct_fg:.1}%"
    );

    // Dump artifacts for visual inspection into SVGIT_TEST_OUT (default: temp).
    let out_dir = std::env::var("SVGIT_TEST_OUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    let matte: Vec<u8> = (0..w * h).map(|i| out[i * 4 + 3]).collect();
    let matte_path = out_dir.join(format!("bg-matte-{model_name}.png"));
    let cutout_path = out_dir.join(format!("bg-cutout-{model_name}.png"));
    image::save_buffer(&matte_path, &matte, w as u32, h as u32, image::ColorType::L8)
        .expect("write matte png");
    image::save_buffer(&cutout_path, &out, w as u32, h as u32, image::ColorType::Rgba8)
        .expect("write cutout png");
    eprintln!("wrote {} and {}", matte_path.display(), cutout_path.display());
}
