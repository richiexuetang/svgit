//! sRGB ↔ CIELAB conversion (D65 white point).
//!
//! The quantizer clusters in LAB so that Euclidean distance approximates
//! perceptual difference — the reason LAB k-means produces noticeably better
//! palettes than clustering in raw RGB.

#[inline]
fn srgb_to_linear(u: u8) -> f32 {
    let c = u as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let v = if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (v * 255.0).round().clamp(0.0, 255.0) as u8
}

// D65 reference white.
const XN: f32 = 0.95047;
const YN: f32 = 1.0;
const ZN: f32 = 1.08883;

#[inline]
fn lab_f(t: f32) -> f32 {
    const D: f32 = 6.0 / 29.0;
    if t > D * D * D {
        t.cbrt()
    } else {
        t / (3.0 * D * D) + 4.0 / 29.0
    }
}

#[inline]
fn lab_finv(t: f32) -> f32 {
    const D: f32 = 6.0 / 29.0;
    if t > D {
        t * t * t
    } else {
        3.0 * D * D * (t - 4.0 / 29.0)
    }
}

/// Convert an sRGB triple (0–255) to CIELAB.
#[inline]
pub fn rgb_to_lab(r: u8, g: u8, b: u8) -> [f32; 3] {
    let (rl, gl, bl) = (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b));
    let x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
    let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
    let z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl;
    let fx = lab_f(x / XN);
    let fy = lab_f(y / YN);
    let fz = lab_f(z / ZN);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// Convert a CIELAB triple back to an sRGB triple (0–255), clamped to gamut.
#[inline]
pub fn lab_to_rgb(lab: [f32; 3]) -> [u8; 3] {
    let [l, a, b] = lab;
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let x = XN * lab_finv(fx);
    let y = YN * lab_finv(fy);
    let z = ZN * lab_finv(fz);
    let rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let gl = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    [linear_to_srgb(rl), linear_to_srgb(gl), linear_to_srgb(bl)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_is_near_identity() {
        // LAB is wider than sRGB, but sRGB → LAB → sRGB should return close to
        // the original for in-gamut colors.
        for &(r, g, b) in &[
            (0, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 128, 64),
            (12, 200, 240),
            (130, 70, 200),
        ] {
            let [r2, g2, b2] = lab_to_rgb(rgb_to_lab(r, g, b));
            assert!((r as i32 - r2 as i32).abs() <= 2, "r {r}->{r2}");
            assert!((g as i32 - g2 as i32).abs() <= 2, "g {g}->{g2}");
            assert!((b as i32 - b2 as i32).abs() <= 2, "b {b}->{b2}");
        }
    }

    #[test]
    fn black_and_white_extremes() {
        assert_eq!(rgb_to_lab(0, 0, 0)[0], 0.0);
        assert!((rgb_to_lab(255, 255, 255)[0] - 100.0).abs() < 0.1);
    }
}
