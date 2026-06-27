//! Color quantization — reduce an image to `N` representative colors via
//! k-means clustering in CIELAB space.
//!
//! This is the first owned stage of the Level-2 pipeline and, per the project
//! plan, the single biggest quality lever for tracing: flattening an image to a
//! small set of perceptually-chosen colors yields cleaner regions and far fewer
//! paths than tracing raw photographic noise.

use crate::color::{lab_to_rgb, rgb_to_lab};

#[derive(Debug, Clone)]
pub struct QuantizeConfig {
    /// Target number of colors (k). Effectively clamped to at least 1 and to
    /// the number of distinct sampled colors.
    pub num_colors: usize,
    /// Maximum k-means (Lloyd) iterations.
    pub max_iterations: usize,
    /// Hard cap on the number of pixels sampled when fitting centroids. The
    /// full image is always mapped; only centroid *fitting* is subsampled.
    pub max_samples: usize,
    /// Pixels with alpha at or below this are excluded from clustering and
    /// passed through unchanged.
    pub alpha_threshold: u8,
    /// Seed for deterministic initialization, so identical inputs produce
    /// identical output (no flicker across live re-runs).
    pub seed: u64,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            num_colors: 16,
            max_iterations: 20,
            max_samples: 40_000,
            alpha_threshold: 0,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

/// Deterministic PRNG (SplitMix64) — avoids a dependency and keeps output
/// reproducible. SplitMix64 has no degenerate seed, so the seed is used as-is.
struct SplitMix64(u64);

impl SplitMix64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform float in [0, 1).
    #[inline]
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform integer in [0, n). `n` must be > 0.
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[inline]
fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    d0 * d0 + d1 * d1 + d2 * d2
}

#[inline]
fn nearest(lab: [f32; 3], centroids: &[[f32; 3]]) -> usize {
    let mut best = 0;
    let mut bestd = f32::INFINITY;
    for (c, &ce) in centroids.iter().enumerate() {
        let d = dist2(lab, ce);
        if d < bestd {
            bestd = d;
            best = c;
        }
    }
    best
}

/// Pack an RGB triple into a 16-bit RGB565 key. Near-identical colors collapse
/// to the same bucket, which is exactly what we want for memoizing the
/// nearest-centroid lookup: the chosen centroid is the same for colors that
/// differ only in low bits.
#[inline]
fn rgb565(r: u8, g: u8, b: u8) -> usize {
    ((r as usize >> 3) << 11) | ((g as usize >> 2) << 5) | (b as usize >> 3)
}

/// Quantize an RGBA buffer to `cfg.num_colors` colors, in place.
///
/// Takes ownership of the buffer and returns it with each opaque pixel's RGB
/// replaced by its cluster's representative color (alpha preserved). Pixels with
/// alpha ≤ `alpha_threshold` are left untouched.
pub fn quantize_rgba(mut pixels: Vec<u8>, cfg: &QuantizeConfig) -> Vec<u8> {
    let n_px = pixels.len() / 4;
    if n_px == 0 {
        return pixels;
    }

    // --- count opaque pixels (cheap alpha-only pass; no allocation) ---
    let n_opaque = pixels
        .chunks_exact(4)
        .filter(|px| px[3] > cfg.alpha_threshold)
        .count();
    if n_opaque == 0 {
        return pixels;
    }

    // --- pick which opaque pixels to fit centroids from ---
    // Jittered stride sampling: hard-capped at `max_samples`, and the per-window
    // random jitter avoids aliasing against periodic image structure. Only the
    // sampled pixels are converted to LAB.
    let mut rng = SplitMix64(cfg.seed);
    let max_samples = cfg.max_samples.max(1);
    let target = max_samples.min(n_opaque);
    let stride = (n_opaque / target).max(1);
    let mut sample_ordinals: Vec<usize> = Vec::with_capacity(target);
    for m in 0..target {
        let ord = m * stride + rng.below(stride);
        if ord < n_opaque {
            sample_ordinals.push(ord);
        }
    }

    let mut samples: Vec<[f32; 3]> = Vec::with_capacity(sample_ordinals.len());
    {
        let mut next = 0usize; // index into sample_ordinals
        let mut o = 0usize; // running opaque-pixel ordinal
        for px in pixels.chunks_exact(4) {
            if next >= sample_ordinals.len() {
                break;
            }
            if px[3] > cfg.alpha_threshold {
                if o == sample_ordinals[next] {
                    samples.push(rgb_to_lab(px[0], px[1], px[2]));
                    next += 1;
                }
                o += 1;
            }
        }
    }

    let k = cfg.num_colors.max(1).min(samples.len());

    // --- k-means++ initialization ---
    let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(k);
    centroids.push(samples[rng.below(samples.len())]);
    let mut d2: Vec<f32> = samples.iter().map(|&s| dist2(s, centroids[0])).collect();
    while centroids.len() < k {
        let total: f64 = d2.iter().map(|&x| x as f64).sum();
        if total <= 0.0 {
            break; // all remaining samples coincide with a centroid
        }
        // Fall back to the farthest sample (guaranteed positive weight).
        let mut chosen = 0;
        let mut maxd = -1.0f32;
        for (i, &w) in d2.iter().enumerate() {
            if w > maxd {
                maxd = w;
                chosen = i;
            }
        }
        let mut target_w = rng.next_f32() as f64 * total;
        for (i, &w) in d2.iter().enumerate() {
            if w > 0.0 {
                target_w -= w as f64;
                if target_w <= 0.0 {
                    chosen = i;
                    break;
                }
            }
        }
        let c = samples[chosen];
        centroids.push(c);
        for (i, &s) in samples.iter().enumerate() {
            let nd = dist2(s, c);
            if nd < d2[i] {
                d2[i] = nd;
            }
        }
    }

    // --- Lloyd iterations ---
    let kk = centroids.len();
    let mut assign = vec![usize::MAX; samples.len()];
    for _ in 0..cfg.max_iterations {
        let mut changed = false;
        for (i, &s) in samples.iter().enumerate() {
            let best = nearest(s, &centroids);
            if assign[i] != best {
                assign[i] = best;
                changed = true;
            }
        }
        let mut sums = vec![[0.0f64; 3]; kk];
        let mut counts = vec![0u64; kk];
        for (i, &s) in samples.iter().enumerate() {
            let c = assign[i];
            sums[c][0] += s[0] as f64;
            sums[c][1] += s[1] as f64;
            sums[c][2] += s[2] as f64;
            counts[c] += 1;
        }
        for c in 0..kk {
            if counts[c] > 0 {
                let n = counts[c] as f64;
                centroids[c] = [
                    (sums[c][0] / n) as f32,
                    (sums[c][1] / n) as f32,
                    (sums[c][2] / n) as f32,
                ];
            }
        }
        if !changed {
            break;
        }
    }

    // Representative sRGB color per centroid.
    let palette: Vec<[u8; 3]> = centroids.iter().map(|&c| lab_to_rgb(c)).collect();

    // --- map every opaque pixel to its nearest centroid, in place ---
    // Memoized by a bounded RGB565 table (256 KB) rather than an unbounded
    // hash map: O(1) lookup, no growth, and a high hit rate on real images.
    let mut lut = vec![i16::MIN; 1 << 16];
    for px in pixels.chunks_exact_mut(4) {
        if px[3] <= cfg.alpha_threshold {
            continue;
        }
        let key = rgb565(px[0], px[1], px[2]);
        let ci = if lut[key] >= 0 {
            lut[key] as usize
        } else {
            let c = nearest(rgb_to_lab(px[0], px[1], px[2]), &centroids);
            lut[key] = c as i16;
            c
        };
        let rep = palette[ci];
        px[0] = rep[0];
        px[1] = rep[1];
        px[2] = rep[2];
        // px[3] (alpha) preserved
    }

    pixels
}

/// Count the distinct opaque RGB colors in an RGBA buffer. Diagnostic helper.
pub fn distinct_colors(pixels: &[u8], alpha_threshold: u8) -> usize {
    let mut set = std::collections::HashSet::new();
    for px in pixels.chunks_exact(4) {
        if px[3] > alpha_threshold {
            set.insert((px[0] as u32) << 16 | (px[1] as u32) << 8 | px[2] as u32);
        }
    }
    set.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an RGBA buffer from a list of opaque RGB triples.
    fn rgba(colors: &[[u8; 3]]) -> Vec<u8> {
        let mut v = Vec::with_capacity(colors.len() * 4);
        for c in colors {
            v.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
        v
    }

    #[test]
    fn reduces_to_requested_color_count() {
        // Four well-separated colors, asked for two -> exactly two distinct out.
        let img = rgba(&[[255, 0, 0], [250, 5, 5], [0, 0, 255], [5, 5, 250]]);
        let out = quantize_rgba(img, &QuantizeConfig { num_colors: 2, ..Default::default() });
        assert_eq!(distinct_colors(&out, 0), 2);
    }

    #[test]
    fn single_color_target_flattens_everything() {
        let img = rgba(&[[10, 20, 30], [200, 100, 50], [0, 255, 0]]);
        let out = quantize_rgba(img, &QuantizeConfig { num_colors: 1, ..Default::default() });
        assert_eq!(distinct_colors(&out, 0), 1);
    }

    #[test]
    fn never_exceeds_requested_colors() {
        // A spread of colors quantized to 5 -> at most 5 distinct out.
        let mut cols = Vec::new();
        for i in 0..50u8 {
            cols.push([i.wrapping_mul(5), 255 - i.wrapping_mul(3), i.wrapping_mul(2)]);
        }
        let out = quantize_rgba(rgba(&cols), &QuantizeConfig { num_colors: 5, ..Default::default() });
        assert!(distinct_colors(&out, 0) <= 5);
    }

    #[test]
    fn is_deterministic() {
        let img = rgba(&[[10, 20, 30], [200, 100, 50], [0, 255, 0], [120, 120, 120]]);
        let a = quantize_rgba(img.clone(), &QuantizeConfig { num_colors: 2, ..Default::default() });
        let b = quantize_rgba(img, &QuantizeConfig { num_colors: 2, ..Default::default() });
        assert_eq!(a, b);
    }

    #[test]
    fn preserves_transparent_pixels() {
        // One opaque, one fully transparent with distinctive RGB.
        let img = vec![255u8, 0, 0, 255, 7, 8, 9, 0];
        let out = quantize_rgba(img.clone(), &QuantizeConfig { num_colors: 4, ..Default::default() });
        // Transparent pixel's bytes are untouched.
        assert_eq!(&out[4..8], &img[4..8]);
    }

    #[test]
    fn handles_empty_and_tiny_inputs() {
        assert!(quantize_rgba(Vec::new(), &QuantizeConfig::default()).is_empty());
        let one = vec![1u8, 2, 3, 255];
        assert_eq!(
            quantize_rgba(one, &QuantizeConfig { num_colors: 8, ..Default::default() }).len(),
            4
        );
    }
}
