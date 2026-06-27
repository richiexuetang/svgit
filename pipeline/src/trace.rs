//! Owned tracer — orchestrates the Level-2 pipeline end to end:
//! quantized raster → palette indices → connected-components segmentation →
//! contour extraction → RDP simplification → layered polygon SVG.
//!
//! The input is expected to already be color-reduced (see [`crate::quantize`]);
//! the tracer derives its palette from the distinct colors present, so feeding
//! it a full-color photo would produce one region per unique color.

use std::collections::HashMap;

use crate::contour::contours_of;
use crate::segment::segment;
use crate::simplify::simplify_closed;
use crate::svg::{to_svg, Region};

#[derive(Debug, Clone)]
pub struct TraceConfig {
    /// Pixels with alpha at or below this are treated as transparent (not drawn).
    pub alpha_threshold: u8,
    /// Merge regions smaller than this many pixels into their largest neighbour.
    pub min_area: u32,
    /// RDP simplification tolerance in pixels (0 = keep exact staircase edges).
    pub simplify: f64,
    /// Emit the largest region as a background rect instead of a full polygon.
    pub background: bool,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            alpha_threshold: 0,
            min_area: 4,
            simplify: 1.2,
            background: true,
        }
    }
}

/// Trace an (already quantized) RGBA buffer into a flat-color SVG document.
pub fn trace_rgba(pixels: &[u8], width: usize, height: usize, cfg: &TraceConfig) -> String {
    let n = width * height;
    if n == 0 || pixels.len() < n * 4 {
        return to_svg(width, height, None, &[]);
    }

    // --- build palette indices: 0 = transparent, 1.. = opaque colors ---
    let mut idx = vec![0u32; n];
    let mut palette: Vec<[u8; 3]> = Vec::new();
    let mut map: HashMap<u32, u32> = HashMap::new();
    for i in 0..n {
        let a = pixels[i * 4 + 3];
        if a <= cfg.alpha_threshold {
            continue; // idx stays 0
        }
        let (r, g, b) = (pixels[i * 4], pixels[i * 4 + 1], pixels[i * 4 + 2]);
        let key = (r as u32) << 16 | (g as u32) << 8 | b as u32;
        let ci = *map.entry(key).or_insert_with(|| {
            palette.push([r, g, b]);
            palette.len() as u32 // first opaque color -> 1
        });
        idx[i] = ci;
    }

    // --- segment + merge speckles ---
    let mut seg = segment(&idx, width, height);
    seg.merge_small(cfg.min_area);
    let bboxes = seg.bboxes();

    // --- choose a background region (largest opaque) to lay down as a rect ---
    let mut bg_region: Option<usize> = None;
    if cfg.background {
        let mut best_area = 0u32;
        for c in 0..seg.num_components {
            if seg.component_color[c] != 0 && seg.component_area[c] > best_area {
                best_area = seg.component_area[c];
                bg_region = Some(c);
            }
        }
    }
    let background = bg_region.map(|c| palette[(seg.component_color[c] - 1) as usize]);

    // --- trace every opaque region (except the background) ---
    let mut regions: Vec<Region> = Vec::new();
    for c in 0..seg.num_components {
        if seg.component_color[c] == 0 || Some(c) == bg_region {
            continue;
        }
        let color = palette[(seg.component_color[c] - 1) as usize];
        let raw_loops = contours_of(&seg.labels, width, height, c as u32, bboxes[c]);
        let mut loops = Vec::with_capacity(raw_loops.len());
        for lp in raw_loops {
            let simp = simplify_closed(&lp, cfg.simplify);
            if simp.len() >= 3 {
                loops.push(simp);
            }
        }
        if !loops.is_empty() {
            regions.push(Region { color, loops });
        }
    }

    to_svg(width, height, background, &regions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba(colors: &[[u8; 4]]) -> Vec<u8> {
        colors.iter().flat_map(|c| c.iter().copied()).collect()
    }

    #[test]
    fn two_flat_regions_produce_valid_svg() {
        // 2x2: top row red, bottom row blue.
        let px = rgba(&[
            [255, 0, 0, 255],
            [255, 0, 0, 255],
            [0, 0, 255, 255],
            [0, 0, 255, 255],
        ]);
        let svg = trace_rgba(&px, 2, 2, &TraceConfig { min_area: 0, simplify: 0.0, ..Default::default() });
        assert!(svg.starts_with("<?xml"));
        assert!(svg.contains("<svg"));
        assert!(svg.ends_with("</svg>\n"));
        // Background rect (one color) + one path (the other region).
        assert!(svg.contains("<rect"));
        assert_eq!(svg.matches("<path").count(), 1);
        // Both colors appear.
        assert!(svg.contains("#ff0000"));
        assert!(svg.contains("#0000ff"));
    }

    #[test]
    fn transparent_pixels_are_not_drawn() {
        // One opaque green, three transparent.
        let px = rgba(&[
            [0, 255, 0, 255],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]);
        let svg = trace_rgba(&px, 2, 2, &TraceConfig { background: false, min_area: 0, simplify: 0.0, ..Default::default() });
        assert_eq!(svg.matches("<path").count(), 1);
        assert!(svg.contains("#00ff00"));
        assert!(!svg.contains("<rect"));
    }

    #[test]
    fn probe_diagonal_holes_full_pipeline_default_cfg() {
        // 4x4 background color A with two diagonally-touching holes of color B
        // at cells (1,1) and (2,2). With default simplify=1.2 the holes get
        // mangled into a degenerate triangle / dropped.
        let a = [10u8, 20, 30, 255];
        let b = [200u8, 100, 50, 255];
        let mut px = vec![];
        let layout = [
            0,0,0,0,
            0,1,0,0,
            0,0,1,0,
            0,0,0,0,
        ];
        for &v in &layout {
            px.extend_from_slice(if v == 0 { &a } else { &b });
        }
        // Default cfg has background=true, simplify=1.2, min_area=4.
        // Use min_area=0 so the 1px holes survive as their own components.
        let cfg = TraceConfig { min_area: 0, ..Default::default() };
        let svg = trace_rgba(&px, 4, 4, &cfg);
        eprintln!("SVG:\n{svg}");
        // Count distinct fill colors actually drawn.
        let n_b = svg.matches("#c86432").count(); // b = (200,100,50)
        eprintln!("paths of color B (the two holes) = {n_b}");
    }

    #[test]
    fn empty_image_is_valid_svg() {
        let svg = trace_rgba(&[], 0, 0, &TraceConfig::default());
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
    }

}
