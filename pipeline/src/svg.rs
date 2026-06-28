//! SVG serialization — stage 5 of the owned pipeline, plus the geometry types
//! the contour/curve-fit stages produce.
//!
//! Emits one `<path>` per region (holes are extra subpaths in the same path,
//! rendered with the nonzero fill rule thanks to their opposite winding). An
//! optional background `<rect>` lets the caller drop the single largest region,
//! which the others then stack on top of — keeping the path count low.

use std::collections::BTreeMap;
use std::fmt::Write;

type Pt = (i32, i32);
pub type Ptf = (f64, f64);

/// A single path segment. Polygon output uses only `Line`; curve fitting emits
/// `Cubic` (two control points + endpoint; the start point is implicit).
pub enum Seg {
    Line(Ptf),
    Cubic(Ptf, Ptf, Ptf),
}

/// A closed subpath: a start point followed by segments back around to it.
pub struct Subpath {
    pub start: Ptf,
    pub segs: Vec<Seg>,
}

/// A linear (axial) gradient fill in user space: a color ramp along the segment
/// `(x1,y1)→(x2,y2)`, with stops at offsets in `[0,1]`.
#[derive(Clone)]
pub struct LinearGradient {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub stops: Vec<(f64, [u8; 3])>,
}

/// A radial gradient fill in user space: concentric rings of color centred at
/// `(cx,cy)`, the stops mapped across radius `0..r`.
#[derive(Clone)]
pub struct RadialGradient {
    pub cx: f64,
    pub cy: f64,
    pub r: f64,
    pub stops: Vec<(f64, [u8; 3])>,
}

/// How a region is painted: a flat color (the common case, merged by color in
/// the output) or a fitted gradient (emitted as its own `<path>` referencing a
/// `<linearGradient>`/`<radialGradient>` in `<defs>`; identical gradients shared
/// by several regions — e.g. merged quantization bands — collapse to one def).
#[derive(Clone)]
pub enum Fill {
    Solid([u8; 3]),
    Linear(LinearGradient),
    Radial(RadialGradient),
}

pub struct Region {
    pub fill: Fill,
    /// Outer subpath plus any hole subpaths.
    pub subpaths: Vec<Subpath>,
}

/// Build a closed polygon subpath (all line segments) from integer corners.
pub fn polygon_subpath(pts: &[Pt]) -> Subpath {
    let start = (pts[0].0 as f64, pts[0].1 as f64);
    let segs = pts[1..]
        .iter()
        .map(|&p| Seg::Line((p.0 as f64, p.1 as f64)))
        .collect();
    Subpath { start, segs }
}

/// Build a closed polygon subpath from float corners (e.g. edge-snapped points).
pub fn polygon_subpath_f(pts: &[Ptf]) -> Subpath {
    let start = pts.first().copied().unwrap_or((0.0, 0.0));
    let segs = pts.get(1..).unwrap_or(&[]).iter().map(|&p| Seg::Line(p)).collect();
    Subpath { start, segs }
}

fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// Format a coordinate with up to 2 decimals, trimming trailing zeros.
fn fnum(v: f64) -> String {
    if !v.is_finite() {
        return "0".to_string(); // never emit NaN/inf into path data
    }
    let r = (v * 100.0).round() / 100.0;
    if r == r.trunc() {
        format!("{}", r as i64)
    } else {
        let s = format!("{r:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

fn write_subpath(d: &mut String, sp: &Subpath) {
    if sp.segs.len() < 2 {
        return;
    }
    let _ = write!(d, "M{} {}", fnum(sp.start.0), fnum(sp.start.1));
    for seg in &sp.segs {
        match seg {
            Seg::Line(p) => {
                let _ = write!(d, "L{} {}", fnum(p.0), fnum(p.1));
            }
            Seg::Cubic(c1, c2, e) => {
                let _ = write!(
                    d,
                    "C{} {} {} {} {} {}",
                    fnum(c1.0),
                    fnum(c1.1),
                    fnum(c2.0),
                    fnum(c2.1),
                    fnum(e.0),
                    fnum(e.1)
                );
            }
        }
    }
    d.push('Z');
}

fn write_stops(s: &mut String, stops: &[(f64, [u8; 3])]) {
    for (off, c) in stops {
        let _ = write!(s, "<stop offset=\"{}\" stop-color=\"{}\"/>", fnum(*off), hex(*c));
    }
}

/// The element name plus everything after the `id="…"` (attributes, `>`, stops),
/// for a gradient fill. The returned string doubles as a dedup key: two regions
/// with byte-identical geometry+stops (e.g. merged quantization bands sharing one
/// fitted gradient) produce the same key and thus share a single `<defs>` entry.
fn grad_inner(fill: &Fill) -> (&'static str, String) {
    let mut s = String::new();
    match fill {
        Fill::Linear(g) => {
            let _ = write!(
                s,
                "gradientUnits=\"userSpaceOnUse\" x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\">",
                fnum(g.x1),
                fnum(g.y1),
                fnum(g.x2),
                fnum(g.y2)
            );
            write_stops(&mut s, &g.stops);
            ("linearGradient", s)
        }
        Fill::Radial(g) => {
            let _ = write!(
                s,
                "gradientUnits=\"userSpaceOnUse\" cx=\"{}\" cy=\"{}\" r=\"{}\">",
                fnum(g.cx),
                fnum(g.cy),
                fnum(g.r)
            );
            write_stops(&mut s, &g.stops);
            ("radialGradient", s)
        }
        Fill::Solid(_) => ("", String::new()),
    }
}

/// Render a set of regions into `<path>` elements. Solid regions are merged by
/// color into one path each (lossless — the tracer's regions tile, so z-order
/// is irrelevant); a solid region matching `skip_solid` is dropped (it's already
/// painted by a background rect). Gradient regions reference a `<defs>` entry
/// appended to `defs` (ids drawn from `next_id`); identical gradients are emitted
/// once and shared. Returns the path-element body; `defs` is filled as a side
/// effect.
fn render_region_paths(
    regions: &[Region],
    skip_solid: Option<[u8; 3]>,
    defs: &mut String,
    next_id: &mut usize,
    grad_ids: &mut BTreeMap<String, usize>,
) -> String {
    // BTreeMap keeps solid output ordered (deterministic) by color.
    let mut by_color: BTreeMap<[u8; 3], String> = BTreeMap::new();
    let mut grad_paths: Vec<(usize, String)> = Vec::new();
    for r in regions {
        match &r.fill {
            Fill::Solid(c) => {
                if Some(*c) == skip_solid {
                    continue; // already painted by the background rect
                }
                let d = by_color.entry(*c).or_default();
                for sp in &r.subpaths {
                    write_subpath(d, sp);
                }
            }
            grad @ (Fill::Linear(_) | Fill::Radial(_)) => {
                let mut d = String::new();
                for sp in &r.subpaths {
                    write_subpath(&mut d, sp);
                }
                if d.is_empty() {
                    continue;
                }
                let (tag, inner) = grad_inner(grad);
                let key = format!("{tag}|{inner}");
                let id = if let Some(&id) = grad_ids.get(&key) {
                    id
                } else {
                    let id = *next_id;
                    *next_id += 1;
                    let _ = write!(defs, "<{tag} id=\"g{id}\" {inner}</{tag}>");
                    grad_ids.insert(key, id);
                    id
                };
                grad_paths.push((id, d));
            }
        }
    }
    let mut body = String::new();
    for (color, d) in &by_color {
        if d.is_empty() {
            continue;
        }
        let _ = writeln!(
            body,
            "<path d=\"{}\" fill=\"{}\" fill-rule=\"nonzero\"/>",
            d,
            hex(*color)
        );
    }
    for (id, d) in &grad_paths {
        let _ = writeln!(
            body,
            "<path d=\"{}\" fill=\"url(#g{})\" fill-rule=\"nonzero\"/>",
            d, id
        );
    }
    body
}

/// Build a full SVG document from regions, optionally laying the largest region
/// down as a background rectangle.
pub fn to_svg(width: usize, height: usize, background: Option<[u8; 3]>, regions: &[Region]) -> String {
    let mut s = String::with_capacity(1024 + regions.len() * 96);
    s.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    s.push_str("<!-- Generator: svgit owned pipeline -->\n");
    let _ = writeln!(
        s,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
    );
    if let Some(bg) = background {
        let _ = writeln!(
            s,
            "<rect width=\"{width}\" height=\"{height}\" fill=\"{}\"/>",
            hex(bg)
        );
    }
    let mut defs = String::new();
    let mut next_id = 0usize;
    let mut grad_ids = BTreeMap::new();
    let body = render_region_paths(regions, background, &mut defs, &mut next_id, &mut grad_ids);
    if !defs.is_empty() {
        let _ = writeln!(s, "<defs>{defs}</defs>");
    }
    s.push_str(&body);
    s.push_str("</svg>\n");
    s
}

/// Escape the few characters that aren't legal inside a double-quoted XML
/// attribute value. Layer labels are svgit-generated and simple, so this is
/// defensive rather than load-bearing.
fn attr_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Build a full SVG document where each layer becomes a `<g data-object="...">`
/// group, emitted bottom-first. Within a layer, same-color subpaths merge into
/// one `<path>` (deterministic via BTreeMap) exactly as [`to_svg`] does — so an
/// object made of several quantized colors stays a single group of color paths.
pub fn to_svg_layered(width: usize, height: usize, layers: &[(String, Vec<Region>)]) -> String {
    let mut s = String::with_capacity(2048);
    s.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    s.push_str("<!-- Generator: svgit owned pipeline (layered) -->\n");
    let _ = writeln!(
        s,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
    );
    let mut defs = String::new();
    let mut next_id = 0usize;
    let mut grad_ids = BTreeMap::new();
    let mut groups = String::new();
    for (label, regions) in layers {
        let body = render_region_paths(regions, None, &mut defs, &mut next_id, &mut grad_ids);
        if body.is_empty() {
            continue;
        }
        let _ = writeln!(groups, "<g data-object=\"{}\">", attr_escape(label));
        groups.push_str(&body);
        groups.push_str("</g>\n");
    }
    if !defs.is_empty() {
        let _ = writeln!(s, "<defs>{defs}</defs>");
    }
    s.push_str(&groups);
    s.push_str("</svg>\n");
    s
}
