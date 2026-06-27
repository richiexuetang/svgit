//! SVG serialization — stage 5 of the owned pipeline.
//!
//! Emits one `<path>` per region (holes are extra subpaths in the same path,
//! rendered with the nonzero fill rule thanks to their opposite winding). An
//! optional background `<rect>` lets the caller drop the single largest region,
//! which the others then stack on top of — keeping the path count low.

use std::fmt::Write;

type Pt = (i32, i32);

pub struct Region {
    pub color: [u8; 3],
    /// Outer loop plus any hole loops, as produced by the contour tracer.
    pub loops: Vec<Vec<Pt>>,
}

fn hex(c: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

fn write_loops(d: &mut String, loops: &[Vec<Pt>]) {
    for lp in loops {
        if lp.len() < 3 {
            continue;
        }
        let _ = write!(d, "M{} {}", lp[0].0, lp[0].1);
        for p in &lp[1..] {
            let _ = write!(d, "L{} {}", p.0, p.1);
        }
        d.push('Z');
    }
}

/// Build a full SVG document from regions, optionally laying the largest region
/// down as a background rectangle.
pub fn to_svg(
    width: usize,
    height: usize,
    background: Option<[u8; 3]>,
    regions: &[Region],
) -> String {
    let mut s = String::with_capacity(1024 + regions.len() * 64);
    s.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    s.push_str("<!-- Generator: svgit owned pipeline -->\n");
    let _ = write!(
        s,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\n"
    );
    if let Some(bg) = background {
        let _ = write!(
            s,
            "<rect width=\"{width}\" height=\"{height}\" fill=\"{}\"/>\n",
            hex(bg)
        );
    }
    let mut d = String::new();
    for r in regions {
        d.clear();
        write_loops(&mut d, &r.loops);
        if d.is_empty() {
            continue;
        }
        let _ = write!(s, "<path d=\"{}\" fill=\"{}\"/>\n", d, hex(r.color));
    }
    s.push_str("</svg>\n");
    s
}
