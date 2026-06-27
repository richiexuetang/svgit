//! Boundary extraction — trace the outline of a labeled region into closed
//! integer polygons on the pixel-corner lattice.
//!
//! Stage 3 of the owned pipeline. For a component we collect every unit edge
//! that separates an "inside" cell from an "outside" cell, oriented so the
//! interior is always on the right, then stitch those edges into closed loops.
//! A solid region yields one loop; a region with holes yields one outer loop
//! plus one loop per hole, with opposite winding — exactly what the nonzero
//! fill rule needs to punch the holes out.

use std::collections::HashMap;

type Pt = (i32, i32);

#[inline]
fn unit(a: Pt, b: Pt) -> Pt {
    ((b.0 - a.0).signum(), (b.1 - a.1).signum())
}

/// Trace all boundary loops of `label` within its bounding box (inclusive).
/// Each loop is a list of lattice corners with collinear runs collapsed; the
/// first point is not repeated at the end.
pub fn contours_of(
    labels: &[u32],
    width: usize,
    height: usize,
    label: u32,
    bbox: (usize, usize, usize, usize),
) -> Vec<Vec<Pt>> {
    let (min_x, min_y, max_x, max_y) = bbox;
    let is = |x: i64, y: i64| -> bool {
        x >= 0
            && y >= 0
            && (x as usize) < width
            && (y as usize) < height
            && labels[y as usize * width + x as usize] == label
    };

    // Collect directed boundary edges (interior on the right).
    let mut edges: Vec<(Pt, Pt)> = Vec::new();
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if labels[y * width + x] != label {
                continue;
            }
            let (xi, yi) = (x as i32, y as i32);
            let (xl, yl) = (x as i64, y as i64);
            // up
            if !is(xl, yl - 1) {
                edges.push(((xi, yi), (xi + 1, yi)));
            }
            // right
            if !is(xl + 1, yl) {
                edges.push(((xi + 1, yi), (xi + 1, yi + 1)));
            }
            // down
            if !is(xl, yl + 1) {
                edges.push(((xi + 1, yi + 1), (xi, yi + 1)));
            }
            // left
            if !is(xl - 1, yl) {
                edges.push(((xi, yi + 1), (xi, yi)));
            }
        }
    }

    // Index edges by their start corner.
    let mut start_map: HashMap<Pt, Vec<usize>> = HashMap::new();
    for (i, e) in edges.iter().enumerate() {
        start_map.entry(e.0).or_default().push(i);
    }

    let mut used = vec![false; edges.len()];
    let mut loops: Vec<Vec<Pt>> = Vec::new();

    for e0 in 0..edges.len() {
        if used[e0] {
            continue;
        }
        let start0 = edges[e0].0;
        let mut pts: Vec<Pt> = Vec::new();
        let mut cur = e0;
        let mut closed = false;
        loop {
            used[cur] = true;
            pts.push(edges[cur].0);
            let end = edges[cur].1;
            if end == start0 {
                closed = true;
                break;
            }
            let din = unit(edges[cur].0, end);
            let Some(cands) = start_map.get(&end) else {
                break;
            };
            // Pick the next edge; at a junction keep the interior on the right
            // by preferring the sharpest right turn.
            let mut next = None;
            let mut best_key = 9;
            for &ei in cands {
                if used[ei] {
                    continue;
                }
                let d = unit(edges[ei].0, edges[ei].1);
                let cross = din.0 * d.1 - din.1 * d.0;
                let dot = din.0 * d.0 + din.1 * d.1;
                let key = if cross > 0 {
                    0
                } else if cross == 0 && dot > 0 {
                    1
                } else if cross < 0 {
                    2
                } else {
                    3
                };
                if key < best_key {
                    best_key = key;
                    next = Some(ei);
                }
            }
            match next {
                Some(ei) => cur = ei,
                None => break,
            }
        }
        // Only emit a loop that actually closed; an early break (dead end)
        // would otherwise produce an unclosed, malformed polygon.
        if closed && pts.len() >= 4 {
            let collapsed = collapse_collinear(&pts);
            if collapsed.len() >= 3 {
                loops.push(collapsed);
            }
        }
    }

    loops
}

/// Drop vertices that lie on a straight run, merging collinear unit edges into
/// a single segment. Operates on a closed loop (wraps around).
fn collapse_collinear(pts: &[Pt]) -> Vec<Pt> {
    let n = pts.len();
    if n < 3 {
        return pts.to_vec();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let prev = pts[(i + n - 1) % n];
        let cur = pts[i];
        let next = pts[(i + 1) % n];
        let d1 = (cur.0 - prev.0, cur.1 - prev.1);
        let d2 = (next.0 - cur.0, next.1 - cur.1);
        let cross = d1.0 * d2.1 - d1.1 * d2.0;
        if cross != 0 {
            out.push(cur);
        }
    }
    out
}

/// Signed area (shoelace) of a polygon; sign indicates winding.
pub fn signed_area(pts: &[Pt]) -> i64 {
    let n = pts.len();
    let mut a: i64 = 0;
    for i in 0..n {
        let p = pts[i];
        let q = pts[(i + 1) % n];
        a += p.0 as i64 * q.1 as i64 - q.0 as i64 * p.1 as i64;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_square_one_loop() {
        // 2x2 all label 0.
        let labels = vec![0u32; 4];
        let loops = contours_of(&labels, 2, 2, 0, (0, 0, 1, 1));
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].len(), 4); // four corners
        assert_eq!(signed_area(&loops[0]).abs(), 2 * 2 * 2); // |2A| = 8 -> area 4
    }

    #[test]
    fn donut_has_outer_and_hole_with_opposite_winding() {
        // 3x3, center (label 1) is a hole inside the ring (label 0).
        let labels = vec![0, 0, 0, 0, 1, 0, 0, 0, 0];
        let loops = contours_of(&labels, 3, 3, 0, (0, 0, 2, 2));
        assert_eq!(loops.len(), 2, "outer + hole");
        let a0 = signed_area(&loops[0]);
        let a1 = signed_area(&loops[1]);
        assert!(
            a0.signum() != a1.signum(),
            "outer and hole must wind oppositely (got {a0}, {a1})"
        );
        // Outer perimeter (3x3) has |2A| = 18; hole (1x1) has |2A| = 2.
        let mut areas = [a0.abs(), a1.abs()];
        areas.sort_unstable();
        assert_eq!(areas, [2, 18]);
    }

    #[test]
    fn l_shape_single_loop() {
        // 2x2 minus bottom-right cell -> L tromino, label 0; other is label 1.
        let labels = vec![0, 0, 0, 1];
        let loops = contours_of(&labels, 2, 2, 0, (0, 0, 1, 1));
        assert_eq!(loops.len(), 1);
        // L-shape outline has 6 corners.
        assert_eq!(loops[0].len(), 6);
    }
}
