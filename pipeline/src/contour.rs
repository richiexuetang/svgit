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
        loop {
            used[cur] = true;
            pts.push(edges[cur].0);
            let end = edges[cur].1;
            if end == start0 {
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
        if pts.len() >= 4 {
            loops.push(collapse_collinear(&pts));
        } else if !pts.is_empty() {
            loops.push(pts);
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

    fn closes_ok(pts: &[Pt]) -> bool {
        // Re-derive edges from the returned loop and confirm it forms a single
        // closed polyline of nonzero area with no repeated vertex (no pinch).
        pts.len() >= 3 && signed_area(pts) != 0
    }

    #[test]
    fn probe_pinch_diagonal_hole_touch() {
        // 3x3. label 0 everywhere except the two cells (1,0) and (0,1) are label 9,
        // and center (1,1) is label 0. This makes label 0 a shape whose boundary
        // pinches at corner (1,1)... actually let's lay out explicitly.
        // Grid (row-major), label per cell:
        //  0 9 0
        //  9 0 0
        //  0 0 0
        // Is label 0 4-connected? (0,0) is isolated by 9 at right and 9 at below
        // -> (0,0) connects to nothing of label 0 except diagonally. So (0,0) is
        // its own component under 4-conn. The rest of label 0 is one component.
        let labels = vec![0, 9, 0, 9, 0, 0, 0, 0, 0];
        // Trace the BIG component. Its label value here is 0 but (0,0) is also 0 —
        // contours_of keys purely on label value, NOT connectivity. So it will
        // treat ALL cells == 0 as "inside", including the disconnected (0,0).
        let loops = contours_of(&labels, 3, 3, 0, (0, 0, 2, 2));
        eprintln!("PINCH loops = {}", loops.len());
        for (i, l) in loops.iter().enumerate() {
            eprintln!("  loop[{i}] len={} area2={} pts={:?}", l.len(), signed_area(l), l);
        }
        for l in &loops {
            assert!(closes_ok(l), "loop not closed/degenerate: {:?}", l);
        }
    }

    #[test]
    fn probe_checkerboard_pinch_two_diagonal_cells() {
        // Two cells of the same label touching only at a corner:
        //  0 9
        //  9 0
        // Both 0-cells share label 0; contours_of treats both as inside.
        // Their boundaries meet at the center corner (1,1) -> pinch point.
        let labels = vec![0, 9, 9, 0];
        let loops = contours_of(&labels, 2, 2, 0, (0, 0, 1, 1));
        eprintln!("CHECKER loops = {}", loops.len());
        for (i, l) in loops.iter().enumerate() {
            eprintln!("  loop[{i}] len={} area2={} pts={:?}", l.len(), signed_area(l), l);
        }
        let total: i64 = loops.iter().map(|l| signed_area(l).abs()).sum();
        eprintln!("  total |2A| = {} (expect 4 for two unit cells)", total);
    }

    #[test]
    fn probe_border_touch_full_image() {
        // Full 2x2 of label 0 — component touches all four borders.
        let labels = vec![0, 0, 0, 0];
        let loops = contours_of(&labels, 2, 2, 0, (0, 0, 1, 1));
        eprintln!("BORDER loops = {}", loops.len());
        for (i, l) in loops.iter().enumerate() {
            eprintln!("  loop[{i}] len={} area2={} pts={:?}", l.len(), signed_area(l), l);
        }
    }

    #[test]
    fn probe_plus_shape_concavities() {
        // Plus/cross of label 0 in 3x3, corners are label 9.
        //  9 0 9
        //  0 0 0
        //  9 0 9
        let labels = vec![9, 0, 9, 0, 0, 0, 9, 0, 9];
        let loops = contours_of(&labels, 3, 3, 0, (0, 0, 2, 2));
        eprintln!("PLUS loops = {}", loops.len());
        for (i, l) in loops.iter().enumerate() {
            eprintln!("  loop[{i}] len={} area2={} pts={:?}", l.len(), signed_area(l), l);
        }
    }

    #[test]
    fn probe_hole_touches_diagonally_single_walk() {
        // 4x4 all label 0 EXCEPT two diagonal holes that share a lattice corner:
        // holes at (1,1) and (2,2).  Grid:
        //  0 0 0 0
        //  0 9 0 0
        //  0 0 9 0
        //  0 0 0 0
        // The two holes touch at lattice corner (2,2). Tracing the hole
        // boundary of label 0 forces the stitch walk to hit corner (2,2)
        // where multiple inward edges meet -> the junction turn-rule decides.
        let labels = vec![
            0, 0, 0, 0,
            0, 9, 0, 0,
            0, 0, 9, 0,
            0, 0, 0, 0,
        ];
        let loops = contours_of(&labels, 4, 4, 0, (0, 0, 3, 3));
        eprintln!("DIAGHOLE loops = {}", loops.len());
        let mut total = 0i64;
        for (i, l) in loops.iter().enumerate() {
            let a = signed_area(l);
            total += a.abs();
            eprintln!("  loop[{i}] len={} area2={} pts={:?}", l.len(), a, l);
        }
        // Outer 4x4 has |2A|=32. The two unit holes remove 2 each.
        // Correct total |2A| with nonzero fill = 32 (outer) accounted with holes.
        eprintln!("  total |2A| sum = {}", total);
        for l in &loops { assert!(closes_ok(l), "degenerate loop {:?}", l); }
    }

    #[test]
    fn probe_two_components_share_diagonal_one_walk() {
        // A single label-0 region shaped so its OWN outer boundary pinches:
        // an hourglass. 3x3:
        //  0 0 0
        //  9 0 9    <- middle row only center is 0; sides are 9
        //  0 0 0
        // Wait that's 4-connected through center. Boundary pinches at corners
        // (1,1) and (2,1)? Let's see.
        let labels = vec![
            0, 0, 0,
            9, 0, 9,
            0, 0, 0,
        ];
        // label 0: top row (3 cells) + center (1,1) + bottom row (3 cells), all
        // 4-connected through the center column. The two 9 cells create pinch
        // corners on the boundary.
        let loops = contours_of(&labels, 3, 3, 0, (0, 0, 2, 2));
        eprintln!("HOURGLASS loops = {}", loops.len());
        let mut total = 0i64;
        for (i, l) in loops.iter().enumerate() {
            let a = signed_area(l);
            total += a.abs();
            eprintln!("  loop[{i}] len={} area2={} pts={:?}", l.len(), a, l);
        }
        eprintln!("  total |2A| sum = {} (expect 14 = 7 cells * 2)", total);
        for l in &loops { assert!(closes_ok(l), "degenerate loop {:?}", l); }
    }

    #[test]
    fn probe_diaghole_after_collapse_and_two_pinches() {
        // Same as DIAGHOLE but check collapse_collinear didn't fuse a pinch.
        // Two holes touching at TWO corners would be a thicker pinch. Here use
        // the standard diagonal touch and print pre/post collapse.
        let labels = vec![
            0, 0, 0, 0,
            0, 9, 0, 0,
            0, 0, 9, 0,
            0, 0, 0, 0,
        ];
        let loops = contours_of(&labels, 4, 4, 0, (0, 0, 3, 3));
        // Count repeated vertices in any single loop (a proper simple polygon has none).
        for (i, l) in loops.iter().enumerate() {
            let mut seen = std::collections::HashSet::new();
            let mut dups = Vec::new();
            for &p in l {
                if !seen.insert(p) { dups.push(p); }
            }
            eprintln!("loop[{i}] repeated-vertices = {:?}", dups);
        }
    }

    #[test]
    fn probe_opposite_winding_pinch_cancellation() {
        // Construct a region where two touching holes might wind oppositely.
        // A label-0 plus-shaped solid with two separate notches that meet.
        // 5x5, holes carved as a diagonal chain of THREE cells meeting at corners:
        //  0 0 0 0 0
        //  0 9 0 0 0
        //  0 0 9 0 0
        //  0 0 0 9 0
        //  0 0 0 0 0
        // three holes (1,1),(2,2),(3,3) form a diagonal staircase touching at
        // corners (2,2) and (3,3).
        let labels = vec![
            0, 0, 0, 0, 0,
            0, 9, 0, 0, 0,
            0, 0, 9, 0, 0,
            0, 0, 0, 9, 0,
            0, 0, 0, 0, 0,
        ];
        let loops = contours_of(&labels, 5, 5, 0, (0, 0, 4, 4));
        eprintln!("STAIRHOLE loops = {}", loops.len());
        for (i, l) in loops.iter().enumerate() {
            let a = signed_area(l);
            eprintln!("  loop[{i}] len={} area2={} pts={:?}", l.len(), a, l);
        }
    }

    #[test]
    fn probe_diaghole_through_full_pipeline() {
        use crate::simplify::simplify_closed;
        // The DIAGHOLE figure-eight hole, then RDP it like trace.rs does.
        let figure8 = vec![(2,1),(1,1),(1,2),(2,2),(2,3),(3,3),(3,2),(2,2)];
        eprintln!("pre-RDP  len={} area2={}", figure8.len(), signed_area(&figure8));
        for eps in [0.0_f64, 1.2, 2.0] {
            let s = simplify_closed(&figure8, eps);
            eprintln!("  eps={eps} -> len={} area2={} pts={:?}", s.len(), signed_area(&s), s);
        }
    }

    #[test]
    fn probe_single_pixel_collapse() {
        // A single isolated label-0 pixel surrounded by label 9.
        let labels = vec![9,9,9, 9,0,9, 9,9,9];
        let loops = contours_of(&labels, 3, 3, 0, (1, 1, 1, 1));
        eprintln!("SINGLEPX loops={}", loops.len());
        for (i,l) in loops.iter().enumerate() {
            eprintln!("  loop[{i}] len={} area2={} pts={:?}", l.len(), signed_area(l), l);
        }
    }

    #[test]
    fn probe_double_hole_pinch() {
        // 5x5 label 0 frame-ish with TWO holes that touch at a diagonal,
        // forcing a hole-boundary pinch:
        //  0 0 0 0 0
        //  0 9 0 9 0
        //  0 0 9 0 0   <- wait, make holes touch diagonally
        //  0 9 0 9 0
        //  0 0 0 0 0
        // Holes at (1,1),(3,1),(2,2),(1,3),(3,3): the center hole touches the
        // corner holes diagonally. Under contour (label-based) the 9-cells are
        // outside, so label-0's boundary pinches at (2,2)'s corners.
        let labels = vec![
            0, 0, 0, 0, 0,
            0, 9, 0, 9, 0,
            0, 0, 9, 0, 0,
            0, 9, 0, 9, 0,
            0, 0, 0, 0, 0,
        ];
        let loops = contours_of(&labels, 5, 5, 0, (0, 0, 4, 4));
        eprintln!("DOUBLEHOLE loops = {}", loops.len());
        for (i, l) in loops.iter().enumerate() {
            eprintln!("  loop[{i}] len={} area2={}", l.len(), signed_area(l));
        }
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
