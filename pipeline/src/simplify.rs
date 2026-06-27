//! Path simplification — Ramer–Douglas–Peucker on closed loops.
//!
//! Stage 4 of the owned pipeline: the contour tracer emits staircase polygons
//! along pixel edges; RDP drops vertices within `epsilon` of a straight chord,
//! turning long stairs into clean diagonals and shrinking the path data. (The
//! later curve-fitting stage will replace these polylines with Béziers.)

type Pt = (i32, i32);

/// Squared perpendicular distance from `p` to the segment `a`–`b`.
fn perp_dist2(p: Pt, a: Pt, b: Pt) -> f64 {
    let (px, py) = (p.0 as f64, p.1 as f64);
    let (ax, ay) = (a.0 as f64, a.1 as f64);
    let (bx, by) = (b.0 as f64, b.1 as f64);
    let dx = bx - ax;
    let dy = by - ay;
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        let ex = px - ax;
        let ey = py - ay;
        return ex * ex + ey * ey;
    }
    let cross = (px - ax) * dy - (py - ay) * dx;
    cross * cross / len2
}

fn rdp(pts: &[Pt], first: usize, last: usize, eps2: f64, keep: &mut [bool]) {
    if last <= first + 1 {
        return;
    }
    let mut idx = first;
    let mut max_d = 0.0;
    for i in (first + 1)..last {
        let d = perp_dist2(pts[i], pts[first], pts[last]);
        if d > max_d {
            max_d = d;
            idx = i;
        }
    }
    if max_d > eps2 {
        keep[idx] = true;
        rdp(pts, first, idx, eps2, keep);
        rdp(pts, idx, last, eps2, keep);
    }
}

/// Simplify a closed loop with tolerance `epsilon`. Returns the kept vertices
/// (first point preserved as an anchor). A non-positive epsilon is a no-op.
pub fn simplify_closed(loop_pts: &[Pt], epsilon: f64) -> Vec<Pt> {
    let n = loop_pts.len();
    if epsilon <= 0.0 || n <= 3 {
        return loop_pts.to_vec();
    }
    // Treat the closed loop as an open polyline that returns to its anchor.
    let mut pts = loop_pts.to_vec();
    pts.push(loop_pts[0]);
    let m = pts.len();
    let mut keep = vec![false; m];
    keep[0] = true;
    keep[m - 1] = true;
    rdp(&pts, 0, m - 1, epsilon * epsilon, &mut keep);
    let mut out = Vec::new();
    for i in 0..(m - 1) {
        if keep[i] {
            out.push(pts[i]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapses_a_staircase_into_a_diagonal() {
        // A staircase from (0,0) to (4,4) closed back along the top edge.
        let loop_pts = vec![
            (0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (0, 4),
        ];
        let s = simplify_closed(&loop_pts, 1.5);
        // The stair should collapse to roughly the corners of a triangle.
        assert!(s.len() < loop_pts.len());
        assert!(s.len() <= 4, "expected a near-triangle, got {}", s.len());
    }

    #[test]
    fn noop_on_small_or_zero_epsilon() {
        let loop_pts = vec![(0, 0), (4, 0), (4, 4), (0, 4)];
        assert_eq!(simplify_closed(&loop_pts, 0.0), loop_pts);
        assert_eq!(simplify_closed(&loop_pts, -1.0), loop_pts);
    }

    #[test]
    fn keeps_real_corners() {
        // A clean square should keep its 4 corners.
        let sq = vec![(0, 0), (10, 0), (10, 10), (0, 10)];
        let s = simplify_closed(&sq, 1.0);
        assert_eq!(s.len(), 4);
    }
}
