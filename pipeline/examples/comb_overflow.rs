// Realistic comb: single connected component whose boundary keeps a real corner
// at every tooth. We build a 1-pixel-thick boustrophedon "comb" of color A on a
// background of color B, all within the 25 MP / 5000px bound, then run the REAL
// contour + simplify path on a 2 MB stack (emulating tokio spawn_blocking).
use svgit_pipeline::contour::contours_of;
use svgit_pipeline::simplify::simplify_closed;
use svgit_pipeline::segment::segment;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let teeth: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(2000);
    let tooth_h: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(2000);

    // Image: width = 2*teeth+1, height = tooth_h+1. Comb tines stick UP from a
    // baseline row; each tine is 1px wide separated by 1px gap. This is ONE
    // 4-connected component (joined along the baseline row at y=tooth_h).
    let w = 2 * teeth + 1;
    let h = tooth_h + 1;
    let mp = (w as u64) * (h as u64);
    eprintln!("image {}x{} = {:.1} MP (limit 25 MP)", w, h, mp as f64 / 1e6);
    assert!(mp <= 25_000_000, "exceeds 25 MP cap");
    assert!(w <= 5000 && h <= 5000, "exceeds 5000px (got {}x{})", w, h);

    // palette idx: 1 = comb color, 0 would be transparent so use 2 for bg.
    let mut idx = vec![2u32; w * h];
    // baseline row (y = tooth_h) entirely comb:
    for x in 0..w { idx[tooth_h * w + x] = 1; }
    // tines: at even x columns, fill from y=0..tooth_h with comb color.
    for t in 0..teeth {
        let x = 2 * t; // even columns are tines
        for y in 0..tooth_h { idx[y * w + x] = 1; }
    }

    // Run the REAL segment + contour path on a 2 MB stack.
    let handle = std::thread::Builder::new().stack_size(2*1024*1024).spawn(move || {
        let seg = segment(&idx, w, h);
        // find the comb component (color 1, largest)
        let mut best = 0usize; let mut barea = 0u32;
        for c in 0..seg.num_components {
            if seg.component_color[c] == 1 && seg.component_area[c] > barea {
                barea = seg.component_area[c]; best = c;
            }
        }
        let bbox = seg.bboxes()[best];
        let raw_loops = contours_of(&seg.labels, w, h, best as u32, bbox);
        let maxlen = raw_loops.iter().map(|l| l.len()).max().unwrap_or(0);
        eprintln!("comb component: {} loop(s), longest loop after collapse_collinear = {} vertices", raw_loops.len(), maxlen);
        for lp in &raw_loops {
            let _s = simplify_closed(lp, 1.2); // default eps -> RDP recursion
        }
        eprintln!("OK no overflow (teeth={})", teeth);
    }).unwrap();
    if handle.join().is_err() { eprintln!("ABORTED (stack overflow) teeth={}", teeth); }
}
