//! Connected-components segmentation of a paletted image (4-connectivity).
//!
//! Stage 2 of the owned pipeline: after quantization flattens the image to a
//! small palette, group pixels into maximal connected regions of the same
//! palette index. A small-region merge pass folds speckles into their largest
//! neighbour so the trace stays compact.

use std::collections::HashSet;

/// A labeled image plus per-component metadata.
pub struct Segmentation {
    pub width: usize,
    pub height: usize,
    /// Component id per pixel (0..num_components).
    pub labels: Vec<u32>,
    pub num_components: usize,
    /// Palette index per component (0 is reserved for "transparent").
    pub component_color: Vec<u32>,
    /// Pixel count per component.
    pub component_area: Vec<u32>,
}

/// Label connected components of `palette_idx` (len = width*height) using
/// 4-connectivity. Palette index 0 is treated as transparent but still forms
/// components (they are simply not drawn downstream).
pub fn segment(palette_idx: &[u32], width: usize, height: usize) -> Segmentation {
    let n = width * height;
    debug_assert_eq!(palette_idx.len(), n);
    let mut labels = vec![u32::MAX; n];
    let mut component_color = Vec::new();
    let mut component_area = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut num: u32 = 0;

    for start in 0..n {
        if labels[start] != u32::MAX {
            continue;
        }
        let color = palette_idx[start];
        let id = num;
        num += 1;
        component_color.push(color);
        let mut area: u32 = 0;
        labels[start] = id;
        stack.push(start);
        while let Some(p) = stack.pop() {
            area += 1;
            let x = p % width;
            let y = p / width;
            let mut neighbors = [usize::MAX; 4];
            let mut nn = 0;
            if x > 0 {
                neighbors[nn] = p - 1;
                nn += 1;
            }
            if x + 1 < width {
                neighbors[nn] = p + 1;
                nn += 1;
            }
            if y > 0 {
                neighbors[nn] = p - width;
                nn += 1;
            }
            if y + 1 < height {
                neighbors[nn] = p + width;
                nn += 1;
            }
            for &q in &neighbors[..nn] {
                if labels[q] == u32::MAX && palette_idx[q] == color {
                    labels[q] = id;
                    stack.push(q);
                }
            }
        }
        component_area.push(area);
    }

    Segmentation {
        width,
        height,
        labels,
        num_components: num as usize,
        component_color,
        component_area,
    }
}

struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl UnionFind {
    fn new(areas: &[u32]) -> Self {
        Self {
            parent: (0..areas.len() as u32).collect(),
            size: areas.to_vec(),
        }
    }
    fn find(&mut self, mut a: u32) -> u32 {
        while self.parent[a as usize] != a {
            self.parent[a as usize] = self.parent[self.parent[a as usize] as usize];
            a = self.parent[a as usize];
        }
        a
    }
    /// Union, keeping `keep` as the surviving root (so the merged region adopts
    /// `keep`'s color). Returns the surviving root.
    fn union_into(&mut self, child: u32, keep: u32) -> u32 {
        let rc = self.find(child);
        let rk = self.find(keep);
        if rc == rk {
            return rk;
        }
        self.parent[rc as usize] = rk;
        self.size[rk as usize] += self.size[rc as usize];
        rk
    }
}

impl Segmentation {
    /// Best-effort merge of undersized opaque components into a larger
    /// neighbour. Each opaque component below `min_area` is folded (smallest
    /// first, cascading via union-find) into its largest adjacent opaque
    /// component. This is best-effort, not a guarantee: a small cluster with no
    /// larger opaque neighbour (e.g. a tiny blob surrounded only by transparency)
    /// has nothing to merge into and is left as-is. Relabels in place and
    /// compacts component ids. A `min_area` of 0 or 1 is a no-op.
    pub fn merge_small(&mut self, min_area: u32) {
        if min_area <= 1 || self.num_components <= 1 {
            return;
        }
        let (w, h) = (self.width, self.height);

        // Build the adjacency set between distinct neighbouring components.
        let mut adj: Vec<HashSet<u32>> = vec![HashSet::new(); self.num_components];
        for y in 0..h {
            for x in 0..w {
                let a = self.labels[y * w + x];
                if x + 1 < w {
                    let b = self.labels[y * w + x + 1];
                    if a != b {
                        adj[a as usize].insert(b);
                        adj[b as usize].insert(a);
                    }
                }
                if y + 1 < h {
                    let b = self.labels[(y + 1) * w + x];
                    if a != b {
                        adj[a as usize].insert(b);
                        adj[b as usize].insert(a);
                    }
                }
            }
        }

        let mut uf = UnionFind::new(&self.component_area);

        // Smallest-area components first.
        let mut order: Vec<u32> = (0..self.num_components as u32).collect();
        order.sort_by_key(|&c| self.component_area[c as usize]);

        for c in order {
            if self.component_color[c as usize] == 0 {
                continue; // never merge transparent regions
            }
            let root = uf.find(c);
            if uf.size[root as usize] >= min_area {
                continue;
            }
            // Pick the largest opaque neighbour (by current root size).
            let mut best: Option<u32> = None;
            let mut best_size = 0u32;
            for &nb in &adj[c as usize] {
                if self.component_color[nb as usize] == 0 {
                    continue;
                }
                let rn = uf.find(nb);
                if rn == root {
                    continue;
                }
                if uf.size[rn as usize] >= best_size {
                    best_size = uf.size[rn as usize];
                    best = Some(rn);
                }
            }
            if let Some(keep) = best {
                uf.union_into(root, keep);
            }
        }

        // Compact roots into a dense 0..m labeling.
        let mut remap: Vec<u32> = vec![u32::MAX; self.num_components];
        let mut new_color = Vec::new();
        let mut new_count = 0u32;
        for old in 0..self.num_components as u32 {
            let r = uf.find(old);
            if remap[r as usize] == u32::MAX {
                remap[r as usize] = new_count;
                new_color.push(self.component_color[r as usize]);
                new_count += 1;
            }
            remap[old as usize] = remap[r as usize];
        }
        for lbl in self.labels.iter_mut() {
            *lbl = remap[*lbl as usize];
        }
        // Recompute areas.
        let mut new_area = vec![0u32; new_count as usize];
        for &lbl in &self.labels {
            new_area[lbl as usize] += 1;
        }

        self.num_components = new_count as usize;
        self.component_color = new_color;
        self.component_area = new_area;
    }

    /// Per-component bounding box as (min_x, min_y, max_x, max_y) inclusive.
    pub fn bboxes(&self) -> Vec<(usize, usize, usize, usize)> {
        let (w, h) = (self.width, self.height);
        let mut bb = vec![(usize::MAX, usize::MAX, 0usize, 0usize); self.num_components];
        for y in 0..h {
            for x in 0..w {
                let c = self.labels[y * w + x] as usize;
                let b = &mut bb[c];
                if x < b.0 {
                    b.0 = x;
                }
                if y < b.1 {
                    b.1 = y;
                }
                if x > b.2 {
                    b.2 = x;
                }
                if y > b.3 {
                    b.3 = y;
                }
            }
        }
        bb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_two_regions() {
        // 2x2: top row color 1, bottom row color 2.
        let idx = vec![1, 1, 2, 2];
        let s = segment(&idx, 2, 2);
        assert_eq!(s.num_components, 2);
        assert_eq!(s.labels[0], s.labels[1]);
        assert_eq!(s.labels[2], s.labels[3]);
        assert_ne!(s.labels[0], s.labels[2]);
        assert_eq!(s.component_area, vec![2, 2]);
    }

    #[test]
    fn diagonal_same_color_is_two_components_under_4conn() {
        // checkerboard of color 1 and 2 -> 4 separate single-pixel components.
        let idx = vec![1, 2, 2, 1];
        let s = segment(&idx, 2, 2);
        assert_eq!(s.num_components, 4);
    }

    #[test]
    fn merge_small_folds_speckle_into_neighbour() {
        // 3x3 of color 1 with a single color-2 speckle in the center.
        let idx = vec![1, 1, 1, 1, 2, 1, 1, 1, 1];
        let mut s = segment(&idx, 3, 3);
        assert_eq!(s.num_components, 2);
        s.merge_small(4);
        // The 1-pixel speckle folds into the surrounding region.
        assert_eq!(s.num_components, 1);
        assert!(s.labels.iter().all(|&l| l == 0));
    }
}
