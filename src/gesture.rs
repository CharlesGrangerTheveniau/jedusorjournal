//! Spiral gesture detection: the user ends a question with a small,
//! deliberate spiral flourish, which triggers the diary to answer.

#[derive(Debug, Clone, Copy)]
pub struct SpiralConfig {
    pub min_turn_degrees: f32,
    pub min_bbox_px: f32,
    pub max_bbox_px: f32,
    pub max_duration_ms: u64,
}

impl Default for SpiralConfig {
    fn default() -> Self {
        Self {
            min_turn_degrees: 720.0,
            min_bbox_px: 15.0,
            max_bbox_px: 60.0,
            max_duration_ms: 1500,
        }
    }
}

/// Detect whether a completed pen stroke (points in virtual screen space,
/// in drawing order) is a spiral: at least two consistent-winding loops,
/// drawn as a small deliberate mark, quickly. This rejects casual circles
/// / letters like 'o' or 'e' (single loop, ~360°) and large scribbles
/// (bbox too big) as well as slow multi-loop doodles (duration too long).
pub fn is_spiral(points: &[(f32, f32)], duration_ms: u64, config: &SpiralConfig) -> bool {
    if points.len() < 8 {
        return false;
    }
    if duration_ms > config.max_duration_ms {
        return false;
    }

    let (min_x, min_y, max_x, max_y) = bounding_box(points);
    let max_dim = (max_x - min_x).max(max_y - min_y);
    if max_dim < config.min_bbox_px || max_dim > config.max_bbox_px {
        return false;
    }

    let turn = cumulative_turn_degrees(points).abs();
    turn >= config.min_turn_degrees
}

fn bounding_box(points: &[(f32, f32)]) -> (f32, f32, f32, f32) {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for &(x, y) in points {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    (min_x, min_y, max_x, max_y)
}

/// Sum of signed turning angles at each interior point. A perfect circle
/// traversed once accumulates ~360°; two consistent-winding loops
/// accumulate ~720°. Reversing direction mid-stroke cancels out, which is
/// intentional — a spiral has a single consistent winding direction.
fn cumulative_turn_degrees(points: &[(f32, f32)]) -> f32 {
    let mut total = 0.0f32;
    for i in 1..points.len() - 1 {
        let (x0, y0) = points[i - 1];
        let (x1, y1) = points[i];
        let (x2, y2) = points[i + 1];
        let d1 = (x1 - x0, y1 - y0);
        let d2 = (x2 - x1, y2 - y1);
        let len1 = (d1.0 * d1.0 + d1.1 * d1.1).sqrt();
        let len2 = (d2.0 * d2.0 + d2.1 * d2.1).sqrt();
        if len1 < 0.01 || len2 < 0.01 {
            continue;
        }
        let cross = d1.0 * d2.1 - d1.1 * d2.0;
        let dot = d1.0 * d2.0 + d1.1 * d2.1;
        total += cross.atan2(dot).to_degrees();
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate an Archimedean spiral of `loops` full turns, centered at
    /// (cx, cy), with the given max radius, sampled at `n` points.
    fn synthetic_spiral(cx: f32, cy: f32, max_radius: f32, loops: f32, n: usize) -> Vec<(f32, f32)> {
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                let angle = t * loops * 2.0 * std::f32::consts::PI;
                let radius = t * max_radius;
                (cx + radius * angle.cos(), cy + radius * angle.sin())
            })
            .collect()
    }

    #[test]
    fn detects_a_deliberate_two_loop_spiral() {
        let points = synthetic_spiral(100.0, 100.0, 20.0, 2.5, 60);
        assert!(is_spiral(&points, 800, &SpiralConfig::default()));
    }

    #[test]
    fn rejects_a_straight_line() {
        let points: Vec<(f32, f32)> = (0..20).map(|i| (i as f32 * 2.0, 0.0)).collect();
        assert!(!is_spiral(&points, 500, &SpiralConfig::default()));
    }

    #[test]
    fn rejects_a_single_loop_like_the_letter_o() {
        let points = synthetic_spiral(100.0, 100.0, 20.0, 1.0, 40);
        assert!(!is_spiral(&points, 500, &SpiralConfig::default()));
    }

    #[test]
    fn rejects_a_spiral_that_is_too_large() {
        let points = synthetic_spiral(100.0, 100.0, 200.0, 2.5, 60);
        assert!(!is_spiral(&points, 800, &SpiralConfig::default()));
    }

    #[test]
    fn rejects_a_spiral_drawn_too_slowly() {
        let points = synthetic_spiral(100.0, 100.0, 20.0, 2.5, 60);
        assert!(!is_spiral(&points, 3000, &SpiralConfig::default()));
    }

    #[test]
    fn rejects_too_few_points() {
        let points = vec![(0.0, 0.0), (1.0, 1.0)];
        assert!(!is_spiral(&points, 500, &SpiralConfig::default()));
    }
}
