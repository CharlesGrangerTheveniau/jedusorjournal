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

// --- Task 10: stateful watcher that reads the real pen input device ---

use anyhow::Result;
use evdev::{Device, EventStream, EventType as EvdevEventType};
use log::{debug, info};
use std::time::Instant;

use crate::cancellation::GhostwriterCancellation;
use crate::device::DeviceModel;

const ABS_X: u16 = 0;
const ABS_Y: u16 = 1;
const BTN_TOUCH: u16 = 330;

// Duplicated from pen.rs / touch.rs by existing project convention (each
// device-facing module keeps its own copy of these small constants rather
// than sharing a central module).
const VIRTUAL_WIDTH: f32 = 768.0;
const VIRTUAL_HEIGHT: f32 = 1024.0;

fn pen_max_x(device_model: DeviceModel) -> f32 {
    match device_model {
        DeviceModel::RemarkablePaperPro => 11180.0,
        _ => 15725.0,
    }
}

fn pen_max_y(device_model: DeviceModel) -> f32 {
    match device_model {
        DeviceModel::RemarkablePaperPro => 15340.0,
        _ => 20966.0,
    }
}

fn input_to_virtual((x, y): (f32, f32), device_model: DeviceModel) -> (f32, f32) {
    let max_x = pen_max_x(device_model);
    let max_y = pen_max_y(device_model);
    match device_model {
        DeviceModel::RemarkablePaperPro => (x / max_x * VIRTUAL_WIDTH, y / max_y * VIRTUAL_HEIGHT),
        // RM2: pen input space is swapped/flipped relative to virtual space,
        // mirroring Pen::virtual_to_input's inverse.
        _ => (y / max_x * VIRTUAL_WIDTH, (1.0 - x / max_y) * VIRTUAL_HEIGHT),
    }
}

pub struct SpiralWatcher {
    event_stream: Option<EventStream>,
    device_model: DeviceModel,
    config: SpiralConfig,
    log_gestures: bool,
}

impl SpiralWatcher {
    pub fn new(no_gesture: bool, config: SpiralConfig, log_gestures: bool) -> Self {
        let device_model = DeviceModel::detect();
        let pen_input_device = match device_model {
            DeviceModel::RemarkablePaperPro => "/dev/input/event2",
            _ => "/dev/input/event1",
        };
        let event_stream = if no_gesture {
            None
        } else {
            Device::open(pen_input_device).ok().and_then(|d| d.into_event_stream().ok())
        };
        Self {
            event_stream,
            device_model,
            config,
            log_gestures,
        }
    }

    /// Wait until a spiral is drawn on the pen digitizer, returning its
    /// bounding-box center in virtual screen coordinates (used to anchor
    /// the answer below the question). Never returns on a non-spiral
    /// stroke — it keeps buffering strokes until one matches.
    pub async fn wait_for_spiral(&mut self, cancellation: &GhostwriterCancellation) -> Result<(f32, f32)> {
        let Some(stream) = &mut self.event_stream else {
            // No-gesture mode: block until cancelled, like Touch's no-stream path.
            loop {
                if cancellation.should_cancel_main() {
                    return Err(anyhow::anyhow!("Gesture waiting cancelled"));
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        };

        let mut points: Vec<(f32, f32)> = Vec::new();
        let mut cur_x = 0.0f32;
        let mut cur_y = 0.0f32;
        let mut stroke_start: Option<Instant> = None;

        loop {
            let event = tokio::select! {
                _ = async {
                    while !cancellation.should_cancel_main() {
                        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    }
                } => return Err(anyhow::anyhow!("Gesture waiting cancelled")),
                ev = stream.next_event() => ev?,
            };

            match (event.event_type(), event.code(), event.value()) {
                (EvdevEventType::KEY, BTN_TOUCH, 1) => {
                    points.clear();
                    stroke_start = Some(Instant::now());
                }
                (EvdevEventType::ABSOLUTE, ABS_X, v) => cur_x = v as f32,
                (EvdevEventType::ABSOLUTE, ABS_Y, v) => cur_y = v as f32,
                (EvdevEventType::KEY, BTN_TOUCH, 0) => {
                    let Some(start) = stroke_start.take() else { continue };
                    let duration_ms = start.elapsed().as_millis() as u64;
                    let virtual_points: Vec<(f32, f32)> =
                        points.iter().map(|&p| input_to_virtual(p, self.device_model)).collect();

                    if self.log_gestures {
                        let (bx0, by0, bx1, by1) = bounding_box(&virtual_points);
                        let bbox_dim = (bx1 - bx0).max(by1 - by0);
                        let turn = cumulative_turn_degrees(&virtual_points).abs();
                        let accepted = is_spiral(&virtual_points, duration_ms, &self.config);
                        info!(
                            "gesture stroke: {} points, duration={}ms, bbox_max_dim={:.1}px, turn={:.1}deg => {}",
                            virtual_points.len(),
                            duration_ms,
                            bbox_dim,
                            turn,
                            if accepted { "ACCEPTED as spiral" } else { "rejected" }
                        );
                    }

                    if is_spiral(&virtual_points, duration_ms, &self.config) {
                        let (min_x, _min_y, max_x, max_y) = virtual_points.iter().fold(
                            (f32::MAX, f32::MAX, f32::MIN, f32::MIN),
                            |(mnx, mny, mxx, mxy), &(x, y)| (mnx.min(x), mny.min(y), mxx.max(x), mxy.max(y)),
                        );
                        return Ok(((min_x + max_x) / 2.0, max_y));
                    }
                    debug!("gesture stroke rejected as non-spiral");
                }
                (EvdevEventType::SYNCHRONIZATION, _, _) => {
                    if stroke_start.is_some() {
                        points.push((cur_x, cur_y));
                    }
                }
                _ => {}
            }
        }
    }
}
