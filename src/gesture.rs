//! Triple-tap gesture detection: the user ends a question with three quick,
//! adjacent taps of the pen (like an ellipsis: "…"), which triggers the
//! diary to answer.
//!
//! Originally a double-tap (two taps), but real on-device testing showed
//! that two small, quick, closely-spaced taps are geometrically
//! indistinguishable from incidental marks in normal handwriting — French
//! accent marks in particular are exactly this shape, and one accidentally
//! paired with a nearby dot fired the trigger mid-sentence, sending an
//! incomplete question to the LLM. Three consecutive taps all pairwise
//! close in time and space is a much rarer accidental coincidence while
//! remaining a fast, deliberate gesture to perform on purpose.

#[derive(Debug, Clone, Copy)]
pub struct TripleTapConfig {
    pub max_tap_duration_ms: u64,
    pub max_tap_bbox_px: f32,
    pub max_pair_gap_ms: u64,
    pub max_pair_distance_px: f32,
}

impl Default for TripleTapConfig {
    fn default() -> Self {
        Self {
            max_tap_duration_ms: 400,
            max_tap_bbox_px: 12.0,
            max_pair_gap_ms: 1200,
            max_pair_distance_px: 20.0,
        }
    }
}

/// Detect whether a completed pen stroke (points in virtual screen space)
/// is a "tap": a small, quick, nearly-stationary mark like a period —
/// clearly smaller and faster than any letter (verified on real device
/// data: even the smallest tested letters were 20px+ and 300ms+).
pub fn is_tap(points: &[(f32, f32)], duration_ms: u64, config: &TripleTapConfig) -> bool {
    if points.is_empty() {
        return false;
    }
    if duration_ms > config.max_tap_duration_ms {
        return false;
    }
    let (min_x, min_y, max_x, max_y) = bounding_box(points);
    let max_dim = (max_x - min_x).max(max_y - min_y);
    max_dim <= config.max_tap_bbox_px
}

/// Detect whether two taps (each already confirmed via `is_tap`) are close
/// enough in time and space to be consecutive members of the same
/// deliberate tap sequence.
pub fn is_tap_pair(first_center: (f32, f32), second_center: (f32, f32), gap_ms: u64, config: &TripleTapConfig) -> bool {
    if gap_ms > config.max_pair_gap_ms {
        return false;
    }
    let dx = second_center.0 - first_center.0;
    let dy = second_center.1 - first_center.1;
    let dist = (dx * dx + dy * dy).sqrt();
    dist <= config.max_pair_distance_px
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

fn bbox_center(points: &[(f32, f32)]) -> (f32, f32) {
    let (min_x, min_y, max_x, max_y) = bounding_box(points);
    ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small cluster of points confined to a tiny area, simulating a
    /// quick pen tap (period-sized mark).
    fn synthetic_tap(cx: f32, cy: f32, radius: f32, n: usize) -> Vec<(f32, f32)> {
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1).max(1) as f32;
                let angle = t * 2.0 * std::f32::consts::PI;
                (cx + radius * angle.cos(), cy + radius * angle.sin())
            })
            .collect()
    }

    #[test]
    fn detects_a_quick_small_tap() {
        let points = synthetic_tap(100.0, 100.0, 3.0, 6);
        assert!(is_tap(&points, 150, &TripleTapConfig::default()));
    }

    #[test]
    fn rejects_a_tap_that_is_too_big() {
        // Bigger than any letter observed in real calibration data (20px+).
        let points = synthetic_tap(100.0, 100.0, 15.0, 6);
        assert!(!is_tap(&points, 150, &TripleTapConfig::default()));
    }

    #[test]
    fn rejects_a_tap_drawn_too_slowly() {
        let points = synthetic_tap(100.0, 100.0, 3.0, 6);
        assert!(!is_tap(&points, 600, &TripleTapConfig::default()));
    }

    #[test]
    fn rejects_an_empty_stroke() {
        assert!(!is_tap(&[], 100, &TripleTapConfig::default()));
    }

    #[test]
    fn accepts_two_adjacent_taps_close_in_time() {
        let config = TripleTapConfig::default();
        assert!(is_tap_pair((100.0, 100.0), (110.0, 102.0), 500, &config));
    }

    #[test]
    fn rejects_two_taps_too_far_apart_in_time() {
        let config = TripleTapConfig::default();
        assert!(!is_tap_pair((100.0, 100.0), (110.0, 102.0), 2000, &config));
    }

    #[test]
    fn rejects_two_taps_too_far_apart_in_space() {
        let config = TripleTapConfig::default();
        assert!(!is_tap_pair((100.0, 100.0), (300.0, 100.0), 500, &config));
    }

    #[test]
    fn bbox_center_of_a_single_point_is_itself() {
        let points = vec![(50.0, 60.0)];
        assert_eq!(bbox_center(&points), (50.0, 60.0));
    }
}

// --- Stateful watcher that reads the real pen input device ---

use anyhow::Result;
use evdev::{Device, EventStream, EventType as EvdevEventType};
use log::{info, warn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
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

/// Shared flag the drawing code sets to `true` for the duration of a
/// write_cursive drawing pass, so the gesture watcher can ignore its own
/// simulated pen events instead of risking a false trigger. Confirmed on
/// real hardware: the diary's own simulated strokes appear on the same
/// pen input device the watcher reads, indistinguishable from real user
/// input by the watcher alone — during one drawing pass, a burst of
/// self-generated strokes accidentally paired into a spurious trigger
/// (caught harmlessly by the existing "ignore triggers received during
/// processing" drain, but a real gap worth closing here directly).
pub type DrawingInProgress = Arc<AtomicBool>;

pub struct TripleTapWatcher {
    event_stream: Option<EventStream>,
    device_model: DeviceModel,
    config: TripleTapConfig,
    log_gestures: bool,
    drawing_in_progress: DrawingInProgress,
}

impl TripleTapWatcher {
    pub fn new(no_gesture: bool, config: TripleTapConfig, log_gestures: bool, drawing_in_progress: DrawingInProgress) -> Self {
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
            drawing_in_progress,
        }
    }

    /// Re-open the pen input device from scratch and replace `event_stream`.
    /// Called right after a drawing pass finishes: the diary's own bursts of
    /// simulated pen events flow through this same fd, and on real hardware
    /// the async event stream has been observed to go completely silent
    /// afterward (no more events ever delivered, even for genuine new
    /// writing) — most likely a lost epoll/mio readiness notification after
    /// that burst. Discarding the old fd and opening a fresh one clears
    /// whatever stuck state caused it, at negligible cost since this only
    /// runs once per answer.
    fn reopen_stream(&mut self) {
        let pen_input_device = match self.device_model {
            DeviceModel::RemarkablePaperPro => "/dev/input/event2",
            _ => "/dev/input/event1",
        };
        match Device::open(pen_input_device).and_then(|d| d.into_event_stream()) {
            Ok(stream) => {
                info!("Gesture watcher: reopened pen input stream after drawing finished");
                self.event_stream = Some(stream);
            }
            Err(e) => {
                warn!("Gesture watcher: failed to reopen pen input stream: {}", e);
            }
        }
    }

    /// Wait until three consecutive, pairwise-adjacent, quick taps are
    /// drawn on the pen digitizer (like "…"), returning the centroid of
    /// the three in virtual screen coordinates. Any non-tap stroke (normal
    /// writing) clears the whole in-progress chain, so only a genuinely
    /// uninterrupted run of three taps counts — a tap that doesn't pair
    /// with the previous one restarts the chain from itself rather than
    /// discarding it entirely, so "tap, [pause], tap, tap" still fires if
    /// the last two are close enough.
    pub async fn wait_for_triple_tap(&mut self, cancellation: &GhostwriterCancellation) -> Result<(f32, f32)> {
        if self.event_stream.is_none() {
            // No-gesture mode: block until cancelled, like Touch's no-stream path.
            loop {
                if cancellation.should_cancel_main() {
                    return Err(anyhow::anyhow!("Gesture waiting cancelled"));
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }

        let mut points: Vec<(f32, f32)> = Vec::new();
        let mut cur_x = 0.0f32;
        let mut cur_y = 0.0f32;
        let mut stroke_start: Option<Instant> = None;
        // Taps confirmed so far in the current chain (0, 1, or 2 entries).
        let mut chain: Vec<((f32, f32), Instant)> = Vec::new();
        // Tracks the drawing_in_progress flag's last-seen value so the Tick
        // branch below can detect the falling edge (drawing just finished)
        // even if no further pen events ever arrive to wake the event branch.
        let mut was_drawing = false;

        enum Woke {
            Event(std::io::Result<evdev::InputEvent>),
            Tick,
        }

        loop {
            let woke = {
                let stream = self.event_stream.as_mut().expect("checked event_stream.is_none() above");
                tokio::select! {
                    _ = async {
                        while !cancellation.should_cancel_main() {
                            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                        }
                    } => return Err(anyhow::anyhow!("Gesture waiting cancelled")),
                    ev = stream.next_event() => Woke::Event(ev),
                    _ = tokio::time::sleep(std::time::Duration::from_millis(250)) => Woke::Tick,
                }
            };

            let event = match woke {
                Woke::Tick => {
                    let now_drawing = self.drawing_in_progress.load(Ordering::Relaxed);
                    if was_drawing && !now_drawing {
                        self.reopen_stream();
                        points.clear();
                        stroke_start = None;
                        chain.clear();
                    }
                    was_drawing = now_drawing;
                    continue;
                }
                Woke::Event(ev) => ev?,
            };

            if self.drawing_in_progress.load(Ordering::Relaxed) {
                // The diary is currently drawing its own answer, which
                // generates real pen events on this same device. Ignore
                // everything until drawing finishes, and drop any
                // in-progress stroke/chain state so we don't misinterpret
                // the tail of a self-generated stroke once we resume.
                points.clear();
                stroke_start = None;
                chain.clear();
                was_drawing = true;
                continue;
            }
            was_drawing = false;

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
                    let tap = is_tap(&virtual_points, duration_ms, &self.config);

                    if self.log_gestures {
                        let (bx0, by0, bx1, by1) = bounding_box(&virtual_points);
                        let bbox_dim = (bx1 - bx0).max(by1 - by0);
                        info!(
                            "gesture stroke: {} points, duration={}ms, bbox_max_dim={:.1}px => {}",
                            virtual_points.len(),
                            duration_ms,
                            bbox_dim,
                            if tap { "TAP" } else { "not a tap (normal writing)" }
                        );
                    }

                    if !tap {
                        // Anything that isn't a tap clears the whole chain — a
                        // triple-tap trigger requires three uninterrupted taps,
                        // with nothing else drawn in between.
                        chain.clear();
                        continue;
                    }

                    let this_center = bbox_center(&virtual_points);
                    let this_time = Instant::now();

                    let pairs_with_last = chain
                        .last()
                        .map(|&(prev_center, prev_time)| {
                            let gap_ms = prev_time.elapsed().as_millis() as u64;
                            let dx = this_center.0 - prev_center.0;
                            let dy = this_center.1 - prev_center.1;
                            let dist = (dx * dx + dy * dy).sqrt();
                            let paired = is_tap_pair(prev_center, this_center, gap_ms, &self.config);
                            if self.log_gestures {
                                info!(
                                    "tap pair check: chain_len={}, gap={}ms, dist={:.1}px => {}",
                                    chain.len(),
                                    gap_ms,
                                    dist,
                                    if paired { "PAIRED" } else { "not paired" }
                                );
                            }
                            paired
                        })
                        .unwrap_or(false);

                    if pairs_with_last {
                        chain.push((this_center, this_time));
                        if chain.len() == 3 {
                            let (sum_x, sum_y) = chain.iter().fold((0.0, 0.0), |(sx, sy), &((x, y), _)| (sx + x, sy + y));
                            let anchor = (sum_x / 3.0, sum_y / 3.0);
                            info!("Triple-tap detected at ({:.1}, {:.1})", anchor.0, anchor.1);
                            return Ok(anchor);
                        }
                    } else {
                        // Didn't pair with the chain's last tap (or chain was
                        // empty) — start a fresh chain from this tap.
                        chain.clear();
                        chain.push((this_center, this_time));
                    }
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
