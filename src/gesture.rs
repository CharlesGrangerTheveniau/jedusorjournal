//! Idle-based auto-trigger: once the user stops writing anywhere on the
//! page for `idle_delay_ms` of continuous pen inactivity, the diary
//! automatically takes a screenshot and answers — no tap or gesture
//! required at all.
//!
//! This replaces an earlier tap-based gesture design (double-tap, then
//! triple-tap) that proved unreliable on real hardware: small quick taps
//! are geometrically similar to incidental marks in normal handwriting
//! (French accents in particular), causing false triggers on incomplete
//! questions. Watching for plain inactivity sidesteps that whole class of
//! problem — any real stroke just resets the timer, and only genuine
//! silence ever fires it. One piece of per-stroke classification survives
//! from the tap era, inverted: tap-shaped strokes (toolbar interactions
//! done with the pen, stray punctuation) DELAY the trigger but never ARM
//! it — see `TAP_MAX_DURATION_MS` below for the on-device incident that
//! made this necessary.

use anyhow::Result;
use evdev::{Device, EventStream, EventType as EvdevEventType};
use log::{info, warn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::cancellation::GhostwriterCancellation;
use crate::device::DeviceModel;

const ABS_X: u16 = 0;
const ABS_Y: u16 = 1;
const BTN_TOOL_RUBBER: u16 = 321;
const BTN_TOUCH: u16 = 330;

/// A "tap": a short, tiny stroke. On this device that's either a toolbar/
/// palette interaction done with the pen (the sidebar UI responds to pen
/// taps, which land on this same digitizer as writing) or stray punctuation
/// (periods, i-dots). Neither is a question: taps must never ARM the idle
/// trigger, only delay it. Confirmed on device: switching pen type via the
/// toolbar produced a chain of taps in the sidebar column, which armed the
/// trigger and fired mid-question with a bogus toolbar-strip content bbox
/// of (0,109)-(115,392). Real writing always contains non-tap strokes, so
/// dots and accents still get answered as part of their question.
/// Thresholds from the earlier triple-tap-era calibration on real strokes:
/// the smallest genuine letters measured 20px+ and 300ms+, while taps sat
/// well under both limits.
const TAP_MAX_DURATION_MS: u64 = 400;
const TAP_MAX_BBOX_PX: f32 = 12.0;

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

#[derive(Debug, Clone, Copy)]
pub struct IdleTriggerConfig {
    /// How long the pen must be continuously inactive (no stroke anywhere on
    /// the page) before the diary auto-triggers.
    pub idle_delay_ms: u64,
}

impl Default for IdleTriggerConfig {
    fn default() -> Self {
        Self { idle_delay_ms: 3500 }
    }
}

/// Shared flag the drawing code sets to `true` for the duration of a
/// write_cursive drawing pass, so the watcher can ignore its own simulated
/// pen events instead of treating them as user activity (which would keep
/// resetting the idle timer forever) or risking a false trigger. Confirmed
/// on real hardware: the diary's own simulated strokes appear on the same
/// pen input device the watcher reads, indistinguishable from real user
/// input by the watcher alone.
pub type DrawingInProgress = Arc<AtomicBool>;

/// What the idle trigger observed, in virtual screen coordinates: where the
/// newest writing ended (used to place the answer below it) and the bounding
/// box of every pen stroke seen since the watcher was (re)armed — i.e. the
/// region containing the newest question. The bbox is ground truth from real
/// pen events, letting downstream code crop the question out of the
/// screenshot instead of asking a vision model to find it spatially.
#[derive(Debug, Clone, Copy)]
pub struct IdleTrigger {
    pub anchor: (f32, f32),
    /// (min_x, min_y, max_x, max_y)
    pub bbox: (f32, f32, f32, f32),
}

fn expand_bbox(bbox: &mut Option<(f32, f32, f32, f32)>, (x, y): (f32, f32)) {
    *bbox = Some(match *bbox {
        None => (x, y, x, y),
        Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
    });
}

pub struct IdleWatcher {
    event_stream: Option<EventStream>,
    device_model: DeviceModel,
    config: IdleTriggerConfig,
    log_gestures: bool,
    drawing_in_progress: DrawingInProgress,
}

impl IdleWatcher {
    pub fn new(no_gesture: bool, config: IdleTriggerConfig, log_gestures: bool, drawing_in_progress: DrawingInProgress) -> Self {
        let device_model = DeviceModel::detect();
        let pen_input_device = match device_model {
            DeviceModel::RemarkablePaperPro => "/dev/input/event2",
            _ => "/dev/input/event1",
        };
        let event_stream = if no_gesture {
            info!("Idle watcher: disabled (no-gesture/no-draw mode)");
            None
        } else {
            // Log both outcomes explicitly: a silent None here means the idle
            // trigger never fires for the whole process lifetime, which
            // otherwise looks identical to "user simply hasn't written yet".
            match Device::open(pen_input_device).and_then(|d| d.into_event_stream()) {
                Ok(stream) => {
                    info!("Idle watcher: armed on {} ({}ms idle delay)", pen_input_device, config.idle_delay_ms);
                    Some(stream)
                }
                Err(e) => {
                    warn!("Idle watcher: FAILED to open {} — idle trigger will never fire: {}", pen_input_device, e);
                    None
                }
            }
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
                info!("Idle watcher: reopened pen input stream after drawing finished");
                self.event_stream = Some(stream);
            }
            Err(e) => {
                warn!("Idle watcher: failed to reopen pen input stream: {}", e);
            }
        }
    }

    /// Wait until the pen has been inactive for `config.idle_delay_ms` after
    /// at least one genuine stroke, returning where the writing ended and the
    /// bounding box of all of it (virtual screen coordinates). Any real
    /// stroke resets the idle clock, so the trigger only fires once the user
    /// has actually stopped writing — no explicit gesture needed.
    pub async fn wait_for_idle_trigger(&mut self, cancellation: &GhostwriterCancellation) -> Result<IdleTrigger> {
        if self.event_stream.is_none() {
            // No-gesture mode: block until cancelled, like Touch's no-stream path.
            loop {
                if cancellation.should_cancel_main() {
                    return Err(anyhow::anyhow!("Gesture waiting cancelled"));
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }

        let mut cur_x = 0.0f32;
        let mut cur_y = 0.0f32;
        let mut in_stroke = false;
        // Per-stroke measurements for the tap classifier: when the current
        // stroke started, and the bbox of just this stroke.
        let mut stroke_start: Option<Instant> = None;
        let mut stroke_bbox: Option<(f32, f32, f32, f32)> = None;
        // Whether the pen's eraser end is currently the active tool. Eraser
        // strokes still reset the idle clock (don't fire mid-erase) but do
        // NOT count as new content — erasing after an answer shouldn't
        // re-trigger an answer to the already-answered page.
        let mut rubber_active = false;
        // Set on every genuine stroke-end; cleared once we've fired (or once
        // a drawing pass finishes, so the diary's own strokes never count).
        let mut last_activity: Option<Instant> = None;
        let mut has_new_content = false;
        // Where the last real pen stroke ENDED, captured at stroke-end time.
        // Deliberately not read from cur_x/cur_y at fire time: the digitizer
        // keeps reporting ABS coordinates while the pen merely hovers, so by
        // the time the idle timer fires the live position may have drifted
        // to wherever the user is hovering, not where they stopped writing.
        let mut last_stroke_end: Option<(f32, f32)> = None;
        // Bounding box of every pen-stroke point seen this call — i.e. of
        // the newest question, since each call spans exactly the writing
        // between two triggers.
        let mut content_bbox: Option<(f32, f32, f32, f32)> = None;
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
                            tokio::time::sleep(Duration::from_millis(50)).await;
                        }
                    } => return Err(anyhow::anyhow!("Gesture waiting cancelled")),
                    ev = stream.next_event() => Woke::Event(ev),
                    _ = tokio::time::sleep(Duration::from_millis(200)) => Woke::Tick,
                }
            };

            let event = match woke {
                Woke::Tick => {
                    let now_drawing = self.drawing_in_progress.load(Ordering::Relaxed);
                    if was_drawing && !now_drawing {
                        self.reopen_stream();
                        // The diary just finished drawing its own answer —
                        // don't let that count as "new content" to answer again.
                        has_new_content = false;
                        last_activity = None;
                        last_stroke_end = None;
                        content_bbox = None;
                        in_stroke = false;
                    }
                    was_drawing = now_drawing;

                    if !now_drawing && has_new_content {
                        if let Some(t) = last_activity {
                            if t.elapsed() >= Duration::from_millis(self.config.idle_delay_ms) {
                                let anchor = last_stroke_end.unwrap_or_else(|| input_to_virtual((cur_x, cur_y), self.device_model));
                                // Degenerate fallback (shouldn't happen: any pen
                                // stroke expands the bbox): a small box around
                                // the anchor.
                                let bbox = content_bbox.unwrap_or((anchor.0 - 50.0, anchor.1 - 30.0, anchor.0 + 50.0, anchor.1 + 10.0));
                                info!(
                                    "Idle trigger fired at ({:.1}, {:.1}), content bbox ({:.0}, {:.0})-({:.0}, {:.0}), after {}ms of inactivity",
                                    anchor.0, anchor.1, bbox.0, bbox.1, bbox.2, bbox.3, self.config.idle_delay_ms
                                );
                                return Ok(IdleTrigger { anchor, bbox });
                            }
                        }
                    }
                    continue;
                }
                Woke::Event(Ok(ev)) => ev,
                Woke::Event(Err(e)) => {
                    // Don't let a transient read error kill the trigger task
                    // for the rest of the process lifetime — the pen fd has
                    // already proven flaky on this hardware (see
                    // `reopen_stream`), so treat errors the same way: drop
                    // the fd, reopen, carry on.
                    warn!("Idle watcher: pen input stream error: {} — reopening stream", e);
                    self.reopen_stream();
                    in_stroke = false;
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    continue;
                }
            };

            if self.drawing_in_progress.load(Ordering::Relaxed) {
                // The diary is currently drawing its own answer, which
                // generates real pen events on this same device. Ignore it
                // entirely rather than treating it as user activity.
                in_stroke = false;
                stroke_start = None;
                stroke_bbox = None;
                was_drawing = true;
                continue;
            }
            was_drawing = false;

            match (event.event_type(), event.code(), event.value()) {
                (EvdevEventType::KEY, BTN_TOOL_RUBBER, v) => {
                    rubber_active = v == 1;
                }
                (EvdevEventType::KEY, BTN_TOUCH, 1) => {
                    in_stroke = true;
                    stroke_start = Some(Instant::now());
                    stroke_bbox = None;
                }
                (EvdevEventType::ABSOLUTE, ABS_X, v) => {
                    cur_x = v as f32;
                    if in_stroke {
                        expand_bbox(&mut stroke_bbox, input_to_virtual((cur_x, cur_y), self.device_model));
                    }
                }
                (EvdevEventType::ABSOLUTE, ABS_Y, v) => {
                    cur_y = v as f32;
                    if in_stroke {
                        expand_bbox(&mut stroke_bbox, input_to_virtual((cur_x, cur_y), self.device_model));
                    }
                }
                (EvdevEventType::KEY, BTN_TOUCH, 0) => {
                    if in_stroke {
                        in_stroke = false;
                        // Any stroke (pen, eraser, or tap) resets the idle
                        // clock so we never fire mid-activity — but only
                        // non-tap pen strokes ARM the trigger and extend the
                        // question bbox. Excluding taps from the bbox also
                        // keeps toolbar taps from stretching the crop to the
                        // sidebar; genuine punctuation sits inside the line's
                        // bbox anyway (the crop adds margin).
                        last_activity = Some(Instant::now());
                        let duration_ms = stroke_start.take().map(|t| t.elapsed().as_millis() as u64).unwrap_or(0);
                        let this_bbox = stroke_bbox.take();
                        let max_dim = this_bbox.map(|(x0, y0, x1, y1)| (x1 - x0).max(y1 - y0)).unwrap_or(0.0);
                        let tap_like = duration_ms <= TAP_MAX_DURATION_MS && max_dim <= TAP_MAX_BBOX_PX;
                        if !rubber_active && !tap_like {
                            has_new_content = true;
                            last_stroke_end = Some(input_to_virtual((cur_x, cur_y), self.device_model));
                            if let Some((x0, y0, x1, y1)) = this_bbox {
                                expand_bbox(&mut content_bbox, (x0, y0));
                                expand_bbox(&mut content_bbox, (x1, y1));
                            }
                        }
                        if self.log_gestures {
                            let (vx, vy) = input_to_virtual((cur_x, cur_y), self.device_model);
                            let kind = if rubber_active {
                                "eraser"
                            } else if tap_like {
                                "tap (UI/punctuation, not arming)"
                            } else {
                                "pen"
                            };
                            info!(
                                "Idle watcher: {} stroke ended near ({:.1}, {:.1}) ({}ms, {:.1}px), resetting idle timer",
                                kind, vx, vy, duration_ms, max_dim
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    }
}
