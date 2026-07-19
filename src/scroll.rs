//! Closed-loop canvas scrolling: reveal blank space below the current
//! viewport before drawing an answer that would otherwise clip past the
//! bottom edge (confirmed on device: ink drawn beyond y=1024 is simply
//! lost — a long answer's tail never appears on the page).
//!
//! reMarkable continuous pages scroll by touch fling, and the fling is
//! velocity-based with momentum — calibrated on device (RM2, 3.27.3.0):
//! a fast 100px drag flings ~250-300px, a fast 200px drag ~800px, and a
//! slow drag or one ending in a stationary hold scrolls NOTHING (the
//! gesture needs real release velocity to register). So exact open-loop
//! scrolling is impossible; instead each fling is measured after the fact
//! by cross-correlating per-row ink profiles of before/after screenshots,
//! and the caller adjusts its drawing coordinates by the TOTAL measured
//! shift. Overshoot is harmless (extra blank space); undershoot triggers
//! another fling, up to a small cap.

use anyhow::Result;
use log::info;
use std::time::Duration;

use crate::screenshot::Screenshot;
use crate::touch::{Touch, TriggerCorner};

/// Ink-luminance threshold and toolbar exclusion mirror the calibration
/// tooling in src/bin/experiment.rs (ShiftBetween / InkBounds).
const INK_LUMA_MAX: u8 = 80;
const PAGE_X_MIN: u32 = 80;

/// Per-row count of dark pixels in the page area of the current screen.
fn row_ink_profile_from_screen() -> Result<Vec<u32>> {
    let mut ss = Screenshot::new()?;
    ss.take_screenshot()?;
    // Decode once via a temp PNG: Screenshot::get_pixel re-decodes the
    // whole PNG per call, which is minutes of CPU at this many samples.
    let tmp = "/tmp/diary_scroll_probe.png";
    ss.save_image(tmp)?;
    let img = image::open(tmp)?.to_luma8();
    let (w, h) = (img.width().min(768), img.height().min(1024));
    Ok((0..h)
        .map(|y| (PAGE_X_MIN..w).step_by(2).filter(|&x| img.get_pixel(x, y).0[0] < INK_LUMA_MAX).count() as u32)
        .collect())
}

/// Best vertical shift aligning profile `a` (before) onto `b` (after),
/// by minimum sum-of-absolute-differences. Positive = content moved UP.
fn measure_shift(a: &[u32], b: &[u32]) -> i32 {
    let n = a.len().min(b.len());
    let mut best = (0i32, u64::MAX);
    for s in -950i32..=950 {
        let mut cost: u64 = 0;
        let mut count = 0u32;
        for y in 0..n as i32 {
            let ya = y + s;
            if ya < 0 || ya >= n as i32 {
                continue;
            }
            cost += (a[ya as usize] as i64 - b[y as usize] as i64).unsigned_abs();
            count += 1;
        }
        if count > 60 {
            let normalized = cost * 1024 / count as u64;
            if normalized < best.1 {
                best = (s, normalized);
            }
        }
    }
    best.0
}

/// One gentle upward fling: fast 100px drag in mid-screen (away from the
/// trigger corner and all toolbar UI). Calibrated to shift ~250-300px.
async fn fling_up() -> Result<()> {
    let mut touch = Touch::new(false, TriggerCorner::UpperRight);
    touch.touch_start((400, 750)).await?;
    for i in 1..=10 {
        touch.goto_xy((400, 750 - 10 * i)).await?;
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    touch.touch_stop().await?;
    Ok(())
}

/// Scroll the canvas up until at least `needed_px` of new blank space is
/// revealed below, measuring each fling's real effect. Returns the total
/// measured shift (0.0 if the page would not scroll — fixed-layout page,
/// or already at the scroll limit — in which case the caller should fall
/// back to drawing as-is).
pub async fn scroll_up_to_make_room(needed_px: f32) -> Result<f32> {
    let mut total = 0.0f32;
    for attempt in 1..=4 {
        if total >= needed_px {
            break;
        }
        let before = row_ink_profile_from_screen()?;
        fling_up().await?;
        // Let the fling's momentum finish and the eInk refresh settle
        // before measuring, or the after-screenshot catches mid-animation.
        tokio::time::sleep(Duration::from_millis(1200)).await;
        let after = row_ink_profile_from_screen()?;
        let shift = measure_shift(&before, &after);
        info!("scroll: fling {} shifted content by {}px (total {:.0}, needed {:.0})", attempt, shift, total + shift.max(0) as f32, needed_px);
        if shift <= 5 {
            break;
        }
        total += shift as f32;
    }
    Ok(total)
}
