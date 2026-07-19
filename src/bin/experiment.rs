use anyhow::Result;
use clap::{Parser, Subcommand};
use evdev::{Device, EventType as EvdevEventType, InputEvent};
use ghostwriter::pen::Pen;
use ghostwriter::screenshot::Screenshot;
use ghostwriter::touch::{Touch, TriggerCorner};
use ghostwriter::util::{svg_to_alpha_bitmap, svg_to_bitmap, svg_to_bitmap_threshold};
use std::thread::sleep as std_sleep;
use std::time::Duration;
use tokio::time::sleep;

// Touch device for RMPP
const TOUCH_DEVICE: &str = "/dev/input/event3";

// Virtual coordinate space
const VIRTUAL_WIDTH: f32 = 768.0;
const VIRTUAL_HEIGHT: f32 = 1024.0;

// RMPP touch input space
const TOUCH_SCREEN_WIDTH: f32 = 2065.0;
const TOUCH_SCREEN_HEIGHT: f32 = 2833.0;

// MT event codes
const ABS_MT_SLOT: u16 = 47;
const ABS_MT_TOUCH_MAJOR: u16 = 48;
const ABS_MT_TOUCH_MINOR: u16 = 49;
const ABS_MT_ORIENTATION: u16 = 52;
const ABS_MT_POSITION_X: u16 = 53;
const ABS_MT_POSITION_Y: u16 = 54;
const ABS_MT_TRACKING_ID: u16 = 57;
const ABS_MT_PRESSURE: u16 = 58;

fn virtual_to_touch(x: i32, y: i32) -> (i32, i32) {
    let tx = (x as f32 / VIRTUAL_WIDTH * TOUCH_SCREEN_WIDTH) as i32;
    let ty = (y as f32 / VIRTUAL_HEIGHT * TOUCH_SCREEN_HEIGHT) as i32;
    (tx, ty)
}

#[derive(Parser)]
#[command(name = "experiment")]
#[command(about = "Ghostwriter pen/touch experiment tool for reMarkable Paper Pro")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Capture the current screen to a PNG file
    Screenshot { output_path: String },
    /// Draw a line between two virtual coordinates (768x1024 space)
    DrawLine { x1: i32, y1: i32, x2: i32, y2: i32 },
    /// Draw a dot at a virtual coordinate with a small wiggle
    DrawDot { x: i32, y: i32 },
    /// Draw a rectangle (four lines) in virtual coordinates
    DrawRect { x1: i32, y1: i32, x2: i32, y2: i32 },
    /// Draw a triangle (three lines) in virtual coordinates
    DrawTriangle { x1: i32, y1: i32, x2: i32, y2: i32, x3: i32, y3: i32 },
    /// Draw a circle as a single continuous pen stroke (path tracing, not bitmap)
    DrawCircle { cx: i32, cy: i32, r: i32 },
    /// Render an SVG string via bitmap to pen strokes
    DrawSvg { svg_string: String },
    /// Render an SVG string via skeleton centerline (single-stroke, no fill needed)
    DrawSvgCenterline { svg_string: String },
    /// Draw text at given font size and y position using skeleton centerline rendering
    DrawText { text: String, font_size: u32, y: Option<u32> },
    /// Render an SVG string via path tracing (smooth strokes, handles text via glyph paths)
    DrawSvgPaths { svg_string: String },
    /// Render SVG via path tracing, bypassing usvg (better for geometric shapes, no text)
    DrawSvgPathsRaw { svg_string: String },
    /// Render a PNG file as dark pixels to pen strokes
    DrawPng { png_path: String },
    /// Single-finger tap at a virtual coordinate
    Tap { x: i32, y: i32 },
    /// Two-finger tap at a virtual coordinate (triggers undo in reMarkable)
    TwoFingerTap { x: i32, y: i32 },
    /// Touch swipe gesture between two virtual coordinates
    Swipe { x1: i32, y1: i32, x2: i32, y2: i32 },
    /// Navigate to a new page (swipe left, then tap new-page button)
    NewPage,
    /// Undo last stroke (two-finger tap at center)
    Undo,
    /// Wait N milliseconds
    SleepMs { ms: u64 },
    /// Switch to fineliner tool, detecting current palette state via pixel check
    SelectFineliner,
    /// Switch back to ballpoint tool
    SelectBallpoint,
    /// Read current tool state (for debugging pixel detection)
    ReadToolState,
    /// Render an SVG string via bidirectional scan (alternating L→R and R→L per row)
    DrawSvgBidi { svg_string: String },
    /// Render an SVG string via column-first scan (top→bottom per column)
    DrawSvgCol { svg_string: String },
    /// Render an SVG string using alpha-to-pressure mapping for anti-aliased rendering
    DrawSvgAlphaPressure { svg_string: String },
    /// Render an SVG string with configurable alpha threshold
    DrawSvgThreshold { svg_string: String, threshold: u8 },
    /// Render an SVG string at 3x scale for highest precision
    DrawSvgScale3x { svg_string: String },
    /// Render an SVG string with configurable threshold + bidirectional scan
    DrawSvgThresholdBidi { svg_string: String, threshold: u8 },
    /// Print the top and bottom rows containing ink (dark pixels), scanning
    /// the page area only (x >= 80, skipping the toolbar column). Used to
    /// measure how far a swipe scrolled the canvas.
    InkBounds,
    /// Measure the vertical scroll shift between two screenshots by
    /// cross-correlating their per-row ink-count profiles (page area only,
    /// x >= 80). Prints the shift in px (positive = content moved UP).
    ShiftBetween { png_a: String, png_b: String },
    /// Slow swipe ending with a stationary hold before lifting, so the
    /// release velocity is zero and no momentum/fling scroll kicks in.
    SwipeHold { x1: i32, y1: i32, x2: i32, y2: i32, steps: u32, step_ms: u64, hold_ms: u64 },
}

/// Per-row count of dark pixels in the page area (x >= 80, toolbar excluded).
fn row_ink_profile(path: &str) -> Result<Vec<u32>> {
    let img = image::open(path)?.to_luma8();
    let (w, h) = (img.width().min(768), img.height().min(1024));
    Ok((0..h)
        .map(|y| (80..w).step_by(2).filter(|&x| img.get_pixel(x, y).0[0] < 80).count() as u32)
        .collect())
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Screenshot { output_path } => {
            let mut screenshot = Screenshot::new()?;
            screenshot.take_screenshot()?;
            screenshot.save_image(&output_path)?;
            println!("Screenshot saved to {}", output_path);
        }

        Commands::DrawLine { x1, y1, x2, y2 } => {
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.draw_line_screen((x1, y1), (x2, y2))
            })
            .await?;
            println!("Drew line from ({}, {}) to ({}, {})", x1, y1, x2, y2);
        }

        Commands::DrawDot { x, y } => {
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.pen_up()?;
                pen.goto_xy_virtual((x, y))?;
                pen.pen_down()?;
                for n in 0..3 {
                    pen.goto_xy_virtual((x + n, y + n))?;
                }
                pen.goto_xy_virtual((x, y))?;
                pen.pen_up()?;
                std_sleep(Duration::from_millis(5));
                Ok(())
            })
            .await?;
            println!("Drew dot at ({}, {})", x, y);
        }

        Commands::DrawRect { x1, y1, x2, y2 } => {
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.draw_line_screen((x1, y1), (x2, y1))?; // top
                pen.draw_line_screen((x2, y1), (x2, y2))?; // right
                pen.draw_line_screen((x2, y2), (x1, y2))?; // bottom
                pen.draw_line_screen((x1, y2), (x1, y1))   // left
            })
            .await?;
            println!("Drew rect ({}, {}) to ({}, {})", x1, y1, x2, y2);
        }

        Commands::DrawTriangle { x1, y1, x2, y2, x3, y3 } => {
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.draw_line_screen((x1, y1), (x2, y2))?;
                pen.draw_line_screen((x2, y2), (x3, y3))?;
                pen.draw_line_screen((x3, y3), (x1, y1))
            })
            .await?;
            println!("Drew triangle ({},{}) ({},{}) ({},{})", x1, y1, x2, y2, x3, y3);
        }

        Commands::DrawCircle { cx, cy, r } => {
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                let circumference = 2.0 * std::f32::consts::PI * r as f32;
                let steps = (circumference / 3.0).ceil() as usize;
                pen.pen_up()?;
                let x0 = cx + r;
                let y0 = cy;
                pen.pen_down_at(pen.virtual_to_input_pub((x0, y0)))?;
                for i in 0..=steps {
                    let angle = 2.0 * std::f32::consts::PI * i as f32 / steps as f32;
                    let x = (cx as f32 + r as f32 * angle.cos()).round() as i32;
                    let y = (cy as f32 + r as f32 * angle.sin()).round() as i32;
                    pen.goto_xy_virtual((x, y))?;
                }
                pen.pen_up()
            })
            .await?;
            println!("Drew circle at ({}, {}) r={}", cx, cy, r);
        }

        Commands::DrawSvg { svg_string } => {
            // Render at 2x resolution for sub-pixel accuracy; draw_bitmap_scaled maps back
            let scale = 2u32;
            let bitmap = svg_to_bitmap(&svg_string, 768 * scale, 1024 * scale)?;
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.draw_bitmap_scaled(&bitmap, scale)
            })
            .await?;
            println!("Drew SVG ({} chars)", svg_string.len());
        }

        Commands::DrawSvgCenterline { svg_string } => {
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.draw_svg_centerline(&svg_string)
            })
            .await?;
            println!("Drew SVG centerline ({} chars)", svg_string.len());
        }

        Commands::DrawText { text, font_size, y } => {
            let y = y.unwrap_or_else(|| 200u32.max(font_size + 150));
            let svg_string = format!(
                r#"<svg width="768" height="1024" xmlns="http://www.w3.org/2000/svg"><text x="80" y="{}" font-family="sans-serif" font-size="{}" fill="black">{}</text></svg>"#,
                y, font_size, text
            );
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.draw_svg_centerline(&svg_string)
            })
            .await?;
            println!("Drew text '{}' at font-size {} (y={})", text, font_size, y);
        }

        Commands::DrawSvgPaths { svg_string } => {
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.draw_svg_paths(&svg_string)
            })
            .await?;
            println!("Drew SVG paths ({} chars)", svg_string.len());
        }

        Commands::DrawSvgPathsRaw { svg_string } => {
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.draw_svg_paths_raw(&svg_string)
            })
            .await?;
            println!("Drew SVG paths raw ({} chars)", svg_string.len());
        }

        Commands::DrawPng { png_path } => {
            let img = image::open(&png_path)?.to_luma8();
            let bitmap: Vec<Vec<bool>> =
                img.rows().map(|row| row.map(|p| p[0] < 128).collect()).collect();
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.draw_bitmap(&bitmap)
            })
            .await?;
            println!("Drew PNG from {}", png_path);
        }

        Commands::Tap { x, y } => {
            let mut touch = Touch::new(false, TriggerCorner::UpperRight);
            touch.touch_start((x, y)).await?;
            sleep(Duration::from_millis(100)).await;
            touch.touch_stop().await?;
            println!("Tapped at ({}, {})", x, y);
        }

        Commands::TwoFingerTap { x, y } => {
            two_finger_tap(x, y).await?;
            println!("Two-finger tapped at ({}, {})", x, y);
        }

        Commands::Swipe { x1, y1, x2, y2 } => {
            let mut touch = Touch::new(false, TriggerCorner::UpperRight);
            touch.touch_start((x1, y1)).await?;
            let steps = 20i32;
            for i in 0..=steps {
                let t = i as f32 / steps as f32;
                let x = (x1 as f32 + (x2 - x1) as f32 * t) as i32;
                let y = (y1 as f32 + (y2 - y1) as f32 * t) as i32;
                touch.goto_xy((x, y)).await?;
                sleep(Duration::from_millis(10)).await;
            }
            touch.touch_stop().await?;
            println!("Swiped from ({}, {}) to ({}, {})", x1, y1, x2, y2);
        }

        Commands::NewPage => {
            let mut touch = Touch::new(false, TriggerCorner::UpperRight);
            // Swipe right-to-left to bring up the page navigation UI
            touch.touch_start((700, 512)).await?;
            let steps = 20i32;
            for i in 0..=steps {
                let t = i as f32 / steps as f32;
                let x = (700.0 - 600.0 * t) as i32;
                touch.goto_xy((x, 512)).await?;
                sleep(Duration::from_millis(10)).await;
            }
            touch.touch_stop().await?;
            sleep(Duration::from_millis(300)).await;
            // Tap the new-page button (dark circle icon at right side ~x=700, y=514)
            touch.touch_start((700, 514)).await?;
            sleep(Duration::from_millis(100)).await;
            touch.touch_stop().await?;
            sleep(Duration::from_millis(300)).await;
            println!("New page command sent");
        }

        Commands::Undo => {
            two_finger_tap(384, 512).await?;
            println!("Undo sent");
        }

        Commands::SleepMs { ms } => {
            sleep(Duration::from_millis(ms)).await;
            println!("Slept {}ms", ms);
        }

        Commands::SelectFineliner => {
            let mut touch = Touch::new(false, TriggerCorner::UpperRight);
            touch.select_fineliner().await?;
            println!("SelectFineliner: done");
        }

        Commands::SelectBallpoint => {
            println!("SelectBallpoint: removed — no verified RM2 coordinate for this yet (see src/touch.rs's tool-palette-helpers comment). Use SelectFineliner or the diary's select_calligraphy_pen instead.");
        }

        Commands::ReadToolState => {
            // Take a screenshot and report whether the pen-tool sidebar slot
            // is currently the highlighted one (matches Touch::pen_slot_is_active).
            let mut ss = Screenshot::new()?;
            ss.take_screenshot()?;
            let pen_slot_pixel = ss.get_pixel(5, 91);
            let pen_active = pen_slot_pixel.map(|(r,_,_)| r < 128).unwrap_or(false);
            println!("Pen slot active: {} | pixel(5,91)={:?}", pen_active, pen_slot_pixel);
        }

        Commands::DrawSvgBidi { svg_string } => {
            let scale = 2u32;
            let bitmap = svg_to_bitmap(&svg_string, 768 * scale, 1024 * scale)?;
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.draw_bitmap_bidi(&bitmap, scale)
            })
            .await?;
            println!("Drew SVG bidi ({} chars)", svg_string.len());
        }

        Commands::DrawSvgCol { svg_string } => {
            let scale = 2u32;
            let bitmap = svg_to_bitmap(&svg_string, 768 * scale, 1024 * scale)?;
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.draw_bitmap_col(&bitmap, scale)
            })
            .await?;
            println!("Drew SVG col ({} chars)", svg_string.len());
        }

        Commands::DrawSvgAlphaPressure { svg_string } => {
            let scale = 2u32;
            let alpha_bitmap = svg_to_alpha_bitmap(&svg_string, 768 * scale, 1024 * scale)?;
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.draw_bitmap_alpha_pressure(&alpha_bitmap, scale)
            })
            .await?;
            println!("Drew SVG alpha-pressure ({} chars)", svg_string.len());
        }

        Commands::DrawSvgThreshold { svg_string, threshold } => {
            let scale = 2u32;
            let bitmap = svg_to_bitmap_threshold(&svg_string, 768 * scale, 1024 * scale, threshold)?;
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.draw_bitmap_scaled(&bitmap, scale)
            })
            .await?;
            println!("Drew SVG threshold={} ({} chars)", threshold, svg_string.len());
        }

        Commands::DrawSvgScale3x { svg_string } => {
            let scale = 3u32;
            let bitmap = svg_to_bitmap(&svg_string, 768 * scale, 1024 * scale)?;
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.draw_bitmap_scaled(&bitmap, scale)
            })
            .await?;
            println!("Drew SVG scale3x ({} chars)", svg_string.len());
        }

        Commands::DrawSvgThresholdBidi { svg_string, threshold } => {
            let scale = 2u32;
            let bitmap = svg_to_bitmap_threshold(&svg_string, 768 * scale, 1024 * scale, threshold)?;
            with_fineliner(|| {
                let mut pen = Pen::new(false);
                pen.draw_bitmap_bidi(&bitmap, scale)
            })
            .await?;
            println!("Drew SVG threshold-bidi threshold={} ({} chars)", threshold, svg_string.len());
        }

        Commands::InkBounds => {
            let mut screenshot = Screenshot::new()?;
            screenshot.take_screenshot()?;
            // Decode ONCE via a temp PNG: Screenshot::get_pixel re-decodes
            // the stored PNG on every call, which at ~350k sampled pixels
            // spins the CPU for minutes on-device.
            let tmp = "/tmp/inkbounds.png";
            screenshot.save_image(tmp)?;
            let img = image::open(tmp)?.to_luma8();
            let (w, h) = (img.width().min(768), img.height().min(1024));
            let mut top: Option<u32> = None;
            let mut bottom: Option<u32> = None;
            for y in 0..h {
                // Skip the toolbar column (x < 80); sample every 2px.
                let has_ink = (80..w).step_by(2).any(|x| img.get_pixel(x, y).0[0] < 80);
                if has_ink {
                    if top.is_none() {
                        top = Some(y);
                    }
                    bottom = Some(y);
                }
            }
            match (top, bottom) {
                (Some(t), Some(b)) => println!("ink top={} bottom={}", t, b),
                _ => println!("no ink found"),
            }
        }

        Commands::ShiftBetween { png_a, png_b } => {
            let a = row_ink_profile(&png_a)?;
            let b = row_ink_profile(&png_b)?;
            let n = a.len().min(b.len());
            // For each candidate shift s (content moved UP by s px between A
            // and B), compare A's row y+s against B's row y over the overlap
            // and score by sum of absolute differences (lower = better).
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
            println!("shift={} (cost={})", best.0, best.1);
        }

        Commands::SwipeHold { x1, y1, x2, y2, steps, step_ms, hold_ms } => {
            let mut touch = Touch::new(false, TriggerCorner::UpperRight);
            touch.touch_start((x1, y1)).await?;
            for i in 0..=steps {
                let t = i as f32 / steps.max(1) as f32;
                let x = (x1 as f32 + (x2 - x1) as f32 * t) as i32;
                let y = (y1 as f32 + (y2 - y1) as f32 * t) as i32;
                touch.goto_xy((x, y)).await?;
                sleep(Duration::from_millis(step_ms)).await;
            }
            sleep(Duration::from_millis(hold_ms)).await;
            touch.touch_stop().await?;
            println!("Swipe-hold from ({}, {}) to ({}, {}), {} steps x {}ms, hold {}ms", x1, y1, x2, y2, steps, step_ms, hold_ms);
        }
    }

    Ok(())
}

/// Switch to fineliner (with correct size/color settings), then run drawing closure.
/// Deliberately does not restore the previous tool afterward — see
/// src/touch.rs's tool-palette-helpers comment for why.
async fn with_fineliner<F: FnOnce() -> Result<()>>(f: F) -> Result<()> {
    let mut touch = Touch::new(false, TriggerCorner::UpperRight);
    touch.select_fineliner().await?;
    sleep(Duration::from_millis(500)).await; // Wait for palette close animation to finish
    f()
}

async fn two_finger_tap(x: i32, y: i32) -> Result<()> {
    let mut device = Device::open(TOUCH_DEVICE)?;
    let (tx, ty) = virtual_to_touch(x, y);
    let (tx2, ty2) = virtual_to_touch(x + 50, y + 50);

    // Press two fingers simultaneously
    device.send_events(&[
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_SLOT, 0),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_TRACKING_ID, 1),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_POSITION_X, tx),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_POSITION_Y, ty),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_PRESSURE, 100),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_TOUCH_MAJOR, 17),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_TOUCH_MINOR, 17),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_ORIENTATION, 4),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_SLOT, 1),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_TRACKING_ID, 2),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_POSITION_X, tx2),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_POSITION_Y, ty2),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_PRESSURE, 100),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_TOUCH_MAJOR, 17),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_TOUCH_MINOR, 17),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_ORIENTATION, 4),
        InputEvent::new(EvdevEventType::SYNCHRONIZATION.0, 0, 0),
    ])?;

    sleep(Duration::from_millis(100)).await;

    // Release both fingers
    device.send_events(&[
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_SLOT, 0),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_TRACKING_ID, -1),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_SLOT, 1),
        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_TRACKING_ID, -1),
        InputEvent::new(EvdevEventType::SYNCHRONIZATION.0, 0, 0),
    ])?;

    Ok(())
}
