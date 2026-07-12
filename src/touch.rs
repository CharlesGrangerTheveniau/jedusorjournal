use anyhow::Result;
use evdev::EventType as EvdevEventType;
use evdev::{Device, EventStream, InputEvent};
use log::{debug, info, trace};

use std::time::Duration;
use tokio::time::sleep;

use crate::cancellation::GhostwriterCancellation;
use crate::device::DeviceModel;
use crate::screenshot::Screenshot;
use crate::simulation::{SimulationConfig, TouchSimulator};


#[derive(Debug, Clone, Copy)]
pub enum TriggerCorner {
    UpperRight,
    UpperLeft,
    LowerRight,
    LowerLeft,
}

impl TriggerCorner {
    pub fn from_string(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "ur" | "upper-right" => Ok(TriggerCorner::UpperRight),
            "ul" | "upper-left" => Ok(TriggerCorner::UpperLeft),
            "lr" | "lower-right" => Ok(TriggerCorner::LowerRight),
            "ll" | "lower-left" => Ok(TriggerCorner::LowerLeft),
            _ => Err(anyhow::anyhow!(
                "Invalid trigger corner: {}. Use UR, UL, LR, LL, upper-right, upper-left, lower-right, or lower-left",
                s
            )),
        }
    }
}

// Output dimensions remain the same for both devices
const VIRTUAL_WIDTH: u16 = 768;
const VIRTUAL_HEIGHT: u16 = 1024;

// Event codes
const ABS_MT_SLOT: u16 = 47;
const ABS_MT_TOUCH_MAJOR: u16 = 48;
const ABS_MT_TOUCH_MINOR: u16 = 49;
const ABS_MT_ORIENTATION: u16 = 52;
const ABS_MT_POSITION_X: u16 = 53;
const ABS_MT_POSITION_Y: u16 = 54;
// const ABS_MT_TOOL_TYPE: u16 = 55;
const ABS_MT_TRACKING_ID: u16 = 57;
const ABS_MT_PRESSURE: u16 = 58;

pub enum TouchMode {
    Real {
        input_device: Option<Device>,      // For sending touch events
        event_stream: Option<EventStream>, // For reading touch events
        device_model: DeviceModel,
    },
    Simulated {
        simulator: TouchSimulator,
    },
}

pub struct Touch {
    mode: TouchMode,
    trigger_corner: TriggerCorner,
}

impl Touch {
    pub fn new(no_touch: bool, trigger_corner: TriggerCorner) -> Self {
        let device_model = DeviceModel::detect();
        info!("Touch using device model: {}", device_model.name());

        let device_path = match device_model {
            DeviceModel::Remarkable2 => "/dev/input/event2",
            DeviceModel::RemarkablePaperPro => "/dev/input/event3",
            DeviceModel::Unknown => "/dev/input/event2", // Default to RM2
        };

        let (input_device, event_stream) = if no_touch {
            (None, None)
        } else {
            let input_dev = Device::open(device_path).unwrap();
            let read_dev = Device::open(device_path).unwrap();
            let stream = read_dev.into_event_stream().unwrap();
            (Some(input_dev), Some(stream))
        };

        Self {
            mode: TouchMode::Real {
                input_device,
                event_stream,
                device_model,
            },
            trigger_corner,
        }
    }

    pub fn new_simulated(simulation_config: SimulationConfig, trigger_corner: TriggerCorner) -> Result<Self> {
        let simulator = TouchSimulator::new(simulation_config, trigger_corner)?;
        info!("Touch using simulation mode");

        Ok(Self {
            mode: TouchMode::Simulated { simulator },
            trigger_corner,
        })
    }

    pub async fn wait_for_trigger(&mut self, cancellation: &GhostwriterCancellation) -> Result<()> {
        debug!("wait_for_trigger: entered, checking mode");
        match &mut self.mode {
            TouchMode::Simulated { simulator } => {
                debug!("wait_for_trigger: using Simulated mode");
                simulator.wait_for_trigger(cancellation).await
            }
            TouchMode::Real {
                event_stream, device_model, ..
            } => {
                debug!("wait_for_trigger: using Real device mode");
                let trigger_corner = self.trigger_corner;
                Self::wait_for_real_trigger(event_stream, device_model, trigger_corner, cancellation).await
            }
        }
    }

    async fn wait_for_real_trigger(
        event_stream: &mut Option<EventStream>,
        device_model: &DeviceModel,
        trigger_corner: TriggerCorner,
        cancellation: &GhostwriterCancellation,
    ) -> Result<()> {
        debug!("wait_for_real_trigger: entered");
        let mut position_x = 0;
        let mut position_y = 0;

        if let Some(events) = event_stream {
            debug!("wait_for_real_trigger: event stream available, entering wait loop");

            loop {
                debug!("wait_for_real_trigger: loop iteration starting");
                tokio::select! {
                    // Check for cancellation (only main token, not execution cycles)
                    _ = async {
                        while !cancellation.should_cancel_main() {
                            sleep(Duration::from_millis(50)).await;
                        }
                    } => {
                        debug!("wait_for_real_trigger: cancellation detected");
                        debug!("Touch waiting cancelled due to shutdown");
                        return Err(anyhow::anyhow!("Touch waiting cancelled"));
                    }

                    // Wait for next event
                    event_result = events.next_event() => {
                        debug!("wait_for_real_trigger: received event");
                        match event_result {
                            Ok(event) => {
                                if event.code() == ABS_MT_POSITION_X {
                                    position_x = event.value();
                                }
                                if event.code() == ABS_MT_POSITION_Y {
                                    position_y = event.value();
                                }
                                if event.code() == ABS_MT_TRACKING_ID && event.value() == -1 {
                                    let (x, y) = Self::input_to_virtual((position_x, position_y), device_model);
                                    debug!("Touch release detected at ({}, {}) normalized ({}, {})", position_x, position_y, x, y);
                                    if Self::is_in_trigger_zone(x, y, trigger_corner) {
                                        debug!("Touch release in target zone!");
                                        debug!("wait_for_real_trigger: returning Ok()");
                                        return Ok(());
                                    } else {
                                        debug!("Touch release NOT in trigger zone, continuing");
                                    }
                                }
                            }
                            Err(e) => {
                                debug!("Error reading touch events: {}", e);
                                return Err(e.into());
                            }
                        }
                    }
                }
            }
        } else {
            debug!("wait_for_real_trigger: no event stream available, entering cancellation wait loop");
            // No event stream available, just wait for cancellation
            loop {
                if cancellation.should_cancel_main() {
                    debug!("wait_for_real_trigger: cancellation detected in no-stream path");
                    debug!("Touch waiting cancelled due to shutdown");
                    return Err(anyhow::anyhow!("Touch waiting cancelled"));
                }
                sleep(Duration::from_millis(50)).await;
            }
        }
    }

    pub async fn touch_start(&mut self, xy: (i32, i32)) -> Result<()> {
        match &mut self.mode {
            TouchMode::Simulated { .. } => {
                debug!("Simulated touch_start at ({}, {})", xy.0, xy.1);
                Ok(())
            }
            TouchMode::Real {
                input_device, device_model, ..
            } => {
                let (x, y) = Self::virtual_to_input(xy, device_model);
                if let Some(device) = input_device {
                    trace!("touch_start at ({}, {})", x, y);
                    device.send_events(&[
                        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_SLOT, 0),
                        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_TRACKING_ID, 1),
                        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_POSITION_X, x),
                        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_POSITION_Y, y),
                        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_PRESSURE, 100),
                        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_TOUCH_MAJOR, 17),
                        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_TOUCH_MINOR, 17),
                        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_ORIENTATION, 4),
                        InputEvent::new(EvdevEventType::SYNCHRONIZATION.0, 0, 0), // SYN_REPORT
                    ])?;
                    sleep(Duration::from_millis(1)).await;
                }
                Ok(())
            }
        }
    }

    pub async fn touch_stop(&mut self) -> Result<()> {
        match &mut self.mode {
            TouchMode::Simulated { .. } => {
                debug!("Simulated touch_stop");
                Ok(())
            }
            TouchMode::Real { input_device, .. } => {
                if let Some(device) = input_device {
                    trace!("touch_stop");
                    device.send_events(&[
                        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_SLOT, 0),
                        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_TRACKING_ID, -1),
                        InputEvent::new(EvdevEventType::SYNCHRONIZATION.0, 0, 0), // SYN_REPORT
                    ])?;
                    sleep(Duration::from_millis(1)).await;
                }
                Ok(())
            }
        }
    }

    pub async fn goto_xy(&mut self, xy: (i32, i32)) -> Result<()> {
        match &mut self.mode {
            TouchMode::Simulated { .. } => {
                debug!("Simulated goto_xy at ({}, {})", xy.0, xy.1);
                Ok(())
            }
            TouchMode::Real {
                input_device, device_model, ..
            } => {
                let (x, y) = Self::virtual_to_input(xy, device_model);
                if let Some(device) = input_device {
                    device.send_events(&[
                        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_SLOT, 0),
                        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_TRACKING_ID, 1),
                        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_POSITION_X, x),
                        InputEvent::new(EvdevEventType::ABSOLUTE.0, ABS_MT_POSITION_Y, y),
                        InputEvent::new(EvdevEventType::SYNCHRONIZATION.0, 0, 0), // SYN_REPORT
                    ])?;
                }
                Ok(())
            }
        }
    }

    pub async fn tap_middle_bottom(&mut self) -> Result<()> {
        self.touch_start((384, 1023)).await?; // middle bottom
        sleep(Duration::from_millis(100)).await;
        self.touch_stop().await?;
        // sleep(Duration::from_millis(10));
        // sleep(Duration::from_millis(100));
        Ok(())
    }

    // ── Tool palette helpers ────────────────────────────────────────────────
    //
    // Recalibrated 2026-07-12 against reMarkable 2 software 3.27.3.0's
    // actual toolbar. The previous version of this code (see git history)
    // was calibrated for a reMarkable Paper Pro's UI layout, which uses
    // different coordinates entirely — on RM2 those coordinates landed on
    // the Eraser icon instead of a pen type, and caused real, unrecoverable
    // content loss during on-device testing of this feature (a drawing
    // pass ran with the eraser active instead of a pen). Every coordinate
    // below was individually verified on real hardware: tapped once, then
    // confirmed via a screenshot showing the expected on-screen result
    // (e.g. the settings panel's header text, or the sidebar's
    // highlighted-icon pixel state) before being hardcoded here. Do not
    // add or change a coordinate without the same verification, and do not
    // reuse coordinates from any other reMarkable device or software
    // version — verify fresh.
    //
    // RM2's toolbar has a single pen-type sidebar slot at PEN_SLOT: tapping
    // it activates pen mode (recalling whichever type was last used) if
    // some other tool is currently active, or opens a settings panel
    // (pen type / stroke size / color) if pen mode is already active. The
    // eraser lives in a separate, adjacent sidebar slot — intentionally
    // not referenced anywhere below.

    /// The single pen-tool sidebar slot (RM2, software 3.27.3.0).
    const PEN_SLOT: (i32, i32) = (28, 91);

    /// Calligraphy pen icon within the settings panel's pen-type grid.
    /// Verified: tapping this shows the panel header "Calligraphy pen".
    const PEN_TYPE_CALLIGRAPHY: (i32, i32) = (231, 197);

    /// Fineliner icon within the same grid. Verified: header "Fineliner".
    const PEN_TYPE_FINELINER: (i32, i32) = (170, 136);

    /// Thin stroke-width icon within the settings panel.
    const SETTINGS_SIZE_THIN: (i32, i32) = (109, 380);

    /// Medium stroke-width icon within the settings panel. Verified:
    /// tapping this shows the panel header "Medium" (was "Thin").
    const SETTINGS_SIZE_MEDIUM: (i32, i32) = (170, 380);

    /// Black color icon within the settings panel.
    const SETTINGS_COLOR_BLACK: (i32, i32) = (109, 478);

    /// Tap a point (touch_start + brief hold + touch_stop).
    async fn tap(&mut self, xy: (i32, i32)) -> Result<()> {
        self.touch_start(xy).await?;
        sleep(Duration::from_millis(100)).await;
        self.touch_stop().await?;
        sleep(Duration::from_millis(300)).await;
        Ok(())
    }

    /// Detect whether the pen-tool sidebar slot is the currently highlighted
    /// one, by checking whether its background is dark (pen mode active) or
    /// light (some other tool, e.g. eraser, is active instead). Verified
    /// against real screenshots in both states: pixel (5, 91) reads pure
    /// black when pen mode is active, pure white when it is not.
    async fn pen_slot_is_active(&self) -> bool {
        let Ok(mut ss) = Screenshot::new() else {
            return false;
        };
        if ss.take_screenshot().is_err() {
            return false;
        }
        ss.get_pixel(5, 91).map(|(r, _, _)| r < 128).unwrap_or(false)
    }

    /// Select a specific pen type and stroke width (color is always
    /// forced to black), using only verified RM2 coordinates. Never taps
    /// the eraser icon or any unverified location.
    async fn select_pen_type(&mut self, type_xy: (i32, i32), size_xy: (i32, i32)) -> Result<()> {
        if !self.pen_slot_is_active().await {
            // Some other tool (eraser, text, select, ...) is active: one
            // tap here reactivates pen mode (recalling the last-used pen
            // type) without opening the settings panel, so a second tap
            // is needed to actually open it.
            self.tap(Self::PEN_SLOT).await?;
        }
        // Pen mode is active either way now: this tap opens its settings panel.
        self.tap(Self::PEN_SLOT).await?;
        self.tap(type_xy).await?;
        self.tap(size_xy).await?;
        self.tap(Self::SETTINGS_COLOR_BLACK).await?;
        // Close the panel.
        self.tap(Self::PEN_SLOT).await?;
        Ok(())
    }

    /// Select the calligraphy pen (medium, black) — used for the diary's
    /// cursive handwriting.
    pub async fn select_calligraphy_pen(&mut self) -> Result<()> {
        self.select_pen_type(Self::PEN_TYPE_CALLIGRAPHY, Self::SETTINGS_SIZE_MEDIUM).await?;
        info!("select_calligraphy_pen: done");
        Ok(())
    }

    /// Select the fineliner pen (thin, black) — used for draw_svg.
    pub async fn select_fineliner(&mut self) -> Result<()> {
        self.select_pen_type(Self::PEN_TYPE_FINELINER, Self::SETTINGS_SIZE_THIN).await?;
        info!("select_fineliner: done");
        Ok(())
    }

    fn is_in_trigger_zone(x: i32, y: i32, trigger_corner: TriggerCorner) -> bool {
        const CORNER_SIZE: i32 = 68; // Size of the trigger zone (68x68 pixels)

        match trigger_corner {
            TriggerCorner::UpperRight => x > VIRTUAL_WIDTH as i32 - CORNER_SIZE && y < CORNER_SIZE,
            TriggerCorner::UpperLeft => x < CORNER_SIZE && y < CORNER_SIZE,
            TriggerCorner::LowerRight => x > VIRTUAL_WIDTH as i32 - CORNER_SIZE && y > VIRTUAL_HEIGHT as i32 - CORNER_SIZE,
            TriggerCorner::LowerLeft => x < CORNER_SIZE && y > VIRTUAL_HEIGHT as i32 - CORNER_SIZE,
        }
    }

    fn virtual_to_input((x, y): (i32, i32), device_model: &DeviceModel) -> (i32, i32) {
        // Swap and normalize the coordinates
        let x_normalized = x as f32 / VIRTUAL_WIDTH as f32;
        let y_normalized = y as f32 / VIRTUAL_HEIGHT as f32;
        let (screen_width, screen_height) = Self::screen_dimensions(device_model);

        match device_model {
            DeviceModel::RemarkablePaperPro => {
                let x_input = (x_normalized * screen_width as f32) as i32;
                let y_input = (y_normalized * screen_height as f32) as i32;
                (x_input, y_input)
            }
            _ => {
                // RM2 coordinate transformation
                let x_input = (x_normalized * screen_width as f32) as i32;
                let y_input = ((1.0 - y_normalized) * screen_height as f32) as i32;
                (x_input, y_input)
            }
        }
    }

    fn input_to_virtual((x, y): (i32, i32), device_model: &DeviceModel) -> (i32, i32) {
        // Swap and normalize the coordinates
        let (screen_width, screen_height) = Self::screen_dimensions(device_model);
        let x_normalized = x as f32 / screen_width as f32;
        let y_normalized = y as f32 / screen_height as f32;

        match device_model {
            DeviceModel::RemarkablePaperPro => {
                let x_input = (x_normalized * VIRTUAL_WIDTH as f32) as i32;
                let y_input = (y_normalized * VIRTUAL_HEIGHT as f32) as i32;
                (x_input, y_input)
            }
            _ => {
                // RM2 coordinate transformation
                let x_input = (x_normalized * VIRTUAL_WIDTH as f32) as i32;
                let y_input = ((1.0 - y_normalized) * VIRTUAL_HEIGHT as f32) as i32;
                (x_input, y_input)
            }
        }
    }

    fn screen_dimensions(device_model: &DeviceModel) -> (u32, u32) {
        match device_model {
            DeviceModel::Remarkable2 => (1404, 1872),
            DeviceModel::RemarkablePaperPro => (2065, 2833),
            DeviceModel::Unknown => (1404, 1872), // Default to RM2
        }
    }

    /// Update the trigger corner (called when config changes)
    pub fn set_trigger_corner(&mut self, new_corner: TriggerCorner) {
        self.trigger_corner = new_corner;
        if let TouchMode::Simulated { simulator } = &mut self.mode {
            simulator.set_trigger_corner(new_corner);
        }
    }

    /// Get handle for manual triggering (for web API in simulation mode)
    pub fn get_manual_trigger_handle(&self) -> Option<std::sync::Arc<std::sync::Mutex<Vec<TriggerCorner>>>> {
        match &self.mode {
            TouchMode::Simulated { simulator } => Some(simulator.get_manual_trigger_handle()),
            TouchMode::Real { .. } => None,
        }
    }

    /// Add a manual trigger (for web API in simulation mode)
    pub fn add_manual_trigger(&self, corner: TriggerCorner) {
        if let TouchMode::Simulated { simulator } = &self.mode {
            simulator.add_manual_trigger(corner);
        }
    }
}
