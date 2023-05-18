use bevy::{
    core::FrameCount,
    diagnostic::{Diagnostic, DiagnosticId, Diagnostics},
    prelude::*,
};

/// Adds "frame time" diagnostic to an App, specifically "frame time", "fps" and "frame count"
#[derive(Default)]
pub struct FrameTimeDiagnosticsPlugin;

impl Plugin for FrameTimeDiagnosticsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, Self::setup_system)
            .add_systems(Update, Self::diagnostic_system);
    }
}

impl FrameTimeDiagnosticsPlugin {
    pub const FPS: DiagnosticId = DiagnosticId::from_u128(228146834822086093791974401528866909483);
    pub const FRAME_TIME: DiagnosticId =
        DiagnosticId::from_u128(73471630925188532774622109383099159199);

    pub fn setup_system(mut diagnostics: ResMut<Diagnostics>) {
        diagnostics.add(
            Diagnostic::new(Self::FRAME_TIME, "frame_time", 30)
                .with_suffix("ms")
                .with_smoothing_factor(2.0),
        );
        diagnostics.add(Diagnostic::new(Self::FPS, "fps", 30).with_smoothing_factor(2.0));
    }

    pub fn diagnostic_system(
        mut diagnostics: ResMut<Diagnostics>,
        time: Res<Time>,
        frame_count: Res<FrameCount>,
    ) {
        let delta_seconds = time.raw_delta_seconds_f64();
        if delta_seconds == 0.0 {
            return;
        }

        diagnostics.add_measurement(Self::FRAME_TIME, || delta_seconds * 1000.0);

        diagnostics.add_measurement(Self::FPS, || 1.0 / delta_seconds);
    }
}
