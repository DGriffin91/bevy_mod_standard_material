//! Loads and renders a glTF file as a scene.

mod camera_controller;
mod helmet;
mod load_sponza;
mod pbr_material;

use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    pbr::DirectionalLightShadowMap,
    prelude::*,
};
use helmet::HelmetScenePlugin;
use load_sponza::SponzaPlugin;
use pbr_material::{swap_standard_material, CustomStandardMaterial};

fn main() {
    App::new()
        //.add_plugin(SponzaPlugin)
        .add_plugin(HelmetScenePlugin)
        .insert_resource(Msaa::Off)
        // 2048 is default
        .insert_resource(DirectionalLightShadowMap { size: 2048 })
        .add_plugins(DefaultPlugins.set(AssetPlugin {
            watch_for_changes: true,
            ..default()
        }))
        .add_plugin(MaterialPlugin::<CustomStandardMaterial>::default())
        .add_system(swap_standard_material)
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .run();
}
