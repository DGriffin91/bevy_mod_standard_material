//! Loads and renders a glTF file as a scene.

mod camera_controller;
mod copy_frame;
mod helmet;
mod load_sponza;
mod pbr_material;
mod sphere;

use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    pbr::DirectionalLightShadowMap,
    prelude::*,
};
use copy_frame::CopyFramePlugin;
use helmet::HelmetScenePlugin;
use load_sponza::SponzaPlugin;
use pbr_material::{load_blue_noise, swap_standard_material, CustomStandardMaterial};
use sphere::SphereScenePlugin;

fn main() {
    App::new()
        .add_plugin(SponzaPlugin)
        //.add_plugin(HelmetScenePlugin)
        //.add_plugin(SphereScenePlugin)
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
        .add_plugin(CopyFramePlugin)
        .add_startup_system(load_blue_noise)
        .run();
}
