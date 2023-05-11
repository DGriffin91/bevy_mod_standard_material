//! Loads and renders a glTF file as a scene.

mod bistro;
mod camera_controller;
mod copy_frame;
mod helmet;
mod image_window_auto_size;
mod kitchen;
mod load_sponza;
mod path_trace;
mod pbr_material;
mod prepass_downsample;
mod screen_space_passes;
mod sphere;

use bevy::{
    core_pipeline::{core_3d, experimental::taa::TemporalAntiAliasPlugin},
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    pbr::DirectionalLightShadowMap,
    prelude::*,
    render::{render_graph::RenderGraphApp, RenderApp},
    window::PresentMode,
};
use bevy_coordinate_systems::CoordinateTransformationsPlugin;
use bevy_mod_bvh::{DynamicTLAS, StaticTLAS};
use bistro::BistroPlugin;
use copy_frame::CopyFramePlugin;
use helmet::HelmetScenePlugin;
use kitchen::KitchenPlugin;
use load_sponza::SponzaPlugin;
use path_trace::PathTracePlugin;
use pbr_material::{load_blue_noise, swap_standard_material, CustomStandardMaterial};
use prepass_downsample::PrepassDownsample;
use screen_space_passes::ScreenSpacePassesPlugin;
use sphere::SphereScenePlugin;

fn main() {
    let mut app = App::new();
    app
        //.add_plugin(BistroPlugin)
        .add_plugin(KitchenPlugin)
        //.add_plugin(SponzaPlugin)
        //.add_plugin(HelmetScenePlugin)
        //.add_plugin(SphereScenePlugin)
        .insert_resource(Msaa::Off)
        // 2048 is default
        .insert_resource(DirectionalLightShadowMap { size: 2048 })
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    watch_for_changes: true,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: PresentMode::Immediate,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugin(CopyFramePlugin)
        .add_plugin(PrepassDownsample)
        .add_plugin(PathTracePlugin)
        //.add_plugin(ScreenSpacePassesPlugin)
        .add_plugin(CoordinateTransformationsPlugin)
        .add_plugin(MaterialPlugin::<CustomStandardMaterial>::default())
        .add_system(swap_standard_material)
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(TemporalAntiAliasPlugin)
        .add_startup_system(load_blue_noise)
        .add_startup_system(no_empty_tlas);

    app.run();
}

//TODO don't require this
fn no_empty_tlas(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // cube
    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
            transform: Transform::from_xyz(0.0, -1.5, 0.0),
            ..default()
        })
        .insert(StaticTLAS);
    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),
            material: materials.add(Color::rgb(0.8, 0.7, 0.6).into()),
            transform: Transform::from_xyz(0.0, -1.5, 0.0),
            ..default()
        })
        .insert(DynamicTLAS);
}
