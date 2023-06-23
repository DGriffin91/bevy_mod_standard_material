//! Loads and renders a glTF file as a scene.

pub mod bind_group_utils;
mod bistro;
pub mod camera_controller;
mod copy_frame;
mod debug_view;
mod deferred_lighting_pass;
mod helmet;
mod kitchen;
mod load_sponza;
mod path_trace;
mod prepass_downsample;
mod screen_space_passes;
pub mod taa;
//mod slow_frame_diag;
mod sphere;
mod voxel_pass;

use std::time::Duration;

use bevy::{
    asset::ChangeWatcher,
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    input::mouse::MouseMotion,
    pbr::{DefaultOpaqueRendererMethod, DirectionalLightShadowMap, OpaqueRendererMethod},
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        settings::{WgpuFeatures, WgpuSettings},
        RenderPlugin,
    },
    window::PresentMode,
};
use bevy_coordinate_systems::CoordinateTransformationsPlugin;
use bevy_hanabi::{BillboardModifier, HanabiPlugin, SetSizeModifier};
use bevy_mod_bvh::{DynamicTLAS, StaticTLAS};

use bistro::BistroPlugin;
use copy_frame::CopyFramePlugin;
use debug_view::DebugViewPlugin;
use deferred_lighting_pass::CustomDeferredLightingPlugin;
use kitchen::KitchenPlugin;

use bevy::pbr::deferred::BypassPBRDeferredLightingPlugin;
use load_sponza::SponzaPlugin;
use path_trace::PathTracePlugin;
use prepass_downsample::PrepassDownsamplePlugin;
use screen_space_passes::ScreenSpacePassesPlugin;
use taa::TemporalAntiAliasPlugin;
use voxel_pass::VoxelPassPlugin;

fn main() {
    let mut wgpu_settings = WgpuSettings::default();
    wgpu_settings
        .features
        .set(WgpuFeatures::VERTEX_WRITABLE_STORAGE, true);

    let mut app = App::new();
    app.insert_resource(BypassPBRDeferredLightingPlugin)
        .insert_resource(Msaa::Off)
        .insert_resource(DefaultOpaqueRendererMethod(OpaqueRendererMethod::Deferred))
        // 2048 is default
        .insert_resource(DirectionalLightShadowMap { size: 2048 })
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    watch_for_changes: ChangeWatcher::with_delay(Duration::from_millis(200)),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: PresentMode::Immediate,
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin { wgpu_settings }),
        )
        .add_plugin(CustomDeferredLightingPlugin)
        .add_plugin(HanabiPlugin)
        //-----------
        .add_plugin(BistroPlugin)
        //.add_plugin(KitchenPlugin)
        //.add_plugin(SponzaPlugin)
        //.add_plugin(HelmetScenePlugin)
        //.add_plugin(SphereScenePlugin)
        //-----------
        .add_plugin(CopyFramePlugin)
        .add_plugin(PrepassDownsamplePlugin)
        .add_plugin(VoxelPassPlugin)
        .add_plugin(PathTracePlugin)
        .add_plugin(ScreenSpacePassesPlugin)
        .add_plugin(CoordinateTransformationsPlugin)
        .add_plugin(DebugViewPlugin)
        //.add_plugin(MaterialPlugin::<CustomStandardMaterial>::default())
        //.add_systems(Update, swap_standard_material)
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(TemporalAntiAliasPlugin)
        .add_plugin(ExtractResourcePlugin::<BlueNoise>::default())
        .add_systems(Startup, load_blue_noise)
        .add_systems(Startup, no_empty_tlas)
        //.add_systems(Startup, setup_part)
        .add_systems(Update, move_directional_light);

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

fn move_directional_light(
    mut query: Query<&mut Transform, With<DirectionalLight>>,
    mut motion_evr: EventReader<MouseMotion>,
    keys: Res<Input<KeyCode>>,
) {
    if !keys.pressed(KeyCode::L) {
        return;
    }
    for mut trans in &mut query {
        let euler = trans.rotation.to_euler(EulerRot::XYZ);
        for ev in motion_evr.iter() {
            trans.rotation = Quat::from_euler(
                EulerRot::XYZ,
                (euler.0.to_degrees() + ev.delta.y).to_radians(),
                (euler.1.to_degrees() + ev.delta.x).to_radians(),
                euler.2,
            );
        }
    }
}

#[allow(dead_code)]
fn setup_part(mut commands: Commands, mut effects: ResMut<Assets<bevy_hanabi::EffectAsset>>) {
    let mut gradient = bevy_hanabi::Gradient::new();
    gradient.add_key(0.0, Vec4::new(200.0, 50.0, 5000.0, 1.0));
    gradient.add_key(1.0, Vec4::new(10.0, 1.0, 500.0, 0.0));

    let effect = effects.add(
        bevy_hanabi::EffectAsset {
            name: "emit:burst".to_string(),
            capacity: 32768,
            spawner: bevy_hanabi::Spawner::rate((20.0).into()),
            ..Default::default()
        }
        .init(bevy_hanabi::InitPositionSphereModifier {
            center: Vec3::ZERO,
            radius: 0.3,
            dimension: bevy_hanabi::ShapeDimension::Volume,
        })
        .init(bevy_hanabi::InitVelocitySphereModifier {
            center: Vec3::ZERO,
            speed: 1.0.into(),
        })
        .init(bevy_hanabi::InitLifetimeModifier {
            lifetime: 1.5.into(),
        })
        .update(bevy_hanabi::AccelModifier::constant(Vec3::new(0., 1., 0.)))
        .render(bevy_hanabi::ColorOverLifetimeModifier { gradient })
        .render(SetSizeModifier {
            size: bevy_hanabi::Value::Single(Vec2::splat(0.04)),
        })
        .render(BillboardModifier),
    );

    commands.spawn((
        Name::new("emit:random"),
        bevy_hanabi::ParticleEffectBundle {
            effect: bevy_hanabi::ParticleEffect::new(effect),
            transform: Transform::from_translation(Vec3::new(0., 0., 0.)),
            ..Default::default()
        },
    ));
}

#[derive(Resource, ExtractResource, Clone)]
pub struct BlueNoise(pub Handle<Image>);

pub fn load_blue_noise(mut commands: Commands, ass: Res<AssetServer>) {
    //commands.insert_resource(BlueNoise(ass.load("textures/blue_noise_64x64_l64_s16.png")));
    commands.insert_resource(BlueNoise(ass.load("textures/blue_noise_64x64_l64.dds")));
    //commands.insert_resource(BlueNoise(ass.load("textures/stochastic_noise.ktx2")));
}
