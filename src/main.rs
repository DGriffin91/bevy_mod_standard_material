//! Loads and renders a glTF file as a scene.

mod camera_controller;
mod load_sponza;
mod pbr_material;

use std::f32::consts::*;

use bevy::{
    core_pipeline::prepass::{DepthPrepass, NormalPrepass},
    math::vec3,
    pbr::{CascadeShadowConfigBuilder, DirectionalLightShadowMap},
    prelude::*,
};
use load_sponza::SponzaPlugin;
use pbr_material::{swap_standard_material, CustomStandardMaterial};

fn main() {
    App::new()
        .add_plugin(SponzaPlugin)
        //.insert_resource(AmbientLight {
        //    color: Color::WHITE,
        //    brightness: 1.0 / 5.0f32,
        //})
        .insert_resource(Msaa::Off)
        // 2048 is default
        //.insert_resource(DirectionalLightShadowMap { size: 2048 })
        .add_plugins(DefaultPlugins.set(AssetPlugin {
            watch_for_changes: true,
            ..default()
        }))
        .add_plugin(MaterialPlugin::<CustomStandardMaterial>::default())
        .add_system(swap_standard_material)
        //.add_startup_system(setup)
        //.add_system(animate_light_direction)
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.7, 0.7, 1.0)
                .looking_at(Vec3::new(0.0, 0.3, 0.0), Vec3::Y),
            ..default()
        },
        EnvironmentMapLight {
            diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
            specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
        },
        DepthPrepass,
        NormalPrepass,
    ));
    // plane
    commands.spawn(PbrBundle {
        mesh: meshes.add(shape::Plane::from_size(10.0).into()),
        material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
        ..default()
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            shadows_enabled: true,
            shadow_depth_bias: 0.02,
            shadow_normal_bias: 1.0,
            ..default()
        },
        cascade_shadow_config: CascadeShadowConfigBuilder::default().into(),
        transform: Transform::IDENTITY.looking_at(vec3(1.0, -1.0, 0.5), vec3(0.0, 1.0, 0.0)),
        ..default()
    });
    commands.spawn(SceneBundle {
        scene: asset_server.load("models/FlightHelmet/FlightHelmet.gltf#Scene0"),
        transform: Transform::from_xyz(-1.0, 0.0, 0.0),
        ..default()
    });
}

fn animate_light_direction(
    time: Res<Time>,
    mut query: Query<&mut Transform, With<DirectionalLight>>,
) {
    for mut transform in &mut query {
        transform.rotation = Quat::from_euler(
            EulerRot::ZYX,
            0.0,
            time.elapsed_seconds() * PI / 5.0,
            -FRAC_PI_4 * 0.5,
        );
    }
}
