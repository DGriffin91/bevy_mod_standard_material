use std::f32::consts::{FRAC_PI_4, PI};

use crate::camera_controller::{CameraController, CameraControllerPlugin};
use bevy::{
    core_pipeline::{
        fxaa::Fxaa,
        prepass::{DepthPrepass, NormalPrepass},
    },
    math::vec3,
    pbr::CascadeShadowConfigBuilder,
    prelude::*,
};

pub struct HelmetScenePlugin;
impl Plugin for HelmetScenePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(CameraControllerPlugin)
            //.add_system(animate_light_direction)
            .add_startup_system(setup);
    }
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
        Fxaa::default(),
        CameraController {
            orbit_focus: vec3(0.0, 0.3, 0.0),
            orbit_mode: true,
            walk_speed: 1.0,
            ..default()
        },
    ));
    // plane
    commands.spawn(PbrBundle {
        mesh: meshes.add(shape::Plane::from_size(10.0).into()),
        material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
        ..default()
    });
    let a = vec3(1.0, -1.0, 0.5);
    let _b = vec3(0.21899201, -0.38268343, 0.89754987);
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            shadows_enabled: true,
            shadow_depth_bias: 0.02,
            shadow_normal_bias: 1.0,
            ..default()
        },
        cascade_shadow_config: CascadeShadowConfigBuilder::default().into(),
        transform: Transform::IDENTITY.looking_at(a, vec3(0.0, 1.0, 0.0)),
        ..default()
    });

    commands.spawn(SceneBundle {
        scene: asset_server.load("models/FlightHelmet/FlightHelmet.gltf#Scene0"),
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        ..default()
    });
    commands.spawn(SceneBundle {
        scene: asset_server.load("large_models/example_trees.glb#Scene0"),
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
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
        //dbg!(transform.forward());
    }
}
