use std::f32::consts::PI;

use crate::{
    camera_controller::{CameraController, CameraControllerPlugin},
    pbr_material::CustomStandardMaterial,
};
use bevy::{
    core_pipeline::{
        bloom::BloomSettings,
        experimental::taa::{TemporalAntiAliasBundle, TemporalAntiAliasSettings},
        fxaa::Fxaa,
        prepass::{DepthPrepass, MotionVectorPrepass, NormalPrepass},
        tonemapping::Tonemapping,
    },
    pbr::{CascadeShadowConfigBuilder, DirectionalLightShadowMap},
    prelude::*,
    render::camera::TemporalJitter,
};

pub struct BistroPlugin;
impl Plugin for BistroPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DirectionalLightShadowMap { size: 4096 })
            .insert_resource(ClearColor(Color::rgb(1.75, 1.9, 1.99)))
            .insert_resource(AmbientLight {
                color: Color::rgb(1.0, 1.0, 1.0),
                //brightness: 0.02,
                brightness: 0.0,
            })
            .add_plugin(CameraControllerPlugin)
            .add_startup_system(setup)
            .add_system(fix_sky_brightness);
    }
}

pub fn fix_sky_brightness(
    entites: Query<(&Handle<CustomStandardMaterial>, &Parent)>,
    mut custom_materials: ResMut<Assets<CustomStandardMaterial>>,
    mut fixed: Local<bool>,
    names: Query<&Name>,
) {
    if *fixed {
        return;
    }
    for (mat_h, parent) in &entites {
        if let Ok(name) = names.get(**parent) {
            if name.to_lowercase().contains("sky") {
                if let Some(mat) = custom_materials.get_mut(mat_h) {
                    mat.emissive *= 3.0;
                    *fixed = true;
                }
            }
        }
    }
}

pub fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(SceneBundle {
        scene: asset_server.load("large_models/BistroExterior_NoMat.gltf#Scene0"),
        ..default()
    });
    commands.spawn(SceneBundle {
        scene: asset_server.load("large_models/BistroInterior_NoMat.gltf#Scene0"),
        ..default()
    });

    let mut bloom_settings = BloomSettings::NATURAL;
    bloom_settings.intensity *= 0.35;
    // Camera
    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                hdr: true,
                ..default()
            },
            tonemapping: Tonemapping::None,
            transform: Transform::from_xyz(3.4, 1.7, 1.0)
                .looking_at(Vec3::new(-0.1, 1.2, 0.2), Vec3::Y),
            projection: Projection::Perspective(PerspectiveProjection {
                fov: std::f32::consts::PI / 3.0,
                near: 0.1,
                far: 1000.0,
                aspect_ratio: 1.0,
            }),
            ..default()
        },
        bloom_settings,
        CameraController {
            walk_speed: 1.0,
            ..default()
        },
        NormalPrepass,
        //DepthPrepass,
        //MotionVectorPrepass,
        TemporalAntiAliasBundle::default(),
        // {
        //    settings: TemporalAntiAliasSettings { reset: true },
        //    jitter: TemporalJitter { offset: Vec2::ZERO },
        //    depth_prepass: DepthPrepass,
        //    motion_vector_prepass: MotionVectorPrepass,
        //},
    ));
}
