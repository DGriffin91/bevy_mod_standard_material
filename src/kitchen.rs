use std::f32::consts::PI;

use crate::{
    camera_controller::{CameraController, CameraControllerPlugin},
    path_trace::TraceSettings,
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
    input::mouse::MouseMotion,
    pbr::{CascadeShadowConfig, CascadeShadowConfigBuilder, DirectionalLightShadowMap},
    prelude::*,
    render::camera::TemporalJitter,
};

pub struct KitchenPlugin;
impl Plugin for KitchenPlugin {
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
        scene: asset_server.load("large_models/kitchen_gltf.gltf#Scene0"), //_no_window_cover
        ..default()
    });

    //commands.spawn(SceneBundle {
    //    scene: asset_server.load("large_models/wet_ground.gltf#Scene0"),
    //    ..default()
    //});
    //
    //commands.spawn(SceneBundle {
    //    scene: asset_server.load("large_models/spheres.gltf#Scene0"),
    //    ..default()
    //});

    // Sun
    commands.spawn(DirectionalLightBundle {
        transform: Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            (13.3f32).to_radians(),
            (180.0 - 14.2f32).to_radians(),
            0.0,
        )),
        directional_light: DirectionalLight {
            color: Color::rgb(0.95, 0.93, 0.85),
            illuminance: 120000.0,
            shadows_enabled: true,
            shadow_depth_bias: 0.10,
            shadow_normal_bias: 1.5,
        },
        //cascade_shadow_config: CascadeShadowConfigBuilder {
        //    num_cascades: 2,
        //    minimum_distance: 0.0,
        //    maximum_distance: 12.0,
        //    first_cascade_far_bound: 3.0,
        //    overlap_proportion: 0.2,
        //}
        //.into(),
        cascade_shadow_config: CascadeShadowConfig {
            /// The (positive) distance to the far boundary of each cascade.
            bounds: vec![6.0],
            /// The proportion of overlap each cascade has with the previous cascade.
            overlap_proportion: 0.5,
            /// The (positive) distance to the near boundary of the first cascade.
            minimum_distance: -6.0,
        },
        ..default()
    });

    let mut bloom_settings = BloomSettings::NATURAL;
    bloom_settings.intensity *= 0.35;
    // Camera
    commands
        .spawn((
            Camera3dBundle {
                camera: Camera {
                    hdr: true,
                    ..default()
                },
                tonemapping: Tonemapping::TonyMcMapface,
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
            DepthPrepass,
            MotionVectorPrepass,
            Fxaa::default(),
            //TemporalAntiAliasBundle::default(),
            // {
            //    settings: TemporalAntiAliasSettings { reset: true },
            //    jitter: TemporalJitter { offset: Vec2::ZERO },
            //    depth_prepass: DepthPrepass,
            //    motion_vector_prepass: MotionVectorPrepass,
            //},
        ))
        .insert(TraceSettings { frame: 0, fps: 0.0 });
}
