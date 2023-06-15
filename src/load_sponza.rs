use std::f32::consts::PI;

use crate::{
    camera_controller::{CameraController, CameraControllerPlugin},
    copy_frame::CopyFrame,
    path_trace::{PathTrace, TraceSettings},
    prepass_downsample::PrepassDownsample,
    screen_space_passes::ScreenSpacePasses,
    voxel_pass::VoxelPass,
};
use bevy::{
    core_pipeline::{
        bloom::BloomSettings, experimental::taa::TemporalAntiAliasBundle, fxaa::Fxaa,
        prepass::NormalPrepass, tonemapping::Tonemapping,
    },
    pbr::CascadeShadowConfigBuilder,
    prelude::*,
};

pub struct SponzaPlugin;
impl Plugin for SponzaPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClearColor(Color::rgb(1.75, 1.9, 1.99)))
            .insert_resource(AmbientLight {
                color: Color::rgb(1.0, 1.0, 1.0),
                //brightness: 0.02,
                brightness: 0.0,
            })
            .add_plugin(CameraControllerPlugin)
            .add_systems(Startup, setup)
            .add_systems(Update, proc_scene);
    }
}

#[derive(Component)]
pub struct PostProcScene;

#[derive(Component)]
pub struct GrifLight;

pub fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    println!("Loading models, generating mipmaps");

    // sponza
    commands
        .spawn(SceneBundle {
            scene: asset_server.load("main_sponza/NewSponza_Main_glTF_002.gltf#Scene0"),
            ..default()
        })
        .insert(PostProcScene);

    // curtains
    commands
        .spawn(SceneBundle {
            scene: asset_server.load("PKG_A_Curtains/NewSponza_Curtains_glTF.gltf#Scene0"),
            ..default()
        })
        .insert(PostProcScene);

    commands.spawn(SceneBundle {
        scene: asset_server.load("models/FlightHelmet/FlightHelmet.gltf#Scene0"),
        transform: Transform::from_xyz(0.0, 0.0, -2.0),
        ..default()
    });

    // Sun
    commands
        .spawn(DirectionalLightBundle {
            transform: Transform::from_rotation(Quat::from_euler(
                EulerRot::XYZ,
                PI * -0.43,
                PI * -0.08,
                0.0,
            )),
            directional_light: DirectionalLight {
                color: Color::rgb(1.0, 1.0, 0.99),
                illuminance: 400000.0,
                shadows_enabled: true,
                shadow_depth_bias: 0.2,
                shadow_normal_bias: 0.2,
            },
            cascade_shadow_config: CascadeShadowConfigBuilder {
                num_cascades: 4,
                minimum_distance: 0.1,
                maximum_distance: 100.0,
                first_cascade_far_bound: 5.0,
                overlap_proportion: 0.2,
            }
            .into(),
            ..default()
        })
        .insert(GrifLight);
    /*
    // Sun Refl
    commands
        .spawn(SpotLightBundle {
            transform: Transform::from_xyz(2.0, -0.0, -2.0)
                .looking_at(Vec3::new(0.0, 999.0, 0.0), Vec3::X),
            spot_light: SpotLight {
                range: 15.0,
                intensity: 1000.0,
                color: Color::rgb(1.0, 0.97, 0.85),
                shadows_enabled: false,
                inner_angle: PI * 0.4,
                outer_angle: PI * 0.5,
                ..default()
            },
            ..default()
        })
        .insert(GrifLight);

    // Sun refl 2nd bounce / misc bounces
    commands
        .spawn(SpotLightBundle {
            transform: Transform::from_xyz(2.0, 5.5, -2.0)
                .looking_at(Vec3::new(0.0, -999.0, 0.0), Vec3::X),
            spot_light: SpotLight {
                range: 13.0,
                intensity: 800.0,
                color: Color::rgb(1.0, 0.97, 0.85),
                shadows_enabled: false,
                inner_angle: PI * 0.3,
                outer_angle: PI * 0.4,
                ..default()
            },
            ..default()
        })
        .insert(GrifLight);


    // sky
    // seems to be making blocky artifacts. Even if it's the only light.
    commands
        .spawn(PointLightBundle {
            point_light: PointLight {
                color: Color::rgb(0.8, 0.9, 0.97),
                intensity: 100000.0,
                shadows_enabled: false,
                range: 24.0,
                radius: 3.0,
                ..default()
            },
            transform: Transform::from_xyz(0.0, 30.0, 0.0),
            ..default()
        })
        .insert(GrifLight);

    // sky refl
    commands
        .spawn(SpotLightBundle {
            transform: Transform::from_xyz(0.0, -2.0, 0.0)
                .looking_at(Vec3::new(0.0, 999.0, 0.0), Vec3::X),
            spot_light: SpotLight {
                range: 11.0,
                intensity: 300.0,
                color: Color::rgb(0.8, 0.9, 0.97),
                shadows_enabled: false,
                inner_angle: PI * 0.46,
                outer_angle: PI * 0.49,
                ..default()
            },
            ..default()
        })
        .insert(GrifLight);

    // sky low
    commands
        .spawn(SpotLightBundle {
            transform: Transform::from_xyz(3.0, 2.0, 0.0)
                .looking_at(Vec3::new(0.0, -999.0, 0.0), Vec3::X),
            spot_light: SpotLight {
                range: 12.0,
                radius: 0.0,
                intensity: 1800.0,
                color: Color::rgb(0.8, 0.9, 0.95),
                shadows_enabled: false,
                inner_angle: PI * 0.34,
                outer_angle: PI * 0.5,
                ..default()
            },
            ..default()
        })
        .insert(GrifLight);

    */
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
                transform: Transform::from_xyz(-10.5, 1.7, -1.0)
                    .looking_at(Vec3::new(0.0, 3.5, 0.0), Vec3::Y),
                projection: Projection::Perspective(PerspectiveProjection {
                    fov: std::f32::consts::PI / 3.0,
                    near: 0.1,
                    far: 1000.0,
                    aspect_ratio: 1.0,
                }),
                ..default()
            },
            NormalPrepass,
            //DepthPrepass,
            //MotionVectorPrepass,
            TemporalAntiAliasBundle::default(),
            bloom_settings,
            CameraController {
                walk_speed: 1.0,
                ..default()
            },
            Fxaa::default(),
            CopyFrame,
            PrepassDownsample,
            VoxelPass,
            ScreenSpacePasses,
            PathTrace,
        ))
        .insert(TraceSettings { frame: 0, fps: 0.0 });
}

pub fn all_children<F: FnMut(Entity)>(
    children: &Children,
    children_query: &Query<&Children>,
    closure: &mut F,
) {
    for child in children {
        if let Ok(children) = children_query.get(*child) {
            all_children(children, children_query, closure);
        }
        closure(*child);
    }
}

#[allow(clippy::type_complexity)]
pub fn proc_scene(
    mut commands: Commands,
    flip_normals_query: Query<Entity, With<PostProcScene>>,
    children_query: Query<&Children>,
    has_std_mat: Query<&Handle<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    lights: Query<
        Entity,
        (
            Or<(With<PointLight>, With<DirectionalLight>, With<SpotLight>)>,
            Without<GrifLight>,
        ),
    >,
    cameras: Query<Entity, With<Camera>>,
) {
    for entity in flip_normals_query.iter() {
        if let Ok(children) = children_query.get(entity) {
            all_children(children, &children_query, &mut |entity| {
                // Sponza needs flipped normals
                if let Ok(mat_h) = has_std_mat.get(entity) {
                    if let Some(mat) = materials.get_mut(mat_h) {
                        mat.flip_normal_map_y = true;
                    }
                }

                // Sponza has a bunch of lights by default
                if lights.get(entity).is_ok() {
                    commands.entity(entity).despawn_recursive();
                }

                // Sponza has a bunch of cameras by default
                if cameras.get(entity).is_ok() {
                    commands.entity(entity).despawn_recursive();
                }
            });
            commands.entity(entity).remove::<PostProcScene>();
        }
    }
}
