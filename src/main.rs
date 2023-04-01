//! Loads and renders a glTF file as a scene.

mod pbr_material;

use std::f32::consts::*;

use bevy::{
    pbr::{CascadeShadowConfigBuilder, DirectionalLightShadowMap},
    prelude::*,
};
use pbr_material::CustomStandardMaterial;

fn main() {
    App::new()
        .insert_resource(AmbientLight {
            color: Color::WHITE,
            brightness: 1.0 / 5.0f32,
        })
        .insert_resource(DirectionalLightShadowMap { size: 4096 })
        .add_plugins(DefaultPlugins.set(AssetPlugin {
            watch_for_changes: true,
            ..default()
        }))
        .add_plugin(MaterialPlugin::<CustomStandardMaterial>::default())
        .add_startup_system(setup)
        .add_system(animate_light_direction)
        .add_system(swap_standard_material)
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
            ..default()
        },
        ..default()
    });
    commands.spawn(SceneBundle {
        scene: asset_server.load("models/FlightHelmet/FlightHelmet.gltf#Scene0"),
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
            -FRAC_PI_4,
        );
    }
}

fn swap_standard_material(
    mut commands: Commands,
    mut material_events: EventReader<AssetEvent<StandardMaterial>>,
    entites: Query<(Entity, &Handle<StandardMaterial>)>,
    standard_materials: Res<Assets<StandardMaterial>>,
    mut custom_materials: ResMut<Assets<CustomStandardMaterial>>,
) {
    for event in material_events.iter() {
        let handle = match event {
            AssetEvent::Created { handle } => handle,
            _ => continue,
        };
        if let Some(material) = standard_materials.get(handle) {
            let custom_mat_h = custom_materials.add(CustomStandardMaterial {
                base_color: material.base_color,
                base_color_texture: material.base_color_texture.clone(),
                emissive: material.emissive,
                emissive_texture: material.emissive_texture.clone(),
                perceptual_roughness: material.perceptual_roughness,
                metallic: material.metallic,
                metallic_roughness_texture: material.metallic_roughness_texture.clone(),
                reflectance: material.reflectance,
                normal_map_texture: material.normal_map_texture.clone(),
                flip_normal_map_y: material.flip_normal_map_y,
                occlusion_texture: material.occlusion_texture.clone(),
                double_sided: material.double_sided,
                cull_mode: material.cull_mode,
                unlit: material.unlit,
                fog_enabled: material.fog_enabled,
                alpha_mode: material.alpha_mode,
                depth_bias: material.depth_bias,
            });
            for (entity, entity_mat_h) in entites.iter() {
                if entity_mat_h == handle {
                    let mut ecmds = commands.entity(entity);
                    ecmds.remove::<Handle<StandardMaterial>>();
                    ecmds.insert(custom_mat_h.clone());
                }
            }
        }
    }
}
