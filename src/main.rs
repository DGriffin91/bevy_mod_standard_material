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
        .add_plugins((
            DefaultPlugins,
            MaterialPlugin::<CustomStandardMaterial>::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (animate_light_direction, swap_standard_material))
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.7, 0.7, 1.0)
                .looking_at(Vec3::new(0.0, 0.3, 0.0), Vec3::Y),
            ..default()
        },
        EnvironmentMapLight {
            diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
            specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            intensity: 1000.0,
        },
    ));

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        // This is a relatively small scene, so use tighter shadow
        // cascade bounds than the default for better quality.
        // We also adjusted the shadow map to be larger since we're
        // only using a single cascade.
        cascade_shadow_config: CascadeShadowConfigBuilder {
            num_cascades: 1,
            maximum_distance: 1.6,
            ..default()
        }
        .into(),
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
    for event in material_events.read() {
        let handle = match event {
            AssetEvent::Added { id } => id,
            AssetEvent::LoadedWithDependencies { id } => id,
            _ => continue,
        };
        if let Some(material) = standard_materials.get(*handle) {
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
                depth_map: material.depth_map.clone(),
                parallax_depth_scale: material.parallax_depth_scale,
                parallax_mapping_method: material.parallax_mapping_method,
                max_parallax_layer_count: material.max_parallax_layer_count,
                diffuse_transmission: material.diffuse_transmission,
                specular_transmission: material.specular_transmission,
                thickness: material.thickness,
                ior: material.ior,
                attenuation_distance: material.attenuation_distance,
                attenuation_color: material.attenuation_color,
                opaque_render_method: material.opaque_render_method,
                deferred_lighting_pass_id: material.deferred_lighting_pass_id,
                lightmap_exposure: material.lightmap_exposure,
            });
            for (entity, entity_mat_h) in entites.iter() {
                if entity_mat_h.id() == *handle {
                    let mut ecmds = commands.entity(entity);
                    ecmds.remove::<Handle<StandardMaterial>>();
                    ecmds.insert(custom_mat_h.clone());
                }
            }
        }
    }
}
