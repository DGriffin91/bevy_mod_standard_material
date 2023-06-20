@group(0) @binding(12)
var blue_noise_tex: texture_2d_array<f32>;
const BLUE_NOISE_TEX_DIMS = vec3<u32>(64u, 64u, 64u);

#import "shaders/sampling.wgsl"
#import "shaders/pathtrace/trace_gpu_types.wgsl"

#import bevy_pbr::mesh_types
#import bevy_render::view
#import bevy_render::globals
#import bevy_pbr::utils

struct MaterialData {
    color: vec3<f32>,
    perceptual_roughness: f32,
    metallic: f32,
    reflectance: f32,
}

@group(0) @binding(0)
var<uniform> view: View;
@group(0) @binding(1)
var<uniform> globals: Globals;
@group(0) @binding(2)
var prev_frame_tex: texture_2d<f32>;
@group(0) @binding(3)
var linear_sampler: sampler;
struct TraceSettings {
    frame: u32,
    fps: f32,
}
@group(0) @binding(4)
var<uniform> settings: TraceSettings;
@group(0) @binding(5)
var<storage> vertex_buffer: array<VertexData>;
@group(0) @binding(6)
var<storage> index_buffer: array<VertexIndices>;
@group(0) @binding(7)
var<storage> blas_buffer: array<BVHData>;
@group(0) @binding(8)
var<storage> static_tlas_buffer: array<BVHData>;
@group(0) @binding(9)
var<storage> dynamic_tlas_buffer: array<BVHData>;
@group(0) @binding(10)
var<storage> static_mesh_instance_buffer: array<InstanceData>;
@group(0) @binding(11)
var<storage> dynamic_mesh_instance_buffer: array<InstanceData>;
@group(0) @binding(13)
var<storage> static_material_instance_buffer: array<MaterialData>;
@group(0) @binding(14)
var<storage> dynamic_material_instance_buffer: array<MaterialData>;

@group(0) @binding(15)
var depth_prepass_texture: texture_depth_2d;
@group(0) @binding(16)
var normal_prepass_texture: texture_2d<f32>;
@group(0) @binding(17)
var motion_vector_prepass_texture: texture_2d<f32>;
@group(0) @binding(18)
var prev_tex: texture_2d_array<f32>;
@group(0) @binding(19)
var target_tex: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(20)
var prepass_downsample: texture_2d<f32>;

@group(0) @binding(21)
var screen_passes_processed: texture_2d_array<f32>;
@group(0) @binding(22)
var screen_passes_target: texture_storage_2d_array<rgba16float, write>;

@group(0) @binding(23)
var voxel_cache: texture_3d<f32>;
@group(0) @binding(24)
var voxel_cache_write: texture_storage_3d<rgba32float, write>;

#import "shaders/voxel_cache.wgsl"


#import bevy_coordinate_systems::transformations
#import bevy_pbr::prepass_utils
#import "shaders/pathtrace/traverse_tlas.wgsl"
#import "shaders/pathtrace/tracing.wgsl"

#import "shaders/contact_shadows.wgsl"
#import "shaders/depth_buffer_raymarching.wgsl"


fn get_hit_material(query: SceneQuery) -> MaterialData {
    if query.static_tlas {
        return static_material_instance_buffer[query.hit.instance_idx];
    } else {
        return dynamic_material_instance_buffer[query.hit.instance_idx];
    }
}

fn material_get_f0(material: MaterialData) -> vec3<f32> {
    let F0 = 0.16 * material.reflectance * material.reflectance * (1.0 - material.metallic) + material.color * material.metallic;
    return F0;
}

fn perceptualRoughnessToRoughness(perceptualRoughness: f32) -> f32 {
    let clampedPerceptualRoughness = clamp(perceptualRoughness, 0.089, 1.0);
    return clampedPerceptualRoughness * clampedPerceptualRoughness;
}

// -1.0 is a miss
fn get_screen_color_from_pos(hit_pos: vec3<f32>, ray_direction: vec3<f32>) -> vec3<f32> {
    let hit_pos_ndc = position_world_to_ndc(hit_pos);
    let hit_uv = ndc_to_uv(hit_pos_ndc.xy);
    // check if we hit screen, if so use that
    if distance(hit_pos_ndc, clamp(hit_pos_ndc, vec3(-1.0), vec3(1.0))) < 0.01 {
        let screen_depth = prepass_depth(vec4<f32>(hit_uv * view.viewport.zw, 0.0, 0.0), 0u);
        if distance(hit_pos_ndc.z, screen_depth) < 0.01 {
            let hit_n = normalize(prepass_normal(vec4<f32>(hit_uv * view.viewport.zw, 0.0, 0.0), 0u));
            let backface = dot(hit_n, ray_direction);
            if backface < 0.01 {
                let closest_motion_vector = prepass_motion_vector(vec4<f32>(hit_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
                let history_uv = hit_uv - closest_motion_vector;
                if history_uv.x > 0.0 && history_uv.x < 1.0 && history_uv.y > 0.0 && history_uv.y < 1.0 {
                    return textureSampleLevel(prev_frame_tex, linear_sampler, history_uv, 0.0).rgb;
                }
            }
        }
    }
    return vec3(-1.0);
}