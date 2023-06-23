
// TODO find better name than common

//#define USE_VOXEL_FALLBACK

//#define RAY_MARCH
#define USE_PATH_TRACED
#define SPATIAL_REUSE
#define FILTER_SSGI
//#define APPLY_FILTER_TO_OUTPUT
//#define SSAO_FOCUS
//#define SSAO
#define RESTIR_ANTI_CLUMPING
const CLUMP_RESET_MIN = 0.15;
const CLUMP_RESET_MAX = 0.3;
#define SSR
const SSR_SAMPLES = 0u;
const FULL_SSR_SAMPLES = 3u;
const MAX_M = 512u; //current max possible 2048
const GI_HYSTERISIS = 1.0;//0.08; //0.2
const SSGI_PIX_RADIUS_DIST = 100.0;

const RESERVIOR_LAYER0 = 0u;
const RESERVIOR_LAYER1 = 1u;
const COLOR_LAYER = 2u;
const SSR_LAYER = 3u;

@group(0) @binding(4)
var blue_noise_tex: texture_2d_array<f32>;
const BLUE_NOISE_TEX_DIMS = vec3<u32>(64u, 64u, 64u);

#import "shaders/sampling.wgsl"
#import "shaders/bicubic.wgsl"
//#import "common.wgsl"

#import bevy_pbr::mesh_types
#import bevy_render::view
#import bevy_render::globals
#import bevy_pbr::utils

#import "shaders/screen_space_passes/bindings.wgsl"

#import bevy_coordinate_systems::transformations

#import bevy_pbr::prepass_utils


// needed by pbr_deferred_functions
fn calculate_view(world_position: vec4<f32>, is_orthographic: bool) -> vec3<f32> {
    if is_orthographic {
        return normalize(vec3<f32>(view.view_proj[0].z, view.view_proj[1].z, view.view_proj[2].z));
    } else {
        return normalize(view.world_position.xyz - world_position.xyz);
    }
}

#import bevy_pbr::pbr_types
#import bevy_pbr::pbr_deferred_types
#import bevy_pbr::pbr_deferred_functions

#import "shaders/contact_shadows.wgsl"
#import "shaders/depth_buffer_raymarching.wgsl"
#import "shaders/bad_ssr.wgsl"
#import "shaders/bad_ssgi.wgsl"
#import "shaders/bad_gtao.wgsl"
#import "shaders/voxel_cache.wgsl"
#import "shaders/screen_space_passes/probe.wgsl"




fn new_drm_for_restir() -> DepthRayMarch {
    var drm = DepthRayMarch_new_from_depth(view.viewport.zw);
    drm.linear_steps = 12u;
    drm.depth_thickness_linear_z = 1.5;
    drm.march_behind_surfaces = false;
    drm.use_secant = true;
    drm.bisection_steps = 4u;
    drm.use_bilinear = false;
    drm.mip_min_max = vec2(0.0, 3.0);
    return drm;
}

// set distance a bit under less than something you want to check vis to so you don't hit it
fn vis_blocked(origin: vec3<f32>, direction: vec3<f32>, max_dist: f32, jitter: f32) -> bool {
    var drm = DepthRayMarch_new_from_depth(view.viewport.zw);
    drm.linear_steps = 2u;
    drm.depth_thickness_linear_z = 1.5;
    drm.march_behind_surfaces = false;
    drm.use_secant = false;
    drm.bisection_steps = 0u;
    drm.use_bilinear = false;
    drm.mip_min_max = vec2(1.0, 3.0);
    drm.ray_start_cs = position_world_to_ndc(origin);
    var ray_end_ws = origin + direction * max_dist;
    drm = to_ws(drm, ray_end_ws);
    drm.jitter = jitter;
    let raymarch_result = march(drm, 0u);
    if (raymarch_result.hit) {
        return true;
    }
    return false;
}

struct ScreenTraceResult {
    pos: vec3<f32>,
    color: vec3<f32>,
    derate: f32,
    hit: bool,
    backface: bool,
}

fn get_f0(reflectance: f32, metallic: f32, metal_color: vec3<f32>) -> vec3<f32> {
    let F0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + metal_color * metallic;
    return F0;
}

fn perceptualRoughnessToRoughness(perceptualRoughness: f32) -> f32 {
    let clampedPerceptualRoughness = clamp(perceptualRoughness, 0.089, 1.0);
    return clampedPerceptualRoughness * clampedPerceptualRoughness;
}

fn pbr_from_frag_coord(full_screen_frag_coord: ptr<function, vec4<f32>>) -> PbrInput {
    let deferred_data = textureLoad(deferred_prepass_texture, vec2<i32>((*full_screen_frag_coord).xy), 0);
#ifdef WEBGL
    (*full_screen_frag_coord).z = unpack_unorm3x4_plus_unorm_20(deferred_data.b).w;
#else
    (*full_screen_frag_coord).z = prepass_depth((*full_screen_frag_coord), 0u);
#endif
    return pbr_input_from_deferred_gbuffer((*full_screen_frag_coord), deferred_data);
}