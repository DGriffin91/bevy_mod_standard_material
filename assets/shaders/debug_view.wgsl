#import bevy_pbr::utils
#import bevy_core_pipeline::fullscreen_vertex_shader

#import bevy_pbr::mesh_types
#import bevy_render::view
#import bevy_render::globals
#import bevy_coordinate_systems::transformations

@group(0) @binding(0)
var<uniform> view: View;
@group(0) @binding(1)
var<uniform> globals: Globals;
@group(0) @binding(2)
var linear_sampler: sampler;
@group(0) @binding(3)
var prepass_downsample: texture_2d<f32>;
@group(0) @binding(4)
var post_process_src: texture_2d<f32>;
@group(0) @binding(5)
var prev_frame_tex: texture_2d<f32>;
@group(0) @binding(6)
var screen_passes_target: texture_2d_array<f32>;
@group(0) @binding(7)
var screen_passes_processed: texture_2d_array<f32>;
@group(0) @binding(8)
var voxel_cache: texture_3d<f32>;
@group(0) @binding(9)
var motion_vector_prepass_texture: texture_2d<f32>;
@group(0) @binding(10)
var depth_prepass_texture: texture_depth_2d;
@group(0) @binding(11)
var normal_prepass_texture: texture_2d<f32>;

#import "shaders/voxel_cache.wgsl"


@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let frag_coord = in.position;
    let frame_col = textureSampleLevel(post_process_src, linear_sampler, in.uv, 0.0);
    let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, in.uv, 0.0);
    let world_position = position_ndc_to_world(vec3(uv_to_ndc(in.uv), nor_depth.w));
    let V = normalize(world_position - view.world_position.xyz);

    let last_world_cache = textureLoad(voxel_cache, vec3<i32>(position_world_to_fvoxel(world_position)), 0);

    let hit = march_voxel_grid(view.world_position.xyz, V, 512u, 1u, 1000.0);

    var col = hit.color;

    if hit.t < 0.0 {
        col = frame_col.rgb;
    }

    //return frame_col;
    //return vec4(vec3(last_world_cache.rgb), 1.0);
    return vec4(vec3(frame_col.rgb), 1.0); //hit.color * 
}