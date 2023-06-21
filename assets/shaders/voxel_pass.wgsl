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

@group(0) @binding(0)
var<uniform> view: View;
@group(0) @binding(1)
var<uniform> globals: Globals;
@group(0) @binding(2)
var prev_frame_tex: texture_2d<f32>;
@group(0) @binding(3)
var linear_sampler: sampler;
@group(0) @binding(5)
var depth_prepass_texture: texture_depth_2d;
@group(0) @binding(6)
var normal_prepass_texture: texture_2d<f32>;
@group(0) @binding(7)
var motion_vector_prepass_texture: texture_2d<f32>;
@group(0) @binding(8)
var prepass_downsample: texture_2d<f32>;
@group(0) @binding(9)
var screen_passes_processed: texture_2d_array<f32>;
@group(0) @binding(10)
var voxel_cache: texture_3d<f32>;
@group(0) @binding(11)
var voxel_cache_write: texture_storage_3d<rgba32float, write>;

/*
This is a voxel rate shader that checks if the fragment depth is intersecting with a voxel
    if so it writes the color to the voxel, 
    if the voxel is further, it doesn't update it,
    if it's closer, it removes it
    all off screen voxels locations are updated as the camera moves
*/

#import bevy_coordinate_systems::transformations
#import bevy_pbr::prepass_utils

#import "shaders/contact_shadows.wgsl"
#import "shaders/depth_buffer_raymarching.wgsl"
#import "shaders/bad_ssr.wgsl"
#import "shaders/bad_ssgi.wgsl"

#define VOXEL_3D
#import "shaders/voxel_cache.wgsl"

#define PREVIEW_MODE

// Morton order of 3D space in GLSL
// https://gist.github.com/franjaviersans/885c136932ef37d8905a6433d0828be6
// https://en.wikipedia.org/wiki/Z-order_curve
fn part1by2(n: u32) -> u32 {
	var n: u32 = n & 0x000003ffu;
	n = (n ^ (n << 16u)) & 0xff0000ffu;
	n = (n ^ (n << 8u)) & 0x0300f00fu;
	n = (n ^ (n << 4u)) & 0x030c30c3u;
	n = (n ^ (n << 2u)) & 0x09249249u;
	return n;
}

fn unpart1by2(n: u32) -> u32 {
	var n: u32 = n & 0x09249249u;
	n = (n ^ (n >> 2u)) & 0x030c30c3u;
	n = (n ^ (n >> 4u)) & 0x0300f00fu;
	n = (n ^ (n >> 8u)) & 0xff0000ffu;
	n = (n ^ (n >> 16u)) & 0x000003ffu;
	return n;
}

fn interleave3(v: vec3<i32>) -> u32 {
	return part1by2(u32(v.x)) | (part1by2(u32(v.y)) << 1u) | (part1by2(u32(v.z)) << 2u);
}

fn deinterleave3(n: u32) -> vec3<i32> {
	return vec3(
        i32(unpart1by2(n)),
        i32(unpart1by2(n >> 1u)),
        i32(unpart1by2(n >> 2u))
    );
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    //if true {return;}

    // something something cache locality
    //let looking_at = position_view_to_world(vec3(0.0, 0.0, -1.0));
    //let looking_at_dir = normalize(looking_at - view.world_position.xyz);
    //let invocation_id = select(vec3(u32(VOXEL_GRID_SIZE)) - invocation_id, invocation_id, looking_at_dir < 0.0);

    let location = vec3<i32>(i32(invocation_id.x), i32(invocation_id.y), i32(invocation_id.z));

    //let location = deinterleave3(
    //    invocation_id.x +
    //    invocation_id.y * 96u +
    //    invocation_id.z * 96u * 96u
    //);



    let prev_view_world_position = textureLoad(voxel_cache, vec3(0), 0).xyz;

    let voxel_delta = vec3<i32>(view.world_position.xyz / VOXEL_SIZE) - vec3<i32>(prev_view_world_position / VOXEL_SIZE);
    let prev_voxel = textureLoad(voxel_cache, location + voxel_delta, 0);

    var updated = true;
#ifndef PREVIEW_MODE
    let prepass_downsample_dims = vec2<f32>(textureDimensions(prepass_downsample).xy);
    let vox_ws = position_voxel_to_world(location);
    let vox_ws_center = vox_ws + VOXEL_SIZE * 0.5;

    let dist_to_vox = distance(vox_ws_center, view.world_position.xyz);

    let V = normalize(vox_ws_center - view.world_position.xyz);

    let vox_vs_center = position_world_to_view(vox_ws_center);

    var voxel_ndc = position_world_to_ndc(vox_ws_center); 
    let n = 2;
    if voxel_ndc.z > 0.0 && all(abs(voxel_ndc) <= 1.0) {
        var closest = F32_MAX;
        var closest_nor_depth = vec4(0.0);
        var closer_than_screen = false;
        var color = vec3(0.0);
        var closest_coord = vec2(0, 0);
        // check a few depths +/- n distance in view space, take the closest one
        for (var x = -n; x <= n; x += 1) {
            for (var y = -n; y <= n; y += 1) {
                //f32(n * 4) to focus more toward the middle to void changing what's behind
                let offset = vec3(vec2(f32(x), f32(y)) * (VOXEL_SIZE / f32(n * 4)), 0.0);
                let offset_vox_ws = position_view_to_world(vox_vs_center + offset);
                voxel_ndc = position_world_to_ndc(offset_vox_ws);
                let voxel_uv = ndc_to_uv(voxel_ndc.xy);
                let coord = vec2<i32>(voxel_uv * prepass_downsample_dims);
                let nor_depth = textureLoad(prepass_downsample, coord, 0);
                let frag_ws_pos = position_ndc_to_world(vec3(voxel_ndc.xy, nor_depth.w));
                let dist = distance(frag_ws_pos, vox_ws_center);
                if voxel_ndc.z > 0.0 && dist < closest {
                    closest_nor_depth = nor_depth;
                    closest = dist;
                    closest_coord = coord;
                }
                if dist_to_vox + VOXEL_SIZE * 2.0 < distance(view.world_position.xyz, frag_ws_pos) {
                    closer_than_screen = true;
                }
            }
        }
        if closest < VOXEL_SIZE / 1.0 {
            let closest_motion_vector = prepass_motion_vector(vec4<f32>(vec2<f32>(closest_coord), 0.0, 0.0), 0u).xy;
            let history_uv = vec2<f32>(closest_coord) / view.viewport.zw - closest_motion_vector;
            color = textureLoad(prev_frame_tex, vec2<i32>(history_uv * view.viewport.zw), 0).rgb;
            let hysteresis = 0.1;
            color = mix(prev_voxel.rgb, color.rgb, hysteresis);
            //textureStore(voxel_cache_write, vec3(location), vec4(a, 0.0, 0.0, f32(globals.time)));
            textureStore(voxel_cache_write, vec3(location), vec4(color.rgb, f32(globals.time)));
        } else if closer_than_screen {
            textureStore(voxel_cache_write, vec3(location), vec4(0.0));
        } else {
            updated = false;
        }
    } else {
        updated = false;
    }
#else
    updated = false;
#endif //PREVIEW_MODE

    // checking against vec3(0) so we don't copy the view pos
    if !updated && !all(location == vec3(0)) && !all((location + voxel_delta) == vec3(0)) {
        // for voxels out of sight, move their data relative to the view
        textureStore(voxel_cache_write, vec3(location), prev_voxel);
    }

    if all(location == vec3(0)) {
        // TODO, move view.world_position elsewhere
        textureStore(voxel_cache_write, vec3(0), vec4(view.world_position.xyz, F32_MAX));
    }
}
