// TODO need to at least be able to feedback

@group(0) @binding(4)
var blue_noise_tex: texture_2d_array<f32>;
const BLUE_NOISE_TEX_DIMS = vec3<u32>(64u, 64u, 64u);

#import "shaders/sampling.wgsl"
#import "shaders/pathtrace/printing.wgsl"
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
var prev_tex: texture_2d<f32>;
@group(0) @binding(10)
var target_tex: texture_storage_2d<rgba16float, write>;


#import bevy_coordinate_systems::transformations
#import bevy_pbr::prepass_utils

#import "shaders/contact_shadows.wgsl"
#import "shaders/depth_buffer_raymarching.wgsl"
#import "shaders/bad_ssr.wgsl"


fn get_f0(reflectance: f32, metallic: f32, metal_color: vec3<f32>) -> vec3<f32> {
    let F0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + metal_color * metallic;
    return F0;
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {

    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let flocation = vec2<f32>(location);

    

    let target_dims = textureDimensions(target_tex).xy;
    let depth_tex_dims = vec2<f32>(textureDimensions(prepass_downsample).xy);
    
    let ftarget_dims = vec2<f32>(target_dims);
    let frag_size = 1.0 / ftarget_dims;
    var screen_uv = flocation.xy / ftarget_dims;





    let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 0.0);
    let surface_normal = nor_depth.xyz;
    let depth = nor_depth.w;
    let frag_coord = vec4(flocation, depth, 0.0);
    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);

    let world_position = position_ndc_to_world(vec3(uv_to_ndc(screen_uv), depth));
    
    let roughness = 0.1; //TODO this whole thing probably needs to work differently, return uvs to same or something, not color
    let F0 = get_f0(0.5, 0.0, vec3(0.0));
    var col = bad_ssr(screen_uv, location, surface_normal, world_position, roughness, F0, 0u).rgb;


    let closest_motion_vector = prepass_motion_vector(vec4<f32>(screen_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
    let history_uv = screen_uv - closest_motion_vector;
    let last_image = max(textureSampleLevel(prev_tex, linear_sampler, history_uv + frag_size * 0.5, 0.0).xyz, vec3(0.0));
    
    textureStore(target_tex, location, vec4(mix(last_image, col, 0.1), depth));
}

@compute @workgroup_size(8, 8, 1)
fn blur(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let flocation = vec2<f32>(location);
    let fprepass_size = vec2<f32>(textureDimensions(prepass_downsample).xy);
    let size = vec2<i32>(textureDimensions(target_tex).xy);
    let fsize = vec2<f32>(size);
    let frag_size = 1.0 / fsize;
    let screen_uv = flocation / fsize + frag_size * 0.5; // TODO verify
    let v = textureLoad(prev_tex, location, 0);
    let v2 = textureSampleLevel(prev_tex, linear_sampler, screen_uv, 0.0);
    let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 2.0);
    

    var tot = vec3(0.0);
    var tot_w = 0.0;

    let dist_factor = 20.0;

    let world_position = position_ndc_to_world(vec3(uv_to_ndc(screen_uv), v.w));

    let n = 6;
    for (var x = -n; x < n; x+=1) {
        for (var y = -n; y < n; y+=1) {
            let uv = vec2<f32>(location + vec2(x, y)) / fsize;
            let px_dist = saturate(1.0 - distance(vec2(0.0, 0.0), vec2(f32(x), f32(y))) / f32(n + 1));
            var w = px_dist;
            let nd = textureLoad(prepass_downsample, vec2<i32>(uv * fprepass_size), 0);
            let col = textureLoad(prev_tex, location + vec2(x, y), 0);

            let d = max(dot(nor_depth.xyz, nd.xyz) + 0.001, 0.0);
            w = w * d * d * d * d * d; //lol
            let px_ws = position_ndc_to_world(vec3(uv_to_ndc(uv), col.w));
            let dist = distance(px_ws, world_position);
            w = w * (1.0 - clamp(dist * dist_factor, 0.0, 1.0));
            //if distance(col.xyz, v.xyz) < 0.7 {
                tot += col.rgb * w;
                tot_w += w;
            //}
        }
    }
    tot /= max(tot_w, 1.0);
    
    textureStore(target_tex, location, vec4(tot, v.w));
}
