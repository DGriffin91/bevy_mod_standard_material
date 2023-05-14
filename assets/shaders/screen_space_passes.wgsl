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
var prev_tex: texture_2d_array<f32>;
@group(0) @binding(10)
var target_tex: texture_storage_2d_array<rgba16float, write>;


#import bevy_coordinate_systems::transformations
#import bevy_pbr::prepass_utils

#import "shaders/contact_shadows.wgsl"
#import "shaders/depth_buffer_raymarching.wgsl"
#import "shaders/bad_ssr.wgsl"
#import "shaders/bad_ssgi.wgsl"
#import "shaders/ssr_uv_generate.wgsl"

fn ssgi_store_hits(ifrag_coord: vec2<i32>, surface_normal: vec3<f32>, world_position: vec3<f32>, samples: u32) {

    var proposed_pos = textureLoad(prev_tex, ifrag_coord, 0u, 0).xyz;
    let weight_data = textureLoad(prev_tex, ifrag_coord, 1u, 0).xyz;
    var M = u32(weight_data.x);
    var w_sum = weight_data.y;
    var weight = weight_data.z;

    if M == 0u {
        proposed_pos = vec3(F32_MAX);
    }

    //if M == 1024u {
    //    return;
    //}


    let trace_dist = 16.0;

    let surface_normal = normalize(surface_normal);
    let ufrag_coord = vec2<u32>(ifrag_coord.xy);
    // TODO surface_normal * 0.01 is because of lines from using really low mip for depth
    let ray_start_ndc = position_world_to_ndc(world_position + surface_normal * 0.01);

    let TBN = build_orthonormal_basis(surface_normal);

    var dmr = new_drm_for_ssgi();
    dmr.ray_start_cs = ray_start_ndc;

    for (var i = 0u; i < samples; i += 1u) {
        let white_frame_noise = white_frame_noise(8492u + i);
        let seed = i * samples + globals.frame_count * samples;

        let urand = fract_blue_noise_for_pixel(ufrag_coord, seed, white_frame_noise);  

        var direction = cosine_sample_hemisphere(urand.xy);

        let jitter = fract(blue_noise_for_pixel(ufrag_coord, seed + 2u) + white_frame_noise.z);
        
        direction = normalize(direction * TBN);

        let ray_end_ws = world_position + direction * trace_dist;

        dmr = to_ws(dmr, ray_end_ws);
        dmr.jitter = jitter;
        var miss = false;
        let raymarch_result = march(dmr, 0u);
        if (raymarch_result.hit) {
            let hit_nd = normalize(textureSampleLevel(prepass_downsample, linear_sampler, raymarch_result.hit_uv, 1.0));
            let backface = dot(hit_nd.xyz, direction);
            if backface < 0.01 {
                let closest_motion_vector = prepass_motion_vector(vec4<f32>(raymarch_result.hit_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
                let history_uv = raymarch_result.hit_uv - closest_motion_vector;

                let color = textureSampleLevel(prev_frame_tex, linear_sampler, history_uv, 2.0).rgb;
                let prop_pos = position_ndc_to_world(vec3(uv_to_ndc(raymarch_result.hit_uv), hit_nd.w));

                let distv = prop_pos - world_position;
                let dir = normalize(distv);
                let dist = sqrt(dot(distv, distv));

                let hit_frame = history_uv.x > 0.0 && history_uv.x < 1.0 && history_uv.y > 0.0 && history_uv.y < 1.0;

                var gr = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));

                let brdf = max(dot(surface_normal, dir), 0.0);
                var new_weight = gr * (1.0 / (1.0 + dist * dist));
                new_weight *= brdf * f32(backface < 0.0) * f32(hit_frame);
                new_weight = max(new_weight, 0.0);
                w_sum += new_weight;
                
                var threshold = new_weight / w_sum;

                if M == 0u || proposed_pos.x == F32_MAX {
                    threshold = 1.0;
                }

                M += 1u;

                if threshold > urand.z {
                    proposed_pos = prop_pos;
                    weight = new_weight;
                }
            
                // best test
                //if new_weight > weight {
                //    proposed_pos = prop_pos;
                //    weight = new_weight;
                //}
            } else {
                M += 1u;
                //miss = true;
            }
        } else {
            M += 1u;
            //miss = true;
        }
        //if miss {
        //    proposed_pos = vec3(F32_MAX);
        //    M = u32(0u);
        //    w_sum = 0.0;
        //    weight = 0.0;
        //}
    }


    

    textureStore(target_tex, ifrag_coord, 0u, vec4(proposed_pos, 0.0));
    textureStore(target_tex, ifrag_coord, 1u, vec4(f32(M), w_sum, weight, 0.0));
}

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
    var screen_uv = flocation.xy / ftarget_dims + frag_size * 0.5;





    let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 1.0);
    let surface_normal = nor_depth.xyz;
    let depth = nor_depth.w;
    let frag_coord = vec4(flocation, depth, 0.0);
    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);

    let world_position = position_ndc_to_world(vec3(uv_to_ndc(screen_uv), depth));
    let F0 = get_f0(0.5, 0.0, vec3(1.0));

    ssgi_store_hits(location, surface_normal, world_position, 2u);

    
    

    let closest_motion_vector = prepass_motion_vector(vec4<f32>(screen_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
    let history_uv = screen_uv - closest_motion_vector;



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
    let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 1.0);
    
/*
    var sharp_tot = vec3(0.0);
    var mid_tot = vec3(0.0);
    var rough_tot = vec3(0.0);
    var diffuse_tot = vec3(0.0);

    var sharp_tot_w = 0.0;
    var mid_tot_w = 0.0;
    var rough_tot_w = 0.0;
    var diffuse_tot_w = 0.0;

    let dist_factor_sharp = 80.0 / depth_ndc_to_linear(nor_depth.w);
    let dist_factor_mid = 80.0 / depth_ndc_to_linear(nor_depth.w);
    let dist_factor_rough = 1.0 / depth_ndc_to_linear(nor_depth.w);
    let dist_factor_diffuse = 1.0 / depth_ndc_to_linear(nor_depth.w);

    let world_position = position_ndc_to_world(vec3(uv_to_ndc(screen_uv), nor_depth.w));

    let n = 4;
    for (var x = -n; x < n; x+=1) {
        for (var y = -n; y < n; y+=1) {
            let uv = screen_uv + vec2<f32>(vec2(x, y)) / fsize;
            let px_dist = saturate(1.0 - distance(vec2(0.0, 0.0), vec2(f32(x), f32(y))) / f32(n + 1));
            var w = px_dist;
            let nd = textureSampleLevel(prepass_downsample, linear_sampler, uv, 1.0);

            let d = max(dot(nor_depth.xyz, nd.xyz) + 0.001, 0.0);
            w = w * d * d * d * d * d; //lol
            let px_ws = position_ndc_to_world(vec3(uv_to_ndc(uv), nor_depth.w));
            let dist = distance(px_ws, world_position);
            var w_sharp = w * (1.0 - clamp(dist * dist_factor_sharp, 0.0, 1.0));
            var w_mid = w * (1.0 - clamp(dist * dist_factor_mid, 0.0, 1.0));
            var w_rough = w * (1.0 - clamp(dist * dist_factor_rough, 0.0, 1.0));
            var w_diffuse = w * (1.0 - clamp(dist * dist_factor_diffuse, 0.0, 1.0));
            if x == 0 && y == 0 {
                w_sharp = 1.0;
                w_mid = 1.0;
                w_rough = 1.0;
                w_diffuse = 1.0;
            }
            sharp_tot += textureSampleLevel(prev_tex, linear_sampler, uv, 0, 1.0).rgb * w_sharp;
            mid_tot += textureSampleLevel(prev_tex, linear_sampler, uv, 1, 1.0).rgb * w_mid;
            rough_tot += textureSampleLevel(prev_tex, linear_sampler, uv, 2, 1.0).rgb * w_rough;
            diffuse_tot += textureSampleLevel(prev_tex, linear_sampler, uv, 3, 1.0).rgb * w_diffuse;
            sharp_tot_w += w_sharp;
            mid_tot_w += w_mid;
            rough_tot_w += w_rough;
            diffuse_tot_w += w_diffuse;
        }
    }
    sharp_tot /= max(sharp_tot_w, 1.0);
    mid_tot /= max(mid_tot_w, 1.0);
    rough_tot /= max(rough_tot_w, 1.0);
    diffuse_tot /= max(diffuse_tot_w, 1.0);
    
    textureStore(target_tex, location, 0, vec4(sharp_tot, 0.0));
    textureStore(target_tex, location, 1, vec4(mid_tot, 0.0));
    textureStore(target_tex, location, 2, vec4(rough_tot, 0.0));
    textureStore(target_tex, location, 3, vec4(diffuse_tot, 0.0));

*/
    textureStore(target_tex, location, 0, textureLoad(prev_tex, location, 0, 0));
    textureStore(target_tex, location, 1, textureLoad(prev_tex, location, 1, 0));
    textureStore(target_tex, location, 2, textureLoad(prev_tex, location, 2, 0));
    textureStore(target_tex, location, 3, textureLoad(prev_tex, location, 3, 0));
}
