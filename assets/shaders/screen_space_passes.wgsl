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

fn new_drm_for_restir() -> DepthRayMarch {
    var dmr = DepthRayMarch_new_from_depth(view.viewport.zw);
    dmr.linear_steps = 16u;
    dmr.depth_thickness_linear_z = 1.5;
    dmr.march_behind_surfaces = false;
    dmr.use_secant = true;
    dmr.bisection_steps = 4u;
    dmr.use_bilinear = false;
    dmr.mip_min_max = vec2(0.0, 3.0);
    return dmr;
}

fn ssgi_store_hits(ifrag_coord: vec2<i32>, surface_normal: vec3<f32>, world_position: vec3<f32>, samples: u32) {
    let tex_dims = vec2<f32>(textureDimensions(prev_tex).xy);
    let frag_size = 1.0 / tex_dims;
    let screen_uv = vec2<f32>(ifrag_coord) / tex_dims;

    var world_position_offs = world_position + surface_normal * 0.001;

    var probe_pos = world_position_offs;
    var proposed_pos = vec3(F32_MAX);
    var M = 0u;
    var w_sum = 0.0;
    var weight = 0.0;
    
    let closest_motion_vector = prepass_motion_vector(vec4<f32>(screen_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
    let history_uv = screen_uv - closest_motion_vector;
    if history_uv.x > 0.0 && history_uv.x < 1.0 && history_uv.y > 0.0 && history_uv.y < 1.0 {
        var selected_offset = vec2(0, 0);
        var closest = F32_MAX;
        let first_probe_pos = textureLoad(prev_tex, vec2<i32>(tex_dims * history_uv), 2u, 0).xyz;
        if distance(first_probe_pos, world_position_offs) > 0.001 {
            for (var x = -2; x <= 2; x += 1) {
                for (var y = -2; y <= 2; y += 1) {
                    let offset = vec2(x, y);
                    let test_probe_pos = textureLoad(prev_tex, vec2<i32>(tex_dims * history_uv) + offset, 2u, 0).xyz;
                    let dist = distance(test_probe_pos, world_position_offs);
                    if dist < closest {
                        selected_offset = offset;
                        closest = dist;
                        probe_pos = test_probe_pos;
                    }
                }
            }
        }


        proposed_pos = textureLoad(prev_tex, vec2<i32>(tex_dims * history_uv) + selected_offset, 0u, 0).xyz;
        let weight_data = textureLoad(prev_tex, vec2<i32>(tex_dims * history_uv) + selected_offset, 1u, 0).xyz;
        M = u32(weight_data.x);
        w_sum = weight_data.y;
        weight = weight_data.z;
    }

    if distance(probe_pos, world_position_offs) > 0.01 {
        probe_pos = world_position_offs;
    }

    if M == 0u {
        proposed_pos = vec3(F32_MAX);
        probe_pos = world_position_offs;
    }

    if M > 1024u {
        return;
    }

//    //reset sometimes, TODO restir probably doesn't reset
//    let reset = hash_noise(ifrag_coord, globals.frame_count) * 64.0 + 1024.0; 
//
//    if f32(M) > reset {
//        proposed_pos = vec3(F32_MAX);
//        M = 0u;
//        w_sum = 0.0;
//        weight = 0.0;
//        probe_pos = world_position_offs;
//    }


    let trace_dist = 16.0;

    let surface_normal = normalize(surface_normal);
    let ufrag_coord = vec2<u32>(ifrag_coord.xy);
    // TODO surface_normal * 0.01 is because of lines from using really low mip for depth
    let ray_start_ndc = position_world_to_ndc(probe_pos);

    let TBN = build_orthonormal_basis(surface_normal);

    var dmr = new_drm_for_restir();
    dmr.ray_start_cs = ray_start_ndc;
    /*
    var recheck_fail = false;
    if proposed_pos.x != F32_MAX { // Rechecking
        let direction = normalize(proposed_pos - probe_pos);
        let ray_end_ws = probe_pos + direction * 0.99;
        let ray_end_cs = position_world_to_ndc(ray_end_ws);
        let ray_end_uv = ndc_to_uv(ray_end_cs.xy);
        dmr = to_ws(dmr, ray_end_ws);
        if ray_end_uv.x > 0.0 && ray_end_uv.x < 1.0 && ray_end_uv.y > 0.0 && ray_end_uv.y < 1.0 {

            // TODO checking weight is not working
            //let closest_motion_vector = prepass_motion_vector(vec4<f32>(ray_end_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
            //let history_uv = ray_end_uv - closest_motion_vector;
            //let color = textureSampleLevel(prev_frame_tex, linear_sampler, history_uv, 2.0).rgb;                
            //let dist = distance(proposed_pos, probe_pos);
            //
            //var gr = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
            //let brdf = max(dot(surface_normal, direction), 0.0);
            //var new_weight = gr * (1.0 / (1.0 + dist * dist));
            //new_weight *= brdf;
            //new_weight = max(new_weight, 0.0);
            //if distance(new_weight, weight) > 0.01 {
            //    recheck_fail = true;
            //}

            dmr.jitter = white_frame_noise(1254u).x; //store jitter?
            let raymarch_result = march(dmr, 0u);
            if (raymarch_result.hit) {
                let hit_nd = textureSampleLevel(prepass_downsample, linear_sampler, raymarch_result.hit_uv, 1.0);
                let hit_ndc = vec3(uv_to_ndc(raymarch_result.hit_uv), hit_nd.w);
                if distance(hit_ndc, position_world_to_ndc(proposed_pos)) > 0.01 {
                    recheck_fail = true;
                }
            }
        } else {
            // ray_end is out of frame
            recheck_fail = true;
        }
    }

    if recheck_fail {
        probe_pos = world_position_offs;
        proposed_pos = vec3(F32_MAX);
        M = u32(0u);
        w_sum = 0.0;
        weight = 0.0;
    }
    */

    for (var i = 0u; i < samples; i += 1u) {
        let white_frame_noise = white_frame_noise(8492u + i);
        let seed = i * samples + globals.frame_count * samples;

        let urand = fract_blue_noise_for_pixel(ufrag_coord, seed, white_frame_noise);  

        var direction = cosine_sample_hemisphere(urand.xy);

        let jitter = fract(blue_noise_for_pixel(ufrag_coord, seed + 4u) + white_frame_noise.z);
        
        direction = normalize(direction * TBN);

        var ray_end_ws = probe_pos + direction * trace_dist;


        dmr = to_ws(dmr, ray_end_ws);
        dmr.jitter = jitter;
        let raymarch_result = march(dmr, 0u);
        if (raymarch_result.hit) {
            let hit_nd = textureSampleLevel(prepass_downsample, linear_sampler, raymarch_result.hit_uv, 1.0);
            var backface = dot(hit_nd.xyz, direction);
            if backface < 0.01 {
                let closest_motion_vector = prepass_motion_vector(vec4<f32>(raymarch_result.hit_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
                let history_uv = raymarch_result.hit_uv - closest_motion_vector;

                let color = textureSampleLevel(prev_frame_tex, linear_sampler, history_uv, 2.0).rgb;
                let hit_ndc = vec3(uv_to_ndc(raymarch_result.hit_uv), hit_nd.w);
                let prop_pos = position_ndc_to_world(hit_ndc);

                let distv = prop_pos - probe_pos;
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
            }
        }
    }


    

    textureStore(target_tex, ifrag_coord, 0u, vec4(proposed_pos, 0.0));
    textureStore(target_tex, ifrag_coord, 1u, vec4(f32(M), w_sum, weight, 0.0));
    textureStore(target_tex, ifrag_coord, 2u, vec4(probe_pos, 0.0));
}

fn get_f0(reflectance: f32, metallic: f32, metal_color: vec3<f32>) -> vec3<f32> {
    let F0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + metal_color * metallic;
    return F0;
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let flocation = vec2<f32>(location);
    let target_dims = vec2<i32>(textureDimensions(target_tex).xy);
    if location.x > target_dims.x || location.y > target_dims.y {
        return;
    }

    

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

    ssgi_store_hits(location, surface_normal, world_position, 1u);

    
    

    let closest_motion_vector = prepass_motion_vector(vec4<f32>(screen_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
    let history_uv = screen_uv - closest_motion_vector;



}

@compute @workgroup_size(8, 8, 1)
fn blur(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let size = vec2<i32>(textureDimensions(target_tex).xy);
    if location.x > size.x || location.y > size.y {
        return;
    }
    let flocation = vec2<f32>(location);
    let fprepass_size = vec2<f32>(textureDimensions(prepass_downsample).xy);
    let fsize = vec2<f32>(size);
    let frag_size = 1.0 / fsize;
    let screen_uv = flocation / fsize + frag_size * 0.5; // TODO verify
    

    //var center_proposed_pos = textureLoad(prev_tex, location, 0u, 0).xyz;
    //let center_weight_data = textureLoad(prev_tex, location, 1u, 0).xyz;
    //let center_probe_pos = textureLoad(prev_tex, location, 2u, 0).xyz;
    //var center_M = u32(center_weight_data.x);
    //var center_w_sum = center_weight_data.y;
    //var center_weight = center_weight_data.z;


    
    //for (var x = -1; x <= 1; x += 1) {
    //    for (var y = -1; y <= 1; y += 1) {
    //        if x == 0 && y == 0 {
    //            continue;
    //        }
    //        let offset = vec2(x, y);
    //        let proposed_pos = textureLoad(prev_tex, location + offset, 0u, 0).xyz;
    //        let weight_data = textureLoad(prev_tex, location + offset, 1u, 0).xyz;
    //        let probe_pos = textureLoad(prev_tex, location + offset, 2u, 0).xyz;
    //        let M = u32(weight_data.x);
    //        let w_sum = weight_data.y;
    //        let weight = weight_data.z;
    //        if distance(probe_pos, center_probe_pos) < 0.01 {
    //            if center_M > 4u && center_weight < 0.1 && weight > center_weight + 0.1 {
    //                center_M += M;
    //                center_w_sum += w_sum;
    //                center_proposed_pos = proposed_pos;
    //                center_weight = weight;
    //            }
    //        }
    //    }
    //}

    //textureStore(target_tex, location, 0u, vec4(center_proposed_pos, 0.0));
    //textureStore(target_tex, location, 1u, vec4(f32(center_M), center_w_sum, center_weight, 0.0));
    //textureStore(target_tex, location, 2u, vec4(center_probe_pos, 0.0));


    textureStore(target_tex, location, 0, textureLoad(prev_tex, location, 0, 0));
    textureStore(target_tex, location, 1, textureLoad(prev_tex, location, 1, 0));
    textureStore(target_tex, location, 2, textureLoad(prev_tex, location, 2, 0));
    textureStore(target_tex, location, 3, textureLoad(prev_tex, location, 3, 0));
}
