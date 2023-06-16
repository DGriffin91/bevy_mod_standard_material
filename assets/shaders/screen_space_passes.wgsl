/*
TODO:
We are writing to the same voxel cell over and over in the same compute pass
Move voxels with view, needs to happen in voxel rate pass or equivalent 
Optimize voxel traversal, we understep currently so we naively hit every voxel
*/

#define RAY_MARCH
//#define USE_VOXEL_FALLBACK
#define FILTER_SSGI
#define USE_PATH_TRACED
#define SSR
const SSR_SAMPLES = 4u;

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
var prepass_downsample: texture_2d<f32>;
@group(0) @binding(6)
var voxel_cache: texture_3d<f32>;
@group(0) @binding(7)
var path_trace_image: texture_2d_array<f32>;
@group(0) @binding(8)
var screen_passes_processed: texture_2d_array<f32>;
@group(0) @binding(9)
var screen_passes_target: texture_storage_2d_array<rgba16float, write>;

@group(0) @binding(10)
var depth_prepass_texture: texture_depth_2d;
@group(0) @binding(11)
var normal_prepass_texture: texture_2d<f32>;
@group(0) @binding(12)
var motion_vector_prepass_texture: texture_2d<f32>;
@group(0) @binding(13)
var deferred_prepass_texture: texture_2d<u32>;



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
#import "shaders/voxel_cache.wgsl"

fn new_drm_for_restir() -> DepthRayMarch {
    var dmr = DepthRayMarch_new_from_depth(view.viewport.zw);
    dmr.linear_steps = 12u;
    dmr.depth_thickness_linear_z = 1.5;
    dmr.march_behind_surfaces = false;
    dmr.use_secant = true;
    dmr.bisection_steps = 4u;
    dmr.use_bilinear = false;
    dmr.mip_min_max = vec2(0.0, 3.0);
    return dmr;
}

fn ssgi_restir(ifrag_coord: vec2<i32>, pbr: PbrInput, samples: u32) {

    
    let trace_dist = 10.0; //after this distance, we switch to the radiance cache


    let itex_dims = vec2<i32>(textureDimensions(screen_passes_processed).xy);
    let tex_dims = vec2<f32>(itex_dims);
    let frag_size = 1.0 / tex_dims;
    let screen_uv = vec2<f32>(ifrag_coord) / tex_dims + frag_size * 0.5;
    //let ifrag_coord = vec2<i32>(screen_uv * tex_dims); //TODO still not rounding to the same probe

    var world_position_offs = pbr.world_position.xyz + pbr.N * 0.001;

    var probe_pos = world_position_offs;
    var proposed_pos = vec3(F32_MAX);
    var M = 0u;
    var w_sum = 0.0;
    var weight = 0.0;
    var probe_latest_color = vec3(0.0);
    var probe_color = vec3(0.0);
    
    let closest_motion_vector = prepass_motion_vector(vec4<f32>(screen_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
    let screen_history_uv = screen_uv - closest_motion_vector;
    let history_hit_screen = screen_history_uv.x > 0.0 && screen_history_uv.x < 1.0 && screen_history_uv.y > 0.0 && screen_history_uv.y < 1.0;
    let screen_space_passes_coord = vec2<i32>(tex_dims * screen_history_uv);
    var sample_coord = screen_space_passes_coord;

    let first_probe_pos = textureLoad(screen_passes_processed, sample_coord, 2u, 0).xyz;
    let first_probe_pos_ndc = position_world_to_ndc(first_probe_pos);
    let probe_pos_ndc = position_world_to_ndc(world_position_offs);
    let probe_ndc_dist = distance(first_probe_pos_ndc.xyz, probe_pos_ndc.xyz);
    // locate closest probe
    if history_hit_screen {
        var selected_offset = vec2(0, 0);
        var closest = distance(first_probe_pos, world_position_offs);
        probe_pos = first_probe_pos;
        if probe_ndc_dist > 0.002 {
            for (var x = -1; x <= 1; x += 1) {
                for (var y = -1; y <= 1; y += 1) {
                    let offset = vec2(x, y);
                    let coord = screen_space_passes_coord + offset;
                    if coord.x < 0 || coord.y < 0 || coord.x >= itex_dims.x || coord.y >= itex_dims.y {
                        continue;
                    }
                    let test_probe_pos = textureLoad(screen_passes_processed, coord, 2u, 0).xyz;
                    let dist = distance(test_probe_pos, world_position_offs);
                    if dist < closest {
                        selected_offset = offset;
                        closest = dist;
                        probe_pos = test_probe_pos;
                    }
                }
            }
        }

        sample_coord = screen_space_passes_coord + selected_offset;
        let sample_uv = (vec2<f32>(sample_coord) + 0.5) / vec2<f32>(tex_dims);
        proposed_pos = textureLoad(screen_passes_processed, sample_coord, 0u, 0).xyz;
        let weight_data = textureLoad(screen_passes_processed, sample_coord, 1u, 0).xyz;
        probe_latest_color = textureLoad(screen_passes_processed, sample_coord, 3u, 0).xyz;

        //probe_color = textureLoad(screen_passes_processed, sample_coord, 4u, 0).xyz;
        probe_color = textureSampleLevel(screen_passes_processed, linear_sampler, screen_history_uv, 4u, 0.0).xyz;

        M = u32(weight_data.x);
        w_sum = weight_data.y;
        weight = weight_data.z;
    }

    // if closest probe is too far, reset position
    if distance(probe_pos, world_position_offs) > 0.01 {
        probe_pos = world_position_offs;
    }

    // if closest probe is too far, reset reservoir
    if distance(probe_pos, world_position_offs) > 0.1 {
        proposed_pos = vec3(F32_MAX);
        M = 0u;
        w_sum = 0.0;
        weight = 0.0;
        probe_pos = world_position_offs;
    }

    if M == 0u {
        proposed_pos = vec3(F32_MAX);
        probe_pos = world_position_offs;
    }


    //if M > 1024u {
    //    return;
    //}

    // randomly reset position
    let reset = hash_noise(ifrag_coord, globals.frame_count);
    if reset > 0.9 {
        probe_pos = world_position_offs;
    }

    // Works better than resetting TODO still probably not what restir does
    let max_M = 64u;
    if M > max_M {
        let ratio = f32(max_M) / f32(M);
        M = max_M;
        w_sum = w_sum * ratio;
    }



    let ufrag_coord = vec2<u32>(ifrag_coord.xy);
    // TODO pbr.N * 0.01 is because of lines from using really low mip for depth
    let ray_start_ws = probe_pos;
    let ray_start_ndc = position_world_to_ndc(ray_start_ws);

    let TBN = build_orthonormal_basis(pbr.N);

    var dmr = new_drm_for_restir();
    dmr.ray_start_cs = ray_start_ndc;
    
    /*
    let recheck = hash_noise(ifrag_coord, globals.frame_count) * 16.0 + 32.0; 
    var recheck_fail = false;
    if f32(M) > recheck && proposed_pos.x != F32_MAX { // Rechecking
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
            //let brdf = max(dot(pbr.N, direction), 0.0);
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

    //if recheck_fail {
    //    probe_pos = world_position_offs;
    //    proposed_pos = vec3(F32_MAX);
    //    M = u32(0u);
    //    w_sum = 0.0;
    //    weight = 0.0;
    //}
    */
    

    for (var i = 0u; i < samples; i += 1u) {
        let white_frame_noise = white_frame_noise(8492u + i);
        let white_frame_noise2 = white_frame_noise(1638u + i);
        let seed = i * samples + globals.frame_count * samples;

        let urand = fract_blue_noise_for_pixel(ufrag_coord, seed, white_frame_noise);  
        let urand2 = fract_blue_noise_for_pixel(ufrag_coord, seed, white_frame_noise2);  

        var direction = cosine_sample_hemisphere(urand.xy);

        let jitter = fract(blue_noise_for_pixel(ufrag_coord, seed + 4u) + white_frame_noise.z);
        
        direction = normalize(direction * TBN);

        var ray_end_ws = probe_pos + direction * trace_dist;

        var march_t = 0.0;

#ifdef RAY_MARCH
        dmr = to_ws(dmr, ray_end_ws);
        dmr.jitter = jitter;
        let raymarch_result = march(dmr, 0u);
        march_t = raymarch_result.hit_t;
        //if false {
        if (raymarch_result.hit) {
            let hit_nd = textureSampleLevel(prepass_downsample, linear_sampler, raymarch_result.hit_uv, 1.0);
            var backface = dot(hit_nd.xyz, direction);
            if backface < 0.01 {
                let closest_motion_vector = prepass_motion_vector(vec4<f32>(raymarch_result.hit_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
                let history_uv = raymarch_result.hit_uv - closest_motion_vector;

                var color = textureSampleLevel(prev_frame_tex, linear_sampler, history_uv, 2.0).rgb;
                let hit_ndc = vec3(uv_to_ndc(raymarch_result.hit_uv), hit_nd.w);
                let prop_pos = position_ndc_to_world(hit_ndc);

                color = clamp(color, vec3(0.0), vec3(12.0));// firefly suppression TODO, do this in filtering

                let dist = distance(prop_pos, probe_pos);

                let hit_frame = history_uv.x > 0.0 && history_uv.x < 1.0 && history_uv.y > 0.0 && history_uv.y < 1.0;

                var gr = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
                let brdf = max(dot(pbr.N, direction), 0.0);
                var new_weight = gr * (1.0 / (1.0 + dist * dist));
                new_weight *= brdf * f32(backface < 0.0) * f32(hit_frame);
                new_weight = max(new_weight, 0.0);

                w_sum += new_weight;

                var threshold = new_weight / w_sum;

                if M == 0u || proposed_pos.x == F32_MAX {
                    threshold = 1.0;
                }

                if threshold > urand.z {
                    proposed_pos = prop_pos;
                    weight = new_weight;
                    probe_latest_color = color;
                }
            }
        } else {
#endif //RAY_MARCH
            #ifdef USE_VOXEL_FALLBACK
            let ray_pos_ndc = mix(dmr.ray_start_cs, dmr.ray_end_cs, march_t);
            let ray_end_pos_ws = position_ndc_to_world(ray_pos_ndc);
            let ray_one_voxel_away = ray_start_ws + direction * VOXEL_SIZE * 1.5;
            let voxel_ray_start = select(ray_one_voxel_away, ray_end_pos_ws, 
                                         distance(ray_end_pos_ws, ray_start_ws) > 
                                         distance(ray_one_voxel_away, ray_start_ws));

            let skip = u32(urand2.x > 0.5);

            let max_voxel_age = 1000.0;
            let voxel_hit = march_voxel_grid(voxel_ray_start, direction, 512u, skip, max_voxel_age);
            if voxel_hit.t > VOXEL_SIZE * (1.0 + urand.w) {
                let age = max(globals.time - voxel_hit.age, 1.0);
                let dist = voxel_hit.t;
                let voxel_derate = 1.0 * saturate(1.0 - pow(age, 1.0) / max_voxel_age);

                let color = voxel_hit.color * voxel_derate;
                let prop_pos = probe_pos + direction * voxel_hit.t;
                var gr = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
                let brdf = max(dot(pbr.N, direction), 0.0);
                var new_weight = gr * (1.0 / (1.0 + dist * dist));
                new_weight *= brdf * voxel_derate;
                new_weight = max(new_weight, 0.0);

                w_sum += new_weight;

                var threshold = new_weight / w_sum;

                if M == 0u || proposed_pos.x == F32_MAX {
                    threshold = 1.0;
                }

                if threshold > urand.z {
                    proposed_pos = prop_pos;
                    weight = new_weight;
                    probe_latest_color = color;
                }
            }
            #endif
#ifdef RAY_MARCH
        }
#endif //RAY_MARCH
        M += 1u;
    }

#ifdef USE_PATH_TRACED
    { // sample from path traced
        let pt_samples = 5u;
        let scale = 1.25;
        let path_trace_image_dims = vec2<f32>(textureDimensions(path_trace_image).xy);
        for (var i = 0u; i <= pt_samples; i+=1u) {
            let seed = i * pt_samples + globals.frame_count * pt_samples;
            let white_frame_noise = white_frame_noise(i + 1821u);
            let urand = fract_blue_noise_for_pixel(ufrag_coord, seed, white_frame_noise);  
            let offset_rand = (urand.wz * 2.0 - 1.0) / path_trace_image_dims * scale;
            let coord = vec2<i32>((screen_uv + offset_rand) * path_trace_image_dims);
            let path_trace_data = textureLoad(path_trace_image, coord, 0u, 0);
            let color = path_trace_data.rgb; //TODO, why is it so much brighter
            let path_trace_data2 = textureLoad(path_trace_image, coord, 1u, 0);
            let direction = path_trace_data2.xyz;
            let dist = path_trace_data2.w;

            let prop_pos = probe_pos + direction * dist;
            var gr = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
            let brdf = max(dot(pbr.N, direction), 0.0);
            var new_weight = brdf * gr * (1.0 / (1.0 + dist * dist));
            new_weight = max(new_weight, 0.0);

            w_sum += new_weight;

            var threshold = new_weight / w_sum;
            //threshold /= f32(pt_samples); //TODO?

            if M == 0u || proposed_pos.x == F32_MAX {
                threshold = 1.0;
            }

            if threshold > urand.z {
                proposed_pos = prop_pos;
                weight = new_weight;
                probe_latest_color = color;
            }
        }
    }
#endif //USE_PATH_TRACED

    let dist_to_hit = distance(proposed_pos, world_position_offs);

    let falloff = (1.0 / (1.0 + dist_to_hit * dist_to_hit));
    var hysterisis = 0.1;
    let w = w_sum / max(0.00001, f32(M) * weight);

    let resolve_col = min(probe_latest_color, probe_latest_color * w * falloff);

    probe_color = mix(probe_color, resolve_col, hysterisis);


    textureStore(screen_passes_target, ifrag_coord, 0u, vec4(proposed_pos, 0.0));
    textureStore(screen_passes_target, ifrag_coord, 1u, vec4(f32(M), w_sum, weight, 0.0));
    textureStore(screen_passes_target, ifrag_coord, 2u, vec4(probe_pos, 0.0));
    textureStore(screen_passes_target, ifrag_coord, 3u, vec4(probe_latest_color, 0.0));
    textureStore(screen_passes_target, ifrag_coord, 4u, vec4(probe_color, 0.0));


#ifdef SSR
    var ssr_hysterisis = 0.1;

    let F0 = get_f0(pbr.material.reflectance, pbr.material.metallic, pbr.material.base_color.rgb);
    let roughness = perceptualRoughnessToRoughness(pbr.material.perceptual_roughness);
    var ssr = bad_ssr(ifrag_coord, pbr.N, pbr.world_position.xyz, roughness, F0, SSR_SAMPLES, vec2(2.0, 3.0)).rgb;

    if history_hit_screen && probe_ndc_dist < 0.005 {
        let prev_ssr = textureLoad(screen_passes_processed, sample_coord, 6u, 0).rgb;
        ssr = mix(prev_ssr, ssr, ssr_hysterisis);
    }

    ssr = clamp(ssr, vec3(0.0), vec3(100.0));

    textureStore(screen_passes_target, ifrag_coord, 6u, vec4(ssr, 0.0));
#else
    textureStore(screen_passes_target, ifrag_coord, 6u, vec4(0.0));
#endif //SSR
}



fn get_f0(reflectance: f32, metallic: f32, metal_color: vec3<f32>) -> vec3<f32> {
    let F0 = 0.16 * reflectance * reflectance * (1.0 - metallic) + metal_color * metallic;
    return F0;
}

fn perceptualRoughnessToRoughness(perceptualRoughness: f32) -> f32 {
    let clampedPerceptualRoughness = clamp(perceptualRoughness, 0.089, 1.0);
    return clampedPerceptualRoughness * clampedPerceptualRoughness;
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let flocation = vec2<f32>(location);
    let target_dims = vec2<i32>(textureDimensions(screen_passes_target).xy);
    if location.x >= target_dims.x || location.y >= target_dims.y {
        return;
    }

    let depth_tex_dims = vec2<f32>(textureDimensions(prepass_downsample).xy);
    
    let ftarget_dims = vec2<f32>(target_dims);
    let frag_size = 1.0 / ftarget_dims;
    var screen_uv = flocation.xy / ftarget_dims + frag_size * 0.5;

    //let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 1.0);
    //let pbr.N = nor_depth.xyz;
    //let depth = nor_depth.w;
    var frag_coord = vec4(flocation, 0.0, 0.0);
    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);


    let deferred_data = textureLoad(deferred_prepass_texture, ifrag_coord, 0);
#ifdef WEBGL
    frag_coord.z = unpack_unorm3x4_plus_unorm_20(deferred_data.b).w;
#else
    frag_coord.z = prepass_depth(frag_coord, 0u);
#endif
    var pbr_input = pbr_input_from_deferred_gbuffer(frag_coord, deferred_data);


    ssgi_restir(location, pbr_input, 1u);

    let closest_motion_vector = prepass_motion_vector(vec4<f32>(screen_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
    let history_uv = screen_uv - closest_motion_vector;
}

@compute @workgroup_size(8, 8, 1)
fn blur(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let size = vec2<i32>(textureDimensions(screen_passes_target).xy);
    if location.x >= size.x || location.y >= size.y {
        return;
    }
    let ulocation = vec2<u32>(location);
    let flocation = vec2<f32>(location);
    let fprepass_size = vec2<f32>(textureDimensions(prepass_downsample).xy);
    let fsize = vec2<f32>(size);
    let frag_size = 1.0 / fsize;
    let screen_uv = flocation / fsize + frag_size * 0.5; // TODO verify

#ifdef FILTER_SSGI
    let v = textureSampleLevel(screen_passes_processed, linear_sampler, screen_uv, 4, 0.0);
    let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 1.0);
    let world_position = position_ndc_to_world(vec3(uv_to_ndc(screen_uv), v.w));
    let dist_factor = 1.0;
    var tot = vec3(0.0);
    var tot_w = 0.0;
    let samples = 12u;
    let range = 20.0;
    let white_frame_noise = white_frame_noise(8462u);
    tot += textureLoad(screen_passes_processed, location, 4, 0).xyz;
    tot_w += 1.0;
    for (var i = 0u; i <= samples; i+=1u) {
        let seed = i * samples + globals.frame_count * samples;
        let urand = fract_blue_noise_for_pixel(ulocation, seed, white_frame_noise) * 2.0 - 1.0;
        let px_pos = vec2(urand.x * range, urand.y * range);
        let coord = location + vec2<i32>(px_pos);
        if coord.x < 0 || coord.y < 0 || coord.x >= size.x || coord.y >= size.y {
            continue;
        }
        let uv = vec2<f32>(coord) / fsize;
        let px_dist = length(abs(urand.xy));
        var w = px_dist;
        let nd = textureLoad(prepass_downsample, vec2<i32>(uv * fprepass_size), 0);
        let col = textureLoad(screen_passes_processed, coord, 4, 0);
    
        let d = max(dot(nor_depth.xyz, nd.xyz) + 0.001, 0.0);
        w = w * d * d * d; //lol
        let px_ws = position_ndc_to_world(vec3(uv_to_ndc(uv), col.w));
        let dist = distance(px_ws, world_position);
        w = w * (1.0 - clamp(dist * dist_factor, 0.0, 1.0));
        w = saturate(w);
        tot += col.rgb * w;
        tot_w += w;
    }
    textureStore(screen_passes_target, location, 4, vec4(tot/tot_w, 1.0));
#else
    textureStore(screen_passes_target, location, 4, textureLoad(screen_passes_processed, location, 4, 0));
#endif //FILTER_SSGI


    textureStore(screen_passes_target, location, 0, textureLoad(screen_passes_processed, location, 0, 0));
    textureStore(screen_passes_target, location, 1, textureLoad(screen_passes_processed, location, 1, 0));
    textureStore(screen_passes_target, location, 2, textureLoad(screen_passes_processed, location, 2, 0));
    textureStore(screen_passes_target, location, 3, textureLoad(screen_passes_processed, location, 3, 0));

    textureStore(screen_passes_target, location, 5, textureLoad(screen_passes_processed, location, 5, 0));
    textureStore(screen_passes_target, location, 6, textureLoad(screen_passes_processed, location, 6, 0));
    textureStore(screen_passes_target, location, 7, textureLoad(screen_passes_processed, location, 7, 0));
}
