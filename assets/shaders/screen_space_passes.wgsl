/*
TODO:
We are writing to the same voxel cell over and over in the same compute pass
Move voxels with view, needs to happen in voxel rate pass or equivalent 
Optimize voxel traversal, we understep currently so we naively hit every voxel
*/

#define RAY_MARCH
#define USE_VOXEL_FALLBACK
//#define FILTER_SSGI
#define USE_PATH_TRACED
#define SSAO_FOCUS
#define SSAO
#define SSR
const SSR_SAMPLES = 10u;
const FULL_SSR_SAMPLES = 1u;

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
var screen_passes_read: texture_2d_array<f32>;
@group(0) @binding(9)
var screen_passes_write: texture_storage_2d_array<rgba16float, write>;

@group(0) @binding(10)
var fullscreen_passes_read: texture_2d_array<f32>;
@group(0) @binding(11)
var fullscreen_passes_write: texture_storage_2d_array<rgba16float, write>;

@group(0) @binding(12)
var depth_prepass_texture: texture_depth_2d;
@group(0) @binding(13)
var normal_prepass_texture: texture_2d<f32>;
@group(0) @binding(14)
var motion_vector_prepass_texture: texture_2d<f32>;
@group(0) @binding(15)
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
#import "shaders/bad_gtao.wgsl"
#import "shaders/voxel_cache.wgsl"

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

struct ScreenTraceResult {
    pos: vec3<f32>,
    color: vec3<f32>,
    derate: f32,
    hit: bool,
    backface: bool,
}

fn screentrace(ufrag_coord: vec2<u32>, probe_pos: vec3<f32>, TBN: mat3x3<f32>) -> ScreenTraceResult {
    var drm = new_drm_for_restir();
    let ray_start_ws = probe_pos;
    drm.ray_start_cs = position_world_to_ndc(ray_start_ws);

    let trace_dist = 10.0; //after this distance, we switch to the radiance cache

    let white_frame_noise = white_frame_noise(8492u);
    let urand = fract_blue_noise_for_pixel(ufrag_coord, globals.frame_count, white_frame_noise);  
    var direction = cosine_sample_hemisphere(urand.xy);
    let jitter = fract(blue_noise_for_pixel(ufrag_coord, globals.frame_count + 4u) + white_frame_noise.z);
    
    direction = normalize(direction * TBN);

    var ray_end_ws = probe_pos + direction * trace_dist;


    var res: ScreenTraceResult;
    res.hit = false;
    res.derate = -1.0; // miss

#ifdef RAY_MARCH
    drm = to_ws(drm, ray_end_ws);
    drm.jitter = jitter;
    let raymarch_result = march(drm, 0u);
    var march_t = raymarch_result.hit_t;
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

            let hit = history_uv.x > 0.0 && history_uv.x < 1.0 && history_uv.y > 0.0 && history_uv.y < 1.0;

            res.pos = prop_pos;
            res.derate = 1.0;
            res.color = color;
            res.hit = hit;
            res.backface = backface > -0.01;

            return res;
        }
    } else {
#endif //RAY_MARCH
            #ifdef USE_VOXEL_FALLBACK
            let ray_pos_ndc = mix(drm.ray_start_cs, drm.ray_end_cs, march_t);
            let ray_end_pos_ws = position_ndc_to_world(ray_pos_ndc);
            let ray_one_voxel_away = ray_start_ws + direction * VOXEL_SIZE * 1.5;
            let voxel_ray_start = select(ray_one_voxel_away, ray_end_pos_ws, 
                                         distance(ray_end_pos_ws, ray_start_ws) > 
                                         distance(ray_one_voxel_away, ray_start_ws));

            let skip = u32(urand.z > 0.5);

            let max_voxel_age = 1000.0;
            let voxel_hit = march_voxel_grid(voxel_ray_start, direction, 512u, skip, max_voxel_age);
            if voxel_hit.t > VOXEL_SIZE * (1.0 + urand.w) {
                let age = max(globals.time - voxel_hit.age, 1.0);
                let dist = voxel_hit.t;
                let voxel_derate = 0.5 * saturate(1.0 - pow(age, 1.0) / max_voxel_age);

                let color = voxel_hit.color * voxel_derate;
                let prop_pos = probe_pos + direction * voxel_hit.t;


                
                res.pos = prop_pos;
                res.color = color;
                res.hit = true;
                res.backface = false;
                res.derate = voxel_derate;

                return res;

            }
            #endif
#ifdef RAY_MARCH
        }
#endif //RAY_MARCH
    return res;
}

fn ssgi_restir(ifrag_coord: vec2<i32>, frag_coord: vec4<f32>, pbr: PbrInput, samples: u32) {
    let MAX_M = 48u;

    let ufrag_coord = vec2<u32>(ifrag_coord.xy);
    let itex_dims = vec2<i32>(textureDimensions(fullscreen_passes_read).xy);
    let tex_dims = vec2<f32>(itex_dims);
    let frag_size = 1.0 / tex_dims;
    let screen_uv = vec2<f32>(ifrag_coord) / tex_dims + frag_size * 0.5;

    var world_position_offs = pbr.world_position.xyz + pbr.N * 0.001;

#ifdef SSAO
    var ssao = bad_gtao(frag_coord, world_position_offs, pbr.N).rgb;
#endif
#ifdef SSAO_FOCUS
    let ssao_focus = pow(ssao.x, 0.75);
#elseif
    let ssao_focus = 1.0;
#endif
    
    var col_accum = vec3(0.0);
    var accum_tot_w = 0.0;
    var pt_col_accum = vec3(0.0);
    var pt_accum_tot_w = 0.0;
    var pt_accum_tot_gr = 0.0;

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

    let first_probe_pos = textureLoad(fullscreen_passes_read, sample_coord, 2u, 0).xyz;
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
                    let test_probe_pos = textureLoad(fullscreen_passes_read, coord, 2u, 0).xyz;
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
        proposed_pos = textureLoad(fullscreen_passes_read, sample_coord, 0u, 0).xyz;
        let weight_data = textureLoad(fullscreen_passes_read, sample_coord, 1u, 0).xyz;
        probe_latest_color = textureLoad(fullscreen_passes_read, sample_coord, 3u, 0).xyz;
        probe_color = textureSampleLevel(fullscreen_passes_read, linear_sampler, screen_history_uv, 4u, 0.0).xyz;

        M = u32(weight_data.x);
        w_sum = weight_data.y;
        weight = weight_data.z;
    }

    // if closest probe is too far, reset position
    if distance(probe_pos, world_position_offs) > 0.05 {
        probe_pos = world_position_offs;
    }

    // if closest probe is too far, reset reservoir
    if distance(probe_pos, world_position_offs) > 0.1 || !history_hit_screen {
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

    // randomly reset position
    let reset = hash_noise(ifrag_coord, globals.frame_count);
    if reset > 0.9 {
        probe_pos = world_position_offs;
    }

    // Works better than resetting TODO still probably not what restir does
    if M > MAX_M {
        let ratio = f32(MAX_M) / f32(M);
        M = MAX_M;
        w_sum = w_sum * ratio;
    }
    
    let white_frame_noise = white_frame_noise(3812u);
    let seed = samples + globals.frame_count * samples;
    let urand = fract_blue_noise_for_pixel(ufrag_coord, seed, white_frame_noise);  

    let screen_passes_dims = vec2<f32>(textureDimensions(screen_passes_read).xy);
#ifdef RAY_MARCH
    var st: ScreenTraceResult;
    var candidates_samples = 8u;
    var candidates_radius = 10.0 * ssao_focus;
    let max_dist = 0.2;
    for (var i = 0u; i < candidates_samples; i+=1u) {
        var coord = vec2(0);
        var offset_uv = vec2(0.0);
        var offset = vec2(0.0);
        
        for (var j = 0u; j < candidates_samples; j+=1u) {
            let seed = (i + 1u) * (j + 1u) * candidates_samples + globals.frame_count * candidates_samples;
            let white_frame_noise = white_frame_noise(seed + 2351u);
            let urand = fract_blue_noise_for_pixel(ufrag_coord, seed, white_frame_noise);  
            offset = (urand.wz * 2.0 - 1.0);
            //if i > 0u {
            offset_uv = offset / screen_passes_dims * candidates_radius;
            //}
            coord = vec2<i32>((screen_uv + offset_uv) * screen_passes_dims);
            let clamped_coord = clamp(coord, vec2(0), vec2<i32>(screen_passes_dims));
            // try to find coord that is not off screen
            if all(clamped_coord == coord) {
                break;
            }
            coord = clamped_coord;
        }

        let st_ray_start = textureLoad(screen_passes_read, coord, 5u, 0).xyz;
        let probe_to_cand_distance = distance(st_ray_start, probe_pos);

        if probe_to_cand_distance > max_dist {
            // Sample is too far away
            // Doesn't increase M
            // TODO scale by frag depth?
            continue;
        }

        st.pos = textureLoad(screen_passes_read, coord, 6u, 0).xyz;
        st.color = textureLoad(screen_passes_read, coord, 7u, 0).xyz;
        let st_data = textureLoad(screen_passes_read, coord, 8u, 0);
        st.derate = st_data.x;
        st.hit = bool(u32(st_data.y));
        st.backface = bool(u32(st_data.z));
        let st_distance = distance(st.pos, world_position_offs);
        let st_direction = normalize(st.pos - world_position_offs);

        if st_distance >= 0.0 { // hit
            var gr = dot(st.color, vec3<f32>(0.2126, 0.7152, 0.0722));
            let brdf = max(dot(pbr.N, st_direction), 0.0);
            let dist_falloff = (1.0 / (1.0 + st_distance * st_distance));
            var new_weight = gr * dist_falloff * brdf * f32(!st.backface) * f32(st.hit) * st.derate;
            new_weight = max(new_weight, 0.0);

            w_sum += new_weight;

            var threshold = new_weight / w_sum;

            if i == 0u && (M == 0u || proposed_pos.x == F32_MAX) {
                threshold = 1.0;
            }

            if threshold > urand.z {
                proposed_pos = st.pos;
                weight = new_weight;
                probe_latest_color = st.color;
            }

            {
                let probe_dist_falloff = saturate(max_dist - probe_to_cand_distance) + 1.0;
                let offset_w = saturate(1.0 / (length(offset) * candidates_radius)) + 1.0;
                let w = max(dist_falloff * brdf * f32(!st.backface) * f32(st.hit) * st.derate * offset_w * probe_dist_falloff, 0.0);
                col_accum += st.color * w * brdf;
                accum_tot_w += w;
            }
        }

        M += 1u;
    }
#endif //RAY_MARCH

#ifdef USE_PATH_TRACED
    { // sample from path traced
        let pt_samples = 10u;
        let candidates_radius = 10.0 * ssao_focus;
        let path_trace_image_dims = vec2<f32>(textureDimensions(path_trace_image).xy);
        let pt_max_dist = 0.5;
        for (var i = 0u; i < pt_samples; i+=1u) {
            var coord = vec2(0);
            var offset_uv = vec2(0.0);
            var offset = vec2(0.0);

            for (var j = 0u; j < pt_samples; j+=1u) {
                let seed = (i + 1u) * (j + 1u) * pt_samples + globals.frame_count * pt_samples;
                let white_frame_noise = white_frame_noise(seed + 1821u);
                let urand = fract_blue_noise_for_pixel(ufrag_coord, i + globals.frame_count, white_frame_noise);  
                offset = (urand.wz * 2.0 - 1.0);
                //if i > 0u {
                offset_uv = offset / path_trace_image_dims * candidates_radius;
                //}
                coord = vec2<i32>((screen_uv + offset_uv) * path_trace_image_dims);
                let clamped_coord = clamp(coord, vec2(0), vec2<i32>(path_trace_image_dims));
                // try to find coord that is not off screen
                if all(clamped_coord == coord) {
                    break;
                }
                coord = clamped_coord;
            }

            let color = textureLoad(path_trace_image, coord, 0u, 0).rgb;
            let pt_world_position = textureLoad(path_trace_image, coord, 1u, 0).xyz;
            let ray_hit = textureLoad(path_trace_image, coord, 2u, 0);
            let ray_hit_pos = ray_hit.xyz;
            let direction = normalize(ray_hit_pos - pt_world_position);
            let dist = ray_hit.w;
            
            let probe_to_cand_distance = distance(probe_pos, pt_world_position);

            if probe_to_cand_distance > pt_max_dist {
                // Sample is too far away
                // Doesn't increase M
                // TODO scale by frag depth?
                continue;
            }

            if dist < 0.0 { // miss
                continue;
            }

            var gr = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
            let brdf = max(dot(pbr.N, direction), 0.0);
            var dist_falloff = (1.0 / (1.0 + dist * dist));
            if dist == F32_MAX { // hit sky
                dist_falloff = 1.0;
            }
            var new_weight = gr * brdf * dist_falloff;
            new_weight = max(new_weight, 0.0);

            w_sum += new_weight;

            var threshold = new_weight / w_sum;

            //if M == 0u || proposed_pos.x == F32_MAX {
            //    threshold = 1.0;
            //}



            if threshold > urand.z {
                proposed_pos = ray_hit_pos;
                weight = new_weight;
                probe_latest_color = color;
            }

            {
                let probe_dist_falloff = saturate(pt_max_dist - probe_to_cand_distance) + 1.0;
                let offset_w = saturate(1.0 / (length(offset) * candidates_radius)) + 1.0;
                let w = max( brdf * dist_falloff * offset_w * probe_dist_falloff, 0.0);
                pt_col_accum += color * w * brdf;
                pt_accum_tot_w += w;
                //pt_accum_tot_gr += gr;
            }
            
            M += 1u;
        }
    }
#endif //USE_PATH_TRACED

    let dist_to_hit = distance(proposed_pos, world_position_offs);
    let direction = normalize(proposed_pos - world_position_offs);
    let brdf = max(dot(pbr.N, direction), 0.0);

    let falloff = (1.0 / (1.0 + dist_to_hit * dist_to_hit));
    var hysterisis = 0.1;
    //hysterisis = mix(0.5, hysterisis, saturate(f32(M) / f32(MAX_M)));
    let w = w_sum / max(0.00001, f32(M) * weight);

    let resolve_col = min(probe_latest_color, probe_latest_color * w);

    //probe_color = mix(probe_color, resolve_col, hysterisis);

    let ssgi = max(col_accum / accum_tot_w, vec3(0.0));
    let ptgi = max((pt_col_accum / pt_accum_tot_w), vec3(0.0));
    var gi = (ssgi + ptgi) / 2.0;

#ifdef SSAO
    // GI is too broad for fine corner detail
    gi *= ssao;
#endif

    probe_color = mix(probe_color, gi, hysterisis);

    


    textureStore(fullscreen_passes_write, ifrag_coord, 0u, vec4(proposed_pos, 0.0));
    textureStore(fullscreen_passes_write, ifrag_coord, 1u, vec4(f32(M), w_sum, weight, 0.0));
    textureStore(fullscreen_passes_write, ifrag_coord, 2u, vec4(probe_pos, 0.0));
    textureStore(fullscreen_passes_write, ifrag_coord, 3u, vec4(probe_latest_color, 0.0));
    textureStore(fullscreen_passes_write, ifrag_coord, 4u, vec4(probe_color, 0.0));


#ifdef SSR
    var ssr_hysterisis = 0.1;

    let F0 = get_f0(pbr.material.reflectance, pbr.material.metallic, pbr.material.base_color.rgb);
    let roughness = perceptualRoughnessToRoughness(pbr.material.perceptual_roughness);
    var ssr = bad_ssr(ifrag_coord, pbr.N, pbr.world_position.xyz, roughness, F0, FULL_SSR_SAMPLES, vec2(2.0, 3.0)).rgb;

    
    let half_ssr = textureLoad(screen_passes_read, vec2<i32>(screen_uv * screen_passes_dims), 1u, 0).rgb;
    ssr = mix(half_ssr, ssr, 0.5);

    if history_hit_screen && probe_ndc_dist < 0.1 {
        let prev_ssr = textureLoad(fullscreen_passes_read, sample_coord, 5u, 0).rgb;
        ssr = mix(prev_ssr, ssr, ssr_hysterisis);
    }

    

    ssr = clamp(ssr, vec3(0.0), vec3(100.0));
    

    textureStore(fullscreen_passes_write, ifrag_coord, 5u, vec4(ssr, 0.0));
#else
    textureStore(fullscreen_passes_write, ifrag_coord, 5u, vec4(0.0));
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
fn candidates(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    //if true {return;}
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let flocation = vec2<f32>(location);
    let target_dims = vec2<i32>(textureDimensions(screen_passes_write).xy);
    if location.x >= target_dims.x || location.y >= target_dims.y {
        return;
    }

    let depth_tex_dims = vec2<f32>(textureDimensions(prepass_downsample).xy);
    
    let ftarget_dims = vec2<f32>(target_dims);
    let frag_size = 1.0 / ftarget_dims;
    var screen_uv = flocation.xy / ftarget_dims + frag_size * 0.5;


    var frag_coord = vec4(flocation, 0.0, 0.0);
    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);

    var full_screen_frag_coord = vec4(screen_uv * view.viewport.zw, 0.0, 0.0);
    let deferred_data = textureLoad(deferred_prepass_texture, vec2<i32>(full_screen_frag_coord.xy), 0);
#ifdef WEBGL
    full_screen_frag_coord.z = unpack_unorm3x4_plus_unorm_20(deferred_data.b).w;
#else
    full_screen_frag_coord.z = prepass_depth(full_screen_frag_coord, 0u);
#endif
    var pbr = pbr_input_from_deferred_gbuffer(full_screen_frag_coord, deferred_data);

    var world_position_offs = pbr.world_position.xyz + pbr.N * 0.001;
    let TBN = build_orthonormal_basis(pbr.N);
    var st = screentrace(ufrag_coord, world_position_offs, TBN);

    textureStore(screen_passes_write, ifrag_coord, 5u, vec4(world_position_offs, 0.0));
    textureStore(screen_passes_write, ifrag_coord, 6u, vec4(st.pos, 0.0));
    textureStore(screen_passes_write, ifrag_coord, 7u, vec4(st.color, 0.0));
    textureStore(screen_passes_write, ifrag_coord, 8u, vec4(st.derate, f32(st.hit), f32(st.backface), 0.0));

#ifdef SSR
    var ssr_hysterisis = 0.1;

    let F0 = get_f0(pbr.material.reflectance, pbr.material.metallic, pbr.material.base_color.rgb);
    let roughness = perceptualRoughnessToRoughness(pbr.material.perceptual_roughness);
    var ssr = bad_ssr(ifrag_coord, pbr.N, pbr.world_position.xyz, roughness, F0, SSR_SAMPLES, vec2(2.0, 3.0)).rgb;

    ssr = clamp(ssr, vec3(0.0), vec3(100.0));

    textureStore(screen_passes_write, ifrag_coord, 1u, vec4(ssr, 0.0));
#endif //SSR
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    //if true {return;}
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let flocation = vec2<f32>(location);
    let target_dims = vec2<i32>(textureDimensions(fullscreen_passes_write).xy);
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

    var full_screen_frag_coord = vec4(screen_uv * view.viewport.zw, 0.0, 0.0);
    let deferred_data = textureLoad(deferred_prepass_texture, vec2<i32>(full_screen_frag_coord.xy), 0);
#ifdef WEBGL
    full_screen_frag_coord.z = unpack_unorm3x4_plus_unorm_20(deferred_data.b).w;
#else
    full_screen_frag_coord.z = prepass_depth(full_screen_frag_coord, 0u);
#endif
    var pbr_input = pbr_input_from_deferred_gbuffer(full_screen_frag_coord, deferred_data);


    ssgi_restir(location, full_screen_frag_coord, pbr_input, 1u);
}

@compute @workgroup_size(8, 8, 1)
fn blur(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    //if true {return;}
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let size = vec2<i32>(textureDimensions(fullscreen_passes_write).xy);
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
    let v = textureSampleLevel(fullscreen_passes_read, linear_sampler, screen_uv, 4, 0.0);
    let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 0.0);
    let world_position = position_ndc_to_world(vec3(uv_to_ndc(screen_uv), v.w));
    let dist_factor = 1.0;
    var tot = vec3(0.0);
    var tot_w = 0.0;
    let samples = 2u;
    let range = 5.0;
    let white_frame_noise = white_frame_noise(8462u);
    let center_weight = 10.0;
    tot += textureLoad(fullscreen_passes_read, location, 4, 0).xyz * center_weight;
    tot_w += center_weight;
    let center = textureLoad(fullscreen_passes_read, location, 4, 0);
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
        let col = textureLoad(fullscreen_passes_read, coord, 4, 0);
    
        let d = max(dot(nor_depth.xyz, nd.xyz) + 0.001, 0.0);
        w = w * d * d * d; //lol
        let px_ws = position_ndc_to_world(vec3(uv_to_ndc(uv), col.w));
        let dist = distance(px_ws, world_position);
        w = w * (1.0 - clamp(dist * dist_factor, 0.0, 1.0));
        w = saturate(w);
        tot += col.rgb * w;
        tot_w += w;
    }
    textureStore(fullscreen_passes_write, location, 4, vec4(tot/tot_w, 1.0));
#else
    textureStore(fullscreen_passes_write, location, 4, textureLoad(fullscreen_passes_read, location, 4, 0));
#endif //FILTER_SSGI


    textureStore(screen_passes_write, location, 0, textureLoad(screen_passes_read, location, 0, 0));
    textureStore(screen_passes_write, location, 1, textureLoad(screen_passes_read, location, 1, 0));
    textureStore(screen_passes_write, location, 2, textureLoad(screen_passes_read, location, 2, 0));
    textureStore(screen_passes_write, location, 3, textureLoad(screen_passes_read, location, 3, 0));
    textureStore(screen_passes_write, location, 4, textureLoad(screen_passes_read, location, 4, 0));
    textureStore(screen_passes_write, location, 5, textureLoad(screen_passes_read, location, 5, 0));
    textureStore(screen_passes_write, location, 6, textureLoad(screen_passes_read, location, 6, 0));
    textureStore(screen_passes_write, location, 7, textureLoad(screen_passes_read, location, 7, 0));
    textureStore(screen_passes_write, location, 8, textureLoad(screen_passes_read, location, 8, 0));

    
    textureStore(fullscreen_passes_write, location, 0, textureLoad(fullscreen_passes_read, location, 0, 0));
    textureStore(fullscreen_passes_write, location, 1, textureLoad(fullscreen_passes_read, location, 1, 0));
    textureStore(fullscreen_passes_write, location, 2, textureLoad(fullscreen_passes_read, location, 2, 0));
    textureStore(fullscreen_passes_write, location, 3, textureLoad(fullscreen_passes_read, location, 3, 0));

    textureStore(fullscreen_passes_write, location, 5, textureLoad(fullscreen_passes_read, location, 5, 0));
    textureStore(fullscreen_passes_write, location, 6, textureLoad(fullscreen_passes_read, location, 6, 0));
    textureStore(fullscreen_passes_write, location, 7, textureLoad(fullscreen_passes_read, location, 7, 0));
    textureStore(fullscreen_passes_write, location, 8, textureLoad(fullscreen_passes_read, location, 8, 0));
}
