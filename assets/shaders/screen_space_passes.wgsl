/*
TODO:
We are writing to the same voxel cell over and over in the same compute pass
Move voxels with view, needs to happen in voxel rate pass or equivalent 
Optimize voxel traversal, we understep currently so we naively hit every voxel
*/

//#define USE_VOXEL_FALLBACK

//#define RAY_MARCH
#define USE_PATH_TRACED
#define FILTER_SSGI
#define APPLY_FILTER_TO_OUTPUT
#define SSAO_FOCUS
#define SSAO
#define RESTIR_ANTI_CLUMPING
#define SSR
const SSR_SAMPLES = 0u;
const FULL_SSR_SAMPLES = 0u;
const MAX_M = 1024u; //current max 2048
const GI_HYSTERISIS = 0.2;//0.08; //0.2
const SSGI_PIX_RADIUS_DIST = 100.0;


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
var fullscreen_passes_write: texture_storage_2d_array<rgba32float, write>;

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
#import "shaders/probe.wgsl"




fn coplanar(pos1: vec3<f32>, normal1: vec3<f32>, pos2: vec3<f32>, normal2: vec3<f32>, nor_epsilon: f32, ws_epsilon: f32) -> bool {
    if (dot(normal1, normal2) < clamp(nor_epsilon - 1.0, -1.0, 1.0)) {
        return false;
    }
    let D1 = -dot(normal1, pos1);
    return abs(dot(normal1, pos2) + D1) < ws_epsilon;
}

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

fn screentrace(linear_depth: f32, ufrag_coord: vec2<u32>, probe_pos: vec3<f32>, TBN: mat3x3<f32>) -> ScreenTraceResult {

    var drm = new_drm_for_restir();
    let ray_start_ws = probe_pos;
    drm.ray_start_cs = position_world_to_ndc(ray_start_ws);

    var pixel_radius = pixel_radius_to_world(1.0, linear_depth, projection_is_orthographic());
    let trace_dist = SSGI_PIX_RADIUS_DIST * pixel_radius; //after this distance, we switch to the radiance cache

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

            let hit = all(history_uv > 0.0) && all(history_uv < 1.0);

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
    
    let ufrag_coord = vec2<u32>(ifrag_coord.xy);
    let itex_dims = vec2<i32>(textureDimensions(fullscreen_passes_read).xy);
    let tex_dims = vec2<f32>(itex_dims);
    let frag_size = 1.0 / tex_dims;
    let screen_uv = vec2<f32>(ifrag_coord) / tex_dims + frag_size * 0.5;
    var pixel_radius = pixel_radius_to_world(1.0, depth_ndc_to_linear(frag_coord.z), pbr.is_orthographic);
    // limit minimum pixel radius for things really close to the camera
    pixel_radius = max(pixel_radius, 0.001); 

    var world_position_offs = pbr.world_position.xyz + pbr.N * 0.001;

#ifdef SSAO
    var ssao = bad_gtao(frag_coord, world_position_offs, pbr.N).rgb;
#endif
#ifdef SSAO_FOCUS
    let ssao_focus = ssao.x;
#elseif
    let ssao_focus = 1.0;
#endif
    
    var col_accum = vec3(0.0);
    var accum_tot_w = 0.0;
    var ssgi_conf = 0.0;
    var pt_col_accum = vec3(0.0);
    var pt_accum_tot_w = 0.0;

    var probe = new_probe();
    probe.pos = world_position_offs;
    var probe_proc_color = vec3(0.0);
    
    let closest_motion_vector = prepass_motion_vector(vec4<f32>(screen_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
    let screen_history_uv = screen_uv - closest_motion_vector;
    let history_hit_screen = screen_history_uv.x > 0.0 && screen_history_uv.x < 1.0 && screen_history_uv.y > 0.0 && screen_history_uv.y < 1.0;
    let screen_space_passes_coord = vec2<i32>(tex_dims * screen_history_uv);
    var sample_coord = screen_space_passes_coord;
    var sample_uv = screen_uv;

    let first_probe_pos = load_probe(sample_coord).pos;
    // locate closest probe
    if history_hit_screen {
        var selected_offset = vec2(0, 0);
        var closest = distance(first_probe_pos, world_position_offs);
        probe.pos = first_probe_pos;
        if closest > pixel_radius * 1.5 {
            for (var x = -1; x <= 1; x += 1) {
                for (var y = -1; y <= 1; y += 1) {
                    let offset = vec2(x, y);
                    let coord = screen_space_passes_coord + offset;
                    if any(coord < vec2(0)) || any(coord >= itex_dims) {
                        continue;
                    }
                    let test_probe_pos = textureLoad(fullscreen_passes_read, coord, 2u, 0).xyz;
                    let dist = distance(test_probe_pos, world_position_offs);
                    if dist < closest {
                        selected_offset = offset;
                        closest = dist;
                        probe.pos = test_probe_pos;
                    }
                }
            }
        }

        sample_coord = screen_space_passes_coord + selected_offset;
        sample_uv = (vec2<f32>(sample_coord) + 0.5) / vec2<f32>(tex_dims);
        
        probe_proc_color = textureLoad(fullscreen_passes_read, sample_coord, 4u, 0).xyz;
        probe = load_probe_with_pos(sample_coord, probe.pos);
    }

    

    var max_radius_dist = pixel_radius * 3.0;

    //normal is basically ignored here
    if coplanar(probe.pos, pbr.N, world_position_offs, pbr.N, 0.1, pixel_radius) {
        max_radius_dist *= 8.0;
    }

    probe.reproject_fail = distance(probe.pos, world_position_offs) > max_radius_dist || !history_hit_screen;


    // if closest probe is too far, reset reservoir
    if probe.reproject_fail || probe.M == 0u {
        probe = new_probe();
        probe.pos = world_position_offs;
    }

    // randomly reset position
    if hash_noise(ifrag_coord, globals.frame_count) > 0.9 {
        probe.pos = world_position_offs;
    }

    probe_scale(&probe, MAX_M);

    
    let white_frame_noise = white_frame_noise(3812u);
    let seed = samples + globals.frame_count * samples;
    let urand = fract_blue_noise_for_pixel(ufrag_coord, seed, white_frame_noise);  

    let screen_passes_dims = vec2<f32>(textureDimensions(screen_passes_read).xy);
#ifdef RAY_MARCH
    var st: ScreenTraceResult;
    var candidates_samples = 2u;
    var candidates_radius = 1.5 * ssao_focus;
    let max_dist = 30.0 * pixel_radius * mix(0.5, 1.0, ssao_focus);
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

        let st_ray_start = textureLoad(screen_passes_read, coord, 0u, 0).xyz;
        let probe_to_cand_distance = distance(st_ray_start, probe.pos);

        if probe_to_cand_distance > max_dist {
            // Sample is too far away
            // Doesn't increase M
            // TODO scale by frag depth?
            continue;
        }

        st.pos = textureLoad(screen_passes_read, coord, 1u, 0).xyz;
        st.color = textureLoad(screen_passes_read, coord, 2u, 0).xyz;
        let st_data = textureLoad(screen_passes_read, coord, 3u, 0);
        st.derate = saturate(st_data.x);
        st.hit = bool(u32(st_data.y));
        st.backface = bool(u32(st_data.z));
        let st_distance = distance(st.pos, world_position_offs);
        let st_direction = normalize(st.pos - world_position_offs);

        let hit = st_distance >= 0.0 && st.hit;

#ifdef USE_PATH_TRACED
#ifdef RAY_MARCH
    // if we are filling in PT with SSGI, don't increase M if we missed SSGI
    if !hit {
        continue;
    }
#endif
#endif

        if hit { // hit
            var gr = dot(st.color, vec3<f32>(0.2126, 0.7152, 0.0722));
            let brdf = max(dot(pbr.N, st_direction), 0.0);
            let dist_falloff = (1.0 / (1.0 + st_distance * st_distance));
            var new_weight = gr * dist_falloff * brdf * f32(!st.backface) * f32(st.hit) * st.derate;
            new_weight = max(new_weight, 0.0);

            probe.w_sum += new_weight;

            var threshold = new_weight / w_sum;

            if i == 0u && (probe.M == 0u || ray_hit_pos.x == F32_MAX) {
                threshold = 1.0;
            }

            probe_update(&probe, urand.z, new_weight, st.pos, st.color)

            {
                let probe_dist_falloff = saturate(max_dist - probe_to_cand_distance) + 1.0;
                let offset_w = saturate(1.0 / (length(offset) * candidates_radius)) + 1.0;
                let w = max(dist_falloff * brdf * f32(!st.backface) * f32(st.hit) * st.derate * offset_w * probe_dist_falloff, 0.0);
                col_accum += st.color * w * brdf;
                accum_tot_w += w;
                ssgi_conf += (1.0 / f32(candidates_samples)) * f32(!st.backface) * f32(st.hit);
            }
        }

        M += 1u;
    }
#endif //RAY_MARCH

// for spatial reuse
var s_w_sum = 0.0;
var s_ray_hit_pos = vec3(F32_MAX);
var s_weight = 0.0;
var s_probe_latest_color = vec3(0.0);
var sM = 0u;

#ifdef USE_PATH_TRACED
    { // sample from path traced
        let pt_samples = 7u;
        let retry_coord_samples = 3u;
        var candidates_radius = 7.0 * max(ssao_focus, 0.1);
        let path_trace_image_dims = vec2<f32>(textureDimensions(path_trace_image).xy);
        let pt_max_dist = 200.0 * pixel_radius * mix(0.5, 1.0, ssao_focus);
        for (var i = 0u; i < pt_samples; i+=1u) {
            var max_dist = pt_max_dist;
            var offset_uv = vec2(0.0);
            var offset = vec2(0.0);
            var coord = vec2<i32>((screen_uv + offset_uv) * path_trace_image_dims);
            let seed = (i + 1u) * pt_samples + globals.frame_count;
            let white_frame_noise = white_frame_noise(seed + 28534u);
            let urand = fract_blue_noise_for_pixel(ufrag_coord, i + globals.frame_count, white_frame_noise);  

            for (var j = 0u; j < retry_coord_samples; j+=1u) {
                let seed = (i + 1u) * (j + 1u) * pt_samples + globals.frame_count * pt_samples;
                let urand = fract_blue_noise_for_pixel(ufrag_coord, seed + globals.frame_count, white_frame_noise);  
                offset = (urand.wz * 2.0 - 1.0);
                offset_uv = (offset / path_trace_image_dims) * candidates_radius;
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
            let new_ray_hit_pos = ray_hit.xyz;
            let direction = normalize(new_ray_hit_pos - pt_world_position);
            let dist = ray_hit.w;
            
            // TODO select mip based on path_trace_image_dims relative to tex_dims
            let nor_depth = textureLoad(prepass_downsample, vec2<i32>(tex_dims * (screen_uv + offset_uv)) / 2, 1);

            let probe_to_cand_distance = distance(probe.pos, pt_world_position);

            if coplanar(probe.pos, nor_depth.xyz, world_position_offs, pbr.N, 0.1, pixel_radius * 2.0) {
                max_dist *= 3.0;
            }

            if probe_to_cand_distance > max_dist {
                continue; // Sample is too far away. Doesn't increase M
            }

            let offset_nor_diff = max(dot(nor_depth.xyz, pbr.N) - 0.01, 0.0);

            var gr = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
            let brdf = max(dot(pbr.N, direction), 0.0);
            var dist_falloff = (1.0 / (1.0 + dist * dist));
            if dist >= F16_MAX || dist < 0.0 { // hit sky
                dist_falloff = 1.0;
            }

            let probe_dist_falloff = saturate(max_dist - probe_to_cand_distance);
            
            var new_weight = gr * brdf * dist_falloff * probe_dist_falloff * offset_nor_diff;
            new_weight = max(new_weight, 0.0);
            
            probe_update(&probe, urand.z, new_weight, new_ray_hit_pos, color);
            
        }
    }
#endif //USE_PATH_TRACED



    let hysterisis = select(GI_HYSTERISIS, 1.0, probe.reproject_fail);

    //let ssgi = max(col_accum / accum_tot_w, vec3(0.0));
    //let ptgi = max((pt_col_accum / pt_accum_tot_w), vec3(0.0));
    //var gi = (ssgi + ptgi + resolve_col) / 3.0;

    var gi = probe_resolve(&probe);

#ifdef SSAO
    // GI is too broad for fine corner detail
    gi *= ssao;
#endif

    //probe_proc_color = mix(probe_proc_color, mix(gi, ssgi, saturate(ssgi_conf - 0.1)), hysterisis);
    probe_proc_color = mix(probe_proc_color, gi, hysterisis);

    

    store_probe(probe, ifrag_coord);
    textureStore(fullscreen_passes_write, ifrag_coord, 4u, vec4(probe_proc_color, ssao_focus));
    


#ifdef SSR
    var ssr_hysterisis = 0.1;

    let F0 = get_f0(pbr.material.reflectance, pbr.material.metallic, pbr.material.base_color.rgb);
    let roughness = perceptualRoughnessToRoughness(pbr.material.perceptual_roughness);
    var ssr = bad_ssr(ifrag_coord, pbr.N, pbr.world_position.xyz, roughness, F0, FULL_SSR_SAMPLES, vec2(2.0, 3.0)).rgb;

    
    //let half_ssr = textureLoad(screen_passes_read, vec2<i32>(screen_uv * screen_passes_dims), 4u, 0).rgb;
    //ssr = mix(half_ssr, ssr, 0.5);

    if !probe.reproject_fail {
        let prev_ssr = textureLoad(fullscreen_passes_read, sample_coord, 6u, 0).rgb;
        ssr = mix(prev_ssr, ssr, ssr_hysterisis);
    }

    

    ssr = clamp(ssr, vec3(0.0), vec3(100.0));
    

    textureStore(fullscreen_passes_write, ifrag_coord, 6u, vec4(ssr, 0.0));
#else
    textureStore(fullscreen_passes_write, ifrag_coord, 6u, vec4(0.0));
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
    var st = screentrace(depth_ndc_to_linear(full_screen_frag_coord.z), ufrag_coord, world_position_offs, TBN);

    textureStore(screen_passes_write, ifrag_coord, 0u, vec4(world_position_offs, 0.0));
    textureStore(screen_passes_write, ifrag_coord, 1u, vec4(st.pos, 0.0));
    textureStore(screen_passes_write, ifrag_coord, 2u, vec4(st.color, 0.0));
    textureStore(screen_passes_write, ifrag_coord, 3u, vec4(st.derate, f32(st.hit), f32(st.backface), 0.0));

#ifdef SSR
    var ssr_hysterisis = 0.1;

    let F0 = get_f0(pbr.material.reflectance, pbr.material.metallic, pbr.material.base_color.rgb);
    let roughness = perceptualRoughnessToRoughness(pbr.material.perceptual_roughness);
    var ssr = bad_ssr(ifrag_coord, pbr.N, pbr.world_position.xyz, roughness, F0, SSR_SAMPLES, vec2(2.0, 3.0)).rgb;

    ssr = clamp(ssr, vec3(0.0), vec3(100.0));

    textureStore(screen_passes_write, ifrag_coord, 4u, vec4(ssr, 0.0));
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

    

    var color_distance = 0.0;

    


    var probe = load_probe(location);

#ifdef FILTER_SSGI
    let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 0.0);

    
    let pixel_radius = pixel_radius_to_world(1.0, depth_ndc_to_linear(nor_depth.w), projection_is_orthographic());

    let world_position = position_ndc_to_world(vec3(uv_to_ndc(screen_uv), nor_depth.w));
    let white_frame_noise = white_frame_noise(8462u);
    let urand = fract_blue_noise_for_pixel(ulocation, globals.frame_count, white_frame_noise) * 2.0 - 1.0;
    let center = textureLoad(fullscreen_passes_read, location, 4u, 0);
    let center_color = center.rgb;
    let ssao_focus = center.w;
    let dist_factor = 50.0 * pixel_radius;
    var tot = vec3(0.0);
    var tot_w = 0.0;
    let samples = 6u;
    var range = mix(9.0, 14.0, urand.z) * ssao_focus;
    let center_weight = 0.1;
    tot += center_color * center_weight;
    tot_w += center_weight;
    for (var i = 0u; i <= samples; i+=1u) {
        let seed = i * samples + globals.frame_count * samples;
        let white_frame_noise = white_frame_noise(seed + 8462u);
        let urand = fract_blue_noise_for_pixel(ulocation, seed, white_frame_noise) * 2.0 - 1.0;
        let px_pos = vec2(urand.x * range, urand.y * range);
        let coord = location + vec2<i32>(px_pos);
        if all(coord < 0) || all(coord >= size) {
            continue;
        }
        let uv = vec2<f32>(coord) / fsize;
        let px_dist = length(abs(urand.xy));
        var w = px_dist;
        let nd = textureLoad(prepass_downsample, vec2<i32>(uv * fprepass_size), 0);
        let col = textureLoad(fullscreen_passes_read, coord, 4u, 0);
    
        let d = max(dot(nor_depth.xyz, nd.xyz) + 0.001, 0.0);
        w = w * d * d * d; //lol
        let px_ws = position_ndc_to_world(vec3(uv_to_ndc(uv), nd.w));
        let dist = distance(px_ws, world_position);
        w = w * (1.0 - saturate(dist / dist_factor));
        w = max(w, 0.0);
        tot += col.rgb * w;
        tot_w += w;
    }
    let final_color = tot/tot_w;
    color_distance = distance(center_color, final_color);
#ifdef APPLY_FILTER_TO_OUTPUT
    textureStore(fullscreen_passes_write, location, 4u, vec4(final_color, 1.0));
#else
    textureStore(fullscreen_passes_write, location, 4u, textureLoad(fullscreen_passes_read, location, 4u, 0));
#endif // APPLY_FILTER_TO_OUTPUT
#else
    textureStore(fullscreen_passes_write, location, 4u, textureLoad(fullscreen_passes_read, location, 4u, 0));
#endif // FILTER_SSGI


    textureStore(screen_passes_write, location, 0u, textureLoad(screen_passes_read, location, 0u, 0));
    textureStore(screen_passes_write, location, 1u, textureLoad(screen_passes_read, location, 1u, 0));
    textureStore(screen_passes_write, location, 2u, textureLoad(screen_passes_read, location, 2u, 0));
    textureStore(screen_passes_write, location, 3u, textureLoad(screen_passes_read, location, 3u, 0));
    textureStore(screen_passes_write, location, 4u, textureLoad(screen_passes_read, location, 4u, 0));

    

    textureStore(fullscreen_passes_write, location, 2u, textureLoad(fullscreen_passes_read, location, 2u, 0));
    textureStore(fullscreen_passes_write, location, 3u, textureLoad(fullscreen_passes_read, location, 3u, 0));

    textureStore(fullscreen_passes_write, location, 5u, textureLoad(fullscreen_passes_read, location, 5u, 0));
    textureStore(fullscreen_passes_write, location, 6u, textureLoad(fullscreen_passes_read, location, 6u, 0));

    


#ifdef RESTIR_ANTI_CLUMPING
    var same_pos = 0u;
    var same_wdata = 0u;

    for (var x = -1; x <= 1; x += 1) {
        for (var y = -1; y <= 1; y += 1) {
            let offset = vec2(x, y);
            if all(offset == vec2(0, 0)) {
                continue;
            }
            let s_probe = load_probe(location + offset);
            if all(probe.ray_hit_pos == s_probe.ray_hit_pos) {
                same_pos += 1u;
            }
            if probe.weight == s_probe.weight && probe.w_sum == s_probe.w_sum  {
                same_wdata += 1u;
            }
        }
    }
    
    let noise = fract(blue_noise_for_pixel(ulocation, globals.frame_count) + hash_noise(vec2(0, 0), globals.frame_count + 4275u));

    if same_pos > 0u || color_distance > 1.0 {
        let derate = mix(f32(same_wdata + same_pos), color_distance, 0.85);
        let reset_max = u32(f32(MAX_M) * mix(0.3, 0.6, noise / derate));
        probe_scale(&probe, reset_max);
    }
#endif // RESTIR_ANTI_CLUMPING
    store_probe(probe, location);
}
