#import "shaders/screen_space_passes/common.wgsl"

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

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    //if true {return;}
    let ufrag_coord = invocation_id.xy;
    let ifrag_coord = vec2<i32>(ufrag_coord);
    var frag_coord = vec4(vec2<f32>(ufrag_coord), 0.0, 0.0);

    let target_dims = vec2<i32>(textureDimensions(screen_passes_write).xy);
    if any(ifrag_coord >= target_dims) {
        return;
    }
    
    let ftarget_dims = vec2<f32>(target_dims);
    let frag_size = 1.0 / ftarget_dims;
    var screen_uv = frag_coord.xy / ftarget_dims + frag_size * 0.5;



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

    let F0 = get_f0(pbr.material.reflectance, pbr.material.metallic, pbr.material.base_color.rgb);
    let roughness = perceptualRoughnessToRoughness(pbr.material.perceptual_roughness);
    var ssr = bad_ssr(ifrag_coord, pbr.N, pbr.world_position.xyz, roughness, F0, SSR_SAMPLES, vec2(2.0, 3.0)).rgb;

    ssr = clamp(ssr, vec3(0.0), vec3(100.0));

    textureStore(screen_passes_write, ifrag_coord, 4u, vec4(ssr, 0.0));
#endif //SSR
}