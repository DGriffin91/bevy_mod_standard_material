fn bad_ssr(ifrag_coord: vec2<i32>, surface_normal: vec3<f32>, world_position: vec3<f32>, roughness: f32, F0: vec3<f32>, samples: u32, mip_min_max: vec2<f32>) -> vec4<f32> {
    let roughness_clamped = roughness;//clamp(roughness, 0.0, 0.005);
    let lod_rough = clamp(pow(roughness, 0.25) * 6.0, 1.0, 4.0);

    let surface_normal = normalize(surface_normal);
    let raymarch_distance = 20.0;
    let linear_sample_count = 24u;

    let ufrag_coord = vec2<u32>(ifrag_coord.xy);
    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));
    let ray_start_ndc = position_world_to_ndc(world_position);
    var V = normalize(view.world_position.xyz - world_position);

    // Build a basis for sampling the BRDF, as BRDF sampling functions assume that the normal faces +Z.
    let tangent_to_world = build_orthonormal_basis(surface_normal);

    let white_frame_noise = vec3(
        hash_noise(vec2(0), globals.frame_count), 
        hash_noise(vec2(1), globals.frame_count + 1024u),
        hash_noise(vec2(2), globals.frame_count + 2048u)
    );

    var tot = vec3(0.0);
    for (var i = 0u; i < samples; i += 1u) {
    

        let urand = vec3(
            fract(blue_noise_for_pixel(ufrag_coord, i * samples + 0u + globals.frame_count) + white_frame_noise.x),
            fract(blue_noise_for_pixel(ufrag_coord, i * samples + 1u + globals.frame_count) + white_frame_noise.y),
            fract(blue_noise_for_pixel(ufrag_coord, i * samples + 2u + globals.frame_count) + white_frame_noise.z),
        );   
    
        var wo = V;

        // Get a good quality sample from the BRDF, using VNDF
        let brdf_sample = brdf_sample(roughness_clamped, F0, tangent_to_world * wo, urand.xy);
        let trace_dir_ws = brdf_sample.wi * tangent_to_world;

        //direction = normalize(surface_normal + direction);
        var dmr = DepthRayMarch_new_from_depth(depth_tex_dims);   
        dmr.ray_start_cs = ray_start_ndc;
        dmr = to_ws(dmr, world_position + trace_dir_ws * raymarch_distance);
        //dmr = to_ws_dir(dmr, direction);
        dmr.linear_steps = linear_sample_count;
        dmr.depth_thickness_linear_z = raymarch_distance / f32(linear_sample_count);
        dmr.jitter = urand.z;//
        dmr.march_behind_surfaces = false;
        dmr.use_secant = true;
        dmr.use_bilinear = false;
        dmr.bisection_steps = 4u;
        dmr.mip_min_max = mip_min_max;

        let raymarch_result = march(dmr, 0u);
        var contribution = vec3(0.0);
        if (raymarch_result.hit) {
            //let hit_n = normalize(prepass_normal(vec4(raymarch_result.hit_uv * view.viewport.zw, 0.0, 0.0), 0u));
            let hit_n = normalize(textureSampleLevel(prepass_downsample, linear_sampler, raymarch_result.hit_uv, 3.0).xyz);
            let backface = dot(hit_n, trace_dir_ws);
            if backface < 0.01 {
                let closest_motion_vector = prepass_motion_vector(vec4<f32>(raymarch_result.hit_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
                let history_uv = raymarch_result.hit_uv - closest_motion_vector;
                if history_uv.x > 0.0 && history_uv.x < 1.0 && history_uv.y > 0.0 && history_uv.y < 1.0 {
                    //contribution = textureSampleLevel(prev_frame_tex, prev_frame_sampler, history_uv, 0.0).rgb;
                    
                    //let pt_image = vec3(textureSampleLevel(pathtrace_tex, pathtrace_samp, screen_uv, 0.0).rgb);
                    let prev_frame1 = textureSampleLevel(prev_frame_tex, linear_sampler, history_uv, lod_rough).rgb;
                    contribution += prev_frame1;//prev_frame;//mix(prev_frame, pt_image, raymarch_result.hit_t);
                }
            }
        }
        tot += contribution * brdf_sample.value_over_pdf;
    }
    tot /= f32(samples);

    return vec4(tot, 1.0);
}

fn interpolate_colors(value: f32, pos1: f32, color1: vec3<f32>, pos2: f32, color2: vec3<f32>, pos3: f32, color3: vec3<f32>) -> vec3<f32> {
    let t1 = smoothstep(pos1, pos2, value);
    let t2 = smoothstep(pos2, pos3, value);

    let mid = mix(color1, color2, t1);
    let end = mix(color2, color3, t2);

    return mix(mid, end, t2);
}