fn bad_ssr(frag_coord: vec4<f32>, surface_normal: vec3<f32>, world_position: vec3<f32>, roughness: f32, albedo: vec3<f32>, sample_index: u32) -> vec4<f32> {
    let samples = 2u;
    let surface_normal = normalize(surface_normal);
    let raymarch_distance = 20.0;
    let linear_sample_count = 16u;

    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);
    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));
    let screen_uv = frag_coord.xy / depth_tex_dims;
    let depth = frag_coord.z;
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



    //    let urand = vec3(
    //        hash_noise(ifrag_coord, globals.frame_count), 
    //        hash_noise(ifrag_coord, globals.frame_count + 1024u),
    //        hash_noise(ifrag_coord, globals.frame_count + 2048u),
    //    );   

        var wo = V;

        // Get a good quality sample from the BRDF, using VNDF
        let brdf_sample = brdf_sample(roughness, albedo, tangent_to_world * wo, urand.xy);
        let trace_dir_ws = brdf_sample.wi * tangent_to_world;

        //direction = normalize(surface_normal + direction);
        var dmr = DepthRayMarch_new_from_depth(depth_tex_dims);   
        dmr.ray_start_cs = vec3(uv_to_cs(screen_uv), depth);
        dmr = to_ws(dmr, world_position + trace_dir_ws * raymarch_distance);
        //dmr = to_ws_dir(dmr, direction);
        dmr.linear_steps = linear_sample_count;
        dmr.depth_thickness_linear_z = raymarch_distance / f32(linear_sample_count);
        dmr.jitter = urand.z;//
        dmr.march_behind_surfaces = false;
        dmr.use_secant = true;
        dmr.bisection_steps = 4u;

        let raymarch_result = march(dmr, sample_index);
        var contribution = vec3(0.0);
        if (raymarch_result.hit) {
            let hit_n = normalize(prepass_normal(vec4(raymarch_result.hit_uv * depth_tex_dims, 0.0, 0.0), sample_index));
            let backface = dot(hit_n, trace_dir_ws);
            if backface < 0.01 {
                let closest_motion_vector = prepass_motion_vector(vec4<f32>(raymarch_result.hit_uv * depth_tex_dims, 0.0, 0.0), sample_index).xy;
                let history_uv = raymarch_result.hit_uv - closest_motion_vector;
                if history_uv.x > 0.0 && history_uv.x < 1.0 && history_uv.y > 0.0 && history_uv.y < 1.0 {
                    contribution = textureSampleLevel(prev_frame_tex, prev_frame_sampler, history_uv, 0.0).rgb;
                }
            }
        }
        tot += contribution * brdf_sample.value_over_pdf;
    }
    tot /= f32(samples);

    return vec4(tot, 1.0);
}