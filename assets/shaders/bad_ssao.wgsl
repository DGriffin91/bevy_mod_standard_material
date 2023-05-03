fn noise_test(frag_coord: vec2<f32>, surface_normal: vec3<f32>, sample_index: u32) -> vec4<f32> {
    let ifrag_coord = vec2<i32>(frag_coord);
    let ufrag_coord = vec2<u32>(frag_coord);

//    var direction = uniform_sample_sphere(vec2(
//        hash_noise(ifrag_coord, 0u),
//        hash_noise(ifrag_coord, 1u),
//    ));

//    var direction = uniform_sample_sphere(vec2(
//        blue_noise_for_pixel_r2(ufrag_coord, 0u),
//        blue_noise_for_pixel_r2(ufrag_coord, 1u),
//    ));

    let TBN = build_orthonormal_basis(surface_normal);
    var v = 0.0;
    var sample = 0u;
    for (var i = 0u; i < 1u; i += 1u) {
        //var direction = uniform_sample_sphere(
        //    fract(blue_noise_for_pixel_simple(ufrag_coord) + r2_sequence(0u))
        //);
        //var direction = uniform_sample_sphere(vec2(
        //    blue_noise_for_pixel(ufrag_coord, sample),
        //    blue_noise_for_pixel(ufrag_coord, sample+1u),
        //));
        var direction = cosine_sample_hemisphere(vec2(
            blue_noise_for_pixel(ufrag_coord, sample),
            blue_noise_for_pixel(ufrag_coord, sample+1u),
        ));

        direction = direction * TBN;
        //var direction = uniform_sample_sphere(
        //    fract(hash_noise(ifrag_coord, 0u) + r2_sequence(0u))
        //);
        //direction = normalize(surface_normal + direction);

        v += dot(direction, surface_normal);
        sample += 2u;
    }
    v /= 1.0;


    
    let dots = f32(v > 0.99);

    return vec4(vec3(dots), 1.0);
}

fn bad_ssao(frag_coord: vec2<f32>, surface_normal: vec3<f32>, sample_index: u32) -> vec4<f32> {
    let samples = 5u;
    let linear_steps = 4u;
    let depth_thickness = 0.6;
    let trace_dist = 0.7;
    let dist_jitter_amount = 0.2; // stay below 1.0

    let ifrag_coord = vec2<i32>(frag_coord);
    let ufrag_coord = vec2<u32>(frag_coord);
    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));
    let screen_uv = frag_coord / depth_tex_dims;
    // TODO just use in.world_position
    let depth = get_depth(screen_uv, sample_index);
    let ray_hit_ws = position_from_uv(screen_uv, depth); // + normal_bias_offset

    
    var dmr = DepthRayMarch_new_from_depth(depth_tex_dims);
    dmr.ray_start_cs = vec3(uv_to_cs(screen_uv), depth);
    dmr.linear_steps = linear_steps;
    dmr.march_behind_surfaces = true;
    dmr.use_secant = false;
    dmr.bisection_steps = 0u;
    dmr.use_bilinear = false;

    let white_frame_noise = vec4(
        hash_noise(vec2(0), globals.frame_count + 0u), 
        hash_noise(vec2(1), globals.frame_count + 1u),
        hash_noise(vec2(2), globals.frame_count + 2u),
        hash_noise(vec2(3), globals.frame_count + 3u)
    );
    let TBN = build_orthonormal_basis(surface_normal);

    var tot = 0.0;
    for (var i = 0u; i < samples; i += 1u) {
        let seed = i * samples + globals.frame_count * samples;
        
        var direction = cosine_sample_hemisphere(vec2(
            fract(blue_noise_for_pixel(ufrag_coord, seed + 0u) + white_frame_noise.x),
            fract(blue_noise_for_pixel(ufrag_coord, seed + 1u) + white_frame_noise.y),
        ));
        let jitter = fract(blue_noise_for_pixel(ufrag_coord, seed + 2u) + white_frame_noise.z);

        var dist_jitter = fract(blue_noise_for_pixel(ufrag_coord, seed + 3u) + white_frame_noise.w);
        dist_jitter = mix(dist_jitter, 1.0, dist_jitter_amount);

        dmr.depth_thickness_linear_z = depth_thickness * dist_jitter;

        direction = normalize(direction * TBN);
        dmr = to_ws(dmr, ray_hit_ws + direction * trace_dist * dist_jitter);
        dmr.jitter = jitter;
        let raymarch_result = march(dmr, sample_index);
        var shadow = 0.0;
        if (raymarch_result.hit) {
            tot += pow(mix(1.0, 0.0, raymarch_result.hit_t), 1.0 - dist_jitter) * 0.9;
        }
    }
    tot /= f32(samples);

    return vec4(vec3(1.0 - tot), 1.0);
}