const PHI = 1.618033988749895; // Golden Ratio

fn randomDirection(randomSeed: f32) -> vec3<f32> {
    // Scale randomSeed to a value between 0 and 1
    let scaledRandomSeed = fract(randomSeed);

    // Calculate the inclination angle (theta) and azimuth angle (phi)
    let theta = acos(1.0 - 2.0 * scaledRandomSeed);
    let phi = 2.0 * PI * PHI * scaledRandomSeed;

    // Convert spherical coordinates to Cartesian coordinates
    var direction: vec3<f32>;
    direction.x = sin(theta) * cos(phi);
    direction.y = sin(theta) * sin(phi);
    direction.z = cos(theta);

    return direction;
}

fn bad_ssao(frag_coord: vec2<f32>, surface_normal: vec3<f32>, sample_index: u32) -> vec4<f32> {
    let ifrag_coord = vec2<i32>(frag_coord);
    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));
    let screen_uv = frag_coord / depth_tex_dims;
    let depth = get_depth(screen_uv, sample_index);
    let ray_hit_ws = position_from_uv(screen_uv, depth); // + normal_bias_offset

    var tot = 0.0;
    var n_seed = 0u;
    for (var i = 0u; i < 12u; i += 1u) {
        //let noise = interleaved_gradient_noise(frag_coord, i);
        //let r2 = fract(0.5 + noise * 6.0 * vec2<f32>(0.75487766624669276005, 0.5698402909980532659114));
        //var direction = vec3(noise, r2.x, r2.y);
        //var direction = randomDirection(noise);
        var direction = vec3(
            interleaved_gradient_noise(frag_coord, n_seed * 0u), 
            interleaved_gradient_noise(frag_coord, n_seed * 1u), 
            interleaved_gradient_noise(frag_coord, n_seed * 2u)
        );
        direction = direction * 2.0 - 1.0;
        direction = dot(direction, surface_normal) * direction;
        var rmr = DepthRayMarch_new_from_depth(depth_tex_dims);   
        rmr.ray_start_cs = vec3(uv_to_cs(screen_uv), depth);
        rmr = to_ws(rmr, ray_hit_ws + direction * 1.0);
        rmr.linear_steps = 6u;
        rmr.depth_thickness_linear_z = 1.0;
        rmr.jitter = 0.5; //interleaved_gradient_noise(frag_coord, globals.frame_count);
        rmr.march_behind_surfaces = true;
        let raymarch_result = march(rmr, sample_index);
        var shadow = 0.0;
        if (raymarch_result.hit) {
            tot += smoothstep(1.0, 0.0, raymarch_result.hit_penetration_frac);
        }
        n_seed += 3u;
    }
    tot /= 12.0;

    return vec4(vec3(1.0 - tot), 1.0);
}