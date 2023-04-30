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

    var v = 0.0;
    var sample = 0u;
    for (var i = 0u; i < 32u; i += 1u) {
        //var direction = uniform_sample_sphere(
        //    fract(blue_noise_for_pixel_simple(ufrag_coord) + r2_sequence(0u))
        //);
        var direction = uniform_sample_sphere(vec2(
            blue_noise_for_pixel(ufrag_coord, sample),
            blue_noise_for_pixel(ufrag_coord, sample+1u),
        ));
        //var direction = uniform_sample_sphere(
        //    fract(hash_noise(ifrag_coord, 0u) + r2_sequence(0u))
        //);
        direction = normalize(surface_normal + direction);

        v += dot(direction, surface_normal);
        sample += 2u;
    }
    v /= 32.0;


    
    let dots = f32(v > 0.75);

    return vec4(vec3(dots), 1.0);
}

fn bad_ssao(frag_coord: vec2<f32>, surface_normal: vec3<f32>, sample_index: u32) -> vec4<f32> {
    let ifrag_coord = vec2<i32>(frag_coord);
    let ufrag_coord = vec2<u32>(frag_coord);
    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));
    let screen_uv = frag_coord / depth_tex_dims;
    let depth = get_depth(screen_uv, sample_index);
    let ray_hit_ws = position_from_uv(screen_uv, depth); // + normal_bias_offset

    var tot = 0.0;
    for (var i = 0u; i < 4u; i += 1u) {
        var direction = uniform_sample_sphere(
            fract(blue_noise_for_pixel(ufrag_coord, 0u) + r2_sequence(i))
        );
        //var direction = uniform_sample_sphere(
        //    fract(interleaved_gradient_noise(frag_coord, 0u) + r2_sequence(i))
        //);
        direction = normalize(surface_normal + direction);
        var dmr = DepthRayMarch_new_from_depth(depth_tex_dims);   
        dmr.ray_start_cs = vec3(uv_to_cs(screen_uv), depth);
        dmr = to_ws(dmr, ray_hit_ws + direction * 1.0);
        //dmr = to_ws_dir(dmr, direction);
        dmr.linear_steps = 4u;
        dmr.depth_thickness_linear_z = 0.8;
        dmr.jitter = 1.0; //interleaved_gradient_noise(frag_coord, globals.frame_count);
        dmr.march_behind_surfaces = true;
        let raymarch_result = march(dmr, sample_index);
        var shadow = 0.0;
        if (raymarch_result.hit) {
            tot += smoothstep(1.0, 0.0, raymarch_result.hit_t);
        }
    }
    tot /= 4.0;

    return vec4(vec3(1.0 - tot), 1.0);
}