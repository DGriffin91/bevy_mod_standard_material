fn bad_ssgi(frag_coord: vec2<f32>, surface_normal: vec3<f32>, sample_index: u32) -> vec4<f32> {
    let ifrag_coord = vec2<i32>(frag_coord);
    let ufrag_coord = vec2<u32>(frag_coord);
    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));
    let screen_uv = frag_coord / depth_tex_dims;
    let depth = get_depth(screen_uv, sample_index);
    let ray_hit_ws = position_from_uv(screen_uv, depth); // + normal_bias_offset

    var tot = vec3(0.0);
    for (var i = 0u; i < 4u; i += 1u) {
        //var direction = uniform_sample_sphere(
        //    fract(blue_noise_for_pixel(ufrag_coord, globals.frame_count) + r2_sequence(i))
        //);
        var direction = uniform_sample_sphere(vec2(
            hash_noise(ifrag_coord, i + globals.frame_count),
            hash_noise(ifrag_coord, i + 64u + globals.frame_count * 64u)
        ));
        direction = normalize(surface_normal + direction);
        var dmr = DepthRayMarch_new_from_depth(depth_tex_dims);   
        dmr.ray_start_cs = vec3(uv_to_cs(screen_uv), depth);
        dmr = to_ws(dmr, ray_hit_ws + direction * 16.0);
        //dmr = to_ws_dir(dmr, direction);
        dmr.linear_steps = 24u;
        dmr.depth_thickness_linear_z = 1.5;
        dmr.jitter = interleaved_gradient_noise(frag_coord, globals.frame_count);
        dmr.march_behind_surfaces = true;
        //dmr.use_secant = true;
        //dmr.bisection_steps = 4u;
        
        let raymarch_result = march(dmr, sample_index);
        var shadow = 0.0;
        if (raymarch_result.hit) {
            let d = dot(surface_normal, direction);
            tot += textureSampleLevel(prev_frame_tex, prev_frame_sampler, raymarch_result.hit_uv, 0.0).rgb;
            // * d * (1.0 - raymarch_result.hit_t) ?
        }
    }
    tot /= 4.0;

    return vec4(tot, 1.0);
}