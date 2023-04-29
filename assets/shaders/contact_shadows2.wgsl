fn contact_shadow2(frag_coord: vec2<f32>, dir_to_light: vec3<f32>, surface_normal: vec3<f32>, sample_index: u32) -> vec4<f32> {
    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));
    let screen_uv = frag_coord / depth_tex_dims;

    let depth = get_depth(screen_uv, sample_index);
    let ray_hit_ws = position_from_uv(screen_uv, depth); // + normal_bias_offset
    
    var rmr = DepthRayMarch_new_from_depth(depth_tex_dims);   
    // March from a clip-space position (w = 1)
    // clip-space coord of the start pixel
    rmr.ray_start_cs = vec3(uv_to_cs(screen_uv), depth);
    rmr = to_ws(rmr, ray_hit_ws + dir_to_light * 0.18);
    //rmr = to_ws_dir(rmr, dir_to_light);
    rmr.linear_steps = 8u;
    rmr.depth_thickness_linear_z = 0.18;
    rmr.jitter = 0.0; //interleaved_gradient_noise(frag_coord, globals.frame_count);
    rmr.march_behind_surfaces = true;
    let raymarch_result = march(rmr, sample_index);
    var shadow = 0.0;
    if (raymarch_result.hit) {
        shadow = smoothstep(1.0, 0.5, raymarch_result.hit_penetration_frac);
    }
    return vec4(vec3(f32(1.0 - shadow)), 1.0);
}