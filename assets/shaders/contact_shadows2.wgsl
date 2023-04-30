fn contact_shadow2(frag_coord: vec2<f32>, dir_to_light: vec3<f32>, surface_normal: vec3<f32>, sample_index: u32) -> vec4<f32> {
    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));
    let screen_uv = frag_coord / depth_tex_dims;

    var distance = 0.18;

    let depth = get_depth(screen_uv, sample_index);
    let ray_hit_ws = position_from_uv(screen_uv, depth); // + normal_bias_offset

    //let linear_depth = .1 / depth;
    //distance = mix(0.18, 0.05, saturate(linear_depth / 10.0));
    
    var dmr = DepthRayMarch_new_from_depth(depth_tex_dims);   
    // March from a clip-space position (w = 1)
    // clip-space coord of the start pixel
    dmr.ray_start_cs = vec3(uv_to_cs(screen_uv), depth);
    dmr = to_ws(dmr, ray_hit_ws + dir_to_light * distance);
    //dmr = to_ws_dir(dmr, dir_to_light);
    dmr.linear_steps = 5u;
    dmr.depth_thickness_linear_z = 0.18;
    dmr.jitter = 1.0;
    dmr.march_behind_surfaces = true;
    let raymarch_result = march(dmr, sample_index);
    var shadow = 0.0;
    if (raymarch_result.hit) {
        shadow = smoothstep(1.0, 0.5, raymarch_result.hit_penetration_frac);
    }
    return vec4(vec3(f32(1.0 - shadow)), 1.0);
    //return vec4(vec3(f32(depth)), 1.0);
}