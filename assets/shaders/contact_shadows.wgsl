fn contact_shadow(frag_coord: vec4<f32>, dir_to_light: vec3<f32>, surface_normal: vec3<f32>, sample_index: u32) -> vec4<f32> {
    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));
    let screen_uv = frag_coord_to_uv(frag_coord.xy);
    let ray_start_ndc = frag_coord_to_ndc(frag_coord);

    var distance = 0.15;

    let ray_hit_ws = position_ndc_to_world(ray_start_ndc);

    var dmr = DepthRayMarch_new_from_depth(depth_tex_dims);   
    // March from a clip-space position (w = 1)
    // clip-space coord of the start pixel
    dmr.ray_start_cs = ray_start_ndc;
    dmr = to_ws(dmr, ray_hit_ws + dir_to_light * distance);
    dmr.linear_steps = 6u;
    dmr.depth_thickness_linear_z = 0.5;
    dmr.jitter = interleaved_gradient_noise(frag_coord.xy, globals.frame_count);
    dmr.march_behind_surfaces = true;
    let raymarch_result = march(dmr, sample_index);
    var shadow = 0.0;
    if (raymarch_result.hit) {
        shadow = smoothstep(1.0, 0.5, raymarch_result.hit_penetration_frac);
    }
    return vec4(vec3(f32(1.0 - shadow)), 1.0);
}