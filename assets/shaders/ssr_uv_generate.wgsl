fn ssr_uv_generate(ifrag_coord: vec2<i32>, surface_normal: vec3<f32>, world_position: vec3<f32>) -> vec4<f32> {

    let surface_normal = normalize(surface_normal);
    let raymarch_distance = 20.0;
    let linear_sample_count = 32u;

    let ufrag_coord = vec2<u32>(ifrag_coord.xy);
    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));
    let ray_start_ndc = position_world_to_ndc(world_position);
    var V = normalize(view.world_position.xyz - world_position);

    var res_uv = vec2(-1.0); //-1 for miss
    let trace_dir_ws = reflect(-V, surface_normal);
    
    var dmr = DepthRayMarch_new_from_depth(depth_tex_dims);   
    dmr.ray_start_cs = ray_start_ndc;
    dmr = to_ws(dmr, world_position + trace_dir_ws * raymarch_distance);
    dmr.linear_steps = linear_sample_count;
    dmr.depth_thickness_linear_z = raymarch_distance / f32(linear_sample_count);
    dmr.jitter = blue_noise_for_pixel(ufrag_coord, globals.frame_count);
    dmr.march_behind_surfaces = false;
    dmr.use_secant = true;
    dmr.bisection_steps = 4u;
    dmr.mip_min_max = vec2(2.0, 3.0);

    let raymarch_result = march(dmr, 0u);
    var contribution = vec3(0.0);
    if (raymarch_result.hit) {
        let hit_n = normalize(prepass_normal(vec4(raymarch_result.hit_uv * view.viewport.zw, 0.0, 0.0), 0u));
        let backface = dot(hit_n, trace_dir_ws);
        if backface < 0.01 {
            res_uv = raymarch_result.hit_uv;
        } else {
            res_uv = vec2(-2.0); //-2 for hit backface
        }
    }

    return vec4(res_uv, 0.0, 1.0);
}
