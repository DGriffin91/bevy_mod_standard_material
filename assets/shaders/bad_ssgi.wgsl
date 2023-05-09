
fn bad_ssgi(frag_coord: vec4<f32>, surface_normal: vec3<f32>, world_position: vec3<f32>, sample_index: u32) -> vec4<f32> {
    let samples = 1u;
    let linear_steps = 8u;
    let bisection_steps = 4u;
    let depth_thickness = 1.5;
    let trace_dist = 8.0;


    let surface_normal = normalize(surface_normal);
    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);
    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));
    let screen_uv = frag_coord_to_uv(frag_coord.xy);
    let ray_start_ndc = frag_coord_to_ndc(frag_coord);

    let TBN = build_orthonormal_basis(surface_normal);
    var tot = vec3(0.0);

    let white_frame_noise = white_frame_noise(8492u);

    var dmr = DepthRayMarch_new_from_depth(depth_tex_dims);
    dmr.ray_start_cs = ray_start_ndc;
    dmr.linear_steps = linear_steps;
    dmr.depth_thickness_linear_z = depth_thickness;
    dmr.march_behind_surfaces = false;
    dmr.use_secant = true;
    dmr.bisection_steps = bisection_steps;
    dmr.use_bilinear = false;

    for (var i = 0u; i < samples; i += 1u) {
        let seed = i * samples + globals.frame_count * samples;

        let urand = fract_blue_noise_for_pixel(ufrag_coord, seed, white_frame_noise);  
        //let urand = fract_white_noise_for_pixel(ifrag_coord, seed, white_frame_noise); 

        var direction = cosine_sample_hemisphere(urand.xy);

        let jitter = fract(blue_noise_for_pixel(ufrag_coord, seed + 2u) + white_frame_noise.z);
        //let jitter = hash_noise(ifrag_coord, seed + 2u);
        
        direction = normalize(direction * TBN);

        let ray_end_ws = world_position + direction * trace_dist;

        dmr = to_ws(dmr, ray_end_ws);
        dmr.jitter = jitter;
        
        let raymarch_result = march(dmr, sample_index);
        var shadow = 0.0;
        if (raymarch_result.hit) {
            let hit_n = normalize(prepass_normal(vec4<f32>(raymarch_result.hit_uv * depth_tex_dims, 0.0, 0.0), sample_index));
            let backface = dot(hit_n, direction);
            if backface < 0.01 {
                
                let closest_motion_vector = prepass_motion_vector(vec4<f32>(raymarch_result.hit_uv * depth_tex_dims, 0.0, 0.0), sample_index).xy;
                let history_uv = raymarch_result.hit_uv - closest_motion_vector;
                if history_uv.x > 0.0 && history_uv.x < 1.0 && history_uv.y > 0.0 && history_uv.y < 1.0 {
                    tot += textureSampleLevel(prev_frame_tex, prev_frame_sampler, history_uv, 0.0).rgb;
                }
            }
        }
    }
    tot /= f32(samples);

    return vec4(vec3(tot), 1.0);
}