
fn new_drm_for_ssgi() -> DepthRayMarch {
    var dmr = DepthRayMarch_new_from_depth(view.viewport.zw);
    dmr.linear_steps = 16u;
    dmr.depth_thickness_linear_z = 1.5;
    dmr.march_behind_surfaces = false;
    dmr.use_secant = true;
    dmr.bisection_steps = 3u;
    dmr.use_bilinear = false;
    dmr.mip_min_max = vec2(0.0, 3.0);
    return dmr;
}

fn bad_ssgi(ifrag_coord: vec2<i32>, surface_normal: vec3<f32>, world_position: vec3<f32>, samples: u32) -> vec4<f32> {
    let trace_dist = 16.0;


    let surface_normal = normalize(surface_normal);
    let ufrag_coord = vec2<u32>(ifrag_coord.xy);
    // TODO surface_normal * 0.01 is because of lines from using really low mip for depth
    let ray_start_ndc = position_world_to_ndc(world_position + surface_normal * 0.01);

    let TBN = build_orthonormal_basis(surface_normal);
    var tot = vec3(0.0);

    var dmr = new_drm_for_ssgi();
    dmr.ray_start_cs = ray_start_ndc;

    for (var i = 0u; i < samples; i += 1u) {
        let white_frame_noise = white_frame_noise(8492u + i);
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
        
        let raymarch_result = march(dmr, 0u);
        var shadow = 0.0;
        if (raymarch_result.hit) {
            let hit_n = normalize(textureSampleLevel(prepass_downsample, linear_sampler, raymarch_result.hit_uv, 1.0).xyz);
            let backface = dot(hit_n, direction);
            if backface < 0.01 {
                let closest_motion_vector = prepass_motion_vector(vec4<f32>(raymarch_result.hit_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
                let history_uv = raymarch_result.hit_uv - closest_motion_vector;
                if history_uv.x > 0.0 && history_uv.x < 1.0 && history_uv.y > 0.0 && history_uv.y < 1.0 {
                    //let pt_image = vec3(textureSampleLevel(pathtrace_tex, pathtrace_samp, raymarch_result.hit_uv, 0.0).rgb);
                    let prev_frame = textureSampleLevel(prev_frame_tex, linear_sampler, history_uv, 2.0).rgb;
                    //tot += mix(prev_frame, pt_image, raymarch_result.hit_t);
                    tot += prev_frame;
                    //tot += pt_image;
                }
            }
        }
    }
    tot /= f32(samples);

    return vec4(vec3(tot), 1.0);
}

