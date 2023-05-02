// A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel Rendering
// Alexander Majercik, Cyril Crassin, Peter Shirley, and Morgan McGuire
fn slabs(origin: vec3<f32>, direction: vec3<f32>, minv: vec3<f32>, maxv: vec3<f32>) -> bool {
    let t0 = (minv - origin) / direction;
    let t1 = (maxv - origin) / direction;

    let tmin = min(t0, t1);
    let tmax = max(t0, t1);

    return max(tmin.x, max(tmin.y, tmin.z)) <= min(tmax.x, min(tmax.y, tmax.z));
}

fn bad_ssgi(frag_coord: vec4<f32>, surface_normal: vec3<f32>, world_position: vec3<f32>, sample_index: u32) -> vec4<f32> {
    let surface_normal = normalize(surface_normal);
    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);
    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));
    let screen_uv = frag_coord.xy / depth_tex_dims;
    let depth = frag_coord.z;


    let white_frame_noise = vec3(
        hash_noise(vec2(0), globals.frame_count), 
        hash_noise(vec2(1), globals.frame_count + 1024u),
        hash_noise(vec2(2), globals.frame_count + 2048u)
    );

    let TBN = build_orthonormal_basis(surface_normal);
    var tot = vec3(0.0);
    for (var i = 0u; i < 4u; i += 1u) {
//        var direction = cosine_sample_hemisphere(vec2(
//            hash_noise(ifrag_coord, i + globals.frame_count),
//            hash_noise(ifrag_coord, i + 64u * 64u + globals.frame_count)
//        ));
//        var direction = cosine_sample_hemisphere(vec2(
//            blue_noise_for_pixel(ufrag_coord, i * 2u + 0u),
//            blue_noise_for_pixel(ufrag_coord, i * 2u + 1u),
//        ));
//        direction = normalize(surface_normal + direction);

        var direction = cosine_sample_hemisphere(vec2(
            fract(blue_noise_for_pixel(ufrag_coord, i * 3u + 0u + globals.frame_count * 12u) + white_frame_noise.x),
            fract(blue_noise_for_pixel(ufrag_coord, i * 3u + 1u + globals.frame_count * 12u) + white_frame_noise.y),
        ));
        
        //var direction = cosine_sample_hemisphere(vec2(
        //    blue_noise_for_pixel(vec2<u32>((world_position.xy + 1000.0) * 64.0), i * 2u + 0u),
        //    blue_noise_for_pixel(vec2<u32>((world_position.yz + 1000.0) * 64.0), i * 2u + 1u),
        //));
        
        direction = normalize(direction * TBN);

        let trace_dist = 10.0;
        let ray_end_ws = world_position + direction * trace_dist;

        var dmr = DepthRayMarch_new_from_depth(depth_tex_dims);
        dmr.ray_start_cs = vec3(uv_to_cs(screen_uv), depth);
        dmr = to_ws(dmr, ray_end_ws);
        //dmr = to_ws_dir(dmr, direction);
        dmr.linear_steps = 8u;
        dmr.depth_thickness_linear_z = 1.5;
        //interleaved_gradient_noise(frag_coord, globals.frame_count);
        dmr.jitter = fract(blue_noise_for_pixel(ufrag_coord, i * 3u + 2u + globals.frame_count * 12u) + white_frame_noise.z);
        dmr.march_behind_surfaces = false;
        dmr.use_secant = true;
        dmr.bisection_steps = 4u;
        
        let raymarch_result = march(dmr, sample_index);
        var shadow = 0.0;
        if (raymarch_result.hit) {
            let hit_n = normalize(prepass_normal(vec4<f32>(raymarch_result.hit_uv * depth_tex_dims, 0.0, 0.0), sample_index));
            let backface = dot(hit_n, direction);
            if backface < 0.01 {
                tot += textureSampleLevel(prev_frame_tex, prev_frame_sampler, raymarch_result.hit_uv, 0.0).rgb;
            }
        }
    }
    tot /= 4.0;

    return vec4(vec3(tot), 1.0);
}