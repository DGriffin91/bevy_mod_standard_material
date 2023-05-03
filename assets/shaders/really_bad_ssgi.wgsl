
fn really_bad_ssgi(frag_coord: vec4<f32>, surface_normal: vec3<f32>, world_position: vec3<f32>, sample_index: u32) -> vec4<f32> {
    let samples = 8u;
    let surface_normal = normalize(surface_normal);
    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);
    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));
    let px_size = 1.0 / depth_tex_dims;
    let screen_uv = frag_coord.xy * px_size;
    let depth = frag_coord.z;

    
    let csn = view.view_proj * vec4(surface_normal, 0.0);
    let cs_nor = normalize(csn.xyz);


    let white_frame_noise = vec3(
        hash_noise(vec2(0), globals.frame_count), 
        hash_noise(vec2(1), globals.frame_count + 1024u),
        hash_noise(vec2(2), globals.frame_count + 2048u)
    );

    var tot = vec3(0.0);
    for (var i = 0u; i < samples; i += 1u) {


        //var direction = cosine_sample_hemisphere(vec2(
        //    fract(blue_noise_for_pixel(ufrag_coord, i * samples + 0u + globals.frame_count * samples) + white_frame_noise.x),
        //    fract(blue_noise_for_pixel(ufrag_coord, i * samples + 1u + globals.frame_count * samples) + white_frame_noise.y),
        //));

        //var direction = cosine_sample_hemisphere(vec2(
        //    hash_noise(ifrag_coord, i + globals.frame_count),
        //    hash_noise(ifrag_coord, i + 1024u + globals.frame_count)
        //));
            
        var rdist = fract(blue_noise_for_pixel(ufrag_coord, i * samples + 3u + globals.frame_count * samples) + white_frame_noise.z);

        var uv = vec2(
            hash_noise(ifrag_coord, i + globals.frame_count),
            hash_noise(ifrag_coord, i + 1024u + globals.frame_count)
        );

        let hit_depth = prepass_depth(vec4<f32>(uv * depth_tex_dims, 0.0, 0.0), sample_index);


        let cs_start = vec3(uv_to_cs(screen_uv), depth);
        let cs_end = vec3(uv_to_cs(uv), hit_depth);

        let direction = normalize(cs_end - cs_start);


        let dist = distance(cs_start, cs_end);


        let hit_n = normalize(prepass_normal(vec4<f32>(uv * depth_tex_dims, 0.0, 0.0), sample_index));
        let backface = dot(hit_n, direction);
        if backface < 0.01 {
            
            let closest_motion_vector = prepass_motion_vector(vec4<f32>(uv * depth_tex_dims, 0.0, 0.0), sample_index).xy;
            let history_uv = uv - closest_motion_vector;
            let d = saturate(dot(direction, cs_nor));
            if history_uv.x > 0.0 && history_uv.x < 1.0 && history_uv.y > 0.0 && history_uv.y < 1.0 {
                tot += mix(textureSampleLevel(prev_frame_tex, prev_frame_sampler, history_uv, 0.0).rgb, vec3(0.0), vec3(dist/(dist + 1.0))) * d;
            }
        }
    }
    tot /= f32(samples);

    return vec4(vec3(tot * 2.0), 1.0);
}