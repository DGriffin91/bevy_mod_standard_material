const PHI = 1.618033988749895; // Golden Ratio
const M_PLASTIC = 1.32471795724474602596;


fn uniform_sample_sphere(urand: vec2<f32>) -> vec3<f32> {
    let theta = 2.0 * PI * urand.y;
    let z = 1.0 - 2.0 * urand.x;
    let xy = sqrt(max(0.0, 1.0 - z * z));
    let sn = sin(theta);
    let cs = cos(theta);
    return vec3(cs * xy, sn * xy, z);
}

fn uniform_sample_disc(urand: vec2<f32>) -> vec2<f32> {
    let theta = 2.0 * PI * urand.y;
    let radius = sqrt(urand.x);
    let x = cos(theta) * radius;
    let y = sin(theta) * radius;
    return vec2(x, y);
}

fn r2_sequence(i: u32) -> vec2<f32> {
    let a1 = 1.0 / M_PLASTIC;
    let a2 = 1.0 / (M_PLASTIC * M_PLASTIC);
    return fract(vec2(a1, a2) * f32(i) + 0.5);
}

fn blue_noise_for_pixel(px: vec2<u32>, layer: u32) -> f32 {
    let offset = vec2((layer % 8u), ((layer / 8u) % 8u)) * 8u;
    //let offset = vec2(layer % BLUE_NOISE_TEX_DIMS.x, 0u);
    return textureLoad(blue_noise_tex, px % BLUE_NOISE_TEX_DIMS + offset, 0).x * 255.0 / 256.0 + 0.5 / 256.0;
}

fn blue_noise_for_pixel_r2(px: vec2<u32>, n: u32) -> f32 {
    let offset = vec2<u32>(r2_sequence(n) * vec2<f32>(BLUE_NOISE_TEX_DIMS));

    return textureLoad(blue_noise_tex, (px + offset) % BLUE_NOISE_TEX_DIMS, 0).x * 255.0 / 256.0 + 0.5 / 256.0;
}

fn blue_noise_for_pixel_simple(px: vec2<u32>) -> f32 {
    return textureLoad(blue_noise_tex, px % BLUE_NOISE_TEX_DIMS, 0).x * 255.0 / 256.0 + 0.5 / 256.0;
}

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
    for (var i = 0u; i < 1u; i += 1u) {
        //var direction = uniform_sample_sphere(
        //    fract(blue_noise_for_pixel_simple(ufrag_coord) + r2_sequence(0u))
        //);
        var direction = uniform_sample_sphere(
            fract(hash_noise(ifrag_coord, 0u) + r2_sequence(0u))
        );
        direction = normalize(surface_normal + direction);

        v += dot(direction, surface_normal);
    }
    v /= 1.0;


    
    let dots = f32(v > 0.98);

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
        var rmr = DepthRayMarch_new_from_depth(depth_tex_dims);   
        rmr.ray_start_cs = vec3(uv_to_cs(screen_uv), depth);
        rmr = to_ws(rmr, ray_hit_ws + direction * 1.0);
        //rmr = to_ws_dir(rmr, direction);
        rmr.linear_steps = 4u;
        rmr.depth_thickness_linear_z = 0.8;
        rmr.jitter = 1.0; //interleaved_gradient_noise(frag_coord, globals.frame_count);
        rmr.march_behind_surfaces = true;
        let raymarch_result = march(rmr, sample_index);
        var shadow = 0.0;
        if (raymarch_result.hit) {
            tot += smoothstep(1.0, 0.0, raymarch_result.hit_t);
        }
    }
    tot /= 4.0;

    return vec4(vec3(1.0 - tot), 1.0);
}