#import "shaders/screen_space_passes/common.wgsl"

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    //if true {return;}
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let size = vec2<i32>(textureDimensions(fullscreen_passes_write).xy);
    if location.x >= size.x || location.y >= size.y {
        return;
    }
    let ulocation = vec2<u32>(location);
    let flocation = vec2<f32>(location);
    let fprepass_size = vec2<f32>(textureDimensions(prepass_downsample).xy);
    let fsize = vec2<f32>(size);
    let frag_size = 1.0 / fsize;
    let screen_uv = flocation / fsize + frag_size * 0.5; // TODO verify

    


    


    var probe = load_probe(location);

#ifdef FILTER_SSGI
    let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 0.0);

    
    let pixel_radius = pixel_radius_to_world(1.0, depth_ndc_to_linear(nor_depth.w), projection_is_orthographic());

    let world_position = position_ndc_to_world(vec3(uv_to_ndc(screen_uv), nor_depth.w));
    let white_frame_noise = white_frame_noise(8462u);
    let urand = fract_blue_noise_for_pixel(ulocation, globals.frame_count, white_frame_noise) * 2.0 - 1.0;
    let center = textureLoad(fullscreen_passes_read, location, COLOR_LAYER, 0);
    let center_color = center.rgb;
    let ssao_focus = center.w;
    let dist_factor = 50.0 * pixel_radius;
    var tot = vec3(0.0);
    var tot_w = 0.0;
    let samples = 6u;
    var range = mix(9.0, 14.0, urand.z) * ssao_focus;
    let center_weight = 0.1;
    tot += center_color * center_weight;
    tot_w += center_weight;
    for (var i = 0u; i <= samples; i+=1u) {
        let seed = i * samples + globals.frame_count * samples;
        let white_frame_noise = white_frame_noise(seed + 8462u);
        let urand = fract_blue_noise_for_pixel(ulocation, seed, white_frame_noise) * 2.0 - 1.0;
        let px_pos = vec2(urand.x * range, urand.y * range);
        let coord = location + vec2<i32>(px_pos);
        if all(coord < 0) || all(coord >= size) {
            continue;
        }
        let uv = vec2<f32>(coord) / fsize;
        let px_dist = length(abs(urand.xy));
        var w = px_dist;
        let nd = textureLoad(prepass_downsample, vec2<i32>(uv * fprepass_size), 0);
        let col = textureLoad(fullscreen_passes_read, coord, COLOR_LAYER, 0);
    
        let d = max(dot(nor_depth.xyz, nd.xyz) + 0.001, 0.0);
        w = w * d * d * d; //lol
        let px_ws = position_ndc_to_world(vec3(uv_to_ndc(uv), nd.w));
        let dist = distance(px_ws, world_position);
        w = w * (1.0 - saturate(dist / dist_factor));
        w = max(w, 0.0);
        tot += col.rgb * w;
        tot_w += w;
    }
    let final_color = tot/tot_w;
#ifdef APPLY_FILTER_TO_OUTPUT
    textureStore(fullscreen_passes_write, location, COLOR_LAYER, vec4(final_color, 1.0));
#else
    textureStore(fullscreen_passes_write, location, COLOR_LAYER, textureLoad(fullscreen_passes_read, location, COLOR_LAYER, 0));
#endif // APPLY_FILTER_TO_OUTPUT
#else
    textureStore(fullscreen_passes_write, location, COLOR_LAYER, textureLoad(fullscreen_passes_read, location, COLOR_LAYER, 0));
#endif // FILTER_SSGI


    textureStore(screen_passes_write, location, 0u, textureLoad(screen_passes_read, location, 0u, 0));
    textureStore(screen_passes_write, location, 1u, textureLoad(screen_passes_read, location, 1u, 0));
    textureStore(screen_passes_write, location, 2u, textureLoad(screen_passes_read, location, 2u, 0));
    textureStore(screen_passes_write, location, 3u, textureLoad(screen_passes_read, location, 3u, 0));
    textureStore(screen_passes_write, location, 4u, textureLoad(screen_passes_read, location, 4u, 0));

    


    textureStore(fullscreen_passes_write, location, 5u, textureLoad(fullscreen_passes_read, location, 5u, 0));
    textureStore(fullscreen_passes_write, location, SSR_LAYER, textureLoad(fullscreen_passes_read, location, SSR_LAYER, 0));

    


#ifdef RESTIR_ANTI_CLUMPING
    var same_pos = 0u;
    var same_wdata = 0u;
    var avg_color = vec3(0.0);
    for (var x = -1; x <= 1; x += 1) {
        for (var y = -1; y <= 1; y += 1) {
            let offset = vec2(x, y);
            if all(offset == vec2(0, 0)) {
                continue;
            }
            var s_probe = load_probe(location + offset);
            if all(probe.ray_hit_pos == s_probe.ray_hit_pos) {
                same_pos += 1u;
            }
            if probe.weight == s_probe.weight && probe.w_sum == s_probe.w_sum  {
                same_wdata += 1u;
            }
            avg_color += probe_resolve(&s_probe);
        }
    }
    avg_color /= 6.0;
    let color_distance = distance(avg_color, probe_resolve(&probe));
    
    let noise = fract(blue_noise_for_pixel(ulocation, globals.frame_count) + hash_noise(vec2(0, 0), globals.frame_count + 4275u));

    if same_pos > 0u || color_distance > CLUMP_RESET_COLOR_DIST_THRESH {
        let derate = mix(f32(same_wdata + same_pos), color_distance, 0.85);
        let reset_max = u32(f32(MAX_M) * mix(CLUMP_RESET_MIN, CLUMP_RESET_MAX, noise / derate));
        probe_scale(&probe, reset_max);
    }
#endif // RESTIR_ANTI_CLUMPING
    store_probe(probe, location);
}