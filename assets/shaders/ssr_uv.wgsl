
fn ssr_uv(screen_uv: vec2<f32>, surface_normal: vec3<f32>, world_position: vec3<f32>, roughness: f32) -> vec4<f32> {
    let lod_rough = clamp(pow(roughness, 0.25) * 6.0, 1.0, 4.0);
    var tot = vec3(0.0);
    let frag_size = 1.0 / view.viewport.zw;
    var tot_w = 0.0;
    let n = 3;
    for (var x = -n; x < n; x+=1) {
        for (var y = -n; y < n; y+=1) {
            let foffset = vec2(f32(x), f32(y)) * frag_size * roughness * 4.0;

            var uv = textureSampleLevel(screen_passes_processed, prev_frame_sampler, screen_uv + foffset, 0, 0.0).xy;
            let closest_motion_vector = prepass_motion_vector(vec4<f32>(uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
            let history_uv = uv - closest_motion_vector;
            if history_uv.x > 0.0 && history_uv.x < 1.0 && history_uv.y > 0.0 && history_uv.y < 1.0 {
                let px_dist = saturate(1.0 - distance(vec2(0.0, 0.0), vec2(f32(x), f32(y))) / f32(n + 1));
                var w = px_dist;
                if uv.x > 0.0 && uv.y > 0.0 {
                    tot_w += w;
                    let prev_frame1 = textureSampleLevel(prev_frame_tex, linear_sampler, history_uv, lod_rough).rgb;
                    tot += prev_frame1 * 0.4;
                }
            }
        }
    }
    tot /= f32(n*n*n);
    return vec4(tot, 1.0);
}
