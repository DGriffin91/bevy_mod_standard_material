

#import "shaders/pathtrace/raytrace_bindings_types.wgsl"

@compute @workgroup_size(8, 8, 1)
fn blur(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let flocation = vec2<f32>(location);
    let fprepass_size = vec2<f32>(textureDimensions(prepass_downsample).xy);
    let size = vec2<i32>(textureDimensions(target_tex).xy);
    let fsize = vec2<f32>(size);
    let screen_uv = flocation / fsize;

/*
    //let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 1.0);
    let nor_depth = textureLoad(prepass_downsample, vec2<i32>(screen_uv * fprepass_size) / 2, 1);
    

    var tot = vec4(0.0);
    var tot_w = 0.0;

    let dist_factor = 6.0;

    let world_position = position_ndc_to_world(vec3(uv_to_ndc(screen_uv), v.w));

    let n = 8;
    for (var x = -n; x < n; x+=1) {
        for (var y = -n; y < n; y+=1) {
            let uv = vec2<f32>(location + vec2(x, y)) / fsize;
            let px_dist = saturate(1.0 - distance(vec2(0.0, 0.0), vec2(f32(x), f32(y))) / f32(n + 1));
            var w = px_dist;
            let nd = textureLoad(prepass_downsample, vec2<i32>(uv * fprepass_size), 0);
            let col = textureLoad(prev_tex, location + vec2(x, y), 0u, 0);

            //let nd = textureSampleLevel(prepass_downsample, linear_sampler, uv, 1.0);
            //let col = textureSampleLevel(prev_tex, linear_sampler, uv, 1.0);

            let d = max(dot(nor_depth.xyz, nd.xyz) + 0.0, 0.0);
            w = w * d;
            let px_ws = position_ndc_to_world(vec3(uv_to_ndc(uv), col.w));
            let dist = distance(px_ws, world_position);
            w = w * (1.0 - clamp(dist * dist_factor, 0.0, 1.0));
            //if distance(col.xyz, v.xyz) < 0.7 {
                tot += vec4(col.rgb, nd.w) * w;
                tot_w += w;
            //}
        }
    }
    tot /= max(tot_w, 1.0);
    
    //tot = textureLoad(prev_tex, location, 0);
*/
    textureStore(target_tex, location, 0u, textureLoad(prev_tex, location, 0u, 0));
    textureStore(target_tex, location, 1u, textureLoad(prev_tex, location, 1u, 0));
    textureStore(target_tex, location, 2u, textureLoad(prev_tex, location, 2u, 0));
    textureStore(target_tex, location, 3u, textureLoad(prev_tex, location, 3u, 0));
}
