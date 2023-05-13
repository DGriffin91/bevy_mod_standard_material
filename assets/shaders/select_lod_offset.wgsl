
// TODO find better way, try textureLoad, maybe blend with weights
// Find the lod offset the best matches the current lod 0 normal/depth
fn select_lod_offset(lod: f32, scale: f32, screen_uv: vec2<f32>) -> vec2<f32> {
    let pt_px = (1.0 / vec2<f32>(textureDimensions(prepass_downsample).xy)) * scale;
    let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 0.0);
    let nor_depth0 = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv + vec2( 0.0,  0.0) * pt_px, lod);
    let nor_depth1 = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv + vec2(-0.5, -0.5) * pt_px, lod);
    let nor_depth2 = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv + vec2(-0.5,  0.5) * pt_px, lod);
    let nor_depth3 = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv + vec2( 0.5, -0.5) * pt_px, lod);
    let nor_depth4 = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv + vec2( 0.5,  0.5) * pt_px, lod);
    let nor_depth5 = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv + vec2( 0.5,  0.0) * pt_px, lod);
    let nor_depth6 = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv + vec2(-0.5,  0.0) * pt_px, lod);
    let nor_depth7 = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv + vec2( 0.0,  0.5) * pt_px, lod);
    let nor_depth8 = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv + vec2( 0.0, -0.5) * pt_px, lod);
    let d0 = distance(nor_depth, nor_depth0);
    let d1 = distance(nor_depth, nor_depth1);
    let d2 = distance(nor_depth, nor_depth2);
    let d3 = distance(nor_depth, nor_depth3);
    let d4 = distance(nor_depth, nor_depth4);
    let d5 = distance(nor_depth, nor_depth5);
    let d6 = distance(nor_depth, nor_depth6);
    let d7 = distance(nor_depth, nor_depth7);
    let d8 = distance(nor_depth, nor_depth8);
    let depth = nor_depth.w;
    var best = vec3(-0.5, -0.5, d1);
    best = select(best, vec3(-0.5,  0.5, d2), best.z >= d2);
    best = select(best, vec3( 0.5, -0.5, d3), best.z >= d3);
    best = select(best, vec3( 0.5,  0.5, d4), best.z >= d4);
    best = select(best, vec3( 0.5,  0.0, d5), best.z >= d5);
    best = select(best, vec3(-0.5,  0.0, d6), best.z >= d6);
    best = select(best, vec3( 0.0,  0.5, d7), best.z >= d7);
    best = select(best, vec3( 0.0, -0.5, d8), best.z >= d8);
    best = select(best, vec3( 0.0,  0.0, d0), best.z >= d0); //prefer
    return best.xy * pt_px;
}