
fn sample_restir_gi(diffuse_color: vec3<f32>, screen_uv: vec2<f32>, surface_normal: vec3<f32>, world_position: vec3<f32>) -> vec3<f32> {
    let screen_passes_processed_size = vec2<f32>(textureDimensions(screen_passes_processed).xy);
    let hit_nd_a = normalize(textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 1.0));

    let depth = distance(world_position, view.world_position.xyz);

    var proposed_pos = textureLoad(screen_passes_processed, vec2<i32>(screen_uv * screen_passes_processed_size), 0u, 0).xyz;
    let weight_data = textureLoad(screen_passes_processed, vec2<i32>(screen_uv * screen_passes_processed_size), 1u, 0).xyz;
    var proposed_col = textureLoad(screen_passes_processed, vec2<i32>(screen_uv * screen_passes_processed_size), 4u, 0).xyz;
    var M = u32(weight_data.x);
    var w_sum = weight_data.y;
    var weight = weight_data.z;
    let ray_dir = normalize(proposed_pos - world_position);
    let dist = distance(proposed_pos, world_position);

    let backface = max(dot(ray_dir, surface_normal), 0.0);

    let c_weight = w_sum / max(0.00001, f32(M) * weight);
    // idk why this 2 is here https://github.com/EmbarkStudios/kajiya/blob/main/assets/shaders/rtdgi/restir_resolve.hlsl#LL172C22-L172C22
    let gw = 2.0 * max(dot(surface_normal, ray_dir), 0.0);
    //let limit_w = min(c_weight * gw, gw); //force conservation TODO probably shouldn't be needed
    // proposed_col has the weight baked in
    let col = proposed_col; //TODO use SH so we can use gw, can't use this with filtering

    return max(col, vec3(0.0)) * diffuse_color;
}