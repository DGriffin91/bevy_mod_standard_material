// Settings
const g_sss_max_steps        = 6u;      // Max ray steps, affects quality and performance.
const g_sss_ray_max_distance = 0.04;     // Max shadow length, longer shadows are less accurate.
const g_sss_thickness        = 0.015;    // Depth testing thickness.
const g_sss_depth_bias       = 0.0;      // TODO, should this be necessary?
const g_sss_normal_bias      = 0.008;
const g_sss_dither           = 0.001;
const g_sss_step_noise       = 0.001;

//sponza:
//g_sss_ray_max_distance 0.24
//g_sss_thickness 0.215

fn project_uv(pos: vec3<f32>) -> vec2<f32> {
    let clipSpace = view.projection * (view.inverse_view * vec4(pos, 1.0));
    let ndc = clipSpace.xyz / clipSpace.w;
    var prev_uv = (ndc.xy * .5 + .5);
    prev_uv.y = 1.0 - prev_uv.y;
    return prev_uv;
}

fn project_uv_view_space(pos: vec3<f32>) -> vec2<f32> {
    let clipSpace = view.projection * vec4(pos, 1.0);
    let ndc = clipSpace.xyz / clipSpace.w;
    var prev_uv = (ndc.xy * .5 + .5);
    prev_uv.y = 1.0 - prev_uv.y;
    return prev_uv;
}

fn get_linear_depth(uv: vec2<f32>, sample_index: u32) -> f32 {
    let resolution = vec2<f32>(textureDimensions(depth_prepass_texture));
    let near = view.projection[3][2];
    return near / prepass_depth(vec4<f32>(uv * resolution, 0.0, 0.0), sample_index);
}

fn get_depth(uv: vec2<f32>, sample_index: u32) -> f32 {
    let resolution = vec2<f32>(textureDimensions(depth_prepass_texture));
    return prepass_depth(vec4<f32>(uv * resolution, 0.0, 0.0), sample_index);
}

fn get_normal(uv: vec2<f32>, sample_index: u32) -> vec3<f32> {
    let resolution = vec2<f32>(textureDimensions(normal_prepass_texture));
    return prepass_normal(vec4<f32>(uv * resolution, 0.0, 0.0), sample_index);
}

fn position_from_uv(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    var ndc = vec3(uv * 2.0 - 1.0, depth);
    ndc.y = -ndc.y;
    let homogeneousPosition = view.inverse_view_proj * vec4(ndc, 1.0);
    let worldSpacePosition = homogeneousPosition.xyz / homogeneousPosition.w;
    return worldSpacePosition.xyz;
}


// TODO we need sampler for depth_prepass_texture so we can use textureSampleCompare
fn software_bilinear(uv: vec2<f32>, size: vec2<f32>) -> f32 {
	let pos = uv * size - 0.5;
    let f = fract(pos);
    
    let pos_top_left = floor(pos);
    
    // we are sample center, so it's the same as point sample
    let tl = textureLoad(depth_prepass_texture, vec2<i32>((pos_top_left + vec2(0.5, 0.5))), 0);
    let tr = textureLoad(depth_prepass_texture, vec2<i32>((pos_top_left + vec2(1.5, 0.5))), 0);
    let bl = textureLoad(depth_prepass_texture, vec2<i32>((pos_top_left + vec2(0.5, 1.5))), 0);
    let br = textureLoad(depth_prepass_texture, vec2<i32>((pos_top_left + vec2(1.5, 1.5))), 0);
    
    return mix(mix(tl, tr, f.x), mix(bl, br, f.x), f.y);
}

fn screen_fade(uv: vec2<f32>, thickness: f32, sharpness: f32) -> f32 {
    let distance_to_edge = min(uv, 1.0 - uv);
    let fade = smoothstep(vec2(0.0), vec2(thickness), distance_to_edge * sharpness);
    return min(fade.x, fade.y);
}

fn contact_shadow(frag_coord: vec2<f32>, dir_to_light: vec3<f32>, surface_normal: vec3<f32>, sample_index: u32) -> f32 {
    let resolution = vec2<f32>(textureDimensions(depth_prepass_texture));
    let ifrag_coord = vec2<i32>(frag_coord);
    let screen_uv = frag_coord / resolution;
    let screen_size = 1.0 / resolution;

    let depth = get_depth(screen_uv, sample_index);
    let linear_depth = get_linear_depth(screen_uv, sample_index);

    // depth is inf
    if depth == 0.0 {
        return 1.0;
    }
    let normal_bias_offset = surface_normal * g_sss_normal_bias;
    let ws_pos = position_from_uv(screen_uv, depth * 1.0) + normal_bias_offset;

    // Compute ray position and direction (in view-space)
    var ray_pos = (vec4(ws_pos - view.world_position.xyz, 1.0) * view.view).xyz;
    var ray_dir = (vec4(dir_to_light, 0.0) * view.view).xyz;

    let noise = hash_noise(vec2<i32>(frag_coord), 1u) * 2.0 - 1.0;

    let max_dist = g_sss_ray_max_distance * mix(1.0, 0.0, depth); //vary max_dist by distance
    var step_length = max_dist / f32(g_sss_max_steps);
    step_length = step_length + noise * g_sss_step_noise * step_length;
    // Compute ray step
    let ray_step = ray_dir * step_length;

    // Ray march towards the light
    var occlusion = 0.0;
    var ray_uv = vec2(0.0);    
    
    for (var i = 0u; i < g_sss_max_steps; i += 1u) {
        // Step the ray
        ray_pos += ray_step;
        // add noise to position (noise could be added to direction instead, but then noise quality does not go up with more steps)
        //i + globals.frame_count
        let noise = interleaved_gradient_noise(frag_coord, i) * 2.0 - 1.0;
        ray_pos += noise * (g_sss_dither / f32(g_sss_max_steps));
        ray_uv  = project_uv_view_space(ray_pos);
        
        let ray_frag = vec2<i32>(ray_uv * resolution);

        // Ensure the UV coordinates are inside the screen
        if ray_uv.x > 0.0 && ray_uv.x <= 1.0 && ray_uv.y > 0.0 && ray_uv.y <= 1.0 && 
            // if we alias to the same coord, continue
           !(ray_frag.x == ifrag_coord.x && ray_frag.y == ifrag_coord.y) {
            // Compute the difference between the ray's and the camera's depth
            let depth_z     = get_linear_depth(ray_uv, sample_index);
            let depth_delta = -ray_pos.z - depth_z;

            // Check if the camera can't "see" the ray (ray depth must be larger than the camera depth, so positive depth_delta)
            if depth_delta > g_sss_depth_bias && depth_delta < g_sss_thickness {

                var strength = 1.0;
                // fade out as we aproach max steps
                strength = 1.0 - f32(i) / f32(g_sss_max_steps); 
                occlusion += strength;

                // Fade out as we approach the edges of the screen (needs tweaking)
                occlusion *= screen_fade(ray_uv, 0.015, .6);
                
                break;
            }
        }
    }

    // Convert to visibility
    occlusion = 1.0 - saturate(occlusion);
    return saturate(occlusion * occlusion * occlusion * occlusion);
}