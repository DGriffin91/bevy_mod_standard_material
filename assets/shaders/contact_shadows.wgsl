// Settings
const g_sss_max_steps        = 5u;       // Max ray steps, affects quality and performance.
//const g_sss_ray_max_distance = 0.05;    // Max shadow length, longer shadows are less accurate.
const g_sss_thickness        = 0.13;      // Depth testing thickness.
const g_sss_step_length      = 0.05;     // g_sss_ray_max_distance / (float)g_sss_max_steps;
const g_sss_depth_bias       = 0.0;       // TODO, should this be necessary?
const g_sss_normal_bias      = 0.01;

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

fn uhash(a: u32, b: u32) -> u32 { 
    var x = ((a * 1597334673u) ^ (b * 3812015801u));
    // from https://nullprogram.com/blog/2018/07/31/
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn unormf(n: u32) -> f32 { 
    return f32(n) * (1.0 / f32(0xffffffffu)); 
}

fn hash_noise(ifrag_coord: vec2<i32>, frame: u32) -> f32 {
    let urnd = uhash(u32(ifrag_coord.x), (u32(ifrag_coord.y) << 11u) + frame);
    return unormf(urnd);
}

fn screen_fade(uv: vec2<f32>, thickness: f32, sharpness: f32) -> f32 {
    let distance_to_edge = min(uv, 1.0 - uv);
    let fade = smoothstep(vec2(0.0), vec2(thickness), distance_to_edge * sharpness);
    return min(fade.x, fade.y);
}

fn contact_shadow(frag_coord: vec2<f32>, dir_to_light: vec3<f32>, sample_index: u32) -> f32 {
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

    let normal_bias_offset = get_normal(screen_uv, sample_index) * g_sss_normal_bias;
    let ws_pos = position_from_uv(screen_uv, depth * 1.0) + normal_bias_offset;


    // Compute ray position and direction (in view-space)
    var ray_pos = (vec4(ws_pos - view.world_position.xyz, 1.0) * view.view).xyz;
    let ray_dir = (vec4(dir_to_light, 0.0) * view.view).xyz;
    //var clip_space_dir = view.projection * vec4(ray_dir, 1.0);
    //var uv_dir = ((clip_space_dir.xyz / clip_space_dir.w).xy * 0.5 + 0.5);
    //uv_dir.y = -uv_dir.y;
    //uv_dir *= screen_size;

    // Compute ray step
    let ray_step = ray_dir * g_sss_step_length;
    //let ray_step = ray_dir * get_linear_depth(screen_uv) * 0.002;

    // Ray march towards the light
    var occlusion = 0.0;
    var ray_uv = vec2(0.0);    
    
    for (var i = 0u; i < g_sss_max_steps; i += 1u) {
        // Step the ray
        ray_pos += ray_step;
        ray_uv  = project_uv_view_space(ray_pos);
        //ray_uv += uv_dir;
        

        let ray_frag = vec2<i32>(ray_uv * resolution);

        if (ray_frag.x == ifrag_coord.x && ray_frag.y == ifrag_coord.y) {
            // if we alias to the same coord, continue
            continue;
        }

        // Ensure the UV coordinates are inside the screen
        if ray_uv.x > 0.0 && ray_uv.x <= 1.0 && ray_uv.y > 0.0 && ray_uv.y <= 1.0 {
            // Compute the difference between the ray's and the camera's depth
            let depth_z     = get_linear_depth(ray_uv, sample_index);
            let depth_delta = -ray_pos.z - depth_z;
            //let depth_delta = linear_depth - depth_z;

            // Check if the camera can't "see" the ray (ray depth must be larger than the camera depth, so positive depth_delta)
            if depth_delta > g_sss_depth_bias && depth_delta < g_sss_thickness {
                // Mark as occluded
                occlusion += 1.0 - f32(i) / f32(g_sss_max_steps);

                // Fade out as we approach the edges of the screen (needs tweaking)
                occlusion *= screen_fade(ray_uv, 0.02, .5);

                break;
            }
        }
    }

    // Convert to visibility
    occlusion = 1.0 - occlusion;
    return saturate(occlusion * occlusion * occlusion);

    //return vec4(project_uv_view_space(ray_pos), 0.0, 1.0);
    //return vec4(ray_pos, 1.0);
    //return vec4(vec3(-ray_pos.z * 0.5), 1.0);
    //return vec4(vec3(depth), 1.0);
    //return vec4(project_uv(ws_pos), 0.0, 1.0);
    //return vec4(ray_uv, 0.0, 1.0);
}