struct NotAReservoir {
    M: u32, //proposal samples count
    w_sum: f32,
    weight: f32,
    proposed_pos: vec3<f32>,
    proposed_col: vec3<f32>, //shouldn't really be here
}

fn new_not_a_reservoir() -> NotAReservoir {
    var rr: NotAReservoir;
    rr.M = 0u;
    rr.w_sum = 0.0;
    rr.weight = 0.0;
    rr.proposed_pos = vec3(0.0);
    rr.proposed_col = vec3(0.0);
    return rr;
}

fn do_something(rr: NotAReservoir, samples: u32, ifrag_coord: vec2<i32>, surface_normal: vec3<f32>, world_position: vec3<f32>, white_frame_noise: vec4<f32>) -> NotAReservoir {
    var rr = rr;
    //var best = new_not_a_reservoir();
    for (var i = 0u; i < samples; i += 1u) {
        let seed = i * samples + globals.frame_count * samples;
        let urand = fract_white_noise_for_pixel(ifrag_coord, seed, white_frame_noise); 
        //let urand = fract_blue_noise_for_pixel(vec2<u32>(ifrag_coord), seed, white_frame_noise);  

        let prop_uv = urand.xy;
        let closest_motion_vector = prepass_motion_vector(vec4<f32>(prop_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
        let history_uv = prop_uv - closest_motion_vector;

        //let nor = normalize(prepass_normal(vec4<f32>(history_uv * view.viewport.zw, 0.0, 0.0), 0u));
        //let depth = prepass_depth(vec4(history_uv * view.viewport.zw, 0.0, 0.0), 0u);
        let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, history_uv, 3.0);
        let nor = nor_depth.xyz;
        let depth = nor_depth.w;
        // TODO reproject last frame
        //let color = textureLoad(prev_frame_tex, vec2<i32>(history_uv * (view.viewport.zw / 8.0)), 3).rgb;
        let color = textureSampleLevel(prev_frame_tex, prev_frame_sampler, history_uv, 3.0).rgb;


        let prop_pos = position_ndc_to_world(vec3(uv_to_ndc(history_uv), depth));
        let distv = prop_pos - world_position;
        let dir = normalize(distv);
        let dist = sqrt(dot(distv, distv));

        let backface = dot(nor, dir);

        var gr = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));

        var weight = gr * (1.0 / (1.0 + dist * dist));
        let brdf = max(dot(surface_normal, dir), 0.0);
        weight *= brdf * f32(backface < 0.01);
        rr.w_sum += weight;

        var threshold = weight / rr.w_sum;

        if rr.M == 0u {
            threshold = 1.0;
        }

        rr.M += 1u;

        if threshold > urand.z {
            rr.proposed_pos = prop_pos;
            rr.weight = weight;
            rr.proposed_col = color;
        }

        //if weight > best.weight {
        //    best.proposed_pos = prop_pos;
        //    best.weight = weight;
        //    best.proposed_col = color;
        //}
    }
    //best.w_sum = rr.w_sum;
    //best.M = rr.M;
    return rr;
}


fn not_restir(frag_coord: vec4<f32>, surface_normal: vec3<f32>, world_position: vec3<f32>, sample_index: u32) -> vec4<f32> {

    var rr = new_not_a_reservoir();

    let samples = 1u;
    let linear_steps = 32u;
    let bisection_steps = 0u;
    let depth_thickness = 0.4;


    let surface_normal = normalize(surface_normal);
    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);
    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));
    let screen_uv = frag_coord_to_uv(frag_coord.xy);
    let ray_start_ndc = frag_coord_to_ndc(frag_coord);

    
    let r_noise = white_frame_noise(6235u);
    rr = do_something(rr, 32u, ifrag_coord, surface_normal, world_position, r_noise);
    

    var tot = vec3(0.0);

    let white_frame_noise = white_frame_noise(8492u);

    var dmr = DepthRayMarch_new_from_depth(depth_tex_dims);
    

    dmr.ray_start_cs = ray_start_ndc;
    dmr.linear_steps = linear_steps;
    dmr.depth_thickness_linear_z = depth_thickness;
    dmr.march_behind_surfaces = false;
    dmr.use_secant = false;
    dmr.bisection_steps = bisection_steps;
    dmr.use_bilinear = true;



    let direction = normalize(rr.proposed_pos - world_position);
    let dist = distance(rr.proposed_pos, world_position);
    let ray_end_ws = rr.proposed_pos - direction * 0.0001;
    //let ray_end_ws = world_position + direction * dist * 0.999;

    dmr = to_ws(dmr, ray_end_ws);
    dmr.jitter = fract(blue_noise_for_pixel(ufrag_coord, 22u * globals.frame_count) + white_frame_noise.x);
    
    let raymarch_result = march(dmr, sample_index);
    var shadow = 0.0;
    if (!raymarch_result.hit) {
        var gr = dot(rr.proposed_col, vec3<f32>(0.2126, 0.7152, 0.0722));
        let c = clamp((rr.proposed_col * rr.weight) / rr.w_sum, vec3(0.0), vec3(100.0));
        tot += c * max(dot(surface_normal, direction), 0.0) * (1.0 / (1.0 + dist * dist));
    }
    
    tot /= f32(samples);

    return vec4(vec3(tot), 1.0);
}