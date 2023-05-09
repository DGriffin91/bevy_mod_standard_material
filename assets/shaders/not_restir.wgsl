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
    for (var i = 0u; i < samples; i += 1u) {
        let seed = i * samples + globals.frame_count * samples;
        let urand = fract_white_noise_for_pixel(ifrag_coord, seed, white_frame_noise); 
        //let urand = fract_blue_noise_for_pixel(vec2<u32>(ifrag_coord), seed, white_frame_noise);  

        let prop_uv = urand.xy;

        let nor = normalize(prepass_normal(vec4<f32>(prop_uv * view.viewport.zw, 0.0, 0.0), 0u));
        // TODO reproject last frame
        let color = textureLoad(prev_frame_tex, vec2<i32>(prop_uv * view.viewport.zw), 0).rgb;

        let depth = prepass_depth(vec4(prop_uv * view.viewport.zw, 0.0, 0.0), 0u);

        let prop_pos = position_ndc_to_world(vec3(uv_to_ndc(prop_uv), depth));
        let dir = normalize(prop_pos - world_position);
        let dist = distance(prop_pos, world_position);

        let backface = dot(nor, -dir);

        //if backface < -0.01 {
        //    continue;
        //}

        let gr = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));

        var weight = 0.0;
        //weight += max(backface * 1.0, 0.0); //too restrictive?
        //weight += dot(color, vec3<f32>(0.2126, 0.7152, 0.0722)) * 10.0;
        //weight += 1.0 / dist;//1.0 - dist / (dist + 1.0); //lol idk
        let brdf = dot(surface_normal, dir) * gr * (1.0 / (1.0 + dist * dist));
        weight += brdf * f32(backface > -0.01);


        var threshold = weight / (weight + rr.w_sum);

        if rr.M == 0u {
            threshold = -1.0;
        }

        rr.M += 1u;
        rr.w_sum += weight;

        if urand.z > threshold {
            rr.proposed_pos = prop_pos;
            rr.weight = weight;
            rr.proposed_col = color;
        }
    }
    return rr;
}


fn not_restir(frag_coord: vec4<f32>, surface_normal: vec3<f32>, world_position: vec3<f32>, sample_index: u32) -> vec4<f32> {

    var rr = new_not_a_reservoir();

    let samples = 1u;
    let linear_steps = 32u;
    let bisection_steps = 0u;
    let depth_thickness = 0.1;


    let surface_normal = normalize(surface_normal);
    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);
    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));
    let screen_uv = frag_coord_to_uv(frag_coord.xy);
    let ray_start_ndc = frag_coord_to_ndc(frag_coord);

    
    let r_noise = white_frame_noise(6235u);
    rr = do_something(rr, 128u, ifrag_coord, surface_normal, world_position, r_noise);
    

    var tot = vec3(0.0);

    let white_frame_noise = white_frame_noise(8492u);

    var dmr = DepthRayMarch_new_from_depth(depth_tex_dims);
    

    dmr.ray_start_cs = ray_start_ndc;
    dmr.linear_steps = linear_steps;
    dmr.depth_thickness_linear_z = depth_thickness;
    dmr.march_behind_surfaces = false;
    dmr.use_secant = false;
    dmr.bisection_steps = bisection_steps;
    dmr.use_bilinear = false;



    let direction = normalize(rr.proposed_pos - world_position);
    let dist = distance(rr.proposed_pos, world_position);
    let ray_end_ws = rr.proposed_pos - direction * 0.0001;

    dmr = to_ws(dmr, ray_end_ws);
    dmr.jitter = fract(blue_noise_for_pixel(ufrag_coord, 22u * globals.frame_count) + white_frame_noise.x);
    
    let raymarch_result = march(dmr, sample_index);
    var shadow = 0.0;
    if (!raymarch_result.hit) {
            
        //let closest_motion_vector = prepass_motion_vector(vec4<f32>(raymarch_result.hit_uv * depth_tex_dims, 0.0, 0.0), sample_index).xy;
        //let history_uv = raymarch_result.hit_uv - closest_motion_vector;
        //if history_uv.x > 0.0 && history_uv.x < 1.0 && history_uv.y > 0.0 && history_uv.y < 1.0 {
            let c = clamp((rr.proposed_col * rr.weight) / (rr.w_sum / f32(rr.M)), vec3(0.0), vec3(50.0));
            tot += c * max(dot(surface_normal, direction), 0.0) * (1.0 / (1.0 + dist * dist));

            //tot += textureLoad(prev_frame_tex, vec2<i32>(ndc_to_uv(position_world_to_ndc(rr.proposed_pos).xy) * view.viewport.zw), 0).rgb;

        //}
        
    }
    
    tot /= f32(samples);

    return vec4(vec3(tot), 1.0);
}