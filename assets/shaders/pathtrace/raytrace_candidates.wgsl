




// comment out to disable
//#define SPECULAR_SCREEN_SAMPLE
//#define DIFFUSE_SCREEN_SAMPLE
#define SPECULAR
#define CDIFFUSE_INDIRECT
//#define CDIFFUSE_DIRECT
//#define DEPTH_MARCH
#define CAPPLY_PRIMARY_COLOR

struct Candidate {
    color: vec3<f32>,
    direction: vec3<f32>,
    distance: f32,
}

fn candidates_update(invocation_id: vec3<u32>) -> Candidate {
    let pri_normal_bias = 0.01;
    let pri_dist_bias = 0.01;

    var cand: Candidate;
    cand.color = vec3(0.0);
    cand.direction = vec3(0.0);
    cand.distance = -1.0;

    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let flocation = vec2<f32>(location);

    

    let target_dims = textureDimensions(target_tex).xy;
    let prepass_tex_dims = vec2<f32>(textureDimensions(prepass_downsample).xy);
    
    let ftarget_dims = vec2<f32>(target_dims);
    let frag_size = 1.0 / ftarget_dims;
    var screen_uv = flocation.xy / ftarget_dims + frag_size * 0.5;

    
    let white_aa_noise = white_frame_noise(653u);
    let urand_aa = fract_white_noise_for_pixel(location, globals.frame_count + 1425u, white_aa_noise);
    let aa_jitter = (urand_aa.xy * 2.0 - 1.0) * frag_size * 0.25;
    screen_uv += aa_jitter;


    let samples = 1u;
    var sun_dir = vec3(-0.25, -0.24, 1.0);
    let sun_color = vec3(0.95, 0.79268, 0.637758) * 10.0;
    //var sun_dir = vec3(0.22, -1.0, -0.2); //sponza
    //let sun_color = vec3(0.95, 0.79268, 0.637758) * 10.0; //sponza
    let sky_color = vec3(1.75, 1.9, 1.99) * 2.0;
    let nee = 0.05; // TODO NEE


    // TODO depth here won't quite be accurate
    let nor_depth = textureLoad(prepass_downsample, vec2<i32>(screen_uv * prepass_tex_dims) / 4, 2);
    //let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 1.0);
    let surface_normal = normalize(nor_depth.xyz);
    let depth = nor_depth.w;
    let frag_coord = vec4(flocation, depth, 0.0);
    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);


    let world_position = position_ndc_to_world(vec3(uv_to_ndc(screen_uv), depth));
    var V = normalize(view.world_position.xyz - world_position);


    let tangent_to_world = build_orthonormal_basis(surface_normal);
    let white_frame_noise = white_frame_noise(78954u);

    var disc_direction = uniform_sample_disc(vec2(
        fract(hash_noise(ifrag_coord, 5345u) + white_frame_noise.x),
        fract(hash_noise(ifrag_coord, 6784u) + white_frame_noise.y),
    )) * 0.007;
    sun_dir = normalize(sun_dir + disc_direction); //make the sun a disc

        
    let urand = fract_blue_noise_for_pixel(ufrag_coord, globals.frame_count + 1425u, white_frame_noise);  
    //let urand = fract_white_noise_for_pixel(ifrag_coord, globals.frame_count + 1425u, white_frame_noise);

    //cand.direction = urand.xyz;
    cand.direction = cosine_sample_hemisphere(urand.xy);
    cand.direction = normalize(cand.direction * tangent_to_world);


    // random direction trace
    var ray = new_ray(world_position + V * pri_dist_bias + surface_normal * pri_normal_bias, cand.direction);

    var query = scene_query(ray);

    cand.distance = query.hit.distance;
    var ray_hit_pos = ray.origin + ray.direction * cand.distance;
    var hit_color = get_hit_material(query).color;

    // first see if we hit somewhere on the screen

    // Trace to sun
    if cand.distance != F32_MAX {
        var sray = new_ray(ray_hit_pos, -sun_dir);
        query = scene_query(sray);
        if query.hit.distance == F32_MAX {
            cand.color = hit_color * sun_color * nee;
        }
    } else {
        cand.color = hit_color * sky_color;    
    }
    //cand.color = vec3(saturate(cand.distance));

    return cand;
}

/*
@fragment
fn fragment_primary_rays(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let depth = prepass_depth(vec4<f32>(in.position.xy, 0.0, 0.0), 0u);
    let frag_coord = vec4(in.position.xy, depth, 0.0);
    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);
    let screen_uv = frag_coord_to_uv(in.position.xy);
    let ray_start_ndc = frag_coord_to_ndc(frag_coord);

    let surface_normal = normalize(prepass_normal(vec4<f32>(in.position.xy, 0.0, 0.0), 0u));

    var col = textureSample(screen_tex, texture_sampler, screen_uv);

    let ray = get_screen_ray(screen_uv);

    let query = scene_query(ray);

    if query.hit.distance != F32_MAX {
        var normal = vec3(0.0);

        var instance: InstanceData;
        var color: vec3<f32>;
        if query.static_tlas {
            instance = static_mesh_instance_buffer[query.hit.instance_idx];
            color = static_material_instance_buffer[query.hit.instance_idx].color;
        } else {
            instance = dynamic_mesh_instance_buffer[query.hit.instance_idx];
            color = dynamic_material_instance_buffer[query.hit.instance_idx].color;
        }
        normal = get_surface_normal(instance, query.hit);
        

        col = vec4(color, 1.0);//vec4(vec3(normal), 1.0);
    } else {
        col = vec4(0.0);    
    }

    col = print_value(frag_coord.xy, col, 0, f32(settings.fps));
    col = print_value(frag_coord.xy, col, 1, f32(settings.frame));
    col = print_value(frag_coord.xy, col, 2, f32(arrayLength(&dynamic_tlas_buffer)));
    col = print_value(frag_coord.xy, col, 3, f32(arrayLength(&static_tlas_buffer)));

    return col;
}
*/