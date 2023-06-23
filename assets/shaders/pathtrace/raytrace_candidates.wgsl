




// comment out to disable
//#define SPECULAR_SCREEN_SAMPLE
//#define CDIFFUSE_SCREEN_SAMPLE
#define VOXEL_1ST_BOUNCE_FALLBACK
#define SPECULAR
#define CDIFFUSE_INDIRECT
//#define CDIFFUSE_DIRECT
//#define DEPTH_MARCH
#define CAPPLY_PRIMARY_COLOR

struct Candidate {
    color: vec3<f32>,
    world_position: vec3<f32>,
    ray_hit_pos: vec3<f32>,
    direction: vec3<f32>,
    distance: f32,
}

fn candidates_update(invocation_id: vec3<u32>) -> Candidate {

    var cand: Candidate;
    cand.color = vec3(0.0);
    cand.world_position = vec3(0.0);
    cand.ray_hit_pos = vec3(0.0);
    cand.distance = -1.0;
    cand.direction = vec3(0.0);

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
    //var sun_dir = normalize(vec3(-0.25, -0.24, 1.0)); // kitchen
    //let sun_color = pow(vec3(0.95, 0.79268, 0.637758), vec3(2.2)) * 20.0; // kitchen
    var sun_dir = vec3(0.22, -1.0, -0.2); //sponza
    let sun_color = vec3(0.95, 0.79268, 0.637758) * 20.0; //sponza
    let sky_color = pow(vec3(0.875, 0.95, 0.995) * 2.0, vec3(2.2));
    let real_sun_angular_radius = 0.53 * 0.5 * PI / 180.0;
    let sun_angular_radius_cos = cos(real_sun_angular_radius);
    let nee = 1.0;// TODO - sun_angular_radius_cos;


    // TODO depth here won't quite be accurate
    let nor_depth = textureLoad(prepass_downsample, vec2<i32>(screen_uv * prepass_tex_dims) / 2, 1);


    //let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 1.0);
    let surface_normal = normalize(nor_depth.xyz);
    let depth = nor_depth.w;
    let frag_coord = vec4(flocation, depth, 0.0);
    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);
    let linear_depth = depth_ndc_to_linear(depth);
    let pixel_radius = pixel_radius_to_world(1.0, linear_depth, projection_is_orthographic());

    //--------
    //--------
    let pri_normal_bias = pixel_radius * 1.5;
    let pri_dist_bias = pixel_radius * 1.5;
    let min_hit_dist = pixel_radius * 1.5;
    //--------
    //--------


    let world_position = position_ndc_to_world(vec3(uv_to_ndc(screen_uv), depth));
    var V = normalize(view.world_position.xyz - world_position);


    let tangent_to_world = build_orthonormal_basis(surface_normal);
    let white_frame_noise = white_frame_noise(78954u);

    let sun_rnd = vec2(
        fract(hash_noise(ifrag_coord, 5345u) + white_frame_noise.x),
        fract(hash_noise(ifrag_coord, 6784u) + white_frame_noise.y),
    );

    //var disc_direction = uniform_sample_disc() * 0.007;
    //sun_dir = normalize(sun_dir + disc_direction); //make the sun a disc
    
    let sun_basis = build_orthonormal_basis(sun_dir);
    sun_dir = normalize(uniform_sample_cone(sun_rnd, sun_angular_radius_cos) * sun_basis);

        
    let urand = fract_blue_noise_for_pixel(ufrag_coord, globals.frame_count + 1425u, white_frame_noise);  
    //let urand = fract_white_noise_for_pixel(ifrag_coord, globals.frame_count + 1425u, white_frame_noise);

    //cand.direction = urand.xyz;
    var direction = cosine_sample_hemisphere(urand.xy);
    direction = normalize(direction * tangent_to_world);


    // random direction trace
    var ray = new_ray(world_position + V * pri_dist_bias + surface_normal * pri_normal_bias, direction);

    var query = scene_query(ray);

    cand.world_position = ray.origin;
    cand.distance = query.hit.distance;
    cand.direction = direction;
    var bounce1_falloff = 0.0;
    var bounce2_falloff = 0.0;
    var bounce1_color = vec3(0.0);
    var bounce2_color = vec3(0.0);
    var bounce1_mat_color = vec3(0.0);
    var bounce2_mat_color = vec3(0.0);
    var direct_color = vec3(0.0);
    if query.hit.distance != F32_MAX && query.hit.distance > 0.0 {
        cand.ray_hit_pos = ray.origin + ray.direction * cand.distance;
        bounce1_falloff = (1.0 / (1.0 + query.hit.distance * query.hit.distance));
        bounce1_mat_color = get_hit_material(query).color;
    }

    // first see if we hit somewhere on the screen
    var screen_color = vec3(-1.0);
#ifdef CDIFFUSE_SCREEN_SAMPLE
    if query.hit.distance != F32_MAX && query.hit.distance > 0.0 {
        screen_color = get_screen_color_from_pos(cand.ray_hit_pos, ray.direction);
    }
#endif

    var ray_hit_voxel = ivoxel_clamp(vec3<i32>(position_world_to_fvoxel(cand.ray_hit_pos)));
    var cache_color = vec4(0.0);
    var use_voxel_for_1st_bounce = false;
    var only_use_voxel_for_1st_bounce = false;
    var same_voxel_as_origin = false;
    if query.hit.distance != F32_MAX {
        cache_color = textureLoad(voxel_cache, ray_hit_voxel, 0);
        if cache_color.w <= 0.0 { // TODO use time
            cache_color = vec4(0.0);
        } else if query.hit.distance > VOXEL_SIZE * 3.0 {
            use_voxel_for_1st_bounce = true;
#ifdef VOXEL_1ST_BOUNCE_FALLBACK            
            if urand.w > 0.5 {
                only_use_voxel_for_1st_bounce = true;
            }
#endif
        }
        if all(ivoxel_clamp(vec3<i32>(position_world_to_fvoxel(ray.origin))) == ray_hit_voxel) {
            same_voxel_as_origin = true;
            use_voxel_for_1st_bounce = false;
        }
    }   
        
    if query.hit.distance != F32_MAX && query.hit.distance > min_hit_dist {
        if !only_use_voxel_for_1st_bounce {
            if urand.z > 0.5 {
                // Trace to sun
                var sray = new_ray(cand.ray_hit_pos, -sun_dir);
                let sun_query = scene_query(sray);
                if sun_query.hit.distance == F32_MAX {
                    bounce1_color += sun_color * nee;
                }
            } else {
                // trace 2nd bounce
                let hit_surface_normal = compute_tri_normal(query);
                let tangent_to_world = build_orthonormal_basis(hit_surface_normal);
                let white_frame_noise = white_frame_noise(28471u);
                let urand = fract_blue_noise_for_pixel(ufrag_coord, globals.frame_count + 6329u, white_frame_noise);  
                var bounce_direction = cosine_sample_hemisphere(urand.xy);
                bounce_direction = normalize(bounce_direction * tangent_to_world);
                let origin = cand.ray_hit_pos + -ray.direction * pri_dist_bias + hit_surface_normal * pri_normal_bias;
                var bray = new_ray(origin, bounce_direction);
                let bounce_query = scene_query(bray);        
                if bounce_query.hit.distance != F32_MAX && bounce_query.hit.distance > min_hit_dist {
                    let bray_hit_pos = bray.origin + bray.direction * bounce_query.hit.distance;
                    bounce2_mat_color = get_hit_material(bounce_query).color;
                    var bray_hit_voxel = ivoxel_clamp(vec3<i32>(position_world_to_fvoxel(bray_hit_pos)));
                    let cache_color = textureLoad(voxel_cache, bray_hit_voxel, 0);
                    bounce2_falloff = (1.0 / (1.0 + bounce_query.hit.distance * bounce_query.hit.distance));
                    if all(ivoxel_clamp(vec3<i32>(position_world_to_fvoxel(bray.origin))) != bray_hit_voxel) {
                        bounce2_color += cache_color.rgb;    
                    }
                } else if bounce_query.hit.distance == F32_MAX {
                    bounce1_color += sky_color;   
                }
            }
            // compensate for only tracing 50% of the time:
            bounce1_color *= 2.0;
            bounce2_color *= 2.0;
        }
    } else if query.hit.distance == F32_MAX {
        cand.ray_hit_pos = ray.origin + ray.direction * 9999.9;
        direct_color += sky_color;    
    }


 

    var bounce1_color_cache = bounce1_color;
    if use_voxel_for_1st_bounce {
        // if we have some distance from the hit voxel, include its contribution
        bounce1_color_cache += cache_color.rgb;
        if !only_use_voxel_for_1st_bounce {
            bounce1_color_cache *= 0.5;
        }
    }

    cand.color = max((
        direct_color
        + bounce1_color_cache * bounce1_mat_color * bounce1_falloff
        + bounce2_color * bounce1_mat_color * bounce2_mat_color * bounce1_falloff * bounce2_falloff
        ), vec3(0.0));


    if query.hit.distance != F32_MAX && query.hit.distance > min_hit_dist {
        if !all(ray_hit_voxel == vec3(0)) {
            var hysterisis = 0.05;
            if cache_color.w <= 0.0 {
                hysterisis = 1.0;
            }
            // + bounce2_color * bounce2_mat_color
            let new_cache_color = max(mix(cache_color.rgb, 
                bounce1_color
              + bounce2_color * bounce2_mat_color * bounce2_falloff
                                        , hysterisis), vec3(0.0));
            textureStore(voxel_cache_write, ray_hit_voxel, vec4(new_cache_color, f32(globals.time)));
        }
    }

    
    //{
    //    var hysterisis = 0.05;
    //    var primary_ray_voxel = ivoxel_clamp(vec3<i32>(position_world_to_fvoxel(world_position)));
    //    if !all(primary_ray_voxel == vec3(0)) {
    //        let primary_cache_color = textureLoad(voxel_cache, primary_ray_voxel, 0);
    //        let new_cache_color = max(mix(primary_cache_color.rgb, cand.color, hysterisis), vec3(0.0));
    //        textureStore(voxel_cache_write, primary_ray_voxel, vec4(new_cache_color, f32(globals.time)));
    //    }
    //}

    


    // TODO don't be ridiculous
    if all(invocation_id == vec3(0u)) {
        textureStore(voxel_cache_write, vec3(0), textureLoad(voxel_cache, vec3(0), 0));
    }

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