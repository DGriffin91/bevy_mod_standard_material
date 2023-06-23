
#import "shaders/screen_space_passes/common.wgsl"

fn ssgi_restir(ifrag_coord: vec2<i32>, frag_coord: vec4<f32>, pbr: PbrInput, samples: u32) {
    
    let ufrag_coord = vec2<u32>(ifrag_coord.xy);
    let itex_dims = vec2<i32>(textureDimensions(fullscreen_passes_read).xy);
    let tex_dims = vec2<f32>(itex_dims);
    let frag_size = 1.0 / tex_dims;
    let screen_uv = vec2<f32>(ifrag_coord) / tex_dims + frag_size * 0.5;
    var pixel_radius = pixel_radius_to_world(1.0, depth_ndc_to_linear(frag_coord.z), pbr.is_orthographic);
    // limit minimum pixel radius for things really close to the camera
    pixel_radius = max(pixel_radius, 0.001); 

    var world_position_offs = pbr.world_position.xyz + pbr.N * 0.001;

#ifdef SSAO
    var ssao = bad_gtao(frag_coord, world_position_offs, pbr.N).rgb;
#endif
#ifdef SSAO_FOCUS
    let ssao_focus = ssao.x;
#elseif
    let ssao_focus = 1.0;
#endif
    
    var col_accum = vec3(0.0);
    var accum_tot_w = 0.0;
    var ssgi_conf = 0.0;
    var pt_col_accum = vec3(0.0);
    var pt_accum_tot_w = 0.0;

    var probe = new_probe();
    probe.pos = world_position_offs;
    
    let closest_motion_vector = prepass_motion_vector(vec4<f32>(screen_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
    let screen_history_uv = screen_uv - closest_motion_vector;
    let history_hit_screen = screen_history_uv.x > 0.0 && screen_history_uv.x < 1.0 && screen_history_uv.y > 0.0 && screen_history_uv.y < 1.0;
    let screen_space_passes_coord = vec2<i32>(tex_dims * screen_history_uv);
    var sample_coord = screen_space_passes_coord;
    var sample_uv = screen_uv;

    let first_probe_pos = load_probe_pos(sample_coord).xyz;
    // locate closest probe
    if history_hit_screen {
        var selected_offset = vec2(0, 0);
        var closest = distance(first_probe_pos, world_position_offs);
        probe.pos = first_probe_pos;
        if closest > pixel_radius * 1.5 {
            for (var x = -1; x <= 1; x += 1) {
                for (var y = -1; y <= 1; y += 1) {
                    let offset = vec2(x, y);
                    let coord = screen_space_passes_coord + offset;
                    if any(coord < vec2(0)) || any(coord >= itex_dims) {
                        continue;
                    }
                    let test_probe_pos = load_probe_pos(coord).xyz;
                    let dist = distance(test_probe_pos, world_position_offs);
                    if dist < closest {
                        selected_offset = offset;
                        closest = dist;
                        probe.pos = test_probe_pos;
                    }
                }
            }
        }

        sample_coord = screen_space_passes_coord + selected_offset;
        sample_uv = (vec2<f32>(sample_coord) + 0.5) / vec2<f32>(tex_dims);
        
        probe = load_probe_with_pos(sample_coord, probe.pos);
    }

    

    var max_radius_dist = pixel_radius * 3.0;

    //normal is basically ignored here
    if coplanar(probe.pos, pbr.N, world_position_offs, pbr.N, 0.1, pixel_radius) {
        max_radius_dist *= 8.0;
    }

    probe.reproject_fail = distance(probe.pos, world_position_offs) > max_radius_dist || !history_hit_screen;


    // if closest probe is too far, reset reservoir
    if probe.reproject_fail || probe.M == 0u {
        probe = new_probe();
        probe.pos = world_position_offs;
    }

    // randomly reset position
    if hash_noise(ifrag_coord, globals.frame_count) > 0.9 {
        probe.pos = world_position_offs;
    }

    probe_scale(&probe, MAX_M);

    
    let white_frame_noise = white_frame_noise(3812u);
    let seed = samples + globals.frame_count * samples;
    let urand = fract_blue_noise_for_pixel(ufrag_coord, seed, white_frame_noise);  

    let screen_passes_dims = vec2<f32>(textureDimensions(screen_passes_read).xy);
#ifdef RAY_MARCH
    var st: ScreenTraceResult;
    var candidates_samples = 2u;
    var candidates_radius = 1.5 * ssao_focus;
    let max_dist = 30.0 * pixel_radius * mix(0.5, 1.0, ssao_focus);
    for (var i = 0u; i < candidates_samples; i+=1u) {
        var coord = vec2(0);
        var offset_uv = vec2(0.0);
        var offset = vec2(0.0);
        
        for (var j = 0u; j < candidates_samples; j+=1u) {
            let seed = (i + 1u) * (j + 1u) * candidates_samples + globals.frame_count * candidates_samples;
            let white_frame_noise = white_frame_noise(seed + 2351u);
            let urand = fract_blue_noise_for_pixel(ufrag_coord, seed, white_frame_noise);  
            offset = (urand.wz * 2.0 - 1.0);
            //if i > 0u {
            offset_uv = offset / screen_passes_dims * candidates_radius;
            //}
            coord = vec2<i32>((screen_uv + offset_uv) * screen_passes_dims);
            let clamped_coord = clamp(coord, vec2(0), vec2<i32>(screen_passes_dims));
            // try to find coord that is not off screen
            if all(clamped_coord == coord) {
                break;
            }
            coord = clamped_coord;
        }

        let st_ray_start = textureLoad(screen_passes_read, coord, 0u, 0).xyz;
        let probe_to_cand_distance = distance(st_ray_start, probe.pos);

        if probe_to_cand_distance > max_dist {
            // Sample is too far away
            // Doesn't increase M
            // TODO scale by frag depth?
            continue;
        }

        st.pos = textureLoad(screen_passes_read, coord, 1u, 0).xyz;
        st.color = textureLoad(screen_passes_read, coord, 2u, 0).xyz;
        let st_data = textureLoad(screen_passes_read, coord, 3u, 0);
        st.derate = saturate(st_data.x);
        st.hit = bool(u32(st_data.y));
        st.backface = bool(u32(st_data.z));
        let st_distance = distance(st.pos, world_position_offs);
        let st_direction = normalize(st.pos - world_position_offs);

        let hit = st_distance >= 0.0 && st.hit;

#ifdef USE_PATH_TRACED
#ifdef RAY_MARCH
    // if we are filling in PT with SSGI, don't increase M if we missed SSGI
    if !hit {
        continue;
    }
#endif
#endif

        if hit { // hit
            var gr = dot(st.color, vec3<f32>(0.2126, 0.7152, 0.0722));
            let brdf = max(dot(pbr.N, st_direction), 0.0);
            let dist_falloff = (1.0 / (1.0 + st_distance * st_distance));
            var new_weight = gr * dist_falloff * brdf * f32(!st.backface) * f32(st.hit) * st.derate;
            new_weight = max(new_weight, 0.0);

            probe.w_sum += new_weight;

            var threshold = new_weight / w_sum;

            if i == 0u && (probe.M == 0u || ray_hit_pos.x == F32_MAX) {
                threshold = 1.0;
            }

            probe_update(&probe, urand.z, new_weight, st.pos, st.color)

            {
                let probe_dist_falloff = saturate(max_dist - probe_to_cand_distance) + 1.0;
                let offset_w = saturate(1.0 / (length(offset) * candidates_radius)) + 1.0;
                let w = max(dist_falloff * brdf * f32(!st.backface) * f32(st.hit) * st.derate * offset_w * probe_dist_falloff, 0.0);
                col_accum += st.color * w * brdf;
                accum_tot_w += w;
                ssgi_conf += (1.0 / f32(candidates_samples)) * f32(!st.backface) * f32(st.hit);
            }
        }

        M += 1u;
    }
#endif //RAY_MARCH

#ifdef USE_PATH_TRACED
    { // sample from path traced
        let pt_samples = 4u;
        var candidates_radius = 7.0 * max(ssao_focus, 0.2);
        let path_trace_image_dims = vec2<f32>(textureDimensions(path_trace_image).xy);
        let pt_max_dist = 30.0 * pixel_radius * mix(0.5, 1.0, ssao_focus);
        let coplanar_max_dist = 300.0;
        for (var i = 0u; i < pt_samples; i+=1u) {
            var max_dist = pt_max_dist;
            let seed = (i + 1u) * pt_samples + globals.frame_count;
            let white_frame_noise = white_frame_noise(seed + 28534u);
            let urand = fract_blue_noise_for_pixel(ufrag_coord, seed + globals.frame_count, white_frame_noise);  
            var offset = (urand.wz * 2.0 - 1.0);
            var offset_uv = (offset / path_trace_image_dims) * candidates_radius;
            var coord = vec2<i32>((screen_uv + offset_uv) * path_trace_image_dims);
            
            let clamped_coord = clamp(coord, vec2(0), vec2<i32>(path_trace_image_dims));

            // if the coord is off the screen, mirror it back onto the screen
            let flip = vec2<f32>(clamped_coord == coord) * 2.0 - 1.0;
            offset = offset * flip;
            offset_uv = offset_uv * flip;
            coord = vec2<i32>((screen_uv + offset_uv) * path_trace_image_dims);

            let color = textureLoad(path_trace_image, coord, 0u, 0).rgb;
            let pt_world_position = textureLoad(path_trace_image, coord, 1u, 0).xyz;
            let ray_hit = textureLoad(path_trace_image, coord, 2u, 0);
            let new_ray_hit_pos = ray_hit.xyz;
            let direction = normalize(new_ray_hit_pos - pt_world_position);
            let dist = ray_hit.w;
            
            // TODO select mip based on path_trace_image_dims relative to tex_dims
            let nor_depth = textureLoad(prepass_downsample, vec2<i32>(tex_dims * (screen_uv + offset_uv)) / 2, 1);

            let probe_to_cand_distance = distance(probe.pos, pt_world_position);

            if coplanar(probe.pos, nor_depth.xyz, pt_world_position, pbr.N, 0.2, pixel_radius * 4.0) {
                max_dist = coplanar_max_dist;
            }

            if probe_to_cand_distance > max_dist {
                continue; // Sample is too far away. Doesn't increase M
            }

            let offset_nor_diff = max(dot(nor_depth.xyz, pbr.N) - 0.01, 0.0);

            var gr = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
            let brdf = max(dot(pbr.N, direction), 0.0);
            var dist_falloff = (1.0 / (1.0 + dist * dist));
            if dist >= F16_MAX || dist < 0.0 { // hit sky
                dist_falloff = 1.0;
            }

            let probe_dist_falloff = saturate(max_dist - probe_to_cand_distance);
            
            var new_weight = gr * brdf * dist_falloff * probe_dist_falloff * offset_nor_diff;
            new_weight = max(new_weight, 0.0);
            
            probe_update(&probe, urand.x, new_weight, new_ray_hit_pos, color);
            
        }
    }
#endif //USE_PATH_TRACED

    var probe_proc_color = vec3(0.0);

    if !probe.reproject_fail {
        probe_proc_color = textureLoad(fullscreen_passes_read, sample_coord, COLOR_LAYER, 0).xyz;
    }

    textureStore(fullscreen_passes_write, ifrag_coord, COLOR_LAYER, vec4(probe_proc_color, ssao_focus));

    store_probe(probe, ifrag_coord);
    
    


#ifdef SSR
    var ssr_hysterisis = 0.1;

    let F0 = get_f0(pbr.material.reflectance, pbr.material.metallic, pbr.material.base_color.rgb);
    let roughness = perceptualRoughnessToRoughness(pbr.material.perceptual_roughness);
    var ssr = bad_ssr(ifrag_coord, pbr.N, pbr.world_position.xyz, roughness, F0, FULL_SSR_SAMPLES, vec2(2.0, 3.0)).rgb;

    
    //let half_ssr = textureLoad(screen_passes_read, vec2<i32>(screen_uv * screen_passes_dims), 4u, 0).rgb;
    //ssr = mix(half_ssr, ssr, 0.5);

    if !probe.reproject_fail {
        let prev_ssr = textureLoad(fullscreen_passes_read, sample_coord, SSR_LAYER, 0).rgb;
        ssr = mix(prev_ssr, ssr, ssr_hysterisis);
    }

    ssr = clamp(ssr, vec3(0.0), vec3(100.0));

    textureStore(fullscreen_passes_write, ifrag_coord, SSR_LAYER, vec4(ssr, 0.0));
#else
    textureStore(fullscreen_passes_write, ifrag_coord, SSR_LAYER, vec4(0.0));
#endif //SSR
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    //if true {return;}
    let ufrag_coord = invocation_id.xy;
    let ifrag_coord = vec2<i32>(ufrag_coord);
    var frag_coord = vec4(vec2<f32>(ufrag_coord), 0.0, 0.0);

    let target_dims = vec2<i32>(textureDimensions(fullscreen_passes_write).xy);
    if any(ifrag_coord >= target_dims) {
        return;
    }
    
    let ftarget_dims = vec2<f32>(target_dims);
    let frag_size = 1.0 / ftarget_dims;
    var screen_uv = frag_coord.xy / ftarget_dims + frag_size * 0.5;

    var full_screen_frag_coord = vec4(screen_uv * view.viewport.zw, 0.0, 0.0);
    let pbr_input = pbr_from_frag_coord(&full_screen_frag_coord);

    ssgi_restir(ifrag_coord, full_screen_frag_coord, pbr_input, 1u);
}


