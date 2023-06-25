#import "shaders/screen_space_passes/common.wgsl"

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
    let pbr = pbr_from_frag_coord(&full_screen_frag_coord);

    var probe = load_probe(ifrag_coord);

    let d = textureLoad(fullscreen_passes_read, ifrag_coord, COLOR_LAYER, 0);
    var probe_proc_color = d.xyz;
    var ssao_focus = d.w;

    
    var pixel_radius = pixel_radius_to_world(1.0, depth_ndc_to_linear(full_screen_frag_coord.z), pbr.is_orthographic);
    // limit minimum pixel radius for things really close to the camera
    pixel_radius = max(pixel_radius, 0.001); 

    //-------------------------------------------------
    //-------------------------------------------------
    //-------------------------------------------------
    
#ifdef SPATIAL_REUSE


    var combined_probe = new_probe();
    combined_probe.pos = probe.pos;
    combined_probe.reproject_fail = false;



    { // sample from path traced
        var samples = 4u;
        var candidates_radius = 16.0 * max(ssao_focus, 0.0);
        let max_dist = 80.0 * pixel_radius * mix(0.5, 1.0, ssao_focus);
        let coplanar_max_dist = 300.0 * pixel_radius;
        //if probe.reproject_fail {
        //    samples *= 2u;
        //    candidates_radius *= 10.0;
        //}
        let low_m_mult = 1.0 + (1.0 - saturate(probe_get_progress(&probe) + 0.8)) * 10.0;
        candidates_radius *= low_m_mult;
    
        for (var i = 0u; i < samples; i+=1u) {
            var max_dist = max_dist;
            let seed = (i + 1u) * samples;
            let white_frame_noise = white_frame_noise(seed + 34531u);
            let urand = fract_blue_noise_for_pixel(ufrag_coord, seed + globals.frame_count, white_frame_noise);  
            var offset = (urand.wz * 2.0 - 1.0);
            var offset_uv = (offset / ftarget_dims) * candidates_radius;
            var coord = vec2<i32>((screen_uv + offset_uv) * ftarget_dims);
            
            let clamped_coord = clamp(coord, vec2(0), vec2<i32>(ftarget_dims));

            // if the coord is off the screen, mirror it back onto the screen
            let flip = vec2<f32>(clamped_coord == coord) * 2.0 - 1.0;
            offset = offset * flip;
            offset_uv = offset_uv * flip;
            coord = vec2<i32>((screen_uv + offset_uv) * ftarget_dims);

            
            var s_probe = load_probe(coord);

            let new_ray_hit_pos = s_probe.ray_hit_pos.xyz;
            let direction = normalize(new_ray_hit_pos - s_probe.pos);
            let dist = distance(new_ray_hit_pos, s_probe.pos);
            
            // TODO select mip based on ftarget_dims relative to ftarget_dims
            let nor_depth = textureLoad(prepass_downsample, vec2<i32>(ftarget_dims * (screen_uv + offset_uv)), 0);

            let probe_to_cand_distance = distance(probe.pos, s_probe.pos);

            let offset_nor_diff = max(dot(nor_depth.xyz, pbr.N) - 0.01, 0.0);

            //var gr = dot(s_probe.color, vec3<f32>(0.2126, 0.7152, 0.0722));
            let brdf = max(dot(pbr.N, direction), 0.0);
            var dist_falloff = (1.0 / (1.0 + dist * dist));
            if dist >= F16_MAX || dist < 0.0 { // hit sky
                dist_falloff = 1.0;
            }

            if coplanar(probe.pos, nor_depth.xyz, s_probe.pos, pbr.N, 0.2, pixel_radius * 4.0) {
                max_dist = coplanar_max_dist;
            }

            if probe_to_cand_distance > max_dist {
                continue;
            }

            let probe_dist_falloff = saturate(max_dist - probe_to_cand_distance) + 1.0;
            
            var new_weight = brdf * probe_dist_falloff * offset_nor_diff; // * (1.0 - saturate(length(offset)))
            new_weight = max(new_weight, 0.0);

            // TODO not the correct way to combine reservoirs
            probe_update(&combined_probe, urand.x, new_weight * f32(s_probe.M), s_probe.ray_hit_pos, probe_resolve(&s_probe));
            //probe_update(&combined_probe, urand.x, new_weight * s_probe.weight * f32(s_probe.M), s_probe.ray_hit_pos, s_probe.color);
            combined_probe.M += s_probe.M;
        }
        
        // Also use the center probe, but make it a bit less likely to be selected (urand.x * 4.0)
        let white_frame_noise = white_frame_noise(74319u);
        let urand = fract_blue_noise_for_pixel(ufrag_coord, globals.frame_count, white_frame_noise); 
        probe_update(&combined_probe, urand.x * 4.0, probe.weight * f32(probe.M), probe.ray_hit_pos, probe_resolve(&probe));
        //let new_brdf = saturate(dot(pbr.N, normalize(combined_probe.ray_hit_pos - probe.pos)));
        combined_probe.weight = (combined_probe.w_sum / f32(combined_probe.M));
    }

#endif //SPATIAL_REUSE

    //-------------------------------------------------
    //-------------------------------------------------
    //-------------------------------------------------



    let hysterisis = select(GI_HYSTERISIS, 1.0, probe.reproject_fail);
#ifdef SPATIAL_REUSE
    var gi = probe_resolve(&combined_probe);
#else
    var gi = probe_resolve(&probe);
#endif

#ifdef SSAO
    // GI is too broad for fine corner detail
    gi *= ssao_focus;
#endif

    probe_proc_color = mix(probe_proc_color, gi, hysterisis);
    textureStore(fullscreen_passes_write, ifrag_coord, COLOR_LAYER, vec4(probe_proc_color, ssao_focus));
    //----------------
    
    store_probe(probe, ifrag_coord);

    textureStore(fullscreen_passes_write, ifrag_coord, SSR_LAYER, textureLoad(fullscreen_passes_read, ifrag_coord, SSR_LAYER, 0));
}