// Ground Truth-based Ambient Occlusion (GTAO)
// Paper: https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
// Presentation: https://blog.selfshadow.com/publications/s2016-shading-course/activision/s2016_pbs_activision_occlusion.pdf

// Source code heavily based on XeGTAO v1.30 from Intel
// https://github.com/GameTechDev/XeGTAO/blob/0d177ce06bfa642f64d8af4de1197ad1bcb862d4/Source/Rendering/Shaders/XeGTAO.hlsli


const HALF_PI: f32 = 1.57079632679;

fn rotation_matrix(to: vec3<f32>) -> mat3x3<f32> {
    let fromm = vec3(0.0, 0.0, -1.0);

    let e = dot(fromm, to);
    let f = abs(e);

    if f > 0.9997 {
        return mat3x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    }

    let v = cross(fromm, to);
    let h = 1.0 / (1.0 + e);
    let hvx = h * v.x;
    let hvz = h * v.z;
    let hvxy = hvx * v.y;
    let hvxz = hvx * v.z;
    let hvyz = hvz * v.y;

    var mtx: mat3x3<f32>;
    mtx[0][0] = e + hvx * v.x;
    mtx[0][1] = hvxy - v.z;
    mtx[0][2] = hvxz + v.y;

    mtx[1][0] = hvxy + v.z;
    mtx[1][1] = e + h * v.y * v.y;
    mtx[1][2] = hvyz - v.x;

    mtx[2][0] = hvxz - v.y;
    mtx[2][1] = hvyz + v.x;
    mtx[2][2] = e + hvz * v.z;

    return mtx;
}


//ScreenSpaceAmbientOcclusionSettings::Low => (1, 2), // 4 spp (1 * (2 * 2)), plus optional temporal samples
//ScreenSpaceAmbientOcclusionSettings::Medium => (2, 2), // 8 spp (2 * (2 * 2)), plus optional temporal samples
//ScreenSpaceAmbientOcclusionSettings::High => (3, 3), // 18 spp (3 * (3 * 2)), plus optional temporal samples
//ScreenSpaceAmbientOcclusionSettings::Ultra => (9, 3), // 54 spp (9 * (3 * 2)), plus optional temporal samples
const SLICE_COUNT = 3u;
const SAMPLES_PER_SLICE_SIDE = 3u;
const DIST_MULT = 1.0;

fn bad_gtao(frag_coord: vec4<f32>, world_position: vec3<f32>, surface_normal: vec3<f32>) -> vec4<f32> {
    let surface_normal = normalize(surface_normal);

    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);

    let depth_tex_dims = vec2<f32>(textureDimensions(depth_prepass_texture));

    let slice_count = f32(SLICE_COUNT);
    let samples_per_slice_side = f32(SAMPLES_PER_SLICE_SIDE);
    let effect_radius = 0.5 * 1.457;
    let falloff_range = 0.615 * effect_radius;
    let falloff_from = effect_radius * (1.0 - 0.615);
    let falloff_mul = -1.0 / falloff_range;
    let falloff_add = falloff_from / falloff_range + 1.0;

    let pixel_coordinates = frag_coord.xy;
    let uv = frag_coord_to_uv(frag_coord.xy);

    var pixel_depth = frag_coord.z;
    pixel_depth += 0.00001; // Avoid depth precision issues

    let pixel_position = position_world_to_view(world_position);
    let pixel_normal = direction_world_to_view(surface_normal); //load_normal_view_space(uv);
    let view_vec = normalize(-pixel_position);

    let white_frame_noise = white_frame_noise(123u);

    //let noise = vec2(
    //    hash_noise(ifrag_coord, globals.frame_count), 
    //    hash_noise(ifrag_coord, globals.frame_count + 1024u),
    //);

    var noise = vec2(
        fract(blue_noise_for_pixel(ufrag_coord, 0u + globals.frame_count) + white_frame_noise.x),
        fract(blue_noise_for_pixel(ufrag_coord, 1u + globals.frame_count) + white_frame_noise.y),
    );

    //let noise = load_noise(pixel_coordinates);
    let sample_scale = (-0.5 * effect_radius * view.projection[0][0]) / pixel_position.z;

    var visibility = 0.0;
    var bent_normal = vec3(0.0);

    var color = vec3(0.0);

    for (var slice_t = 0.0; slice_t < slice_count; slice_t += 1.0) {
        let slice = slice_t + noise.x;
        let phi = (PI / slice_count) * slice;
        let omega = vec2<f32>(cos(phi), sin(phi));

        let direction = vec3<f32>(omega.xy, 0.0);
        let orthographic_direction = direction - (dot(direction, view_vec) * view_vec);
        let axis = cross(direction, view_vec);
        let projected_normal = pixel_normal - axis * dot(pixel_normal, axis);
        let projected_normal_length = length(projected_normal);

        let sign_norm = sign(dot(orthographic_direction, projected_normal));
        let cos_norm = saturate(dot(projected_normal, view_vec) / projected_normal_length);
        let n = sign_norm * fast_acos(cos_norm);

        let min_cos_horizon_1 = cos(n + HALF_PI);
        let min_cos_horizon_2 = cos(n - HALF_PI);
        var cos_horizon_1 = min_cos_horizon_1;
        var cos_horizon_2 = min_cos_horizon_2;
        let sample_mul = vec2<f32>(omega.x, -omega.y) * sample_scale;
        var s_color = vec3(0.0);
        for (var sample_t = 0.0; sample_t < samples_per_slice_side; sample_t += 1.0) {
            // TODO check noise method
            //var sample_noise = (slice_t + sample_t * samples_per_slice_side) * 0.6180339887498948482;
            //sample_noise = fract(noise.y + sample_noise);

            var sample_noise = fract(
                blue_noise_for_pixel(ufrag_coord, 2u + u32(sample_t * slice_t) * SAMPLES_PER_SLICE_SIDE * SLICE_COUNT + globals.frame_count) 
                + white_frame_noise.z
            );

            //var sample_noise = fract(
            //    hash_noise(ifrag_coord, 3u + u32(sample_t * slice_t) * SAMPLES_PER_SLICE_SIDE * SLICE_COUNT + globals.frame_count) 
            //    + white_frame_noise.z
            //);

            var s = ((sample_t + sample_noise) / samples_per_slice_side) * DIST_MULT;
            s *= s; // https://github.com/GameTechDev/XeGTAO#sample-distribution
            let sample = s * sample_mul;

            let sample_mip_level = clamp(log2(length(sample)) - 3.3, 0.0, 5.0); // https://github.com/GameTechDev/XeGTAO#memory-bandwidth-bottleneck

            let uv1 = uv + sample;
            let uv2 = uv - sample;

            let depth_1 = prepass_depth(vec4(uv1 * depth_tex_dims, 0.0, 0.0), 0u);
            let sample_position_1 = position_ndc_to_view(vec3(uv_to_ndc(uv1), depth_1));

            let depth_2 = prepass_depth(vec4(uv2 * depth_tex_dims, 0.0, 0.0), 0u);
            let sample_position_2 = position_ndc_to_view(vec3(uv_to_ndc(uv2), depth_2));

            let sample_difference_1 = sample_position_1 - pixel_position;
            let sample_difference_2 = sample_position_2 - pixel_position;
            let sample_distance_1 = length(sample_difference_1);
            let sample_distance_2 = length(sample_difference_2);
            var sample_cos_horizon_1 = dot(sample_difference_1 / sample_distance_1, view_vec);
            var sample_cos_horizon_2 = dot(sample_difference_2 / sample_distance_2, view_vec);

            let weight_1 = saturate(sample_distance_1 * falloff_mul + falloff_add);
            let weight_2 = saturate(sample_distance_2 * falloff_mul + falloff_add);
            sample_cos_horizon_1 = mix(min_cos_horizon_1, sample_cos_horizon_1, weight_1);
            sample_cos_horizon_2 = mix(min_cos_horizon_2, sample_cos_horizon_2, weight_2);

            cos_horizon_1 = max(cos_horizon_1, sample_cos_horizon_1);
            cos_horizon_2 = max(cos_horizon_2, sample_cos_horizon_2);

        }

        let horizon_1 = fast_acos(cos_horizon_1);
        let horizon_2 = -fast_acos(cos_horizon_2);
        let v1 = (cos_norm + 2.0 * horizon_1 * sin(n) - cos(2.0 * horizon_1 - n)) / 4.0;
        let v2 = (cos_norm + 2.0 * horizon_2 * sin(n) - cos(2.0 * horizon_2 - n)) / 4.0;
        let s_visibility = projected_normal_length * (v1 + v2);
        visibility += s_visibility;
        color += s_color;

        let t0 = (6.0 * sin(horizon_2 - n) - sin(3.0 * horizon_2 - n) + 6.0 * sin(horizon_1 - n) - sin(3.0 * horizon_1 - n) + 16.0 * sin(n) - 3.0 * (sin(horizon_2 + n) + sin(horizon_1 + n))) / 12.0;
        let t1 = (-cos(3.0 * horizon_2 - n) - cos(3.0 * horizon_1 - n) + 8.0 * cos(n) - 3.0 * (cos(horizon_2 + n) + cos(horizon_1 + n))) / 12.0;
        bent_normal += (rotation_matrix(view_vec) * vec3(omega * vec2(t0, t1), -t1)) * projected_normal_length;
    }

    visibility /= slice_count;
    //visibility = pow(visibility, 9.0);
    visibility = clamp(visibility, 0.03, 1.0);
    bent_normal = normalize(bent_normal);

    color = color / (samples_per_slice_side * slice_count);

    //textureStore(ambient_occlusion, pixel_coordinates, vec4<f32>(bent_normal, visibility));
    return vec4(vec3(visibility), 1.0);
}