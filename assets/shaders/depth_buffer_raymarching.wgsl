fn bilinear_depth(uv: vec2<f32>, size: vec2<f32>, sample_index: u32) -> f32 {
	let pos = uv * size - 0.5;
    let f = fract(pos);
    
    let pos_top_left = floor(pos);
    
    // we are sample center, so it's the same as point sample
    let tl = prepass_depth(vec4<f32>(pos_top_left + vec2(0.5, 0.5), 0.0, 0.0), sample_index);
    let tr = prepass_depth(vec4<f32>(pos_top_left + vec2(1.5, 0.5), 0.0, 0.0), sample_index);
    let bl = prepass_depth(vec4<f32>(pos_top_left + vec2(0.5, 1.5), 0.0, 0.0), sample_index);
    let br = prepass_depth(vec4<f32>(pos_top_left + vec2(1.5, 1.5), 0.0, 0.0), sample_index);
    
    return mix(mix(tl, tr, f.x), mix(bl, br, f.x), f.y);
}

// Copyright (c) 2023 Tomasz Stachowiak
//
// This contribution is dual licensed under EITHER OF
// 
//     Apache License, Version 2.0, (http://www.apache.org/licenses/LICENSE-2.0)
//     MIT license (http://opensource.org/licenses/MIT)
// 
// at your option.
// https://gist.github.com/h3r2tic/9c8356bdaefbe80b1a22ae0aaee192db

// OpenGL: 1, Vulkan: -1
const CLIP_SPACE_UV_Y_DIR = -1.0;

fn cs_to_uv(cs: vec2<f32>) -> vec2<f32> {
    return cs * vec2(0.5, CLIP_SPACE_UV_Y_DIR * 0.5) + vec2(0.5, 0.5);
}

fn uv_to_cs(uv: vec2<f32>) -> vec2<f32> {
    return (uv - vec2(0.5)) * vec2(2.0, CLIP_SPACE_UV_Y_DIR * 2.0);
}

struct DistanceWithPenetration {
    root: RootFinderInput,
    /// Conservative estimate of depth to which the ray penetrates the marched surface.
    penetration: f32,
};

struct DepthRaymarchDistanceFn {
    depth_tex_size: vec2<f32>,
    sample_index: u32,
    depth_thickness: f32,
    march_behind_surfaces: bool,
}

// Returns: penetration 
// Conservative estimate of depth to which the ray penetrates the marched surface.
fn distance_fn(_this_: DepthRaymarchDistanceFn, ray_point_cs: vec3<f32>) -> DistanceWithPenetration {
    let interp_uv = cs_to_uv(ray_point_cs.xy);

    let ray_depth = 1.0 / ray_point_cs.z;

    // We're using both point-sampled and bilinear-filtered values from the depth buffer.
    //
    // That's really stupid but works like magic. For samples taken near the ray origin,
    // the discrete nature of the depth buffer becomes a problem. It's not a land of continuous surfaces,
    // but a bunch of stacked duplo bricks.
    //
    // Technically we should be taking discrete steps in this duplo land, but then we're at the mercy
    // of arbitrary quantization of our directions -- and sometimes we'll take a step which would
    // claim that the ray is occluded -- even though the underlying smooth surface wouldn't occlude it.
    //
    // If we instead take linear taps from the depth buffer, we reconstruct the linear surface.
    // That fixes acne, but introduces false shadowing near object boundaries, as we now pretend
    // that everything is shrink-wrapped by this continuous 2.5D surface, and our depth thickness
    // heuristic ends up falling apart.
    //
    // The fix is to consider both the smooth and the discrete surfaces, and only claim occlusion
    // when the ray descends below both.
    //
    // The two approaches end up fixing each other's artifacts:
    // * The false occlusions due to duplo land are rejected because the ray stays above the smooth surface.
    // * The shrink-wrap surface is no longer continuous, so it's possible for rays to miss it.

    //let linear_depth = 1.0 / depth_tex.SampleLevel(sampler_llc, interp_uv, 0);
    //let unfiltered_depth = 1.0 / depth_tex.SampleLevel(sampler_nnc, interp_uv, 0);
    
    // Manual bilinear; in case using a linear sampler on the depth format is not supported.
    // TODO use textureGather
    // let depth_vals = textureGather(depth_prepass_texture, interp_uv, 0);
    // let frac_px = frac(interp_uv * depth_tex_size - 0.5.xx);
    // let linear_depth = 1.0 / lerp(
    //     lerp(depth_vals.w, depth_vals.z, frac_px.x),
    //     lerp(depth_vals.x, depth_vals.y, frac_px.x),
    //     frac_px.y
    // );
    
    let linear_depth = 1.0 / bilinear_depth(interp_uv, _this_.depth_tex_size, _this_.sample_index);
    let unfiltered_depth = 1.0 / prepass_depth(vec4<f32>(interp_uv * _this_.depth_tex_size, 0.0, 0.0), _this_.sample_index);

    let max_depth = max(linear_depth, unfiltered_depth);
    let min_depth = min(linear_depth, unfiltered_depth);
    // Just for sanity checking. Should not be used.
    //max_depth = unfiltered_depth;
    //min_depth = unfiltered_depth;

    let bias = 0.000001;
    
    var res: DistanceWithPenetration;
    res.root.distance = max_depth * (1.0 + bias) - ray_depth;

    // This will be used at the end of the ray march to potentially discard the hit.
    res.penetration = ray_depth - min_depth;

    if (_this_.march_behind_surfaces) {
        res.root.valid = res.penetration < _this_.depth_thickness;
    } else {
        res.root.valid = true;
    }

    return res;
}

struct RootFinderInput {
    /// Distance to the surface of which a root we're trying to find
    distance: f32,

    /// Whether to consider this sample valid for intersection.
    /// Mostly relevant for allowing the ray marcher to travel behind surfaces,
    /// as it will mark surfaces it travels under as invalid.
    valid: bool,
}

struct HybridRootFinder {
    linear_steps: u32,
    bisection_steps: u32,
    use_secant: bool,

    jitter: f32,
    min_t: f32,
    max_t: f32,
}

fn new_with_linear_steps(v: u32) -> HybridRootFinder {
    var res: HybridRootFinder;
    res.linear_steps = v;
    res.bisection_steps = 0u;
    res.use_secant = false;
    res.jitter = 0.0;
    res.min_t = 0.0;
    res.max_t = 1.0;
    return res;
}

struct RootResult {
    hit_d: DistanceWithPenetration,
    hit_t: f32,
    intersected: bool,
}

fn find_root(hrf: HybridRootFinder, drd: DepthRaymarchDistanceFn, start: vec3<f32>, end: vec3<f32>) -> RootResult {
    var res: RootResult;

    let dir = end - start;

    var min_t = hrf.min_t;
    var max_t = hrf.max_t;

    var min_d: DistanceWithPenetration;
    var max_d: DistanceWithPenetration;

    let step_size = (max_t - min_t) / f32(hrf.linear_steps);

    var intersected = false;

    //
    // Ray march using linear steps

    if (hrf.linear_steps > 0u) {
        let candidate_t = min_t + step_size * hrf.jitter;
        let candidate = start + dir * candidate_t;
        let candidate_d = distance_fn(drd, candidate);
        intersected = candidate_d.root.distance < 0.0 && candidate_d.root.valid;

        if (intersected) {
            max_t = candidate_t;
            max_d = candidate_d;
            // The `[min_t .. max_t]` interval contains an intersection. End the linear search.
        } else {
            // No intersection yet. Carry on.
            min_t = candidate_t;
            min_d = candidate_d;

            for (var step = 1u; step < hrf.linear_steps; step += 1u) {
                // TODO If using TAA:
                let candidate_t = min_t + step_size;
                // TODO If not using TAA:
                //let candidate_t = min_t + step_size + step_size * interleaved_gradient_noise(cs_to_uv(candidate.xy * drd.depth_tex_size), step);

                let candidate = start + dir * candidate_t;
                let candidate_d = distance_fn(drd, candidate);
                intersected = candidate_d.root.distance < 0.0 && candidate_d.root.valid;

                if (intersected) {
                    max_t = candidate_t;
                    max_d = candidate_d;
                    // The `[min_t .. max_t]` interval contains an intersection. End the linear search.
                    break;
                } else {
                    // No intersection yet. Carry on.
                    min_t = candidate_t;
                    min_d = candidate_d;
                }
            }
        }
    }

    //
    // Refine the hit using bisection

    if (intersected) {
        for (var step = 0u; step < hrf.bisection_steps; step += 1u) {
            let mid_t = (min_t + max_t) * 0.5;
            let candidate = start + dir * mid_t;
            let candidate_d = distance_fn(drd, candidate);

            if (candidate_d.root.distance < 0.0 && candidate_d.root.valid) {
                // Intersection at the mid point. Refine the first half.
                max_t = mid_t;
                max_d = candidate_d;
            } else {
                // No intersection yet at the mid point. Refine the second half.
                min_t = mid_t;
                min_d = candidate_d;
            }
        }

        if (hrf.use_secant) {
            // Finish with one application of the secant method
            let total_d = min_d.root.distance + -max_d.root.distance;

            let mid_t = mix(min_t, max_t, min_d.root.distance / total_d);
            let candidate = start + dir * mid_t;
            let candidate_d = distance_fn(drd, candidate);

            // Only accept the result of the secant method if it improves upon the previous result.
            //
            // Technically this should be `abs(candidate_d.distance) < min(min_d.distance, -max_d.distance) * frac`,
            // but this seems sufficient.
            if (abs(candidate_d.root.distance) < min_d.root.distance * 0.9 && candidate_d.root.valid) {
                res.hit_t = mid_t;
                res.hit_d = candidate_d;
            } else {
                res.hit_t = max_t;
                res.hit_d = max_d;
            }

            res.intersected = true;
            return res;
        } else {
            res.hit_t = max_t;
            res.hit_d = max_d;
            res.intersected = true;
            return res;
        }
    } else {
        // Mark the conservative miss distance.
        res.hit_t = min_t;
        res.intersected = false;
        return res;
    }
}

struct DepthRayMarchResult {
    /// Approximate UV of the hit.
    hit_uv: vec2<f32>,

    /// The distance that the hit point penetrates into the hit surface.
    /// Will normally be non-zero due to limited precision of the ray march.
    hit_penetration: f32,

    /// Ditto, within the range `0..DepthRayMarch::depth_thickness_linear_z`
    hit_penetration_frac: f32,

    /// In case of a hit, the normalized distance to it.
    ///
    /// In case of a miss, the furthest the ray managed to travel, which could either be
    /// exceeding the max range, or getting behind a surface further than the depth thickness.
    ///
    /// Range: `0..=1` as a lerp factor over `ray_start_cs..=ray_end_cs`.
    hit_t: f32,

    /// True if the raymarch hit something.
    hit: bool,
};

struct DepthRayMarch {
    /// Number of steps to be taken at regular intervals to find an initial intersection.
    /// Must not be zero.
    linear_steps: u32,

    /// Number of steps in a bisection (binary search) to perform once the linear search
    /// has found an intersection. Helps narrow down the hit, increasing the chance of
    /// the secant method finding an accurate hit point.
    ///
    /// Useful when sampling color, e.g. SSR or SSGI, but pointless for contact shadows.
    bisection_steps: u32,

    /// Approximate the root position using the secant method -- by solving for line-line
    /// intersection between the ray approach rate and the surface gradient.
    ///
    /// Useful when sampling color, e.g. SSR or SSGI, but pointless for contact shadows.
    use_secant: bool,

    /// Jitter to apply to the first step of the linear search; 0..=1 range, mapping
    /// to the extent of a single linear step in the first phase of the search.
    jitter: f32,

    /// Clip space coordinates (w=1) of the ray.
    ray_start_cs: vec3<f32>,
    ray_end_cs: vec3<f32>,

    /// Should be used for contact shadows, but not for any color bounce, e.g. SSR.
    ///
    /// For SSR etc. this can easily create leaks, but with contact shadows it allows the rays
    /// to pass over invalid occlusions (due to thickness), and find potentially valid ones ahead.
    ///
    /// Note that this will cause the linear search to potentially miss surfaces,
    /// because when the ray overshoots and ends up penetrating a surface further than
    /// `depth_thickness_linear_z`, the ray marcher will just carry on.
    ///
    /// For this reason, this may require a lot of samples, or high depth thickness,
    /// so that `depth_thickness_linear_z >= world space ray length / linear_steps`.
    march_behind_surfaces: bool,

    /// When marching the depth buffer, we only have 2.5D information, and don't know how
    /// thick surfaces are. We shall assume that the depth buffer fragments are litte squares
    /// with a constant thickness defined by this parameter.
    depth_thickness_linear_z: f32,

    /// Size of the depth buffer we're marching in, in pixels.
    depth_tex_size: vec2<f32>,
} 

fn DepthRayMarch_new_from_depth(depth_tex_size: vec2<f32>) -> DepthRayMarch {
    var res: DepthRayMarch;
    res.linear_steps = 4u;
    res.bisection_steps = 0u;
    res.depth_tex_size = depth_tex_size;
    res.depth_thickness_linear_z = 1.0;
    res.march_behind_surfaces = false;
    return res;
}

/// March towards a clip-space direction.
/// If `infinite` is `true`, then the ray is extended to cover the whole view frustum.
/// If `infinite` is `false`, then the ray length is that of the `dir_cs` parameter.
//
/// Must be called after `from_cs`, as it will clip the world-space ray to the view frustum.
fn to_cs_dir_impl(_this_: DepthRayMarch, dir_cs: vec4<f32>, infinite: bool) -> DepthRayMarch {
    var end_cs = vec4(_this_.ray_start_cs, 1.0) + dir_cs;
    end_cs /= end_cs.w;

    var delta_cs = end_cs.xyz - _this_.ray_start_cs;
    
    // Clip the ray to the frustum
    let dist_to_edge = (sign(delta_cs.xy) - _this_.ray_start_cs.xy) / delta_cs.xy;
    let min_dist_to_edge = min(dist_to_edge.x, dist_to_edge.y);

    if (infinite) {
        delta_cs *= min_dist_to_edge;
    } else {
        // If unbounded, would make the ray reach the end of the frustum
        delta_cs *= min(1.0, min_dist_to_edge);
    }

    var res = _this_;
    res.ray_end_cs = _this_.ray_start_cs + delta_cs;
    return res;
}

/// March to a clip-space position (w = 1)
///
/// Must be called after `from_cs`, as it will clip the world-space ray to the view frustum.
fn to_cs(_this_: DepthRayMarch, end_cs: vec3<f32>) -> DepthRayMarch {
    let dir = vec4(end_cs - _this_.ray_start_cs, 0.0) * sign(end_cs.z);
    return to_cs_dir_impl(_this_, dir, false);
}

/// March towards a clip-space direction. Infinite (ray is extended to cover the whole view frustum).
///
/// Must be called after `from_cs`, as it will clip the world-space ray to the view frustum.
fn to_cs_dir(_this_: DepthRayMarch, dir: vec4<f32>) -> DepthRayMarch {
    return to_cs_dir_impl(_this_, dir, true);
}

/// March to a world-space position.
///
/// Must be called after `from_cs`, as it will clip the world-space ray to the view frustum.
fn to_ws(_this_: DepthRayMarch, end: vec3<f32>) -> DepthRayMarch {
    //return to_cs(_this_, main_view().position_world_to_sample(end));
    let clip_pos = view.view_proj * vec4(end, 1.0);
    return to_cs(_this_, clip_pos.xyz / clip_pos.w);
}

/// March towards a world-space direction. Infinite (ray is extended to cover the whole view frustum).
///
/// Must be called after `from_cs`, as it will clip the world-space ray to the view frustum.
fn to_ws_dir(_this_: DepthRayMarch, dir: vec3<f32>) -> DepthRayMarch {
    //return to_cs_dir_impl(_this_, main_view().direction_world_to_clip(dir), true);
    let clipSpace = view.view_proj * vec4(dir, 0.0);
    return to_cs_dir_impl(_this_, clipSpace, true);
}

/// Perform the ray march.
fn march(_this_: DepthRayMarch, sample_index: u32) -> DepthRayMarchResult {
    var res: DepthRayMarchResult;

    let ray_start_uv = cs_to_uv(_this_.ray_start_cs.xy);
    let ray_end_uv = cs_to_uv(_this_.ray_end_cs.xy);

    let ray_uv_delta = ray_end_uv - ray_start_uv;
    let ray_len_px = ray_uv_delta * _this_.depth_tex_size;

    let MIN_PX_PER_STEP = 1u;
    let step_count = min(_this_.linear_steps, u32(floor(length(ray_len_px) / f32(MIN_PX_PER_STEP))));

    //let linear_z_to_scaled_linear_z = main_view().rcp_near_plane_distance();
    let linear_z_to_scaled_linear_z = 1.0 / view.projection[3][2];
    let depth_thickness = _this_.depth_thickness_linear_z * linear_z_to_scaled_linear_z;

    var distance_fn: DepthRaymarchDistanceFn;
    distance_fn.sample_index = sample_index;
    distance_fn.depth_tex_size = _this_.depth_tex_size;
    distance_fn.march_behind_surfaces = _this_.march_behind_surfaces;
    distance_fn.depth_thickness = depth_thickness;


    var hybrid_root_finder = new_with_linear_steps(step_count);
    hybrid_root_finder.bisection_steps = _this_.bisection_steps;
    hybrid_root_finder.use_secant = _this_.use_secant;
    hybrid_root_finder.jitter = _this_.jitter;
    let root = find_root(hybrid_root_finder, distance_fn, _this_.ray_start_cs, _this_.ray_end_cs);
    
    res.hit_t = root.hit_t;
    if (root.intersected && root.hit_d.penetration < depth_thickness && root.hit_d.root.distance < depth_thickness) {
        res.hit = true;
        res.hit_uv = mix(ray_start_uv, ray_end_uv, root.hit_t);
        res.hit_penetration = root.hit_d.penetration / linear_z_to_scaled_linear_z;
        res.hit_penetration_frac = root.hit_d.penetration / depth_thickness;
        res.hit_t = root.hit_t;
        return res;
    }

    return res;
}

/// Convert clip space coordinate to world space
fn conv_pos_cs_to_ws(cs_pos: vec3<f32>) -> vec3<f32> {
    let ws = view.inverse_view_proj * vec4(cs_pos, 1.0);
    return ws.xyz / ws.w;
}

/// Convert world space coordinate to clip space
fn conv_pos_ws_to_cs(ws_pos: vec3<f32>) -> vec3<f32> {
    let cs = view.view_proj * vec4(ws_pos, 1.0);
    return cs.xyz / cs.w;
}