struct Probe {
    M: f32,
    w_sum: f32,
    weight: f32,
    probe_pos: vec3<f32>,
    ray_hit_pos: vec3<f32>,
    latest_color: vec3<f32>,
}

fn update(r: Probe, urand: f32, new_weight: f32, new_ray_hit_pos: f32, color: vec3<f32>) -> Probe {
    r.w_sum += new_weight;

    var threshold = new_weight / r.w_sum;

    if threshold > urand {
        r.ray_hit_pos = new_ray_hit_pos;
        r.weight = new_weight;
        r.probe_latest_color = color;
    }

    r.M += 1u;
}

fn scale(r: Probe, max_m: f32) -> Probe {
    if r.M > max_m {
        let ratio = f32(max_m) / f32(r.M);
        r.M = max_m;
        r.w_sum = r.w_sum * ratio;
    }
}