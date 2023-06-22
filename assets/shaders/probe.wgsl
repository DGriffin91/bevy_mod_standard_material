struct Probe {
    M: u32,
    w_sum: f32,
    weight: f32,
    pos: vec3<f32>,
    ray_hit_pos: vec3<f32>,
    color: vec3<f32>,
    reproject_fail: bool,
}

fn new_probe() -> Probe {
    var p: Probe;
    p.M = 0u;
    p.w_sum = 0.0;
    p.weight = 0.0;
    p.pos = vec3(F32_MAX);
    p.ray_hit_pos = vec3(F32_MAX);
    p.color = vec3(0.0);
    p.reproject_fail = true;
    return p;
}

// TODO packed probe
/*
    w_sum: f16
    weight: f16

    color: RGB9E5 (32)

    M: u16
    ray_hit_offset: vec3<f16>

    pos: vec3<f32>
    reproject_fail (could be made 1bit and moved elsewhere)
*/

fn load_probe_with_pos(coord: vec2<i32>, pos: vec3<f32>) -> Probe {
    var p = load_probe(coord);
    p.pos = pos;
    return p;
}

fn load_probe(coord: vec2<i32>) -> Probe {
    var p: Probe;
    
    let data0 = bitcast<vec4<u32>>(textureLoad(fullscreen_passes_read, coord, 0u, 0));
    let data1 = textureLoad(fullscreen_passes_read, coord, 1u, 0);

    p.color = rgb9e5_to_float3(data0.x);
    let sum_weight = unpack2x16float(data0.y);
    p.w_sum = sum_weight.x;
    p.weight = sum_weight.y;
    let ray_hit_offset_xy = unpack2x16float(data0.z);
    let ray_hit_offset_z_m = unpack2x16float(data0.w);
    p.M = u32(ray_hit_offset_z_m.y); // current max 2048 TODO use u16: //(data0_w >> 16u) & 0xFFFFu;
    p.pos = data1.xyz;
    p.ray_hit_pos = vec3(ray_hit_offset_xy, ray_hit_offset_z_m.x) + p.pos;
    p.reproject_fail = bool(u32(data1.w));


// old format:
//    let weight_data = textureLoad(fullscreen_passes_read, coord, 1u, 0).xyz;
//    p.M = u32(weight_data.x);
//    p.w_sum = weight_data.y;
//    p.weight = weight_data.z;
//    p.pos = textureLoad(fullscreen_passes_read, coord, 2u, 0).xyz;
//    p.ray_hit_pos = textureLoad(fullscreen_passes_read, coord, 0u, 0).xyz;
//    p.color = textureLoad(fullscreen_passes_read, coord, 3u, 0).xyz;
//    p.reproject_fail = true;

    return p;
}

// pos, reprojection fail
fn load_probe_pos(coord: vec2<i32>) -> vec4<f32> {
    return textureLoad(fullscreen_passes_read, coord, 1u, 0);
}

fn store_probe(p: Probe, coord: vec2<i32>) {
    let color = float3_to_rgb9e5(p.color);
    let sum_weight = pack2x16float(vec2(p.w_sum, p.weight));
    let ray_hit_offset = p.ray_hit_pos - p.pos;
    let ray_hit_offset_xy = pack2x16float(vec2(ray_hit_offset.x, ray_hit_offset.y));
    let ray_hit_offset_z_m = pack2x16float(vec2(ray_hit_offset.z, f32(p.M)));
    //let offset_z_m = ((p.M & 0xFFFFu) << 16u) | (ray_hit_offset_z & 0xFFFFu);

    let data0 = bitcast<vec4<f32>>(vec4(
        color, 
        sum_weight, 
        ray_hit_offset_xy, 
        ray_hit_offset_z_m, 
    ));
    let data1 = vec4(
        p.pos,
        f32(p.reproject_fail), // <- spare bits here
    );
    
    textureStore(fullscreen_passes_write, coord, 0u, data0);
    textureStore(fullscreen_passes_write, coord, 1u, data1);
    
// old format:
//    textureStore(fullscreen_passes_write, coord, 0u, vec4(p.ray_hit_pos, f32(p.reproject_fail)));
//    textureStore(fullscreen_passes_write, coord, 1u, vec4(f32(p.M), p.w_sum, p.weight, 0.0));
//    textureStore(fullscreen_passes_write, coord, 2u, vec4(p.pos, 0.0));
//    textureStore(fullscreen_passes_write, coord, 3u, vec4(p.color, 0.0));
}

fn probe_update(p: ptr<function, Probe>, urand: f32, new_weight: f32, new_ray_hit_pos: vec3<f32>, color: vec3<f32>) {
    (*p).w_sum += new_weight;

    var threshold = new_weight / (*p).w_sum;

    if threshold > urand {
        (*p).ray_hit_pos = new_ray_hit_pos;
        (*p).weight = new_weight;
        (*p).color = color;
    }

    (*p).M += 1u;
}

// Works better than resetting TODO still probably not what restir does
fn probe_scale(p: ptr<function, Probe>, max_m: u32) {
    if (*p).M > max_m {
        let ratio = f32(max_m) / f32((*p).M);
        (*p).M = max_m;
        (*p).w_sum = (*p).w_sum * ratio;
    }
}

fn probe_get_progress(p: ptr<function, Probe>) -> f32 {
    return saturate(f32((*p).M) / f32(MAX_M));
}

fn probe_resolve(p: ptr<function, Probe>) -> vec3<f32> {
    let w = (*p).w_sum / max(0.00001, f32((*p).M) * (*p).weight);
    return min((*p).color, (*p).color * w);
}
