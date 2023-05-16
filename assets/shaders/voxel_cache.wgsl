//kitchen
const VOXEL_START = vec3<f32>(-2.0, -2.0, -2.0);
const VOXEL_GIRD_SIZE = vec3<i32>(32, 32, 32);
const VOXEL_SIZE = 0.25;

//sponza
//const VOXEL_START = vec3<f32>(-20.0, -10.0, -5.0);
//const VOXEL_GIRD_SIZE = vec3<i32>(64, 64, 64);
//const VOXEL_SIZE = 0.75;
#ifdef VOXEL_UPDATE_FN
fn update_world_cache(probe_latest_color: vec3<f32>, probe_pos: vec3<f32>) {
    let coord = position_world_to_voxel_coord(probe_pos);
    textureStore(screen_passes_target, coord, 5u, vec4(probe_latest_color, f32(globals.time)));
}
#endif

fn position_world_to_ivoxel(pos: vec3<f32>) -> vec3<i32> {
    let voxel_pos = vec3<i32>((pos - VOXEL_START) / VOXEL_SIZE);
    return clamp(voxel_pos, vec3(0), VOXEL_GIRD_SIZE);
}

fn position_world_to_fvoxel(pos: vec3<f32>) -> vec3<f32> {
    return (pos - VOXEL_START) / VOXEL_SIZE;
}

fn position_voxel_to_world(voxel: vec3<f32>) -> vec3<f32> {
    return voxel * VOXEL_SIZE + VOXEL_START;
}

fn position_inside_voxel_grid(pos: vec3<f32>) -> bool {
    let voxel_pos = vec3<i32>((pos - VOXEL_START) / VOXEL_SIZE);
    return all(clamp(voxel_pos, vec3(0), VOXEL_GIRD_SIZE) == voxel_pos);
}

fn voxel_to_voxel_coord(voxel_pos: vec3<i32>) -> vec2<i32> {
    return vec2(voxel_pos.x + voxel_pos.z * VOXEL_GIRD_SIZE.x, voxel_pos.y);
}

fn position_world_to_voxel_coord(pos: vec3<f32>) -> vec2<i32> {
    let voxel_pos = position_world_to_ivoxel(pos);
    return vec2(voxel_pos.x + voxel_pos.z * VOXEL_GIRD_SIZE.x, voxel_pos.y);
}

fn read_world_cache(pos: vec3<f32>) -> vec4<f32> {
    let coord = position_world_to_voxel_coord(pos);
    return textureLoad(screen_passes_processed, coord, 5u, 0);
}

struct VoxelHit {
    color: vec3<f32>,
    age: f32,
    t: f32,
}

fn march_voxel_grid(origin: vec3<f32>, direction: vec3<f32>, max_steps: u32, skip_steps: u32, max_age: f32) -> VoxelHit {
    var hit_voxel = vec4(0.0); // field 4 of return as 0.0 would be a miss
    var current_voxel = position_world_to_ivoxel(origin);
    var hit: VoxelHit;

    let step = vec3<i32>(sign(direction));
    var t_delta = 1.0 / max(abs(direction), vec3(0.001));
    var t_max = (1.0 - fract(position_world_to_fvoxel(origin) * vec3<f32>(step))) * t_delta;

    hit.t = -1.0;
    for (var i = 0u; i <= max_steps; i += 1u) {
        if i >= skip_steps {
            hit_voxel = textureLoad(screen_passes_processed, voxel_to_voxel_coord(current_voxel), 5u, 0);
            if hit_voxel.w != 0.0 && (f32(globals.time) - hit_voxel.w) < max_age {
                hit.color = hit_voxel.rgb;
                hit.age = hit_voxel.w;
                hit.t = distance(origin, position_voxel_to_world(vec3<f32>(current_voxel)));
                return hit;
            }
        }
        
        let select = vec3<i32>(vec3(t_max < t_max.yzx) && vec3(t_max <= t_max.zxy));
        current_voxel += select * step;
        t_max += vec3<f32>(select) * t_delta;
    }
    return hit;
}


fn march_voxel_grid2(origin: vec3<f32>, direction: vec3<f32>, max_steps: u32, include_origin_voxel: bool, max_age: f32) -> VoxelHit {
    var ray_pos = origin;
    var hit_voxel = vec4(0.0); // field 4 of return as 0.0 would be a miss
    let step_length = VOXEL_SIZE * 0.999;
    let origin_voxel = position_world_to_ivoxel(origin);
    var hit: VoxelHit;
    hit.t = -1.0;
    for (var i = 0u; i <= max_steps; i += 1u) {
        if !position_inside_voxel_grid(ray_pos) {
            return hit;
        }
        let voxel_pos = position_world_to_ivoxel(ray_pos);
        let coord = voxel_to_voxel_coord(voxel_pos);
        hit_voxel = textureLoad(screen_passes_processed, coord, 5u, 0);
        if hit_voxel.w != 0.0 && (f32(globals.time) - hit_voxel.w) < max_age {
            if include_origin_voxel || !all(voxel_pos == origin_voxel) {
                hit.color = hit_voxel.rgb;
                hit.age = hit_voxel.w;
                hit.t = distance(origin, ray_pos);
                return hit;
            }
        }
        ray_pos = ray_pos + direction * step_length;
    }
    return hit;
}
