const VOXEL_START = vec3<f32>(-2.0, -2.0, -2.0);
const VOXEL_GIRD_SIZE = vec3<i32>(64, 64, 64);
const VOXEL_SIZE = 0.5;
fn update_world_cache(probe_latest_color: vec3<f32>, probe_pos: vec3<f32>) {
    let coord = position_world_to_voxel_coord(probe_pos);
    textureStore(target_tex, coord, 5u, vec4(probe_latest_color, f32(globals.time)));
}

fn position_world_to_voxel(pos: vec3<f32>) -> vec3<i32> {
    let voxel_pos = vec3<i32>((pos - VOXEL_START) / VOXEL_SIZE);
    return clamp(voxel_pos, vec3(0), VOXEL_GIRD_SIZE);
}

fn position_inside_voxel_grid(pos: vec3<f32>) -> bool {
    let voxel_pos = vec3<i32>((pos - VOXEL_START) / VOXEL_SIZE);
    return all(clamp(voxel_pos, vec3(0), VOXEL_GIRD_SIZE) == voxel_pos);
}

fn position_world_to_voxel_coord(pos: vec3<f32>) -> vec2<i32> {
    let voxel_pos = position_world_to_voxel(pos);
    return vec2(voxel_pos.x + voxel_pos.z * VOXEL_GIRD_SIZE.x, voxel_pos.y);
}

fn read_world_cache(pos: vec3<f32>) -> vec4<f32> {
    let coord = position_world_to_voxel_coord(pos);
    return textureLoad(prev_tex, coord, 5u, 0);
}

struct VoxelHit {
    color: vec3<f32>,
    age: f32,
    t: f32,
}

fn march_voxel_grid(origin: vec3<f32>, direction: vec3<f32>, max_steps: u32, include_origin_voxel: bool, max_age: f32) -> VoxelHit {
    var ray_pos = origin;
    var hit_voxel = vec4(0.0); // field 4 of return as 0.0 would be a miss
    let step_length = VOXEL_SIZE * 0.999;
    let origin_voxel_coord = position_world_to_voxel_coord(origin);
    var hit: VoxelHit;
    hit.t = -1.0;
    for (var i = 0u; i <= max_steps; i += 1u) {
        if !position_inside_voxel_grid(ray_pos) {
            return hit;
        }
        let coord = position_world_to_voxel_coord(ray_pos);
        hit_voxel = textureLoad(prev_tex, coord, 5u, 0);
        if hit_voxel.w != 0.0 && (f32(globals.time) - hit_voxel.w) < max_age {
            if include_origin_voxel || (coord.x != origin_voxel_coord.x && coord.y != origin_voxel_coord.y) {
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