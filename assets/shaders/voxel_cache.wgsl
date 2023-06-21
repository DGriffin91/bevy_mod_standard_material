//kitchen
const VOXEL_GRID_SIZE = 64;
const VOXEL_SIZE = 0.5; // kitchen 0.25, sponza 1.0



fn ivoxel_clamp(pos: vec3<i32>) -> vec3<i32> {
    return clamp(pos, vec3(0), vec3(VOXEL_GRID_SIZE));
}

fn position_world_to_fvoxel(pos: vec3<f32>) -> vec3<f32> {
    let view_pos = vec3<f32>(vec3<i32>(view.world_position.xyz / VOXEL_SIZE)) * VOXEL_SIZE;
    return (pos - view_pos) / VOXEL_SIZE + f32(VOXEL_GRID_SIZE / 2);
}

fn position_voxel_to_world(voxel: vec3<i32>) -> vec3<f32> {
    let view_pos = vec3<f32>(vec3<i32>(view.world_position.xyz / VOXEL_SIZE)) * VOXEL_SIZE;
    return vec3<f32>(voxel - VOXEL_GRID_SIZE / 2) * VOXEL_SIZE + view_pos;
}

struct VoxelHit {
    color: vec3<f32>,
    age: f32,
    t: f32,
}

fn march_voxel_grid(origin: vec3<f32>, direction: vec3<f32>, max_steps: u32, skip_steps: u32, max_age: f32) -> VoxelHit {
    let time = f32(globals.time);
    var hit_voxel = vec4(0.0); // field 4 of return as 0.0 would be a miss
    var current_voxel = ivoxel_clamp(vec3<i32>(position_world_to_fvoxel(origin)));
    var hit: VoxelHit;

    let step = vec3<i32>(sign(direction));
    var t_delta = 1.0 / max(abs(direction), vec3(0.001));
    var t_max = (1.0 - fract(position_world_to_fvoxel(origin) * vec3<f32>(step))) * t_delta;

    hit.t = -1.0;
    for (var i = 0u; i <= max_steps; i += 1u) {
        if i >= skip_steps {
            hit_voxel = textureLoad(voxel_cache, current_voxel, 0);
            if hit_voxel.w != 0.0 && (time - hit_voxel.w) < max_age {
                hit.color = hit_voxel.rgb;
                hit.age = hit_voxel.w;
                hit.t = distance(origin, position_voxel_to_world(current_voxel));
                return hit;
            }
        }
        
        let select = vec3<i32>(vec3(t_max < t_max.yzx) && vec3(t_max <= t_max.zxy));
        current_voxel += select * step;
        t_max += vec3<f32>(select) * t_delta;
    }
    return hit;
}