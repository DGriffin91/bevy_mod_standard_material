struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    inv_direction: vec3<f32>,
};

fn new_ray(origin: vec3<f32>, direction: vec3<f32>) -> Ray {
    var ray: Ray;
    ray.origin = origin;
    ray.direction = direction;
    ray.inv_direction = 1.0 / ray.direction;
    return ray;
}

struct Hit {
    uv: vec2<f32>,
    distance: f32,
    instance_idx: i32,
    triangle_idx: i32,
};

struct Stats {
    aabb_hit_tlas: u32,
    aabb_hit_blas: u32,
    tri_hit: u32,
    tri_test: u32,
    node_traversal: u32,
}

fn vis_stat(x: f32, step_value: f32) -> vec3<f32> {
    var col = vec3(0.0);
    let stat = x / step_value;

    col = vec3(stat, max(stat - 1.0, 0.0), max(stat - 2.0, 0.0));

    if stat > 1.0 {
        col.r = 0.0;
    } if stat > 2.0 {
        col.g = 0.0;
    }

    return col;
}



// https://developer.nvidia.com/blog/profiling-dxr-shaders-with-timer-instrumentation/
fn temperature(x: f32, scale: f32) -> vec3<f32> {

    // TODO with array() syntax got error:
    // The expression [102] may only be indexed by a constant
    var c: array<vec3<f32>, 10>;
    c[0] = vec3(   0.0/255.0,   2.0/255.0,  91.0/255.0 );
    c[1] = vec3(   0.0/255.0, 108.0/255.0, 251.0/255.0 );
    c[2] = vec3(   0.0/255.0, 221.0/255.0, 221.0/255.0 );
    c[3] = vec3(  51.0/255.0, 221.0/255.0,   0.0/255.0 );
    c[4] = vec3( 255.0/255.0, 252.0/255.0,   0.0/255.0 );
    c[5] = vec3( 255.0/255.0, 180.0/255.0,   0.0/255.0 );
    c[6] = vec3( 255.0/255.0, 104.0/255.0,   0.0/255.0 );
    c[7] = vec3( 226.0/255.0,  22.0/255.0,   0.0/255.0 );
    c[8] = vec3( 191.0/255.0,   0.0/255.0,  83.0/255.0 );
    c[9] = vec3( 145.0/255.0,   0.0/255.0,  65.0/255.0 );


    let s = x / scale;

    var cur = select(9, i32(s), i32(s) <= 9);
    var prv = select(0, cur - 1, cur >= 1);
    var nxt = select(9, cur + 1, cur < 9);

    let blur = 0.8;

    let wc = smoothstep( f32(cur)-blur, f32(cur)+blur, s ) * (1.0 - smoothstep(f32(cur+1)-blur, f32(cur+1)+blur, s) );
    let wp = 1.0 - smoothstep( f32(cur)-blur, f32(cur)+blur, s );
    let wn = smoothstep( f32(cur+1)-blur, f32(cur+1)+blur, s );

    let r = wc * c[cur] + wp * c[prv] + wn * c[nxt];
    return saturate(r);
}

fn stats_new() -> Stats {
    var stats: Stats;
    stats.aabb_hit_tlas = 0u;
    stats.aabb_hit_blas = 0u;
    stats.tri_hit = 0u;
    stats.node_traversal = 0u;
    return stats;
}

struct SceneQuery {
    hit: Hit,
    static_tlas: bool,
#ifdef RT_STATS
    stats: Stats,
#endif
};

struct Aabb {
    min: vec3<f32>,
    max: vec3<f32>,
};

struct Intersection {
    uv: vec2<f32>,
    distance: f32,
};

fn inside_aabb(p: vec3<f32>, minv: vec3<f32>, maxv: vec3<f32>) -> bool {
    return all(p > minv) && all(p < maxv);
}

// returns distance to intersection
fn intersects_aabb(ray: Ray, minv: vec3<f32>, maxv: vec3<f32>) -> f32 {
    let t1 = (minv - ray.origin) * ray.inv_direction;
    let t2 = (maxv - ray.origin) * ray.inv_direction;

    let tmin = min(t1, t2);
    let tmax = max(t1, t2);

    let tmin_n = max(tmin.x, max(tmin.y, tmin.z));
    let tmax_n = min(tmax.x, min(tmax.y, tmax.z));

    return select(F32_MAX, tmin_n, tmax_n >= tmin_n && tmax_n >= 0.0);
}

fn intersects_aabb_seg(ray: Ray, minv: vec3<f32>, maxv: vec3<f32>) -> vec2<f32> {
    let t1 = (minv - ray.origin) * ray.inv_direction;
    let t2 = (maxv - ray.origin) * ray.inv_direction;

    let tmin = min(t1, t2);
    let tmax = max(t1, t2);

    let tmin_n = max(tmin.x, max(tmin.y, tmin.z));
    let tmax_n = min(tmax.x, min(tmax.y, tmax.z));

    return select(vec2(F32_MAX), vec2(tmin_n, tmax_n), tmax_n >= tmin_n && tmax_n >= 0.0);
}

// A Ray-Box Intersection Algorithm and Efficient Dynamic Voxel Rendering
// Alexander Majercik, Cyril Crassin, Peter Shirley, and Morgan McGuire
fn slabs(ray: Ray, minv: vec3<f32>, maxv: vec3<f32>) -> bool {
    let t0 = (minv - ray.origin) * ray.inv_direction;
    let t1 = (maxv - ray.origin) * ray.inv_direction;

    let tmin = min(t0, t1);
    let tmax = max(t0, t1);

    return max(tmin.x, max(tmin.y, tmin.z)) <= min(tmax.x, min(tmax.y, tmax.z));
}

fn intersects_plane(ray: Ray, planePoint: vec3<f32>, planeNormal: vec3<f32>) -> f32 {
    let denom = dot(ray.direction, planeNormal);

    // Check if ray is parallel to the plane
    if (abs(denom) < F32_EPSILON) {
        return F32_MAX;
    }

    return dot(planePoint - ray.origin, planeNormal) / denom;
}



fn intersects_triangle(ray: Ray, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>) -> Intersection {
    var result: Intersection;
    result.distance = F32_MAX;

    let ab = p1 - p2;
    let ac = p3 - p2;

    let u_vec = cross(ray.direction, ac);
    let det = dot(ab, u_vec);

    // If backface culling on: det, off: abs(det)
    if abs(det) < F32_EPSILON  {
        return result;
    }

    let inv_det = 1.0 / det;
    let ao = ray.origin - p2;
    let u = dot(ao, u_vec) * inv_det;
    if u < 0.0 || u > 1.0 {
        return result;
    }

    let v_vec = cross(ao, ab);
    let v = dot(ray.direction, v_vec) * inv_det;
    result.uv = vec2(u, v);
    if v < 0.0 || u + v > 1.0 {
        return result;
    }

    let distance = dot(ac, v_vec) * inv_det;
    result.distance = select(result.distance, distance, distance > F32_EPSILON);


    return result;
}

fn traverse_blas(instance: MeshData, ray: Ray, min_dist: f32, any_hit: bool,
#ifdef RT_STATS
    stats: ptr<function, Stats>
#endif
) -> Hit {    
    //TODO Should we start at 1 since we already tested aginst the first AABB in the TLAS?
    var next_idx = 0; 
    var hit: Hit;
    hit.distance = F32_MAX;
    var aabb_inter = vec2(0.0);
    var min_dist = min(min_dist, F32_MAX);
    while (next_idx < instance.blas_count) {
        let blas = blas_buffer[next_idx + instance.blas_start];
        if blas.entry_or_shape_idx < 0 {
            let triangle_idx = (blas.entry_or_shape_idx + 1) * -3;
            // If the entry_index is negative, then it's a leaf node.
            let ind1 = i32(index_buffer[triangle_idx + 0 + instance.vert_idx_start].idx);
            let ind2 = i32(index_buffer[triangle_idx + 1 + instance.vert_idx_start].idx);
            let ind3 = i32(index_buffer[triangle_idx + 2 + instance.vert_idx_start].idx);            
            let p1 = VertexData_unpack_pos(vertex_buffer[ind1 + instance.vert_data_start]);
            let p2 = VertexData_unpack_pos(vertex_buffer[ind2 + instance.vert_data_start]);
            let p3 = VertexData_unpack_pos(vertex_buffer[ind3 + instance.vert_data_start]);

            // vert order is acb?
            let intr = intersects_triangle(ray, p1, p3, p2);
            if any_hit && intr.distance < F32_MAX {
                hit.distance = 0.0;
                return hit;
            }
            if intr.distance < min_dist {
                hit.distance = intr.distance;
                hit.triangle_idx = triangle_idx;
                hit.uv = intr.uv;
                min_dist = min(min_dist, hit.distance);
            }
            // Exit the current node.
            next_idx = blas.exit_idx;
#ifdef RT_STATS                  
            (*stats).node_traversal += 1u;               
            (*stats).tri_test += 1u;
            if intr.distance < F32_MAX {        
                (*stats).tri_hit += 1u;
            }
#endif
        } else {
            // If entry_index is not -1 and the AABB test passes, then
            // proceed to the node in entry_index (which goes down the bvh branch).

            // If entry_index is not -1 and the AABB test fails, then
            // proceed to the node in exit_index (which defines the next untested partition).
            
            let aabb_minxy = unpack2x16float(blas.aabb_minxy);
            let aabb_maxxy = unpack2x16float(blas.aabb_maxxy);
            let aabb_z = unpack2x16float(blas.aabb_z);
            let aabb_min = vec3(aabb_minxy, aabb_z.x);
            let aabb_max = vec3(aabb_maxxy, aabb_z.y) + aabb_min;
            next_idx = select(blas.exit_idx, 
                              blas.entry_or_shape_idx, 
                              intersects_aabb(ray, aabb_min, aabb_max) < min_dist);
#ifdef RT_STATS                  
                (*stats).aabb_hit_blas += 1u;
#endif
            //} else {
            //    next_idx = blas.exit_idx;
            //}
        }
    }
    return hit;
}

fn scene_query(ray: Ray, max_ray_dist: f32) -> SceneQuery {
    var max_ray_dist = clamp(max_ray_dist, 0.0, F32_MAX);
#ifdef RT_STATS
    var stats = stats_new();
#endif
    let hit_static = static_traverse_tlas(ray, max_ray_dist, false,
#ifdef RT_STATS
    &stats
#endif
    ); 
    max_ray_dist = clamp(hit_static.distance, 0.0, max_ray_dist);
    let hit_dynamic = dynamic_traverse_tlas(ray, max_ray_dist, false,
#ifdef RT_STATS
    &stats
#endif
    );

    var query: SceneQuery;
    if hit_static.distance < hit_dynamic.distance {
        query.hit = hit_static;
        query.static_tlas = true;
    } else {
        query.hit = hit_dynamic;
        query.static_tlas = false;
    }
#ifdef RT_STATS
    query.stats = stats;
#endif
    return query;
}

fn scene_query_any_hit(ray: Ray, max_ray_dist: f32) -> bool {
    var max_ray_dist = clamp(max_ray_dist, 0.0, F32_MAX);
#ifdef RT_STATS
    var stats = stats_new();
#endif
    let hit_static = static_traverse_tlas(ray, max_ray_dist, true,
#ifdef RT_STATS
    &stats
#endif
    ); 
    if hit_static.distance < F32_MAX {
        return true;
    }
    max_ray_dist = clamp(hit_static.distance, 0.0, max_ray_dist);
    let hit_dynamic = dynamic_traverse_tlas(ray, max_ray_dist, true,
#ifdef RT_STATS
    &stats
#endif
    );
    if hit_dynamic.distance < F32_MAX {
        return true;
    }
    return false;
}

// Inefficient, don't use this if getting more than normal.
fn get_surface_normal(query: SceneQuery) -> vec3<f32> {
    
    var instance: InstanceData;
    if query.static_tlas {
        instance = static_mesh_instance_buffer[query.hit.instance_idx];
    } else {
        instance = dynamic_mesh_instance_buffer[query.hit.instance_idx];
    }

    let mesh_pos_start = i32(instance.mesh_data.vert_data_start);
    let mesh_index_start = i32(instance.mesh_data.vert_idx_start);

    let ind1 = i32(index_buffer[query.hit.triangle_idx + 0 + mesh_index_start].idx);
    let ind2 = i32(index_buffer[query.hit.triangle_idx + 1 + mesh_index_start].idx);
    let ind3 = i32(index_buffer[query.hit.triangle_idx + 2 + mesh_index_start].idx);
    
    let a = VertexData_unpack(vertex_buffer[ind1 + mesh_pos_start]).normal;
    let b = VertexData_unpack(vertex_buffer[ind2 + mesh_pos_start]).normal;
    let c = VertexData_unpack(vertex_buffer[ind3 + mesh_pos_start]).normal;

    // Barycentric Coordinates
    let u = query.hit.uv.x;
    let v = query.hit.uv.y;
    var normal = u * a + v * b + (1.0 - u - v) * c;
    
    // transform local space normal into world space
    // TODO try right hand mult instead
    normal = normalize(instance.local_to_world * vec4(normal, 0.0)).xyz;

    return normal;
}

fn compute_tri_normal(query: SceneQuery) -> vec3<f32> {
    var instance: InstanceData;
    if query.static_tlas {
        instance = static_mesh_instance_buffer[query.hit.instance_idx];
    } else {
        instance = dynamic_mesh_instance_buffer[query.hit.instance_idx];
    }

    let mesh_pos_start = i32(instance.mesh_data.vert_data_start);
    let mesh_index_start = i32(instance.mesh_data.vert_idx_start);
    
    let ind1 = i32(index_buffer[query.hit.triangle_idx + 0 + mesh_index_start].idx);
    let ind2 = i32(index_buffer[query.hit.triangle_idx + 1 + mesh_index_start].idx);
    let ind3 = i32(index_buffer[query.hit.triangle_idx + 2 + mesh_index_start].idx);
    
    let a = VertexData_unpack_pos(vertex_buffer[ind1 + mesh_pos_start]);
    let b = VertexData_unpack_pos(vertex_buffer[ind2 + mesh_pos_start]);
    let c = VertexData_unpack_pos(vertex_buffer[ind3 + mesh_pos_start]);

    let v1 = b - a;
    let v2 = c - a;
    var normal = normalize(cross(v1, v2));

    // transform local space normal into world space
    normal = normalize(instance.local_to_world * vec4(normal, 0.0)).xyz;

    return normal; 
}
