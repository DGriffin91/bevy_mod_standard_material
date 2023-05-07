struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    inv_direction: vec3<f32>,
};

struct Hit {
    uv: vec2<f32>,
    distance: f32,
    instance_idx: i32,
    triangle_idx: i32,
};

struct SceneQuery {
    hit: Hit,
    static_tlas: bool,
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

// just check if the ray intersects a plane in the aabb with the normal of the tri
fn traverse_blas_fast(instance: MeshData, ray: Ray, min_dist: f32) -> Hit {    
    //TODO Should we start at 1 since we already tested aginst the first AABB in the TLAS?
    var next_idx = 0; 
    var hit: Hit;
    hit.distance = F32_MAX;
    var aabb_inter = vec2(0.0);
    var last_aabb_min = vec3(0.0);
    var last_aabb_max = vec3(0.0);
    var min_dist = min_dist;
    while (next_idx < instance.blas_count) {
        let blas = blas_buffer[next_idx + instance.blas_start];
        if blas.entry_or_shape_idx < 0 {
            let triangle_idx = (blas.entry_or_shape_idx + 1) * -3;
            var normal = blas.tri_nor;
            // TODO improve accuracy with distance to plane along normal (stored in normal.w)
            let t = intersects_plane(ray, (last_aabb_min + last_aabb_max) / 2.0, normal.xyz);
            if  t > aabb_inter.x - 0.005 && t < aabb_inter.y + 0.005 && t < min_dist {
                hit.distance = t;
                hit.triangle_idx = triangle_idx;
                hit.uv = vec2(0.5, 0.5);
                min_dist = min(min_dist, hit.distance);
            }
            // Exit the current node.
            next_idx = blas.exit_idx;
        } else {
            // If entry_index is not -1 and the AABB test passes, then
            // proceed to the node in entry_index (which goes down the bvh branch).

            // If entry_index is not -1 and the AABB test fails, then
            // proceed to the node in exit_index (which defines the next untested partition).
            last_aabb_min = blas.aabb_min;
            last_aabb_max = blas.aabb_max;
            aabb_inter = intersects_aabb_seg(ray, blas.aabb_min, blas.aabb_max);
            next_idx = select(blas.exit_idx, 
                              blas.entry_or_shape_idx, 
                              aabb_inter.x < min(min_dist, hit.distance));
        }
    }
    return hit;
}

fn traverse_blas(instance: MeshData, ray: Ray, min_dist: f32) -> Hit {    
    //TODO Should we start at 1 since we already tested aginst the first AABB in the TLAS?
    var next_idx = 0; 
    var hit: Hit;
    hit.distance = F32_MAX;
    var aabb_inter = vec2(0.0);
    var min_dist = min_dist;
    while (next_idx < instance.blas_count) {
        let blas = blas_buffer[next_idx + instance.blas_start];
        if blas.entry_or_shape_idx < 0 {
            let triangle_idx = (blas.entry_or_shape_idx + 1) * -3;
            // If the entry_index is negative, then it's a leaf node.
            let ind1 = i32(index_buffer[triangle_idx + 0 + instance.vert_idx_start].idx);
            let ind2 = i32(index_buffer[triangle_idx + 1 + instance.vert_idx_start].idx);
            let ind3 = i32(index_buffer[triangle_idx + 2 + instance.vert_idx_start].idx);            
            let p1 = vertex_buffer[ind1 + instance.vert_data_start].position;
            let p2 = vertex_buffer[ind2 + instance.vert_data_start].position;
            let p3 = vertex_buffer[ind3 + instance.vert_data_start].position;

            // vert order is acb?
            let intr = intersects_triangle(ray, p1, p3, p2);
            if intr.distance < min_dist {
                hit.distance = intr.distance;
                hit.triangle_idx = triangle_idx;
                hit.uv = intr.uv;
                min_dist = min(min_dist, hit.distance);
            }
            // Exit the current node.
            next_idx = blas.exit_idx;
        } else {
            // If entry_index is not -1 and the AABB test passes, then
            // proceed to the node in entry_index (which goes down the bvh branch).

            // If entry_index is not -1 and the AABB test fails, then
            // proceed to the node in exit_index (which defines the next untested partition).
            next_idx = select(blas.exit_idx, 
                              blas.entry_or_shape_idx, 
                              intersects_aabb(ray, blas.aabb_min.xyz, blas.aabb_max.xyz) < min_dist);
        }
    }
    return hit;
}

fn scene_query(ray: Ray) -> SceneQuery {
    let hit_static = static_traverse_tlas(ray, F32_MAX);
    let hit_dynamic = dynamic_traverse_tlas(ray, hit_static.distance);

    if hit_static.distance < hit_dynamic.distance {
        var query: SceneQuery;
        query.hit = hit_static;
        query.static_tlas = true;
        return query;
    } else {
        var query: SceneQuery;
        query.hit = hit_dynamic;
        query.static_tlas = false;
        return query;
    }
}

// Inefficient, don't use this if getting more than normal.
fn get_surface_normal(instance: InstanceData, hit: Hit) -> vec3<f32> {
    let mesh_pos_start = i32(instance.mesh_data.vert_data_start);
    let mesh_index_start = i32(instance.mesh_data.vert_idx_start);

    let ind1 = i32(index_buffer[hit.triangle_idx + 0 + mesh_index_start].idx);
    let ind2 = i32(index_buffer[hit.triangle_idx + 1 + mesh_index_start].idx);
    let ind3 = i32(index_buffer[hit.triangle_idx + 2 + mesh_index_start].idx);
    
    let a = vertex_buffer[ind1 + mesh_pos_start].normal;
    let b = vertex_buffer[ind2 + mesh_pos_start].normal;
    let c = vertex_buffer[ind3 + mesh_pos_start].normal;

    // Barycentric Coordinates
    let u = hit.uv.x;
    let v = hit.uv.y;
    var normal = u * a + v * b + (1.0 - u - v) * c;
    
    // transform local space normal into world space
    // TODO try right hand mult instead
    normal = normalize(instance.local_to_world * vec4(normal, 0.0)).xyz;

    return normal;
}

fn compute_tri_normal(instance: InstanceData, hit: Hit) -> vec3<f32> {
    let mesh_pos_start = i32(instance.mesh_data.vert_data_start);
    let mesh_index_start = i32(instance.mesh_data.vert_idx_start);
    
    let ind1 = i32(index_buffer[hit.triangle_idx + 0 + mesh_index_start].idx);
    let ind2 = i32(index_buffer[hit.triangle_idx + 1 + mesh_index_start].idx);
    let ind3 = i32(index_buffer[hit.triangle_idx + 2 + mesh_index_start].idx);
    
    let a = vertex_buffer[ind1 + mesh_pos_start].position;
    let b = vertex_buffer[ind2 + mesh_pos_start].position;
    let c = vertex_buffer[ind3 + mesh_pos_start].position;

    let v1 = b - a;
    let v2 = c - a;
    var normal = normalize(cross(v1, v2));

    // transform local space normal into world space
    normal = normalize(instance.local_to_world * vec4(normal, 0.0)).xyz;

    return normal; 
}
