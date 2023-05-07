// instance_data: ptr<storage, array<InstanceData>>
// Argument 'instance_data' at index 0 is a pointer of space Storage { access: LOAD }, which can't be passed into functions.
// need to figure out a why to not have to duplicate these functions
// seems like storage arrays cant be passed
// bindless would probably work, but isn't supported in webgpu

// IF ONE OF THESE FUNCTIONS ARE UPDATED, THEY ALL SHOULD BE

fn static_traverse_tlas(ray: Ray, min_dist: f32) -> Hit {
    var next_idx = 0;
    var temp_return = vec4(0.0);
    var hit: Hit;
    hit.distance = F32_MAX;
    var min_dist = min_dist;
    while (next_idx < i32(arrayLength(&static_tlas_buffer))) {
        let tlas = static_tlas_buffer[next_idx];
        if tlas.entry_or_shape_idx < 0 {
            // If the entry_index is negative, then it's a leaf node.
            // Shape index in this case is the mesh entity instance index
            // Look up the equivalent info as: static_tlas.0.aabbs[shape_index].entity
            let instance_idx = (tlas.entry_or_shape_idx + 1) * -1;

            let instance = static_mesh_instance_buffer[instance_idx];

            let world_to_local = instance.world_to_local;
            let local_to_world = instance.local_to_world;

            // Transform ray into local instance space
            var local_ray: Ray;
            local_ray.origin = (world_to_local * vec4(ray.origin, 1.0)).xyz;
            local_ray.direction = normalize((world_to_local * vec4(ray.direction, 0.0)).xyz);
            local_ray.inv_direction = 1.0 / local_ray.direction;

            //because of non uniform scale, TODO is there a faster way?
            let world_minp = ray.origin + ray.direction * min_dist;
            let local_minp = (world_to_local * vec4(world_minp, 1.0)).xyz;

            var new_hit = traverse_blas(instance.mesh_data, local_ray, distance(local_ray.origin, local_minp));
            
            if new_hit.distance < F32_MAX {
                //because of non uniform scale, TODO is there a faster way?
                let local_hitp = local_ray.origin + local_ray.direction * new_hit.distance;
                let world_hitp = (local_to_world * vec4(local_hitp, 1.0)).xyz;
                new_hit.distance = distance(ray.origin, world_hitp);

                if new_hit.distance < min_dist {
                    hit = new_hit;
                    hit.instance_idx = instance_idx;
                    min_dist = min(min_dist, hit.distance);
                }
            }

            // Exit the current node.
            next_idx = tlas.exit_idx;
        } else {
            // If entry_index is not -1 and the AABB test passes, then
            // proceed to the node in entry_index (which goes down the bvh branch).

            // If entry_index is not -1 and the AABB test fails, then
            // proceed to the node in exit_index (which defines the next untested partition).
            next_idx = select(tlas.exit_idx, 
                              tlas.entry_or_shape_idx, 
                              intersects_aabb(ray, tlas.aabb_min.xyz, tlas.aabb_max.xyz) < min_dist);
        }
    }
    return hit;
}

fn dynamic_traverse_tlas(ray: Ray, min_dist: f32) -> Hit {
    var next_idx = 0;
    var temp_return = vec4(0.0);
    var hit: Hit;
    hit.distance = F32_MAX;
    var min_dist = min_dist;
    while (next_idx < i32(arrayLength(&dynamic_tlas_buffer))) {
        let tlas = dynamic_tlas_buffer[next_idx];
        if tlas.entry_or_shape_idx < 0 {
            // If the entry_index is negative, then it's a leaf node.
            // Shape index in this case is the mesh entity instance index
            // Look up the equivalent info as: dynamic_tlas.0.aabbs[shape_index].entity
            let instance_idx = (tlas.entry_or_shape_idx + 1) * -1;

            let instance = dynamic_mesh_instance_buffer[instance_idx];

            let world_to_local = instance.world_to_local;
            let local_to_world = instance.local_to_world;

            // Transform ray into local instance space
            var local_ray: Ray;
            local_ray.origin = (world_to_local * vec4(ray.origin, 1.0)).xyz;
            local_ray.direction = normalize((world_to_local * vec4(ray.direction, 0.0)).xyz);
            local_ray.inv_direction = 1.0 / local_ray.direction;

            //because of non uniform scale, TODO is there a faster way?
            let world_minp = ray.origin + ray.direction * min_dist;
            let local_minp = (world_to_local * vec4(world_minp, 1.0)).xyz;

            var new_hit = traverse_blas(instance.mesh_data, local_ray, distance(local_ray.origin, local_minp));
            
            if new_hit.distance < F32_MAX {
                //because of non uniform scale, TODO is there a faster way?
                let local_hitp = local_ray.origin + local_ray.direction * new_hit.distance;
                let world_hitp = (local_to_world * vec4(local_hitp, 1.0)).xyz;
                new_hit.distance = distance(ray.origin, world_hitp);

                if new_hit.distance < min_dist {
                    hit = new_hit;
                    hit.instance_idx = instance_idx;
                    min_dist = min(min_dist, hit.distance);
                }
            }

            // Exit the current node.
            next_idx = tlas.exit_idx;
        } else {
            // If entry_index is not -1 and the AABB test passes, then
            // proceed to the node in entry_index (which goes down the bvh branch).

            // If entry_index is not -1 and the AABB test fails, then
            // proceed to the node in exit_index (which defines the next untested partition).
            next_idx = select(tlas.exit_idx, 
                              tlas.entry_or_shape_idx, 
                              intersects_aabb(ray, tlas.aabb_min.xyz, tlas.aabb_max.xyz) < min_dist);
        }
    }
    return hit;
}