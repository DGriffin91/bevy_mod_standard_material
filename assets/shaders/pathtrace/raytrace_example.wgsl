
#define RT_STATS

#import "shaders/pathtrace/raytrace_bindings_types.wgsl"
#import "shaders/pathtrace/raytrace_reference.wgsl"
#import "shaders/pathtrace/raytrace_candidates.wgsl"



fn get_screen_ray(uv: vec2<f32>) -> Ray {
    var ndc = uv * 2.0 - 1.0;
    var eye = view.inverse_view_proj * vec4(ndc.x, -ndc.y, 0.0, 1.0);

    var ray: Ray;
    ray.origin = view.world_position.xyz;
    ray.direction = normalize(eye.xyz);
    ray.inv_direction = 1.0 / ray.direction;

    return ray;
}



/*
@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    //if true {return;}
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    //textureStore(target_tex, location, 0u, reference_update(invocation_id));
    let cand = candidates_update(invocation_id);
    textureStore(target_tex, location, 0u, vec4(cand.color, 0.0));
    textureStore(target_tex, location, 1u, vec4(cand.world_position, 0.0));
    textureStore(target_tex, location, 2u, vec4(cand.ray_hit_pos, cand.distance));
    //textureStore(target_tex, location, 3u, vec4(cand.direction, 0.0));
}
*/


@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let ufrag_coord = invocation_id.xy;
    let ifrag_coord = vec2<i32>(ufrag_coord);
    var frag_coord = vec4(vec2<f32>(ufrag_coord), 0.0, 0.0);

    let target_dims = vec2<i32>(textureDimensions(target_tex).xy);
    
    let ftarget_dims = vec2<f32>(target_dims);
    let frag_size = 1.0 / ftarget_dims;
    var screen_uv = frag_coord.xy / ftarget_dims + frag_size * 0.5;

    let surface_normal = normalize(prepass_normal(vec4<f32>(screen_uv * view.viewport.xy, 0.0, 0.0), 0u));

    let ray = get_screen_ray(screen_uv);

    let query = scene_query(ray);

    var col = vec4(0.0);

    if query.hit.distance != F32_MAX {
        var normal = vec3(0.0);

        var instance: InstanceData;
        var color: vec3<f32>;
        if query.static_tlas {
            instance = static_mesh_instance_buffer[query.hit.instance_idx];
            color = static_material_instance_buffer[query.hit.instance_idx].color;
        } else {
            instance = dynamic_mesh_instance_buffer[query.hit.instance_idx];
            color = dynamic_material_instance_buffer[query.hit.instance_idx].color;
        }
        normal = get_surface_normal(query);
        

        col = vec4(vec3(normal), 1.0);
    } else {
        col = vec4(0.0);    
    }

    col = vec4(vis_stat(f32(query.stats.aabb_hit_blas + query.stats.aabb_hit_tlas), 500.0), 1.0);
    
    textureStore(target_tex, ifrag_coord, 0u, vec4(col.rgb, 0.0));
}
