// TODO need to at least be able to feedback

@group(0) @binding(12)
var blue_noise_tex: texture_2d_array<f32>;
const BLUE_NOISE_TEX_DIMS = vec3<u32>(64u, 64u, 64u);

#import "shaders/sampling.wgsl"
#import "shaders/pathtrace/printing.wgsl"
//#import "common.wgsl"
#import "shaders/pathtrace/trace_gpu_types.wgsl"

#import bevy_pbr::mesh_types
#import bevy_pbr::mesh_view_types
#import bevy_pbr::utils
#import bevy_core_pipeline::fullscreen_vertex_shader

@group(0) @binding(0)
var<uniform> view: View;
@group(0) @binding(1)
var<uniform> globals: Globals;
@group(0) @binding(2)
var prev_frame_tex: texture_2d<f32>;
@group(0) @binding(3)
var prev_frame_sampler: sampler;
struct TraceSettings {
    frame: u32,
    fps: f32,
}
@group(0) @binding(4)
var<uniform> settings: TraceSettings;
@group(0) @binding(5)
var<storage> vertex_buffer: array<VertexData>;
@group(0) @binding(6)
var<storage> index_buffer: array<VertexIndices>;
@group(0) @binding(7)
var<storage> blas_buffer: array<BVHData>;
@group(0) @binding(8)
var<storage> static_tlas_buffer: array<BVHData>;
@group(0) @binding(9)
var<storage> dynamic_tlas_buffer: array<BVHData>;
@group(0) @binding(10)
var<storage> static_mesh_instance_buffer: array<InstanceData>;
@group(0) @binding(11)
var<storage> dynamic_mesh_instance_buffer: array<InstanceData>;

struct MaterialData {
    color: vec3<f32>,
    perceptual_roughness: f32,
}
@group(0) @binding(13)
var<storage> static_material_instance_buffer: array<MaterialData>;
@group(0) @binding(14)
var<storage> dynamic_material_instance_buffer: array<MaterialData>;

@group(0) @binding(15)
var depth_prepass_texture: texture_depth_2d;
@group(0) @binding(16)
var normal_prepass_texture: texture_2d<f32>;
@group(0) @binding(17)
var motion_vector_prepass_texture: texture_2d<f32>;

#import bevy_coordinate_systems::transformations
#import bevy_pbr::prepass_utils
#import "shaders/pathtrace/traverse_tlas.wgsl"
#import "shaders/pathtrace/tracing.wgsl"

fn get_screen_ray(uv: vec2<f32>) -> Ray {
    var ndc = uv * 2.0 - 1.0;
    var eye = view.inverse_view_proj * vec4(ndc.x, -ndc.y, 0.0, 1.0);

    var ray: Ray;
    ray.origin = view.world_position.xyz;
    ray.direction = normalize(eye.xyz);
    ray.inv_direction = 1.0 / ray.direction;

    return ray;
}

fn perceptualRoughnessToRoughness(perceptualRoughness: f32) -> f32 {
    let clampedPerceptualRoughness = clamp(perceptualRoughness, 0.089, 1.0);
    return clampedPerceptualRoughness * clampedPerceptualRoughness;
}

// -1.0 is a miss
fn get_screen_color_from_pos(hit_pos: vec3<f32>, ray_direction: vec3<f32>) -> vec3<f32> {
    let hit_pos_ndc = position_world_to_ndc(hit_pos);
    let hit_uv = ndc_to_uv(hit_pos_ndc.xy);
    // check if we hit screen, if so use that
    if distance(hit_pos_ndc, clamp(hit_pos_ndc, vec3(-1.0), vec3(1.0))) < 0.01 {
        let screen_depth = prepass_depth(vec4<f32>(hit_uv * view.viewport.zw, 0.0, 0.0), 0u);
        if distance(hit_pos_ndc.z, screen_depth) < 0.01 {
            let hit_n = normalize(prepass_normal(vec4<f32>(hit_uv * view.viewport.zw, 0.0, 0.0), 0u));
            let backface = dot(hit_n, ray_direction);
            if backface < 0.01 {
                return textureSampleLevel(prev_frame_tex, prev_frame_sampler, hit_uv, 0.0).rgb;
            }
        }
    }
    return vec3(-1.0);
}

fn get_hit_color(query: SceneQuery) -> vec3<f32> {
    if query.static_tlas {
        return static_material_instance_buffer[query.hit.instance_idx].color;
    } else {
        return dynamic_material_instance_buffer[query.hit.instance_idx].color;
    }
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let samples = 1u;
    let sun_dir = vec3(-0.25, -0.24, 1.0);
    let sun_color = vec3(0.95, 0.79268, 0.637758) * 5.0;
    let sky_color = vec3(1.75, 1.9, 1.99);

    let depth = prepass_depth(vec4<f32>(in.position.xy, 0.0, 0.0), 0u);
    let surface_normal = normalize(prepass_normal(vec4<f32>(in.position.xy, 0.0, 0.0), 0u));
    let frag_coord = vec4(in.position.xy, depth, 0.0);
    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);
    let screen_uv = frag_coord_to_uv(in.position.xy);
    let world_position_ndc = frag_coord_to_ndc(frag_coord);
    let world_position_ws = position_ndc_to_world(world_position_ndc) + surface_normal * 0.0001;

    var V = normalize(view.world_position.xyz - world_position_ws);


    let tangent_to_world = build_orthonormal_basis(surface_normal);
    var tot = vec3(0.0);

    let white_frame_noise = vec4(
        hash_noise(vec2(0), globals.frame_count + 0u), 
        hash_noise(vec2(1), globals.frame_count + 1u),
        hash_noise(vec2(2), globals.frame_count + 2u),
        hash_noise(vec2(3), globals.frame_count + 3u)
    );


    //var col = textureSample(screen_tex, texture_sampler, screen_uv);
    var col = vec3(0.0);

    // Primary ray for diffuse color of this frag
    let primary_ray = get_screen_ray(screen_uv);
    var query = scene_query(primary_ray);
    var primary_diffuse_color = sky_color;
    var primary_roughness = 0.0;
    if query.hit.distance != F32_MAX {
        if query.static_tlas {
            let material = static_material_instance_buffer[query.hit.instance_idx];
            primary_diffuse_color = material.color;
            primary_roughness = perceptualRoughnessToRoughness(material.perceptual_roughness);
        } else {
            let material = dynamic_material_instance_buffer[query.hit.instance_idx];
            primary_diffuse_color = material.color;
            primary_roughness = perceptualRoughnessToRoughness(material.perceptual_roughness);
        }
    }

    
    // Trace to sun
    var ray: Ray;
    ray.origin = world_position_ws;
    ray.direction = -sun_dir;
    ray.inv_direction = 1.0 / ray.direction;
    query = scene_query(ray);
    if query.hit.distance == F32_MAX {
        col = sun_color * primary_diffuse_color;    
    }

    for (var i = 0u; i < samples; i += 1u) {
        let seed = i * samples + globals.frame_count * samples;
        
        let urand = vec4(
            fract(blue_noise_for_pixel(ufrag_coord, seed + 0u) + white_frame_noise.x),
            fract(blue_noise_for_pixel(ufrag_coord, seed + 1u) + white_frame_noise.y),
            fract(blue_noise_for_pixel(ufrag_coord, seed + 2u) + white_frame_noise.z),
            fract(blue_noise_for_pixel(ufrag_coord, seed + 3u) + white_frame_noise.w)
        );  

        //let urand = vec4(
        //    fract(hash_noise(ifrag_coord, seed + 0u) + white_frame_noise.x),
        //    fract(hash_noise(ifrag_coord, seed + 1u) + white_frame_noise.y),
        //    fract(hash_noise(ifrag_coord, seed + 2u) + white_frame_noise.z),
        //    fract(hash_noise(ifrag_coord, seed + 3u) + white_frame_noise.w)
        //);  

        var direction = cosine_sample_hemisphere(urand.xy);
        direction = normalize(direction * tangent_to_world);

        // random direction trace
        var ray: Ray;
        ray.origin = world_position_ws;
        ray.direction = direction;
        ray.inv_direction = 1.0 / ray.direction;
        var query = scene_query(ray);

        var hit_pos = ray.origin + ray.direction * query.hit.distance;
        var hit_color = get_hit_color(query);

        // first see if we hit somewhere on the screen
        let ray_hit_pos = ray.origin + ray.direction * query.hit.distance;
        let screen_color = get_screen_color_from_pos(ray_hit_pos, ray.direction);
        if screen_color.x != -1.0 {
            col += primary_diffuse_color * screen_color;
        } else {
            // Trace to sun
            if query.hit.distance != F32_MAX {
                var sray: Ray;
                sray.origin = hit_pos;
                sray.direction = -sun_dir;
                sray.inv_direction = 1.0 / sray.direction;
                query = scene_query(sray);
                if query.hit.distance == F32_MAX {
                    col += hit_color * primary_diffuse_color * sun_color;
                } else {
                    // if we missed sun, see if we hit somewhere on the screen
                    let sray_hit_pos = sray.origin + sray.direction * query.hit.distance;
                    let screen_color = get_screen_color_from_pos(sray_hit_pos, sray.direction);
                    if screen_color.x != -1.0 {
                        col += hit_color * primary_diffuse_color * screen_color;
                    }
                }
            } else {
                col += hit_color * primary_diffuse_color * sky_color;    
            }
        }

        // Specular
        var wo = V;
        let brdf_sample = brdf_sample(primary_roughness, primary_diffuse_color, tangent_to_world * wo, urand.zw);
        let trace_dir_ws = brdf_sample.wi * tangent_to_world;

        
        ray.origin = world_position_ws;
        ray.direction = trace_dir_ws;
        ray.inv_direction = 1.0 / ray.direction;

        query = scene_query(ray);

        if query.hit.distance != F32_MAX {
            var hit_color = get_hit_color(query);

            // first see if we hit somewhere on the screen
            let ray_hit_pos = ray.origin + ray.direction * query.hit.distance;
            let screen_color = get_screen_color_from_pos(ray_hit_pos, ray.direction);
            if screen_color.x != -1.0 {
                col += primary_diffuse_color * screen_color * brdf_sample.value_over_pdf;
            } else {
                // Trace to sun
                var sray: Ray;
                sray.origin = hit_pos;
                sray.direction = -sun_dir;
                sray.inv_direction = 1.0 / sray.direction;
                query = scene_query(sray);
                if query.hit.distance == F32_MAX {
                    col += hit_color * primary_diffuse_color * sun_color * brdf_sample.value_over_pdf;
                }
            }
        } else {
            // TODO leaking
            col += max(sky_color * brdf_sample.value_over_pdf, vec3(0.0));    
        }

    }
    col /= f32(samples);



    let closest_motion_vector = prepass_motion_vector(frag_coord, 0u).xy;
    let history_uv = (frag_coord.xy / view.viewport.zw) - closest_motion_vector;
    let last_image = textureSampleLevel(prev_frame_tex, prev_frame_sampler, screen_uv, 0.0).rgb;
    return vec4(mix(last_image, col, 0.01), 1.0);
}

/*
@fragment
fn fragment_primary_rays(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let depth = prepass_depth(vec4<f32>(in.position.xy, 0.0, 0.0), 0u);
    let frag_coord = vec4(in.position.xy, depth, 0.0);
    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);
    let screen_uv = frag_coord_to_uv(in.position.xy);
    let ray_start_ndc = frag_coord_to_ndc(frag_coord);

    let surface_normal = normalize(prepass_normal(vec4<f32>(in.position.xy, 0.0, 0.0), 0u));

    var col = textureSample(screen_tex, texture_sampler, screen_uv);

    let ray = get_screen_ray(screen_uv);

    let query = scene_query(ray);

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
        normal = get_surface_normal(instance, query.hit);
        

        col = vec4(color, 1.0);//vec4(vec3(normal), 1.0);
    } else {
        col = vec4(0.0);    
    }

    col = print_value(frag_coord.xy, col, 0, f32(settings.fps));
    col = print_value(frag_coord.xy, col, 1, f32(settings.frame));
    col = print_value(frag_coord.xy, col, 2, f32(arrayLength(&dynamic_tlas_buffer)));
    col = print_value(frag_coord.xy, col, 3, f32(arrayLength(&static_tlas_buffer)));

    return col;
}
*/