// TODO need to at least be able to feedback

@group(0) @binding(12)
var blue_noise_tex: texture_2d_array<f32>;
const BLUE_NOISE_TEX_DIMS = vec3<u32>(64u, 64u, 64u);

#import "shaders/sampling.wgsl"
#import "shaders/pathtrace/printing.wgsl"
//#import "common.wgsl"
#import "shaders/pathtrace/trace_gpu_types.wgsl"

#import bevy_pbr::mesh_types
#import bevy_render::view
#import bevy_render::globals
#import bevy_pbr::utils

@group(0) @binding(0)
var<uniform> view: View;
@group(0) @binding(1)
var<uniform> globals: Globals;
@group(0) @binding(2)
var prev_frame_tex: texture_2d<f32>;
@group(0) @binding(3)
var linear_sampler: sampler;
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
@group(0) @binding(18)
var prev_tex: texture_2d<f32>;
@group(0) @binding(19)
var target_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(20)
var prepass_downsample: texture_2d<f32>;


#import bevy_coordinate_systems::transformations
#import bevy_pbr::prepass_utils
#import "shaders/pathtrace/traverse_tlas.wgsl"
#import "shaders/pathtrace/tracing.wgsl"

#import "shaders/contact_shadows.wgsl"
#import "shaders/depth_buffer_raymarching.wgsl"

struct MaterialData {
    color: vec3<f32>,
    perceptual_roughness: f32,
    metallic: f32,
    reflectance: f32,
}

fn material_get_f0(material: MaterialData) -> vec3<f32> {
    let F0 = 0.16 * material.reflectance * material.reflectance * (1.0 - material.metallic) + material.color * material.metallic;
    return F0;
}

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
                let closest_motion_vector = prepass_motion_vector(vec4<f32>(hit_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
                let history_uv = hit_uv - closest_motion_vector;
                if history_uv.x > 0.0 && history_uv.x < 1.0 && history_uv.y > 0.0 && history_uv.y < 1.0 {
                    return textureSampleLevel(prev_frame_tex, linear_sampler, history_uv, 0.0).rgb;
                }
            }
        }
    }
    return vec3(-1.0);
}

fn get_hit_material(query: SceneQuery) -> MaterialData {
    if query.static_tlas {
        return static_material_instance_buffer[query.hit.instance_idx];
    } else {
        return dynamic_material_instance_buffer[query.hit.instance_idx];
    }
}

// comment out to disable
#define SPECULAR_SCREEN_SAMPLE
#define DIFFUSE_SCREEN_SAMPLE
//#define SPECULAR
#define DIFFUSE_INDIRECT
//#define DIFFUSE_DIRECT
#define DEPTH_MARCH



@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {

    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let flocation = vec2<f32>(location);

    

    let target_dims = textureDimensions(target_tex).xy;
    let depth_tex_dims = vec2<f32>(textureDimensions(prepass_downsample).xy);
    
    let ftarget_dims = vec2<f32>(target_dims);
    let screen_uv = flocation.xy / ftarget_dims;

    let samples = 7u;
    var sun_dir = vec3(-0.25, -0.24, 1.0);
    let sun_color = vec3(0.95, 0.79268, 0.637758) * 10.0;
    //var sun_dir = vec3(0.22, -1.0, -0.2); //sponza
    //let sun_color = vec3(0.95, 0.79268, 0.637758) * 10.0; //sponza
    let sky_color = vec3(1.75, 1.9, 1.99) * 2.0;

    let frag_size = 1.0 / ftarget_dims;

    let white_aa_noise = white_frame_noise(653u);
    let aa_jitter = (white_aa_noise.xy * 2.0 - 1.0) * frag_size * 0.125;

    // TODO depth here won't quite be accurate
    let depth = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 1.0).w;
    let frag_coord = vec4(flocation, depth, 0.0);
    let ifrag_coord = vec2<i32>(frag_coord.xy);
    let ufrag_coord = vec2<u32>(frag_coord.xy);
    


    var diffuse_direct = vec3(0.0);
    var diffuse_indirect = vec3(0.0);
    var specular = vec3(0.0);

    // Primary ray for material/normal of this frag
    let primary_ray = get_screen_ray(screen_uv);
    var query = scene_query(primary_ray);
    var primary_mat: MaterialData;
    var primary_roughness = 0.0;
    var surface_normal = vec3(0.0);
    if query.hit.distance != F32_MAX {
        primary_mat = get_hit_material(query);
        primary_roughness = perceptualRoughnessToRoughness(primary_mat.perceptual_roughness);
        surface_normal = get_surface_normal(query);
    } else {
        textureStore(target_tex, location, vec4(sky_color, 1.0));
        return;
    }
    let primary_depth = query.hit.distance;
    let primary_ndc_depth = depth_linear_to_ndc(primary_depth);

    
    let world_position_ws = primary_ray.origin + primary_ray.direction * query.hit.distance * 0.99999 + surface_normal * 0.00001;
    var V = normalize(view.world_position.xyz - world_position_ws);

    let F0 = material_get_f0(primary_mat);
    let tangent_to_world = build_orthonormal_basis(surface_normal);
    let white_frame_noise = white_frame_noise(78954u);

    var direction = uniform_sample_disc(vec2(
        fract(hash_noise(ifrag_coord, 5345u) + white_frame_noise.x),
        fract(hash_noise(ifrag_coord, 6784u) + white_frame_noise.y),
    )) * 0.007;
    sun_dir = normalize(sun_dir + direction); //make the sun a disc

#ifdef DIFFUSE_DIRECT
    // Trace to sun
    var ray = new_ray(world_position_ws, -sun_dir);
    query = scene_query(ray);
    if query.hit.distance == F32_MAX {
        diffuse_direct = sun_color * max(dot(surface_normal, -sun_dir), 0.0);    
    }
#endif

#ifdef DEPTH_MARCH
    let linear_steps = 24u;
    let bisection_steps = 4u;
    let depth_thickness = 1.5;
    let drm_trace_dist = 20.0;

    // TODO surface_normal * 0.01 is because of lines from using really low mip for depth
    let ray_start_ndc = position_world_to_ndc(world_position_ws + surface_normal * 0.01);
    var dmr = DepthRayMarch_new_from_depth(depth_tex_dims);
    dmr.ray_start_cs = ray_start_ndc;
    dmr.linear_steps = linear_steps;
    dmr.depth_thickness_linear_z = depth_thickness;
    dmr.march_behind_surfaces = false;
    dmr.use_secant = true;
    dmr.bisection_steps = bisection_steps;
    dmr.use_bilinear = false;
    dmr.mip_min_max = vec2(2.0, 3.0);
#endif //DEPTH_MARCH

    for (var i = 0u; i < samples; i += 1u) {
        let seed = i * samples + globals.frame_count * samples;
        
        let urand = fract_blue_noise_for_pixel(ufrag_coord, seed, white_frame_noise);  
        //let urand = fract_white_noise_for_pixel(ifrag_coord, seed, white_frame_noise);


        var direction = cosine_sample_hemisphere(urand.xy);
        direction = normalize(direction * tangent_to_world);

        // random direction trace
        var ray = new_ray(world_position_ws, direction);

#ifdef DEPTH_MARCH
        var depth_march_hit = false;
        dmr = to_ws(dmr, world_position_ws + direction * drm_trace_dist);
        dmr.jitter = fract(blue_noise_for_pixel(ufrag_coord, seed + 2u) + white_frame_noise.z);
        
        let raymarch_result = march(dmr, 0u);
        if (raymarch_result.hit) {
            let hit_n = normalize(prepass_normal(vec4<f32>(raymarch_result.hit_uv * depth_tex_dims, 0.0, 0.0), 0u));
            let backface = dot(hit_n, direction);
            if backface < 0.01 {
                let closest_motion_vector = prepass_motion_vector(vec4<f32>(raymarch_result.hit_uv * view.viewport.zw, 0.0, 0.0), 0u).xy;
                let history_uv = raymarch_result.hit_uv - closest_motion_vector;
                if history_uv.x > 0.0 && history_uv.x < 1.0 && history_uv.y > 0.0 && history_uv.y < 1.0 {
                    depth_march_hit = true;
                    diffuse_indirect += textureSampleLevel(prev_frame_tex, linear_sampler, history_uv, 0.0).rgb;
                }
            }
        }

        // if we didn't hit the screen or hit the back side of something
        // with the depth ray march we need to actually trace a ray
        if !depth_march_hit {
#endif //DEPTH_MARCH
            var query = scene_query(ray);

            var hit_dist = query.hit.distance;
            var hit_pos = ray.origin + ray.direction * hit_dist;
            var hit_color = get_hit_material(query).color;

#ifdef DIFFUSE_INDIRECT
        // first see if we hit somewhere on the screen
#ifdef DIFFUSE_SCREEN_SAMPLE
            let ray_hit_pos = ray.origin + ray.direction * hit_dist;
            let screen_color = get_screen_color_from_pos(ray_hit_pos, ray.direction);
            if screen_color.x != -1.0 {
                diffuse_indirect += screen_color;
            } else {
#endif
                // Trace to sun
                if hit_dist != F32_MAX {
                    var sray = new_ray(hit_pos, -sun_dir);
                    query = scene_query(sray);
                    if query.hit.distance == F32_MAX {
                        diffuse_indirect += hit_color * sun_color;
                    }
                } else {
                    diffuse_indirect += hit_color * sky_color;    
                }
#ifdef DIFFUSE_SCREEN_SAMPLE
            }
#endif
#endif //DIFFUSE_INDIRECT
#ifdef DEPTH_MARCH
        }
#endif //DEPTH_MARCH

#ifdef SPECULAR
        // Specular
        var wo = V;
        let brdf_sample = brdf_sample(primary_roughness, F0, tangent_to_world * wo, urand.zw);
        let trace_dir_ws = brdf_sample.wi * tangent_to_world;

        ray = new_ray(world_position_ws, trace_dir_ws);
        query = scene_query(ray);

        if query.hit.distance != F32_MAX {
            var hit_color = get_hit_material(query).color;
#ifdef SPECULAR_SCREEN_SAMPLE
            // first see if we hit somewhere on the screen
            let ray_hit_pos = ray.origin + ray.direction * query.hit.distance;
            let screen_color = get_screen_color_from_pos(ray_hit_pos, ray.direction);
            if screen_color.x != -1.0 {
                specular += screen_color * brdf_sample.value_over_pdf;
            } else {
#endif
                // Trace to sun
                let sray = new_ray(hit_pos, -sun_dir);
                query = scene_query(sray);
                if query.hit.distance == F32_MAX {
                    // TODO we are assuming the surface we hit is not metallic
                    specular += sun_color * brdf_sample.value_over_pdf;
                }
#ifdef SPECULAR_SCREEN_SAMPLE
            }
#endif
        } else {
            specular += max(sky_color * brdf_sample.value_over_pdf, vec3(0.0));    
        }
#endif // SPECULAR

    }

    diffuse_indirect /= f32(samples);
    var col = diffuse_indirect * primary_mat.color + diffuse_direct * primary_mat.color;
    //var col = diffuse_indirect + diffuse_direct;
    
#ifdef SPECULAR
    specular /= f32(samples);
    col += specular;
#endif



    //let last_image = textureSampleLevel(prev_frame_tex, prev_frame_sampler, history_uv, 0.0).rgb;
    var closest_motion_vector = vec2(0.0);
    closest_motion_vector += prepass_motion_vector(vec4(frag_coord.xy * 4.0 + vec2(0.0, 0.0), 0.0, 0.0), 0u).xy;
    closest_motion_vector += prepass_motion_vector(vec4(frag_coord.xy * 4.0 + vec2(0.0, 1.0), 0.0, 0.0), 0u).xy;
    closest_motion_vector += prepass_motion_vector(vec4(frag_coord.xy * 4.0 + vec2(1.0, 0.0), 0.0, 0.0), 0u).xy;
    closest_motion_vector += prepass_motion_vector(vec4(frag_coord.xy * 4.0 + vec2(1.0, 1.0), 0.0, 0.0), 0u).xy;
    let history_uv = screen_uv - closest_motion_vector / 4.0;

    let prev_target_frame = textureSampleLevel(prev_tex, linear_sampler, history_uv + frag_size * 0.5, 0.0).rgb;
    textureStore(target_tex, location, vec4(mix(prev_target_frame, col, 0.03), primary_ndc_depth));
}

@compute @workgroup_size(8, 8, 1)
fn blur(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let flocation = vec2<f32>(location);
    let fprepass_size = vec2<f32>(textureDimensions(prepass_downsample).xy);
    let size = vec2<i32>(textureDimensions(target_tex).xy);
    let fsize = vec2<f32>(size);
    let screen_uv = flocation / fsize;
    let v = textureLoad(prev_tex, location, 0);
    //let nor_depth = textureSampleLevel(prepass_downsample, linear_sampler, screen_uv, 1.0);
    let nor_depth = textureLoad(prepass_downsample, vec2<i32>(screen_uv * fprepass_size) / 2, 1);
    

    var tot = vec4(0.0);
    var tot_w = 0.0;

    let dist_factor = 3.0;

    let world_position = position_ndc_to_world(vec3(uv_to_ndc(screen_uv), v.w));

    let n = 8;
    for (var x = -n; x < n; x+=1) {
        for (var y = -n; y < n; y+=1) {
            let uv = vec2<f32>(location + vec2(x, y)) / fsize;
            let px_dist = saturate(1.0 - distance(vec2(0.0, 0.0), vec2(f32(x), f32(y))) / f32(n + 1));
            var w = px_dist;
            let nd = textureLoad(prepass_downsample, vec2<i32>(uv * fprepass_size), 0);
            let col = textureLoad(prev_tex, location + vec2(x, y), 0);

            //let nd = textureSampleLevel(prepass_downsample, linear_sampler, uv, 1.0);
            //let col = textureSampleLevel(prev_tex, linear_sampler, uv, 1.0);

            let d = max(dot(nor_depth.xyz, nd.xyz) + 0.0, 0.0);
            w = w * d;
            let px_ws = position_ndc_to_world(vec3(uv_to_ndc(uv), col.w));
            let dist = distance(px_ws, world_position);
            w = w * (1.0 - clamp(dist * dist_factor, 0.0, 1.0));
            //if distance(col.xyz, v.xyz) < 0.7 {
                tot += vec4(col.rgb, nd.w) * w;
                tot_w += w;
            //}
        }
    }
    tot /= max(tot_w, 1.0);
    
    //tot = textureLoad(prev_tex, location, 0).rgb;
    textureStore(target_tex, location, tot);
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