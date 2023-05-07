// TODO need to at least be able to feedback

@group(1) @binding(12)
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
var screen_tex: texture_2d<f32>;
@group(0) @binding(3)
var texture_sampler: sampler;
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
    color: vec3<f32>
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

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let coord = in.position.xy;
    let icoord = vec2<i32>(in.position.xy);
    let uv = in.position.xy/view.viewport.zw;
    let frame = settings.frame;
    
    var col = textureSample(screen_tex, texture_sampler, in.uv);

    let ray = get_screen_ray(uv);

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

    col = print_value(coord, col, 0, f32(settings.fps));
    col = print_value(coord, col, 1, f32(frame));
    col = print_value(coord, col, 2, f32(arrayLength(&dynamic_tlas_buffer)));
    col = print_value(coord, col, 3, f32(arrayLength(&static_tlas_buffer)));

    return col;
}