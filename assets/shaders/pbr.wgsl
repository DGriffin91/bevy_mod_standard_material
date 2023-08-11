#define_import_path bevy_pbr::fragment

#import bevy_pbr::pbr_functions as pbr_functions
#import bevy_pbr::pbr_bindings as pbr_bindings
#import bevy_pbr::pbr_types as pbr_types
#import bevy_pbr::prepass_utils

#import bevy_pbr::mesh_vertex_output       MeshVertexOutput
#import bevy_pbr::mesh_bindings            mesh
#import bevy_pbr::mesh_view_bindings       view, fog, screen_space_ambient_occlusion_texture
#import bevy_pbr::mesh_view_types          FOG_MODE_OFF
#import bevy_core_pipeline::tonemapping    screen_space_dither, powsafe, tone_mapping
#import bevy_pbr::parallax_mapping         parallaxed_uv

#import bevy_pbr::prepass_utils

#import bevy_pbr::gtao_utils gtao_multibounce


// For PbrInput and StandardMaterial fields see:
// https://github.com/bevyengine/bevy/blob/v0.11.0/crates/bevy_pbr/src/render/pbr_functions.wgsl#L140
// https://github.com/bevyengine/bevy/blob/v0.11.0/crates/bevy_pbr/src/render/pbr_types.wgsl#L3

@fragment
fn fragment(
    in: MeshVertexOutput,
    @builtin(front_facing) is_front: bool,
) -> @location(0) vec4<f32> {

    var pbr = pbr_functions::pbr_input_new();
    pbr.frag_coord = in.position;
    pbr.world_position = in.world_position;
    pbr.world_normal = in.world_normal;
    pbr.N = in.world_normal;

    pbr.material.base_color = vec4(0.5, 0.5, 0.5, 1.0);
    pbr.material.perceptual_roughness = 0.05;
    var output_color = pbr_functions::pbr(pbr);

    return output_color;
}