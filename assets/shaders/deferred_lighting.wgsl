
#import bevy_core_pipeline::fullscreen_vertex_shader
#import bevy_pbr::mesh_types
#import bevy_pbr::mesh_view_bindings

#import bevy_coordinate_systems::transformations

#import bevy_pbr::prepass_utils
#import bevy_pbr::pbr_types
#import bevy_pbr::utils
#import bevy_pbr::clustered_forward
#import bevy_pbr::lighting
#import "shaders/sampling.wgsl"
#import "shaders/depth_buffer_raymarching.wgsl"
#import "shaders/contact_shadows.wgsl"
#import "shaders/shadows.wgsl"
//#import bevy_pbr::shadows
#import bevy_pbr::fog
//#import bevy_pbr::pbr_functions
#import "shaders/pbr_functions_deferred.wgsl"
#import bevy_pbr::pbr_deferred_types
#import bevy_pbr::pbr_deferred_functions
#import bevy_pbr::pbr_ambient

@group(0) @binding(20)
var screen_passes_processed: texture_2d_array<f32>;
@group(0) @binding(21)
var fullscreen_passes_processed: texture_2d_array<f32>;
@group(0) @binding(22)
var blue_noise_tex: texture_2d_array<f32>;
const BLUE_NOISE_TEX_DIMS = vec3<u32>(64u, 64u, 64u);
@group(0) @binding(23)
var prepass_downsample: texture_2d<f32>;
@group(0) @binding(24)
var linear_sampler: sampler;



@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    var frag_coord = vec4(in.position.xy, 0.0, 0.0);

    let deferred_data = textureLoad(deferred_prepass_texture, vec2<i32>(frag_coord.xy), 0);

#ifdef WEBGL
    frag_coord.z = unpack_unorm3x4_plus_unorm_20(deferred_data.b).w;
#else
    frag_coord.z = prepass_depth(in.position, 0u);
#endif

    var pbr_input = pbr_input_from_deferred_gbuffer(frag_coord, deferred_data);
    var output_color = vec4(0.0);
    
    // NOTE: Unlit bit not set means == 0 is true, so the true case is if lit
    if ((pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u) {
        output_color = pbr(pbr_input);
    } else {
        output_color = pbr_input.material.base_color;
    }

    // fog
    if (fog.mode != FOG_MODE_OFF && (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT) != 0u) {
        output_color = apply_fog(fog, output_color, pbr_input.world_position.xyz, view.world_position.xyz);
    }

#ifdef TONEMAP_IN_SHADER
    output_color = tone_mapping(output_color);
#ifdef DEBAND_DITHER
    var output_rgb = output_color.rgb;
    output_rgb = powsafe(output_rgb, 1.0 / 2.2);
    output_rgb = output_rgb + screen_space_dither(frag_coord.xy);
    // This conversion back to linear space is required because our output texture format is
    // SRGB; the GPU will assume our output is linear and will apply an SRGB conversion.
    output_rgb = powsafe(output_rgb, 2.2);
    output_color = vec4(output_rgb, output_color.a);
#endif
#endif
#ifdef PREMULTIPLY_ALPHA
    output_color = premultiply_alpha(material.flags, output_color);
#endif

    return output_color;
}

