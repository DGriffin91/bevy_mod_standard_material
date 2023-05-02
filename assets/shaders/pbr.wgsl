#import bevy_pbr::mesh_view_bindings
#import bevy_pbr::pbr_bindings
#import bevy_pbr::mesh_bindings

@group(1) @binding(11)
var blue_noise_tex: texture_2d_array<f32>;
const BLUE_NOISE_TEX_DIMS = vec3<u32>(64u, 64u, 64u);
@group(1) @binding(12)
var prev_frame_tex: texture_2d<f32>;
@group(1) @binding(13)
var prev_frame_sampler: sampler;

#import bevy_pbr::utils
#import bevy_pbr::clustered_forward
#import bevy_pbr::lighting
#import bevy_pbr::pbr_ambient
#import bevy_pbr::prepass_utils
//#import bevy_pbr::shadows
#import "shaders/sampling.wgsl"
#import "shaders/contact_shadows.wgsl"
#import "shaders/depth_buffer_raymarching.wgsl"
#import "shaders/contact_shadows2.wgsl"
#import "shaders/bad_ssao.wgsl"
#import "shaders/bad_ssgi.wgsl"
#import "shaders/bad_ssr.wgsl"
#import "shaders/shadows.wgsl"
#import bevy_pbr::fog
//#import bevy_pbr::pbr_functions
#import "shaders/pbr_functions.wgsl"


struct FragmentInput {
    @builtin(front_facing) is_front: bool,
    @builtin(position) frag_coord: vec4<f32>,
    //@builtin(sample_index) sample_index: u32,
    #import bevy_pbr::mesh_vertex_output
};

@fragment
fn fragment(in: FragmentInput) -> @location(0) vec4<f32> {
    var V = normalize(view.world_position.xyz - in.world_position.xyz);
    var N = vec3(0.0);

    var output_color: vec4<f32> = material.base_color;
#ifdef VERTEX_COLORS
    output_color = output_color * in.color;
#endif
#ifdef VERTEX_UVS
    if ((material.flags & STANDARD_MATERIAL_FLAGS_BASE_COLOR_TEXTURE_BIT) != 0u) {
        output_color = output_color * textureSample(base_color_texture, base_color_sampler, in.uv);
    }
#endif

    // NOTE: Unlit bit not set means == 0 is true, so the true case is if lit
    if ((material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u) {
        // Prepare a 'processed' StandardMaterial by sampling all textures to resolve
        // the material members
        var pbr_input: PbrInput;

        pbr_input.material.base_color = output_color;
        pbr_input.material.reflectance = material.reflectance;
        pbr_input.material.flags = material.flags;
        pbr_input.material.alpha_cutoff = material.alpha_cutoff;

        // TODO use .a for exposure compensation in HDR
        var emissive: vec4<f32> = material.emissive;
#ifdef VERTEX_UVS
        if ((material.flags & STANDARD_MATERIAL_FLAGS_EMISSIVE_TEXTURE_BIT) != 0u) {
            emissive = vec4<f32>(emissive.rgb * textureSample(emissive_texture, emissive_sampler, in.uv).rgb, 1.0);
        }
#endif
        pbr_input.material.emissive = emissive;

        var metallic: f32 = material.metallic;
        var perceptual_roughness: f32 = material.perceptual_roughness;
#ifdef VERTEX_UVS
        if ((material.flags & STANDARD_MATERIAL_FLAGS_METALLIC_ROUGHNESS_TEXTURE_BIT) != 0u) {
            let metallic_roughness = textureSample(metallic_roughness_texture, metallic_roughness_sampler, in.uv);
            // Sampling from GLTF standard channels for now
            metallic = metallic * metallic_roughness.b;
            perceptual_roughness = perceptual_roughness * metallic_roughness.g;
        }
#endif
        pbr_input.material.metallic = metallic;
        pbr_input.material.perceptual_roughness = perceptual_roughness;

        var occlusion: f32 = 1.0;
#ifdef VERTEX_UVS
        if ((material.flags & STANDARD_MATERIAL_FLAGS_OCCLUSION_TEXTURE_BIT) != 0u) {
            occlusion = textureSample(occlusion_texture, occlusion_sampler, in.uv).r;
        }
#endif
        pbr_input.frag_coord = in.frag_coord;
        pbr_input.world_position = in.world_position;
        pbr_input.world_normal = prepare_world_normal(
            in.world_normal,
            (material.flags & STANDARD_MATERIAL_FLAGS_DOUBLE_SIDED_BIT) != 0u,
            in.is_front,
        );

        pbr_input.is_orthographic = view.projection[3].w == 1.0;

        pbr_input.N = apply_normal_mapping(
            material.flags,
            pbr_input.world_normal,
#ifdef VERTEX_TANGENTS
#ifdef STANDARDMATERIAL_NORMAL_MAP
            in.world_tangent,
#endif
#endif
#ifdef VERTEX_UVS
            in.uv,
#endif
        );

        N = in.world_normal;

        pbr_input.V = calculate_view(in.world_position, pbr_input.is_orthographic);
        pbr_input.occlusion = occlusion;

        pbr_input.flags = mesh.flags;

        output_color = pbr(pbr_input, 0u);

    } else {
        output_color = alpha_discard(material, output_color);
    }

    // fog
    if (fog.mode != FOG_MODE_OFF && (material.flags & STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT) != 0u) {
        output_color = apply_fog(output_color, in.world_position.xyz, view.world_position.xyz);
    }

//    var ssao = bad_ssao(in.frag_coord.xy, in.world_normal, in.sample_index).x;
//    ssao = ssao;
//
//    output_color = vec4(output_color.xyz * ssao, output_color.w);

#ifdef TONEMAP_IN_SHADER
        output_color = tone_mapping(output_color);
#endif
#ifdef DEBAND_DITHER
    var output_rgb = output_color.rgb;
    output_rgb = powsafe(output_rgb, 1.0 / 2.2);
    output_rgb = output_rgb + screen_space_dither(in.frag_coord.xy);
    // This conversion back to linear space is required because our output texture format is
    // SRGB; the GPU will assume our output is linear and will apply an SRGB conversion.
    output_rgb = powsafe(output_rgb, 2.2);
    output_color = vec4(output_rgb, output_color.a);
#endif
#ifdef PREMULTIPLY_ALPHA
        output_color = premultiply_alpha(material.flags, output_color);
#endif
    //return noise_test(in.frag_coord.xy, normalize(in.world_normal), in.sample_index);
    //let dir_to_light = lights.directional_lights[0].direction_to_light.xyz;
    //let shad = contact_shadow2(in.frag_coord.xy, dir_to_light, in.world_normal, in.sample_index);
    //return vec4(vec3(shad.x), 1.0);
    //return vec4(vec3(ssao), 1.0);


    let closest_motion_vector = prepass_motion_vector(in.frag_coord, 0u).xy;
    let history_uv = (in.frag_coord.xy / view.viewport.zw) - closest_motion_vector;

    let last_image = textureSampleLevel(prev_frame_tex, prev_frame_sampler, history_uv, 0.0).rgb;
    return vec4(mix(last_image, output_color.rgb, 1.0), output_color.a);

    


//    let blue = blue_noise_for_pixel(vec2<u32>(in.frag_coord.xy), globals.frame_count % 2u);
//    let last_image = textureSampleLevel(prev_frame_tex, prev_frame_sampler, in.frag_coord.xy / view.viewport.zw, 0.0);
//    return mix(last_image, vec4(vec3(f32(blue > 0.98)), 1.0), 0.1);

    //return output_color;
}