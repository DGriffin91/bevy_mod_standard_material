
#import bevy_pbr::utils
#import bevy_core_pipeline::fullscreen_vertex_shader

@group(0) @binding(0)
var screen_texture: texture_2d<f32>;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let coords = floor(in.position);
    return textureLoad(screen_texture, vec2<i32>(coords.xy), 0);
}