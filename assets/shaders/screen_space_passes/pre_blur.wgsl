#import "shaders/screen_space_passes/common.wgsl"

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    //if true {return;}
    let ufrag_coord = invocation_id.xy;
    let ifrag_coord = vec2<i32>(ufrag_coord);
    var frag_coord = vec4(vec2<f32>(ufrag_coord), 0.0, 0.0);

    let target_dims = vec2<i32>(textureDimensions(fullscreen_passes_write).xy);
    if any(ifrag_coord >= target_dims) {
        return;
    }
    
    let ftarget_dims = vec2<f32>(target_dims);
    let frag_size = 1.0 / ftarget_dims;
    var screen_uv = frag_coord.xy / ftarget_dims + frag_size * 0.5;

    
    textureStore(fullscreen_passes_write, ifrag_coord, RESERVIOR_LAYER0, textureLoad(fullscreen_passes_read, ifrag_coord, RESERVIOR_LAYER0, 0));
    textureStore(fullscreen_passes_write, ifrag_coord, RESERVIOR_LAYER1, textureLoad(fullscreen_passes_read, ifrag_coord, RESERVIOR_LAYER1, 0));
    textureStore(fullscreen_passes_write, ifrag_coord, COLOR_LAYER, textureLoad(fullscreen_passes_read, ifrag_coord, COLOR_LAYER, 0));
    textureStore(fullscreen_passes_write, ifrag_coord, SSR_LAYER, textureLoad(fullscreen_passes_read, ifrag_coord, SSR_LAYER, 0));
}