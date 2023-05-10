
// TODO handle multisampled
@group(0) @binding(0)
var depth_prepass_texture: texture_depth_2d;
@group(0) @binding(1)
var normal_prepass_texture: texture_2d<f32>;
@group(0) @binding(2)
var motion_vector_prepass_texture: texture_2d<f32>;

@group(0) @binding(3)
var target_0: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(4)
var target_1: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(5)
var target_2: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(6)
var target_3: texture_storage_2d<rgba32float, read_write>;


fn resample_nor(location: vec2<i32>, size: vec2<i32>, n: i32) -> vec3<f32> {
    var v = vec3(0.0);
    for (var x = 0; x < n; x+=1) {
        for (var y = 0; y < n; y+=1) {
            v += textureLoad(normal_prepass_texture, location * n + vec2(x, y), 0).xyz;
        }
    }
    v /= f32(n * n);
    return v;
}


fn resample_depth(location: vec2<i32>, size: vec2<i32>, n: i32) -> f32 {
    var v = 0.0;
    for (var x = 0; x < n; x+=1) {
        for (var y = 0; y < n; y+=1) {
            v += textureLoad(depth_prepass_texture, location * n + vec2(x, y), 0);
        }
    }
    v /= f32(n * n);
    return v;
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let flocation = vec2<f32>(location);
    let size = vec2<i32>(textureDimensions(normal_prepass_texture).xy);
    let N = textureLoad(normal_prepass_texture, location, 0).xyz;
    let D = textureLoad(depth_prepass_texture, location, 0);
    textureStore(target_0, location, vec4(N, D));

    // TODO, do correctly.

    var n = 2;
    if location.x < size.x / n && location.y < size.y / n {
        let N = resample_nor(location, size, n);
        let D = resample_depth(location, size, n);
        textureStore(target_1, location, vec4(N, D));
    }

    n = 4;
    if location.x < size.x / n && location.y < size.y / n {
        let N = resample_nor(location, size, n);
        let D = resample_depth(location, size, n);
        textureStore(target_2, location, vec4(N, D));
    }

    n = 8;
    if location.x < size.x / n && location.y < size.y / n {
        let N = resample_nor(location, size, n);
        let D = resample_depth(location, size, n);
        textureStore(target_3, location, vec4(N, D));
    }

}
