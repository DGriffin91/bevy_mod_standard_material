
@group(0) @binding(0)
var screen_texture: texture_2d<f32>;
@group(0) @binding(1)
var samp: sampler;
@group(0) @binding(2)
var target_0: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(3)
var target_1: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(4)
var target_2: texture_storage_2d<rgba16float, read_write>;


fn resample_color(location: vec2<i32>, size: vec2<i32>, n: i32) -> vec4<f32> {
    var v = vec4(0.0);
    for (var x = 0; x < n; x+=1) {
        for (var y = 0; y < n; y+=1) {
            v += textureLoad(screen_texture, location * n + vec2(x, y), 0);
        }
    }
    v /= f32(n * n);
    return v;
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let size = vec2<i32>(textureDimensions(screen_texture).xy);
    if location.x > size.x || location.y > size.y {
        return;
    }
    let flocation = vec2<f32>(location);
    let v = textureLoad(screen_texture, location, 0);
    textureStore(target_0, location, v);

    // TODO, do correctly.

    var n = 2;
    if location.x < size.x / n && location.y < size.y / n {
        let C = resample_color(location, size, n);
        textureStore(target_1, location, C);
    }

    n = 4;
    if location.x < size.x / n && location.y < size.y / n {
        let C = resample_color(location, size, n);
        textureStore(target_2, location, C);
    }
}

@compute @workgroup_size(8, 8, 1)
fn update2(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let size = vec2<i32>(textureDimensions(screen_texture).xy);
    if location.x > size.x || location.y > size.y {
        return;
    }
    let flocation = vec2<f32>(location);
    //let v = textureLoad(screen_texture, location, 0);
    //textureStore(target_0, location, v);

    // TODO, do correctly.

    var n = 2;
    if location.x < size.x / n && location.y < size.y / n {
        let C = resample_color(location, size, n);
        textureStore(target_0, location, C);
    }

    n = 4;
    if location.x < size.x / n && location.y < size.y / n {
        let C = resample_color(location, size, n);
        textureStore(target_1, location, C);
    }

    n = 8;
    if location.x < size.x / n && location.y < size.y / n {
        let C = resample_color(location, size, n);
        textureStore(target_2, location, C);
    }
}