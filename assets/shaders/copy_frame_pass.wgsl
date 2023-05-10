
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
@group(0) @binding(5)
var target_3: texture_storage_2d<rgba16float, read_write>;


@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let flocation = vec2<f32>(location);
    let size = vec2<i32>(textureDimensions(screen_texture).xy);
    let v = textureLoad(screen_texture, location, 0);
    textureStore(target_0, location, v);

    // TODO, do correctly.

    if location.x < size.x / 2 && location.y < size.y / 2 {
        var v = vec4(0.0);
        v += textureLoad(screen_texture, location * 2 + vec2(0, 0), 0);
        v += textureLoad(screen_texture, location * 2 + vec2(1, 0), 0);
        v += textureLoad(screen_texture, location * 2 + vec2(0, 1), 0);
        v += textureLoad(screen_texture, location * 2 + vec2(1, 1), 0);
        v /= 4.0;
        textureStore(target_1, location, v);
    }

    var n = 4;
    if location.x < size.x / n && location.y < size.y / n {
        var v = vec4(0.0);
        let n = 4;
        for (var x = 0; x < n; x+=1) {
            for (var y = 0; y < n; y+=1) {
                v += textureLoad(screen_texture, location * n + vec2(x, y), 0);
            }
        }
        v /= f32(n * n);
        textureStore(target_2, location, v);
    }

    n = 8;
    if location.x < size.x / n && location.y < size.y / n {
        var v = vec4(0.0);
        for (var x = 0; x < n; x+=1) {
            for (var y = 0; y < n; y+=1) {
                v += textureLoad(screen_texture, location * n + vec2(x, y), 0);
            }
        }
        v /= f32(n * n);
        textureStore(target_3, location, v);
    }
}
