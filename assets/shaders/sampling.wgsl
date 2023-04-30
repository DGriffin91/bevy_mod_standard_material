const PHI = 1.618033988749895; // Golden Ratio

fn uniform_sample_sphere(urand: vec2<f32>) -> vec3<f32> {
    let theta = 2.0 * PI * urand.y;
    let z = 1.0 - 2.0 * urand.x;
    let xy = sqrt(max(0.0, 1.0 - z * z));
    let sn = sin(theta);
    let cs = cos(theta);
    return vec3(cs * xy, sn * xy, z);
}

fn uniform_sample_disc(urand: vec2<f32>) -> vec2<f32> {
    let theta = 2.0 * PI * urand.y;
    let radius = sqrt(urand.x);
    let x = cos(theta) * radius;
    let y = sin(theta) * radius;
    return vec2(x, y);
}

const M_PLASTIC = 1.32471795724474602596;

fn r2_sequence(i: u32) -> vec2<f32> {
    let a1 = 1.0 / M_PLASTIC;
    let a2 = 1.0 / (M_PLASTIC * M_PLASTIC);
    return fract(vec2(a1, a2) * f32(i) + 0.5);
}

fn blue_noise_for_pixel(px: vec2<u32>, layer: u32) -> f32 {
    let offset = vec2((layer % 8u), ((layer / 8u) % 8u)) * 8u;
    //let offset = vec2(layer % BLUE_NOISE_TEX_DIMS.x, 0u);
    return textureLoad(blue_noise_tex, px % BLUE_NOISE_TEX_DIMS + offset, 0).x * 255.0 / 256.0 + 0.5 / 256.0;
}

fn blue_noise_for_pixel_r2(px: vec2<u32>, n: u32) -> f32 {
    let offset = vec2<u32>(r2_sequence(n) * vec2<f32>(BLUE_NOISE_TEX_DIMS));

    return textureLoad(blue_noise_tex, (px + offset) % BLUE_NOISE_TEX_DIMS, 0).x * 255.0 / 256.0 + 0.5 / 256.0;
}

fn blue_noise_for_pixel_simple(px: vec2<u32>) -> f32 {
    return textureLoad(blue_noise_tex, px % BLUE_NOISE_TEX_DIMS, 0).x * 255.0 / 256.0 + 0.5 / 256.0;
}

fn uhash(a: u32, b: u32) -> u32 { 
    var x = ((a * 1597334673u) ^ (b * 3812015801u));
    // from https://nullprogram.com/blog/2018/07/31/
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn unormf(n: u32) -> f32 { 
    return f32(n) * (1.0 / f32(0xffffffffu)); 
}

fn hash_noise(ifrag_coord: vec2<i32>, frame: u32) -> f32 {
    let urnd = uhash(u32(ifrag_coord.x), (u32(ifrag_coord.y) << 11u) + frame);
    return unormf(urnd);
}

// https://blog.demofox.org/2022/01/01/interleaved-gradient-noise-a-different-kind-of-low-discrepancy-sequence
fn interleaved_gradient_noise(pixel_coordinates: vec2<f32>, frame: u32) -> f32 {
    let frame = f32(frame % 64u);
    let xy = pixel_coordinates + 5.588238 * frame;
    return fract(52.9829189 * fract(0.06711056 * xy.x + 0.00583715 * xy.y));
}