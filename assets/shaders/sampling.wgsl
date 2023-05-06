const PHI = 1.618033988749895; // Golden Ratio
const TAU = 6.28318530717958647692528676655900577;

const INV_TAU: f32 = 0.159154943;
const PHIMINUS1: f32 = 0.61803398875;

const F32_EPSILON: f32 = 1.1920929E-7;
const F32_MAX: f32 = 3.402823466E+38;
const U32_MAX: u32 = 0xFFFFFFFFu;


fn uniform_sample_sphere(urand: vec2<f32>) -> vec3<f32> {
    let theta = 2.0 * PI * urand.y;
    let z = 1.0 - 2.0 * urand.x;
    let xy = sqrt(max(0.0, 1.0 - z * z));
    let sn = sin(theta);
    let cs = cos(theta);
    return vec3(cs * xy, sn * xy, z);
}

fn uniform_sample_disc(urand: vec2<f32>) -> vec3<f32> {
    let r = sqrt(urand.x);
    let theta = urand.y * TAU;

    let x = r * cos(theta);
    let y = r * sin(theta);

    return vec3(x, y, 0.0);
}


fn cosine_sample_hemisphere(urand: vec2<f32>) -> vec3<f32> {
    let r = sqrt(urand.x);
    let theta = urand.y * TAU;

    let x = r * cos(theta);
    let y = r * sin(theta);

    return vec3(x, y, sqrt(max(0.0, 1.0 - urand.x)));
}

const M_PLASTIC = 1.32471795724474602596;

fn r2_sequence(i: u32) -> vec2<f32> {
    let a1 = 1.0 / M_PLASTIC;
    let a2 = 1.0 / (M_PLASTIC * M_PLASTIC);
    return fract(vec2(a1, a2) * f32(i) + 0.5);
}

fn blue_noise_for_pixel(px: vec2<u32>, layer: u32) -> f32 {
    return textureLoad(blue_noise_tex, px % BLUE_NOISE_TEX_DIMS.xy, i32(layer % BLUE_NOISE_TEX_DIMS.z), 0).x * 255.0 / 256.0 + 0.5 / 256.0;
}

//fn blue_noise_for_pixel_r2(px: vec2<u32>, n: u32) -> f32 {
//    let offset = vec2<u32>(r2_sequence(n) * vec2<f32>(BLUE_NOISE_TEX_DIMS.xy));
//    // TODO load texture as unorm not srgb
//    return pow(textureLoad(blue_noise_tex, (px + offset) % BLUE_NOISE_TEX_DIMS.xy, 0).x, 1.0/2.2) * 255.0 / 256.0 + 0.5 / 256.0;
//}

//fn blue_noise_for_pixel_simple(px: vec2<u32>) -> f32 {
//    // TODO load texture as unorm not srgb
//    return pow(textureLoad(blue_noise_tex, px % BLUE_NOISE_TEX_DIMS.xy, 0).x, 1.0/2.2) * 255.0 / 256.0 + 0.5 / 256.0;
//}

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

// Warning: only good for 4096 frames. Don't use with super long frame accumulation
fn white_frame_noise(seed: u32) -> vec4<f32> {
    return vec4(
        hash_noise(vec2(0 + i32(seed)), globals.frame_count + seed), 
        hash_noise(vec2(1 + i32(seed)), globals.frame_count + 4096u + seed),
        hash_noise(vec2(2 + i32(seed)), globals.frame_count + 8192u + seed),
        hash_noise(vec2(3 + i32(seed)), globals.frame_count + 12288u + seed)
    );
}

// https://blog.demofox.org/2022/01/01/interleaved-gradient-noise-a-different-kind-of-low-discrepancy-sequence
fn interleaved_gradient_noise(pixel_coordinates: vec2<f32>, frame: u32) -> f32 {
    let frame = f32(frame % 64u);
    let xy = pixel_coordinates + 5.588238 * frame;
    return fract(52.9829189 * fract(0.06711056 * xy.x + 0.00583715 * xy.y));
}

// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
fn build_orthonormal_basis(n: vec3<f32>) -> mat3x3<f32> {
    var b1: vec3<f32>;
    var b2: vec3<f32>;

    if (n.z < 0.0) {
        let a = 1.0 / (1.0 - n.z);
        let b = n.x * n.y * a;
        b1 = vec3(1.0 - n.x * n.x * a, -b, n.x);
        b2 = vec3(b, n.y * n.y * a - 1.0, -n.y);
    } else {
        let a = 1.0 / (1.0 + n.z);
        let b = -n.x * n.y * a;
        b1 = vec3(1.0 - n.x * n.x * a, b, -n.x);
        b2 = vec3(b, 1.0 - n.y * n.y * a, -n.y);
    }

    return mat3x3<f32>(
        b1.x, b2.x, n.x,
        b1.y, b2.y, n.y,
        b1.z, b2.z, n.z
    );
}

// ------------------------
// BRDF stuff from kajiya
// ------------------------

fn ggx_ndf(a2: f32, cos_theta: f32) -> f32 {
    let denom_sqrt = cos_theta * cos_theta * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom_sqrt * denom_sqrt);
}

fn g_smith_ggx1(ndotv: f32, a2: f32) -> f32 {
    let tan2_v = (1.0 - ndotv * ndotv) / (ndotv * ndotv);
    return 2.0 / (1.0 + sqrt(1.0 + a2 * tan2_v));
}

fn pdf_ggx_vn(a2: f32, wo: vec3<f32>, h: vec3<f32>) -> f32 {
    let g1 = g_smith_ggx1(wo.z, a2);
    let d = ggx_ndf(a2, h.z);
    return g1 * d * max(0.f, dot(wo, h)) / wo.z;
}

struct NdfSample {
    m: vec3<f32>,
    pdf: f32,
};

// From https://github.com/h3r2tic/kajiya/blob/d3b6ac22c5306cc9d3ea5e2d62fd872bea58d8d6/assets/shaders/inc/brdf.hlsl#LL182C1-L214C6
// https://github.com/NVIDIAGameWorks/Falcor/blob/c0729e806045731d71cfaae9d31a992ac62070e7/Source/Falcor/Experimental/Scene/Material/Microfacet.slang
// https://jcgt.org/published/0007/04/01/paper.pdf
fn sample_vndf(alpha: f32, wo: vec3<f32>, urand: vec2<f32>) -> NdfSample {
    let alpha_x = alpha;
    let alpha_y = alpha;
    let a2 = alpha_x * alpha_y;

    // Transform the view vector to the hemisphere configuration.
    let Vh = normalize(vec3(alpha_x * wo.x, alpha_y * wo.y, wo.z));

    // Construct orthonormal basis (Vh, T1, T2).
    let T1 = select(normalize(cross(vec3(0.0, 0.0, 1.0), Vh)), vec3(1.0, 0.0, 0.0), (Vh.z < 0.9999)); // TODO: fp32 precision
    let T2 = cross(Vh, T1);

    // Parameterization of the projected area of the hemisphere.
    let r = sqrt(urand.x);
    let phi = (2.0 * PI) * urand.y;
    let t1 = r * cos(phi);
    var t2 = r * sin(phi);
    let s = 0.5f * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    // Reproject onto hemisphere.
    let Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

    // Transform the normal back to the ellipsoid configuration. This is our half vector.
    let h = normalize(vec3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.0, Nh.z)));
    let pdf = pdf_ggx_vn(a2, wo, h);

    var res: NdfSample;
    res.m = h;
    res.pdf = pdf;
    return res;
}

fn eval_fresnel_schlick(f0: vec3<f32>, f90: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    return mix(f0, f90, pow(max(0.0, 1.0 - cos_theta), 5.0));
}

// Defined wrt the projected solid angle metric
struct BrdfSample {
    value_over_pdf: vec3<f32>,
    value: vec3<f32>,
    pdf: f32,

    transmission_fraction: vec3<f32>,

    wi: vec3<f32>,

    // For filtering / firefly suppression
    approx_roughness: f32,
}

fn BrdfSample_invalid() -> BrdfSample {
    var res: BrdfSample;
    res.value_over_pdf = vec3(0.0);
    res.pdf = 0.0;
    res.wi = vec3(0.0, 0.0, -1.0);
    res.transmission_fraction = vec3(0.0);
    res.approx_roughness = 0.0;
    return res;
}

struct SmithShadowingMasking {
    g: f32,
    g_over_g1_wo: f32,
}

fn SmithShadowingMasking_eval(ndotv: f32, ndotl: f32, a2: f32) -> SmithShadowingMasking {
    var res: SmithShadowingMasking;
//#if USE_GGX_CORRELATED_MASKING
//    res.g = g_smith_ggx_correlated(ndotv, ndotl, a2);
//    res.g_over_g1_wo = res.g / g_smith_ggx1(ndotv, a2);
//#else
    res.g = g_smith_ggx1(ndotl, a2) * g_smith_ggx1(ndotv, a2);
    res.g_over_g1_wo = g_smith_ggx1(ndotl, a2);
//#endif
    return res;
}

const BRDF_SAMPLING_MIN_COS = 1e-5;

fn brdf_sample(roughness: f32, F0: vec3<f32>, wo: vec3<f32>, urand: vec2<f32>) -> BrdfSample {
    //#if USE_GGX_VNDF_SAMPLING
    let ndf_sample = sample_vndf(roughness, wo, urand);
    //#else
    //    NdfSample ndf_sample = sample_ndf(urand);
    //#endif

    let wi = reflect(-wo, ndf_sample.m);

    if (ndf_sample.m.z <= BRDF_SAMPLING_MIN_COS || wi.z <= BRDF_SAMPLING_MIN_COS || wo.z <= BRDF_SAMPLING_MIN_COS) {
        return BrdfSample_invalid();
    }

    // Change of variables from the half-direction space to regular lighting geometry.
    let jacobian = 1.0 / (4.0 * dot(wi, ndf_sample.m));

    let fresnel = eval_fresnel_schlick(F0, vec3(1.0), dot(ndf_sample.m, wi));
    let a2 = roughness * roughness;
    let cos_theta = ndf_sample.m.z;

    let shadowing_masking = SmithShadowingMasking_eval(wo.z, wi.z, a2);

    var res: BrdfSample;
    res.pdf = ndf_sample.pdf * jacobian / wi.z;
    res.wi = wi;
    res.transmission_fraction = vec3(1.0) - fresnel;
    res.approx_roughness = roughness;

    //#if USE_GGX_VNDF_SAMPLING
        res.value_over_pdf =
            fresnel * shadowing_masking.g_over_g1_wo;
    //#else
    //    res.value_over_pdf =
    //        fresnel
    //        / (cos_theta * jacobian)
    //        * shadowing_masking.g
    //        / (4 * wo.z);
    //#endif

    res.value =
            fresnel
            * shadowing_masking.g
            * ggx_ndf(a2, cos_theta)
            / (4.0 * wo.z * wi.z);

    return res;
}