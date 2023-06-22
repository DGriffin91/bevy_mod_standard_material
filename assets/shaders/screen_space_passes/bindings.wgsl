@group(0) @binding(0)
var<uniform> view: View;
@group(0) @binding(1)
var<uniform> globals: Globals;
@group(0) @binding(2)
var prev_frame_tex: texture_2d<f32>;
@group(0) @binding(3)
var linear_sampler: sampler;

@group(0) @binding(5)
var prepass_downsample: texture_2d<f32>;
@group(0) @binding(6)
var voxel_cache: texture_3d<f32>;
@group(0) @binding(7)
var path_trace_image: texture_2d_array<f32>;
@group(0) @binding(8)
var screen_passes_read: texture_2d_array<f32>;
@group(0) @binding(9)
var screen_passes_write: texture_storage_2d_array<rgba16float, write>;

@group(0) @binding(10)
var fullscreen_passes_read: texture_2d_array<f32>;
@group(0) @binding(11)
var fullscreen_passes_write: texture_storage_2d_array<rgba32float, write>;

@group(0) @binding(12)
var depth_prepass_texture: texture_depth_2d;
@group(0) @binding(13)
var normal_prepass_texture: texture_2d<f32>;
@group(0) @binding(14)
var motion_vector_prepass_texture: texture_2d<f32>;
@group(0) @binding(15)
var deferred_prepass_texture: texture_2d<u32>;