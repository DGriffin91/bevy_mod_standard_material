use bevy::{
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderRef},
};

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CustomStandardMaterial {}

impl Material for CustomStandardMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/pbr.wgsl".into()
    }
    fn deferred_fragment_shader() -> ShaderRef {
        "shaders/pbr.wgsl".into()
    }
}
