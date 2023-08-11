use bevy::{
    prelude::*,
    reflect::{TypePath, TypeUuid},
    render::render_resource::{AsBindGroup, ShaderRef},
};

#[derive(AsBindGroup, Debug, Clone, TypeUuid, TypePath)]
#[uuid = "131792e1-f241-4dca-8f72-bc75ff12ebaa"]
pub struct CustomStandardMaterial {}

impl Material for CustomStandardMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/pbr.wgsl".into()
    }
}
