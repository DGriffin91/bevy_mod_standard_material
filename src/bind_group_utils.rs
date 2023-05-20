use std::num::NonZeroU64;

use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;

use bevy::pbr::{MAX_CASCADES_PER_LIGHT, MAX_DIRECTIONAL_LIGHTS};
use bevy::prelude::*;

use bevy::render::globals::{GlobalsBuffer, GlobalsUniform};
use bevy::render::render_resource::{
    BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType,
    BufferBindingType, CachedRenderPipelineId, ColorTargetState, ColorWrites, FilterMode,
    FragmentState, MultisampleState, PipelineCache, PrimitiveState, RenderPipelineDescriptor,
    Sampler, SamplerBindingType, SamplerDescriptor, ShaderDefVal, ShaderStages, ShaderType,
    StorageTextureAccess, TextureFormat, TextureSampleType, TextureView, TextureViewDimension,
};
use bevy::render::texture::BevyDefault;
use bevy::render::view::{ViewUniform, ViewUniforms};

#[macro_export]
macro_rules! get_entries {
    ($entry_set:expr, $world:expr, $images:expr, $entries:expr) => {
        for item in &$entry_set.pairs {
            match &item.resource_func {
                EntryResource::TextureFunc(tex_func) => {
                    if let Some(image) = tex_func($world, $images) {
                        $entries.push(BindGroupEntry {
                            binding: item.binding,
                            resource: BindingResource::TextureView(&image.texture_view),
                        })
                    } else {
                        return Ok(());
                    }
                }
                EntryResource::Sampler(sampler) => $entries.push(BindGroupEntry {
                    binding: item.binding,
                    resource: BindingResource::Sampler(&sampler),
                }),
            };
        }
    };
}

#[macro_export]
macro_rules! resource {
    ($world:expr, $resource_type:ty) => {
        if let Some(res) = $world.get_resource::<$resource_type>() {
            res
        } else {
            return Ok(());
        }
    };
}

#[macro_export]
macro_rules! image {
    ($images:expr, $image_handle:expr) => {
        if let Some(res) = $images.get($image_handle) {
            res
        } else {
            return Ok(());
        }
    };
}

#[macro_export]
macro_rules! get_tex_view_entry {
    ($binding:expr, $images:expr, $image_handle:expr) => {
        if let Some(image) = $images.get(&$image_handle) {
            BindGroupEntry {
                binding: $binding,
                resource: BindingResource::TextureView(&image.texture_view),
            }
        } else {
            return Ok(());
        }
    };
}

pub fn tex_view_entry(binding: u32, texture_view: &TextureView) -> BindGroupEntry {
    BindGroupEntry {
        binding,
        resource: BindingResource::TextureView(texture_view),
    }
}

#[macro_export]
macro_rules! gpuimage {
    ($images:expr, $image_handle:expr) => {
        if let Some(image) = $images.get(&$image_handle) {
            image
        } else {
            return Ok(());
        }
    };
}

pub fn sampler_layout_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::Sampler(SamplerBindingType::Filtering),
        count: None,
    }
}

pub fn image_layout_entry(binding: u32, dim: TextureViewDimension) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: dim,
            multisampled: false,
        },
        count: None,
    }
}

pub fn storage_tex_read_layout_entry(
    binding: u32,
    format: TextureFormat,
    dim: TextureViewDimension,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::ReadOnly,
            format,
            view_dimension: dim,
        },
        count: None,
    }
}

pub fn storage_tex_write_layout_entry(
    binding: u32,
    format: TextureFormat,
    dim: TextureViewDimension,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::WriteOnly,
            format,
            view_dimension: dim,
        },
        count: None,
    }
}

pub fn storage_tex_readwrite_layout_entry(
    binding: u32,
    format: TextureFormat,
    dim: TextureViewDimension,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::StorageTexture {
            access: StorageTextureAccess::ReadWrite,
            format,
            view_dimension: dim,
        },
        count: None,
    }
}

pub fn view_layout_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: Some(ViewUniform::min_size()),
        },
        count: None,
    }
}

pub fn globals_layout_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(GlobalsUniform::min_size()),
        },
        count: None,
    }
}

pub fn uniform_layout_entry(
    binding: u32,
    min_binding_size: Option<NonZeroU64>,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: bevy::render::render_resource::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size,
        },
        count: None,
    }
}

pub fn default_full_screen_tri_pipeline_desc(
    mut shader_defs: Vec<ShaderDefVal>,
    layout: BindGroupLayout,
    pipeline_cache: &mut PipelineCache,
    shader: Handle<Shader>,
    hdr: bool,
) -> CachedRenderPipelineId {
    shader_defs.push(ShaderDefVal::UInt(
        "MAX_DIRECTIONAL_LIGHTS".to_string(),
        MAX_DIRECTIONAL_LIGHTS as u32,
    ));
    shader_defs.push(ShaderDefVal::UInt(
        "MAX_CASCADES_PER_LIGHT".to_string(),
        MAX_CASCADES_PER_LIGHT as u32,
    ));

    pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("post_process_pipeline".into()),
        layout: vec![layout],
        vertex: fullscreen_shader_vertex_state(),
        fragment: Some(FragmentState {
            shader,
            shader_defs,
            entry_point: "fragment".into(),
            targets: vec![Some(ColorTargetState {
                format: if hdr {
                    TextureFormat::Rgba16Float
                } else {
                    TextureFormat::bevy_default()
                },
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        push_constant_ranges: vec![],
    })
}

#[macro_export]
macro_rules! bind_group_layout_entry {
    () => {
        pub fn bind_group_layout_entry(
            binding: u32,
        ) -> bevy::render::render_resource::BindGroupLayoutEntry {
            bevy::render::render_resource::BindGroupLayoutEntry {
                binding,
                visibility: bevy::render::render_resource::ShaderStages::FRAGMENT
                    | bevy::render::render_resource::ShaderStages::COMPUTE,
                ty: bevy::render::render_resource::BindingType::Buffer {
                    ty: bevy::render::render_resource::BufferBindingType::Storage {
                        read_only: true,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: Some(Self::min_size()),
                },
                count: None,
            }
        }
    };
}

pub fn linear_sampler() -> SamplerDescriptor<'static> {
    SamplerDescriptor {
        label: Some("debug_view_sampler_descriptor"),
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: FilterMode::Linear,
        ..default()
    }
}

pub fn prepass_get_bind_group_layout_entries(
    bindings: [u32; 3],
    multisampled: bool,
) -> [BindGroupLayoutEntry; 3] {
    [
        // Depth texture
        BindGroupLayoutEntry {
            binding: bindings[0],
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture {
                multisampled,
                sample_type: TextureSampleType::Depth,
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        },
        // Normal texture
        BindGroupLayoutEntry {
            binding: bindings[1],
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture {
                multisampled,
                sample_type: TextureSampleType::Float { filterable: false },
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        },
        // Motion Vectors texture
        BindGroupLayoutEntry {
            binding: bindings[2],
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture {
                multisampled,
                sample_type: TextureSampleType::Float { filterable: false },
                view_dimension: TextureViewDimension::D2,
            },
            count: None,
        },
    ]
}

pub fn view_binding_entry(binding: u32, world: &World) -> BindGroupEntry {
    let view_uniforms = world.resource::<ViewUniforms>();
    let view_uniforms = view_uniforms.uniforms.binding().unwrap();
    BindGroupEntry {
        binding,
        resource: view_uniforms.clone(),
    }
}

pub fn globals_binding_entry(binding: u32, world: &World) -> BindGroupEntry {
    let globals_buffer = world.resource::<GlobalsBuffer>();
    let globals_binding = globals_buffer.buffer.binding().unwrap();
    BindGroupEntry {
        binding,
        resource: globals_binding.clone(),
    }
}

pub fn sampler_binding_entry(binding: u32, sampler: &Sampler) -> BindGroupEntry {
    BindGroupEntry {
        binding,
        resource: BindingResource::Sampler(&sampler),
    }
}

#[macro_export]
macro_rules! binding_entry {
    ($binding:expr, $buffer:expr) => {{
        let Some(resource) = $buffer.binding() else {return Ok(());};
        BindGroupEntry {
            binding: $binding,
            resource,
        }
    }};
}
