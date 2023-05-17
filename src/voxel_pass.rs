use std::borrow::Cow;

use crate::{
    copy_frame::CopyFrameData,
    image_window_auto_size::get_image_bytes_count,
    pbr_material::BlueNoise,
    prepass_downsample::{
        prepass_get_bind_group_layout_entries, PrepassDownsampleImage, PrepassDownsampleNode,
    },
    screen_space_passes::{ScreenSpacePassesNode, ScreenSpacePassesTargetImage},
};
use bevy::{
    core_pipeline::{core_3d, prepass::ViewPrepassTextures},
    prelude::*,
    reflect::TypeUuid,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        globals::GlobalsBuffer,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
        render_resource::{
            BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindingResource, CachedComputePipelineId, ComputePassDescriptor,
            ComputePipelineDescriptor, Extent3d, FilterMode, PipelineCache, Sampler,
            SamplerDescriptor, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat,
            TextureUsages, TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        texture::ImageSampler,
        view::{ExtractedView, ViewTarget, ViewUniformOffset, ViewUniforms},
        RenderApp,
    },
};
use bevy_mod_bvh::pipeline_utils::{
    globals_entry, image_entry, sampler_entry, storage_tex_write, view_entry,
};

//const WORKGROUP_SIZE: u32 = 8;
const SIZE: u32 = 128;

pub struct VoxelPassPlugin;
impl Plugin for VoxelPassPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_image)
            .add_plugin(ExtractResourcePlugin::<VoxelPassesTargetImage>::default());

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_render_graph_node::<VoxelPassNode>(core_3d::graph::NAME, VoxelPassNode::NAME)
            .add_render_graph_edges(
                core_3d::graph::NAME,
                &[
                    PrepassDownsampleNode::NAME,
                    VoxelPassNode::NAME,
                    core_3d::graph::node::MAIN_OPAQUE_PASS,
                ],
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };
        render_app.init_resource::<TracePipeline>();
    }
}

pub struct VoxelPassNode {
    query: QueryState<
        (
            &'static ViewUniformOffset,
            &'static ViewTarget,
            &'static ViewPrepassTextures,
        ),
        With<ExtractedView>,
    >,
}

impl VoxelPassNode {
    pub const NAME: &str = "VoxelPassesNode";
}

impl FromWorld for VoxelPassNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl Node for VoxelPassNode {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        graph_context: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph_context.view_entity();
        let view_uniforms: &ViewUniforms = world.resource::<ViewUniforms>();
        let view_uniforms = view_uniforms.uniforms.binding().unwrap();
        let globals_buffer = world.resource::<GlobalsBuffer>();
        let globals_binding = globals_buffer.buffer.binding().unwrap();
        let images = world.resource::<RenderAssets<Image>>();

        let Some(prepass_downsample) = world.get_resource::<PrepassDownsampleImage>() else {
            return Ok(());
        };
        let Some(prepass_downsample_tex) = images.get(&prepass_downsample.0) else {
            return Ok(());
        };

        let Some(voxel_passes_image) = world.get_resource::<VoxelPassesTargetImage>() else {
            return Ok(());
        };
        let Some(prev_voxel_image) = images.get(&voxel_passes_image.prev) else {
            return Ok(());
        };
        let Some(current_voxel_image) = images.get(&voxel_passes_image.current) else {
            return Ok(());
        };

        let Some(screenspace_target) = world.get_resource::<ScreenSpacePassesTargetImage>() else {
            return Ok(());
        };
        let Some(processed_image) = images.get(&screenspace_target.processed_img) else {
            return Ok(());
        };

        let Some(copy_frame_data) = world.get_resource::<CopyFrameData>() else {
            return Ok(());
        };
        let Some(prev_frame) = images.get(&copy_frame_data.image) else {
            return Ok(());
        };

        let blue_noise = world.resource::<BlueNoise>();
        let Some(blue_noise) = images.get(&blue_noise.0) else {
            return Ok(());
        };

        let Ok((view_uniform_offset, view_target, prepass_textures)) = self.query.get_manual(world, view_entity) else {
            return Ok(());
        };

        if !view_target.is_hdr() {
            println!("view_target is not HDR");
            return Ok(());
        }

        let pipeline = world.resource::<TracePipeline>();

        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(update_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline_id) else {
            return Ok(());
        };

        let depth_binding = prepass_textures.depth.as_ref().unwrap();
        let normal_binding = prepass_textures.normal.as_ref().unwrap();
        let motion_vectors_binding = prepass_textures.motion_vectors.as_ref().unwrap();

        let entries = vec![
            BindGroupEntry {
                binding: 0,
                resource: view_uniforms.clone(),
            },
            BindGroupEntry {
                binding: 1,
                resource: globals_binding.clone(),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureView(&prev_frame.texture_view),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::Sampler(&pipeline.sampler),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(&blue_noise.texture_view),
            },
            BindGroupEntry {
                binding: 5,
                resource: BindingResource::TextureView(&depth_binding.default_view),
            },
            BindGroupEntry {
                binding: 6,
                resource: BindingResource::TextureView(&normal_binding.default_view),
            },
            BindGroupEntry {
                binding: 7,
                resource: BindingResource::TextureView(&motion_vectors_binding.default_view),
            },
            BindGroupEntry {
                binding: 8,
                resource: BindingResource::TextureView(&prepass_downsample_tex.texture_view),
            },
            BindGroupEntry {
                binding: 9,
                resource: BindingResource::TextureView(&processed_image.texture_view),
            },
            BindGroupEntry {
                binding: 10,
                resource: BindingResource::TextureView(&prev_voxel_image.texture_view),
            },
            BindGroupEntry {
                binding: 11,
                resource: BindingResource::TextureView(&current_voxel_image.texture_view),
            },
        ];

        {
            let bind_group =
                render_context
                    .render_device()
                    .create_bind_group(&BindGroupDescriptor {
                        label: Some("voxel_pass_bind_group"),
                        layout: &pipeline.layout,
                        entries: &entries,
                    });

            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());

            pass.set_pipeline(update_pipeline);
            pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
            pass.dispatch_workgroups(SIZE, SIZE, SIZE);
        }
        {
            render_context.command_encoder().copy_texture_to_texture(
                current_voxel_image.texture.as_image_copy(),
                prev_voxel_image.texture.as_image_copy(),
                Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: SIZE,
                },
            );
        }

        Ok(())
    }
}

#[derive(Resource)]
struct TracePipeline {
    layout: BindGroupLayout,
    sampler: Sampler,
    pipeline_id: CachedComputePipelineId,
}

impl FromWorld for TracePipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/voxel_pass.wgsl");

        let render_device = world.resource::<RenderDevice>();

        let mut entries = vec![
            view_entry(0),
            globals_entry(1),
            image_entry(2, TextureViewDimension::D2),
            sampler_entry(3),
            image_entry(4, TextureViewDimension::D2Array),
            image_entry(8, TextureViewDimension::D2),
            image_entry(9, TextureViewDimension::D2Array),
            image_entry(10, TextureViewDimension::D3),
            storage_tex_write(11, TextureFormat::Rgba32Float, TextureViewDimension::D3),
        ];

        // Prepass
        entries.extend_from_slice(&prepass_get_bind_group_layout_entries([5, 6, 7], false));

        let layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("voxel_pass_bind_group_layout"),
            entries: &entries,
        });

        let sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("voxel_pass_sampler_descriptor"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..default()
        });

        let pipeline_cache = world.resource_mut::<PipelineCache>();

        let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("update"),
        });

        Self {
            layout,
            sampler,
            pipeline_id,
        }
    }
}

#[derive(Resource, Default, Clone, ExtractResource, TypeUuid)]
#[uuid = "ca92a954-077c-4bf3-8d86-a9417e89825e"]
pub struct VoxelPassesTargetImage {
    pub prev: Handle<Image>,
    pub current: Handle<Image>,
}

fn setup_image(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let size = Extent3d {
        width: SIZE,
        height: SIZE,
        depth_or_array_layers: SIZE,
    };
    let img = Image {
        data: vec![
            0;
            get_image_bytes_count(size.width, size.height, 1, 4, 4)
                * size.depth_or_array_layers as usize
        ],
        texture_descriptor: TextureDescriptor {
            dimension: TextureDimension::D3,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[TextureFormat::Rgba32Float],
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
        },
        sampler_descriptor: ImageSampler::Descriptor(SamplerDescriptor {
            label: Some("ScreenSpacePassesNode_sampler_descriptor"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..default()
        }),
        texture_view_descriptor: Some(TextureViewDescriptor {
            label: None,
            format: Some(TextureFormat::Rgba32Float),
            dimension: Some(TextureViewDimension::D3),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
            aspect: TextureAspect::All,
        }),
    };

    commands.insert_resource(VoxelPassesTargetImage {
        prev: images.add(img.clone()),
        current: images.add(img),
    });
}
