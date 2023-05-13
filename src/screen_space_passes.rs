use std::borrow::Cow;

use crate::{
    copy_frame::CopyFrameData,
    image_window_auto_size::{auto_resize_image, get_image_bytes_count, FrameData},
    pbr_material::{BlueNoise, CustomStandardMaterial},
    prepass_downsample::{
        prepass_get_bind_group_layout_entries, PrepassDownsampleImage, PrepassDownsampleNode,
    },
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

const WORKGROUP_SIZE: u32 = 8;
const LAYERS: u32 = 4;
pub struct ScreenSpacePassesPlugin;
impl Plugin for ScreenSpacePassesPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_image)
            .add_systems(
                Update,
                auto_resize_image::<CustomStandardMaterial, ScreenSpacePassesTargetImage>,
            )
            .add_plugin(ExtractResourcePlugin::<ScreenSpacePassesTargetImage>::default());

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_render_graph_node::<ScreenSpacePassesNode>(
                core_3d::graph::NAME,
                ScreenSpacePassesNode::NAME,
            )
            .add_render_graph_edges(
                core_3d::graph::NAME,
                &[
                    PrepassDownsampleNode::NAME,
                    ScreenSpacePassesNode::NAME,
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

pub struct ScreenSpacePassesNode {
    query: QueryState<
        (
            &'static ViewUniformOffset,
            &'static ViewTarget,
            &'static ViewPrepassTextures,
        ),
        With<ExtractedView>,
    >,
}

impl ScreenSpacePassesNode {
    pub const NAME: &str = "ScreenSpacePassesNode";
}

impl FromWorld for ScreenSpacePassesNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl Node for ScreenSpacePassesNode {
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

        let Some(screenspace_target) = world.get_resource::<ScreenSpacePassesTargetImage>() else {
            return Ok(());
        };
        let Some(target_image) = images.get(&screenspace_target.current_img) else {
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
        let Some(blur_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.blur_pipeline_id) else {
            return Ok(());
        };

        let depth_binding = prepass_textures.depth.as_ref().unwrap();
        let normal_binding = prepass_textures.normal.as_ref().unwrap();
        let motion_vectors_binding = prepass_textures.motion_vectors.as_ref().unwrap();

        let mut entries = vec![
            // at the start so they are easy to swap for the blur
            BindGroupEntry {
                binding: 9,
                resource: BindingResource::TextureView(&processed_image.texture_view),
            },
            BindGroupEntry {
                binding: 10,
                resource: BindingResource::TextureView(&target_image.texture_view),
            },
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
        ];

        {
            let bind_group =
                render_context
                    .render_device()
                    .create_bind_group(&BindGroupDescriptor {
                        label: Some("ScreenSpacePassesNode_bind_group"),
                        layout: &pipeline.layout,
                        entries: &entries,
                    });

            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());

            pass.set_pipeline(update_pipeline);
            pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
            pass.dispatch_workgroups(
                target_image.size.x as u32 / WORKGROUP_SIZE,
                target_image.size.y as u32 / WORKGROUP_SIZE,
                1,
            );
        }

        {
            // swap prev and next target
            let a = entries[0].binding;
            let b = entries[1].binding;
            entries[0].binding = b;
            entries[1].binding = a;

            let bind_group =
                render_context
                    .render_device()
                    .create_bind_group(&BindGroupDescriptor {
                        label: Some("ScreenSpacePassesNode_bind_group"),
                        layout: &pipeline.layout,
                        entries: &entries,
                    });

            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());

            pass.set_pipeline(blur_pipeline);
            pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
            pass.dispatch_workgroups(
                target_image.size.x as u32 / WORKGROUP_SIZE,
                target_image.size.y as u32 / WORKGROUP_SIZE,
                1,
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
    blur_pipeline_id: CachedComputePipelineId,
}

impl FromWorld for TracePipeline {
    fn from_world(world: &mut World) -> Self {
        let shader = world
            .resource::<AssetServer>()
            .load("shaders/screen_space_passes.wgsl");

        let render_device = world.resource::<RenderDevice>();

        let mut entries = vec![
            view_entry(0),
            globals_entry(1),
            image_entry(2, TextureViewDimension::D2),
            sampler_entry(3),
            image_entry(4, TextureViewDimension::D2Array),
            image_entry(8, TextureViewDimension::D2),
            image_entry(9, TextureViewDimension::D2Array),
            storage_tex_write(
                10,
                TextureFormat::Rgba16Float,
                TextureViewDimension::D2Array,
            ),
        ];

        // Prepass
        entries.extend_from_slice(&prepass_get_bind_group_layout_entries([5, 6, 7], false));

        let layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ScreenSpacePassesNode_bind_group_layout"),
            entries: &entries,
        });

        let sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("ScreenSpacePassesNode_sampler_descriptor"),
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

        let blur_pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("blur"),
        });

        Self {
            layout,
            sampler,
            pipeline_id,
            blur_pipeline_id,
        }
    }
}

#[derive(Resource, Default, Clone, ExtractResource, TypeUuid)]
#[uuid = "c4fe681e-4dd8-47e5-8039-e9678ad4d717"]
pub struct ScreenSpacePassesTargetImage {
    pub current_img: Handle<Image>,
    pub processed_img: Handle<Image>,
}

fn setup_image(mut commands: Commands, windows: Query<&Window>, mut images: ResMut<Assets<Image>>) {
    let window = windows.single();
    let size = Extent3d {
        width: window.physical_width() / 4,
        height: window.physical_height() / 4,
        depth_or_array_layers: 4,
    };
    let img = Image {
        data: vec![0; get_image_bytes_count(size.width, size.height, 1, 2, 4) * LAYERS as usize],
        texture_descriptor: TextureDescriptor {
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC,
            view_formats: &[TextureFormat::Rgba16Float],
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
            format: Some(TextureFormat::Rgba16Float),
            dimension: Some(TextureViewDimension::D2Array),
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(LAYERS),
            aspect: TextureAspect::All,
        }),
    };
    let img2 = img.clone();

    commands.insert_resource(ScreenSpacePassesTargetImage {
        current_img: images.add(img),
        processed_img: images.add(img2),
    });
}

impl FrameData for ScreenSpacePassesTargetImage {
    fn image_h(&self) -> Handle<Image> {
        self.processed_img.clone()
    }

    fn size(&self, width: u32, height: u32) -> (u32, u32) {
        // make sure the size is divisible by work group
        (
            ((width / 3) / WORKGROUP_SIZE) * WORKGROUP_SIZE,
            ((height / 3) / WORKGROUP_SIZE) * WORKGROUP_SIZE,
        )
    }

    fn resize(&self, width: u32, height: u32, images: &mut Assets<Image>) {
        let size: (u32, u32) = self.size(width, height);
        let image = images.get_mut(&self.current_img).unwrap();
        image.texture_descriptor.size = Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: LAYERS,
        };
        image.data = vec![0; get_image_bytes_count(size.0, size.1, 1, 2, 4) * LAYERS as usize];
        let img2 = image.clone();
        let image2 = images.get_mut(&self.processed_img).unwrap();
        *image2 = img2;
    }
}
