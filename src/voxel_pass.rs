use std::borrow::Cow;

use crate::{
    bind_group_utils::{
        globals_binding_entry, globals_layout_entry, image_layout_entry,
        prepass_get_bind_group_layout_entries, sampler_binding_entry, sampler_layout_entry,
        storage_tex_write_layout_entry, tex_view_entry, view_binding_entry, view_layout_entry,
    },
    copy_frame::CopyFrameData,
    get_tex_view_entry, image,
    image_window_auto_size::get_image_bytes_count,
    pbr_material::BlueNoise,
    prepass_downsample::{PrepassDownsampleImage, PrepassDownsampleNode},
    resource,
    screen_space_passes::ScreenSpacePasses,
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
        let images = world.resource::<RenderAssets<Image>>();

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

        let prev_voxel_image = image!(images, &resource!(world, VoxelPassesTargetImage).prev);
        let current_voxel_image = image!(images, &resource!(world, VoxelPassesTargetImage).current);

        let entries = vec![
            view_binding_entry(0, world),
            globals_binding_entry(1, world),
            get_tex_view_entry!(2, images, resource!(world, CopyFrameData).image),
            sampler_binding_entry(3, &pipeline.sampler),
            get_tex_view_entry!(4, images, resource!(world, BlueNoise).0),
            tex_view_entry(5, &depth_binding.default_view),
            tex_view_entry(6, &normal_binding.default_view),
            tex_view_entry(7, &motion_vectors_binding.default_view),
            get_tex_view_entry!(8, images, resource!(world, PrepassDownsampleImage).0),
            get_tex_view_entry!(9, images, resource!(world, ScreenSpacePasses).processed_img),
            tex_view_entry(10, &prev_voxel_image.texture_view),
            tex_view_entry(11, &current_voxel_image.texture_view),
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
            view_layout_entry(0),
            globals_layout_entry(1),
            image_layout_entry(2, TextureViewDimension::D2),
            sampler_layout_entry(3),
            image_layout_entry(4, TextureViewDimension::D2Array),
            image_layout_entry(8, TextureViewDimension::D2),
            image_layout_entry(9, TextureViewDimension::D2Array),
            image_layout_entry(10, TextureViewDimension::D3),
            storage_tex_write_layout_entry(
                11,
                TextureFormat::Rgba32Float,
                TextureViewDimension::D3,
            ),
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
