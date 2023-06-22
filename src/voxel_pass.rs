use std::borrow::Cow;

use crate::{
    bind_group_utils::{
        globals_binding_entry, globals_layout_entry, image_layout_entry,
        prepass_get_bind_group_layout_entries, sampler_binding_entry, sampler_layout_entry,
        storage_tex_write_layout_entry, tex_view_entry, view_binding_entry, view_layout_entry,
    },
    copy_frame::PrevFrameTexture,
    get_tex_view_entry,
    prepass_downsample::{PrepassDownsampleNode, PrepassDownsampleTexture},
    resource,
    screen_space_passes::ScreenSpacePassesTextures,
    BlueNoise,
};
use bevy::{
    core::FrameCount,
    core_pipeline::{core_3d, prepass::ViewPrepassTextures},
    prelude::*,
    render::{
        camera::ExtractedCamera,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
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
        texture::{CachedTexture, TextureCache},
        view::{ExtractedView, ViewTarget, ViewUniformOffset},
        Render, RenderApp, RenderSet,
    },
};

//const WORKGROUP_SIZE: u32 = 8;
const SIZE: u32 = 64;

#[derive(Component, ExtractComponent, Clone)]
pub struct VoxelPass;

pub struct VoxelPassPlugin;
impl Plugin for VoxelPassPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractComponentPlugin::<VoxelPass>::default());

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(Render, prepare_textures.in_set(RenderSet::Prepare))
            .add_render_graph_node::<VoxelPassNode>(core_3d::graph::NAME, VoxelPassNode::NAME)
            .add_render_graph_edges(
                core_3d::graph::NAME,
                &[
                    PrepassDownsampleNode::NAME,
                    VoxelPassNode::NAME,
                    core_3d::graph::node::START_MAIN_PASS,
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
            &'static PrevFrameTexture,
            &'static PrepassDownsampleTexture,
            &'static VoxelPassTextures,
            &'static ScreenSpacePassesTextures,
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

        let Ok((view_uniform_offset, 
            view_target, 
            prepass_textures, 
            prev_frame_tex, 
            prepass_downsample_texture, 
            voxel_pass_textures, 
            screen_space_passes_textures)) = self.query.get_manual(world, view_entity) else {
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

        let depth_view = depth_binding.texture.create_view(&TextureViewDescriptor {
            label: Some("prepass_depth"),
            aspect: TextureAspect::DepthOnly,
            ..default()
        });

        let normal_binding = prepass_textures.normal.as_ref().unwrap();
        let motion_vectors_binding = prepass_textures.motion_vectors.as_ref().unwrap();

        let entries = vec![
            view_binding_entry(0, world),
            globals_binding_entry(1, world),
            tex_view_entry(2, &prev_frame_tex.0.default_view),
            sampler_binding_entry(3, &pipeline.sampler),
            get_tex_view_entry!(4, images, resource!(world, BlueNoise).0),
            tex_view_entry(5, &depth_view),
            tex_view_entry(6, &normal_binding.default_view),
            tex_view_entry(7, &motion_vectors_binding.default_view),
            tex_view_entry(8, &prepass_downsample_texture.0.default_view),
            tex_view_entry(9, &screen_space_passes_textures.sm_tex_b.default_view),
            tex_view_entry(10, &voxel_pass_textures.read.default_view),
            tex_view_entry(11, &voxel_pass_textures.write.default_view),
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
        entries.extend_from_slice(&prepass_get_bind_group_layout_entries([5, 6, 7, 8], false)[..3]);

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

#[derive(Component, Clone)]
pub struct VoxelPassTextures {
    pub write: CachedTexture,
    pub read: CachedTexture,
}

fn prepare_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<Entity, (With<VoxelPass>, With<ExtractedCamera>)>,
    frame_count: Res<FrameCount>,
) {
    for entity in &views {
        let size = Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: SIZE,
        };
        let mut texture_descriptor = TextureDescriptor {
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
        };

        texture_descriptor.label = Some("VoxelPassesTexture1");
        let texture_1 = texture_cache.get(&render_device, texture_descriptor.clone());

        texture_descriptor.label = Some("VoxelPassesTexture2");
        let texture_2 = texture_cache.get(&render_device, texture_descriptor);

        //let textures = if frame_count.0 % 2 == 0 {
        //    VoxelPassTextures {
        //        write: texture_1,
        //        read: texture_2,
        //    }
        //} else {
        //    VoxelPassTextures {
        //        write: texture_2,
        //        read: texture_1,
        //    }
        //};

        let textures = VoxelPassTextures {
            write: texture_1,
            read: texture_2,
        };

        commands.entity(entity).insert(textures);
    }
}

/*
TextureViewDescriptor {
    label: None,
    format: Some(TextureFormat::Rgba32Float),
    dimension: Some(TextureViewDimension::D3),
    base_mip_level: 0,
    mip_level_count: None,
    base_array_layer: 0,
    array_layer_count: None,
    aspect: TextureAspect::All,
}
 */
