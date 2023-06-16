use std::borrow::Cow;

use bevy::{
    core_pipeline::{core_3d, prepass::ViewPrepassTextures},
    prelude::*,
    render::{
        camera::ExtractedCamera,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
        render_resource::{
            BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindingResource, CachedComputePipelineId, ComputePassDescriptor,
            ComputePipelineDescriptor, Extent3d, PipelineCache, TextureAspect, TextureDescriptor,
            TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
            TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        texture::{CachedTexture, TextureCache},
        view::{ExtractedView, ViewTarget},
        Render, RenderApp, RenderSet,
    },
};

const WORKGROUP_SIZE: u32 = 8;
const MIP_LEVELS: u32 = 4;

use crate::bind_group_utils::{
    prepass_get_bind_group_layout_entries, storage_tex_readwrite_layout_entry, tex_view_entry,
};

#[derive(Component, ExtractComponent, Clone)]
pub struct PrepassDownsample;

pub struct PrepassDownsamplePlugin;
impl Plugin for PrepassDownsamplePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractComponentPlugin::<PrepassDownsample>::default());

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
                    return;
                };

        render_app
            .add_systems(Render, prepare_textures.in_set(RenderSet::Prepare))
            .add_render_graph_node::<PrepassDownsampleNode>(
                core_3d::graph::NAME,
                PrepassDownsampleNode::NAME,
            )
            .add_render_graph_edges(
                core_3d::graph::NAME,
                &[
                    core_3d::graph::node::DEFERRED,
                    PrepassDownsampleNode::NAME,
                    core_3d::graph::node::START_MAIN_PASS,
                ],
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };
        render_app.init_resource::<PrepassDownsamplePipeline>();
    }
}

pub struct PrepassDownsampleNode {
    query: QueryState<
        (
            &'static ViewTarget,
            &'static ViewPrepassTextures,
            &'static PrepassDownsampleTexture,
        ),
        With<ExtractedView>,
    >,
}

impl FromWorld for PrepassDownsampleNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl PrepassDownsampleNode {
    pub const NAME: &str = "copy_frame";
}

impl Node for PrepassDownsampleNode {
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

        let Ok((view_target, 
                prepass_textures, 
                prepass_downsample_texture)) = self.query.get_manual(world, view_entity) else {
            return Ok(());
        };

        if !view_target.is_hdr() {
            println!("view_target is not HDR");
            return Ok(());
        }

        let copy_frame_pipeline = world.resource::<PrepassDownsamplePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(copy_frame_pipeline.pipeline_id) else {
            return Ok(());
        };

        let depth_binding = prepass_textures.depth.as_ref().unwrap();
        let normal_binding = prepass_textures.normal.as_ref().unwrap();
        let motion_vectors_binding = prepass_textures.motion_vectors.as_ref().unwrap();

        let depth_view = depth_binding.texture.create_view(&TextureViewDescriptor {
            label: Some("prepass_depth"),
            aspect: TextureAspect::DepthOnly,
            ..default()
        });
        let mut entries: Vec<BindGroupEntry<'_>> = vec![
            tex_view_entry(0, &depth_view),
            tex_view_entry(1, &normal_binding.default_view),
            tex_view_entry(2, &motion_vectors_binding.default_view),
        ];

        let mut views = Vec::new();
        for i in 0..MIP_LEVELS {
            let view = prepass_downsample_texture
                .0
                .texture
                .create_view(&TextureViewDescriptor {
                    label: Some("copy_frame_texture"),
                    format: Some(prepass_downsample_texture.0.texture.format()),
                    dimension: Some(TextureViewDimension::D2),
                    aspect: TextureAspect::All,
                    base_mip_level: i,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: None,
                });
            views.push(view);
        }
        for (i, view) in views.iter().enumerate() {
            entries.push(BindGroupEntry {
                binding: 3 + i as u32,
                resource: BindingResource::TextureView(&view),
            })
        }

        let bind_group = render_context
            .render_device()
            .create_bind_group(&BindGroupDescriptor {
                label: Some("prepass_downsample_bind_group"),
                layout: &copy_frame_pipeline.layout,

                entries: &entries,
            });

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_bind_group(0, &bind_group, &[]);

        pass.set_pipeline(pipeline);
        let w = prepass_downsample_texture.0.texture.width();
        let h = prepass_downsample_texture.0.texture.height();
        pass.dispatch_workgroups(
            // make sure we are >= target_image.size
            (w as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
            (h as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
            1,
        );

        Ok(())
    }
}

#[derive(Resource)]
struct PrepassDownsamplePipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedComputePipelineId,
}

impl FromWorld for PrepassDownsamplePipeline {
    fn from_world(world: &mut World) -> Self {
        let mut entries = Vec::new();

        // Prepass
        entries.extend_from_slice(&prepass_get_bind_group_layout_entries([0, 1, 2, 3], false)[..3]);

        for i in 0..MIP_LEVELS {
            entries.push(storage_tex_readwrite_layout_entry(
                3 + i,
                TextureFormat::Rgba32Float,
                TextureViewDimension::D2,
            ));
        }

        let layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("copy_frame_bind_group_layout"),
                    entries: &entries,
                });

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/prepass_downsample.wgsl");

        let pipeline_cache = world.resource::<PipelineCache>();

        let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("update"),
        });

        Self {
            layout,
            pipeline_id,
        }
    }
}

#[derive(Component, Clone)]
pub struct PrepassDownsampleTexture(pub CachedTexture);

fn prepare_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera), With<PrepassDownsample>>,
) {
    for (entity, camera) in &views {
        if let Some(physical_viewport_size) = camera.physical_viewport_size {
            let mut texture_descriptor = TextureDescriptor {
                label: None,
                size: Extent3d {
                    depth_or_array_layers: 1,
                    width: physical_viewport_size.x,
                    height: physical_viewport_size.y,
                },
                mip_level_count: MIP_LEVELS,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[TextureFormat::Rgba32Float],
            };

            texture_descriptor.label = Some("PrepassDownsampleTexture");
            let prev_frame_texture = texture_cache.get(&render_device, texture_descriptor.clone());

            commands
                .entity(entity)
                .insert(PrepassDownsampleTexture(prev_frame_texture));
        }
    }
}
