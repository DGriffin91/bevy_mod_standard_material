use bevy::{
    core_pipeline::{core_3d, prepass::ViewPrepassTextures},
    prelude::*,
    render::{
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
        render_resource::{
            BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, CachedRenderPipelineId, Operations, PipelineCache,
            RenderPassColorAttachment, RenderPassDescriptor, Sampler, TextureAspect,
            TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        view::{ExtractedView, ViewTarget, ViewUniformOffset},
        RenderApp,
    },
};

use crate::{
    bind_group_utils::{
        default_full_screen_tri_pipeline_desc, globals_binding_entry, globals_layout_entry,
        image_layout_entry, linear_sampler, prepass_get_bind_group_layout_entries,
        sampler_binding_entry, sampler_layout_entry, tex_view_entry, view_binding_entry,
        view_layout_entry,
    },
    copy_frame::PrevFrameTexture,
    prepass_downsample::PrepassDownsampleTexture,
    screen_space_passes::ScreenSpacePassesTextures,
    voxel_pass::VoxelPassTextures, path_trace::PathTraceTextures,
};

pub struct DebugViewPlugin;

impl Plugin for DebugViewPlugin {
    fn build(&self, app: &mut App) {
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_render_graph_node::<DebugViewNode>(core_3d::graph::NAME, DebugViewNode::NAME)
            .add_render_graph_edges(
                core_3d::graph::NAME,
                &[
                    core_3d::graph::node::BLOOM,
                    DebugViewNode::NAME,
                    core_3d::graph::node::END_MAIN_PASS_POST_PROCESSING,
                ],
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };
        render_app.init_resource::<DebugViewPipeline>();
    }
}

struct DebugViewNode {
    query: QueryState<
        (
            &'static ViewUniformOffset,
            &'static ViewTarget,
            &'static ViewPrepassTextures,
            &'static PrevFrameTexture,
            &'static PrepassDownsampleTexture,
            &'static VoxelPassTextures,
            &'static ScreenSpacePassesTextures,
            &'static PathTraceTextures,
        ),
        With<ExtractedView>,
    >,
}

impl DebugViewNode {
    pub const NAME: &str = "debug_view";
}

impl FromWorld for DebugViewNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl Node for DebugViewNode {
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

        let Ok((
            view_uniform_offset, 
            view_target, 
            prepass_textures, 
            prev_frame_tex, 
            prepass_downsample_texture, 
            voxel_pass_textures,
            screen_space_passes_textures,
            path_trace_textures)) = self.query.get_manual(world, view_entity) else {
            return Ok(());
        };

        let debug_view_pipeline = world.resource::<DebugViewPipeline>();

        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pipeline) = pipeline_cache.get_render_pipeline(debug_view_pipeline.pipeline_id) else {
            return Ok(());
        };

        let post_process = view_target.post_process_write();

        let depth_binding = prepass_textures.depth.as_ref().unwrap();
        let normal_binding = prepass_textures.normal.as_ref().unwrap();
        let motion_vectors_binding = prepass_textures.motion_vectors.as_ref().unwrap();

        let depth_view = depth_binding.texture.create_view(&TextureViewDescriptor {
            label: Some("prepass_depth"),
            aspect: TextureAspect::DepthOnly,
            ..default()
        });

        let entries = vec![
            view_binding_entry(0, world),
            globals_binding_entry(1, world),
            sampler_binding_entry(2, &debug_view_pipeline.sampler),
            tex_view_entry(3, &prepass_downsample_texture.0.default_view),
            tex_view_entry(4, &post_process.source),
            tex_view_entry(5, &prev_frame_tex.0.default_view),
            tex_view_entry(6, &screen_space_passes_textures.sm_tex_b.default_view),
            tex_view_entry(7, &screen_space_passes_textures.sm_tex_a.default_view),
            tex_view_entry(8, &screen_space_passes_textures.full_tex_b.default_view),
            tex_view_entry(9, &screen_space_passes_textures.full_tex_a.default_view),
            tex_view_entry(10, &voxel_pass_textures.write.default_view),
            tex_view_entry(11, &path_trace_textures.processed_img.default_view),
            tex_view_entry(12, &depth_view),
            tex_view_entry(13, &normal_binding.default_view),
            tex_view_entry(14, &motion_vectors_binding.default_view),
        ];

        let bind_group = render_context
            .render_device()
            .create_bind_group(&BindGroupDescriptor {
                label: Some("debug_view_bind_group"),
                layout: &debug_view_pipeline.layout,
                entries: &entries,
            });

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("debug_view_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: post_process.destination,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

#[derive(Resource)]
struct DebugViewPipeline {
    layout: BindGroupLayout,
    sampler: Sampler,
    pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for DebugViewPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let mut entries = vec![
            view_layout_entry(0),
            globals_layout_entry(1),
            sampler_layout_entry(2),
            image_layout_entry(3, TextureViewDimension::D2),
            image_layout_entry(4, TextureViewDimension::D2),
            image_layout_entry(5, TextureViewDimension::D2),
            image_layout_entry(6, TextureViewDimension::D2Array),
            image_layout_entry(7, TextureViewDimension::D2Array),
            image_layout_entry(8, TextureViewDimension::D2Array),
            image_layout_entry(9, TextureViewDimension::D2Array),
            image_layout_entry(10, TextureViewDimension::D3),
            image_layout_entry(11, TextureViewDimension::D2Array),
        ];

        // Prepass
        entries.extend_from_slice(&prepass_get_bind_group_layout_entries([12, 13, 14, 15], false)[..3]);

        let layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("debug_view_bind_group_layout"),
            entries: &entries,
        });

        let sampler = render_device.create_sampler(&linear_sampler());

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/debug_view.wgsl");

        let pipeline_id = default_full_screen_tri_pipeline_desc(
            vec![],
            layout.clone(),
            &mut world.resource_mut::<PipelineCache>(),
            shader,
            true,
        );

        Self {
            layout,
            sampler,
            pipeline_id,
        }
    }
}
