use bevy::{
    core_pipeline::{
        core_3d, fullscreen_vertex_shader::fullscreen_shader_vertex_state,
        prepass::ViewPrepassTextures,
    },
    prelude::*,
    render::{
        globals::GlobalsBuffer,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
        render_resource::{
            BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindingResource, CachedRenderPipelineId, ColorTargetState, ColorWrites, FilterMode,
            FragmentState, MultisampleState, Operations, PipelineCache, PrimitiveState,
            RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, Sampler,
            SamplerDescriptor, TextureFormat, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        view::{ExtractedView, ViewTarget, ViewUniformOffset, ViewUniforms},
        RenderApp,
    },
};

use crate::{
    bind_group_utils::{
        default_full_screen_tri_pipeline_desc, globals_entry, image_entry, linear_sampler,
        prepass_get_bind_group_layout_entries, sampler_entry, view_entry,
    },
    copy_frame::CopyFrameData,
    prepass_downsample::PrepassDownsampleImage,
    resource, retrieve_tex_view_entry,
    screen_space_passes::ScreenSpacePasses,
    tex_view_entry,
    voxel_pass::VoxelPassesTargetImage,
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
        let view_uniforms: &ViewUniforms = world.resource::<ViewUniforms>();
        let view_uniforms = view_uniforms.uniforms.binding().unwrap();
        let globals_buffer = world.resource::<GlobalsBuffer>();
        let globals_binding = globals_buffer.buffer.binding().unwrap();
        let images = world.resource::<RenderAssets<Image>>();

        let Ok((view_uniform_offset, view_target, prepass_textures)) = self.query.get_manual(world, view_entity) else {
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
                resource: BindingResource::Sampler(&debug_view_pipeline.sampler),
            },
            retrieve_tex_view_entry!(3, images, resource!(world, PrepassDownsampleImage).0),
            tex_view_entry!(4, &post_process.source),
            retrieve_tex_view_entry!(5, images, resource!(world, CopyFrameData).image),
            retrieve_tex_view_entry!(6, images, resource!(world, ScreenSpacePasses).current_img),
            retrieve_tex_view_entry!(7, images, resource!(world, ScreenSpacePasses).processed_img),
            retrieve_tex_view_entry!(8, images, resource!(world, VoxelPassesTargetImage).current),
            tex_view_entry!(9, &depth_binding.default_view),
            tex_view_entry!(10, &normal_binding.default_view),
            tex_view_entry!(11, &motion_vectors_binding.default_view),
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
            view_entry(0),
            globals_entry(1),
            sampler_entry(2),
            image_entry(3, TextureViewDimension::D2),
            image_entry(4, TextureViewDimension::D2),
            image_entry(5, TextureViewDimension::D2),
            image_entry(6, TextureViewDimension::D2Array),
            image_entry(7, TextureViewDimension::D2Array),
            image_entry(8, TextureViewDimension::D3),
        ];

        // Prepass
        entries.extend_from_slice(&prepass_get_bind_group_layout_entries([9, 10, 11], false));

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
