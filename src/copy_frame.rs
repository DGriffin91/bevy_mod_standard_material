use bevy::{
    core_pipeline::{core_3d, fullscreen_vertex_shader::fullscreen_shader_vertex_state},
    math::vec2,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
        render_resource::{
            BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, BindingResource, BindingType, CachedRenderPipelineId,
            ColorTargetState, ColorWrites, Extent3d, FragmentState, MultisampleState, Operations,
            PipelineCache, PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages,
            TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
            TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        texture::ImageSampler,
        view::{ExtractedView, ViewTarget},
        RenderApp,
    },
};

use crate::{path_trace::PathTraceNode, pbr_material::CustomStandardMaterial};

pub struct CopyFramePlugin;
impl Plugin for CopyFramePlugin {
    fn build(&self, app: &mut App) {
        app.add_startup_system(setup_image)
            .add_system(resize_image)
            .add_plugin(ExtractResourcePlugin::<CopyFrameData>::default());

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
                    return;
                };

        render_app
            .add_render_graph_node::<FrameCopyNode>(core_3d::graph::NAME, FrameCopyNode::NAME)
            .add_render_graph_edges(
                core_3d::graph::NAME,
                &[
                    core_3d::graph::node::END_MAIN_PASS,
                    FrameCopyNode::NAME,
                    core_3d::graph::node::BLOOM,
                ],
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };
        render_app.init_resource::<CopyFramePipeline>();
    }
}

pub struct FrameCopyNode {
    query: QueryState<&'static ViewTarget, With<ExtractedView>>,
}

impl FromWorld for FrameCopyNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl FrameCopyNode {
    pub const NAME: &str = "copy_frame";
}

impl Node for FrameCopyNode {
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

        let Ok(view_target) = self.query.get_manual(world, view_entity) else {
            return Ok(());
        };

        if !view_target.is_hdr() {
            println!("view_target is not HDR");
            return Ok(());
        }

        let post_process_pipeline = world.resource::<CopyFramePipeline>();
        let images = world.resource::<RenderAssets<Image>>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(copy_frame_data) = world.get_resource::<CopyFrameData>() else {
            return Ok(());
        };

        let Some(target_image) = images.get(&copy_frame_data.image) else {
            return Ok(());
        };

        let Some(pipeline) = pipeline_cache.get_render_pipeline(post_process_pipeline.pipeline_id) else {
            return Ok(());
        };

        let bind_group = render_context
            .render_device()
            .create_bind_group(&BindGroupDescriptor {
                label: Some("post_process_bind_group"),
                layout: &post_process_pipeline.layout,

                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(view_target.main_texture()),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&post_process_pipeline.sampler),
                    },
                ],
            });

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("post_process_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &target_image.texture_view,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
        });
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

#[derive(Resource)]
struct CopyFramePipeline {
    layout: BindGroupLayout,
    sampler: Sampler,
    pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for CopyFramePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("post_process_bind_group_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let sampler = render_device.create_sampler(&SamplerDescriptor::default());

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/post_process_pass.wgsl");

        let pipeline_id =
            world
                .resource_mut::<PipelineCache>()
                .queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some("post_process_pipeline".into()),
                    layout: vec![layout.clone()],

                    vertex: fullscreen_shader_vertex_state(),
                    fragment: Some(FragmentState {
                        shader,
                        shader_defs: vec![],

                        entry_point: "fragment".into(),
                        targets: vec![Some(ColorTargetState {
                            format: TextureFormat::Rgba16Float,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        })],
                    }),

                    primitive: PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                    push_constant_ranges: vec![],
                });

        Self {
            layout,
            sampler,
            pipeline_id,
        }
    }
}

#[derive(Resource, Default, Clone, ExtractResource)]
pub struct CopyFrameData {
    pub image: Handle<Image>,
}

fn setup_image(
    mut commands: Commands,
    windows: Query<&Window>,
    mut images: ResMut<Assets<Image>>,
    msaa: Res<Msaa>,
) {
    let window = windows.single();
    let size = Extent3d {
        width: window.physical_width(),
        height: window.physical_height(),
        depth_or_array_layers: 1,
    };

    let mut img = Image {
        data: vec![0; 8],
        texture_descriptor: TextureDescriptor {
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[TextureFormat::Rgba16Float],
            label: None,
            size,
            mip_level_count: 1,
            sample_count: match *msaa {
                Msaa::Off => 1,
                Msaa::Sample2 => 2,
                Msaa::Sample4 => 4,
                Msaa::Sample8 => 8,
            },
        },
        sampler_descriptor: ImageSampler::Descriptor(SamplerDescriptor::default()),
        texture_view_descriptor: None,
    };
    img.resize(size);

    commands.insert_resource(CopyFrameData {
        image: images.add(img),
    });
}

fn resize_image(
    mut copy_frame_data: ResMut<CopyFrameData>,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window>,
    mut custom_materials: ResMut<Assets<CustomStandardMaterial>>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let Some(image) = images.get(&copy_frame_data.image) else {
        return;
    };
    let w = window.physical_width();
    let h = window.physical_height();

    if w == 0 || h == 0 {
        return;
    }

    if image.size() != vec2(w as f32, h as f32) {
        let mut image = image.clone();
        image.resize(Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        });
        let image_h = images.add(image);
        for (_, mat) in custom_materials.iter_mut() {
            mat.prev_image = Some(image_h.clone());
        }
        copy_frame_data.image = image_h.clone();
        // whyyyyyyyyyyyyyyyyyy
    }
}
