use std::borrow::Cow;

use bevy::{
    core_pipeline::core_3d,
    prelude::*,
    reflect::TypeUuid,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
        render_resource::{
            BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindGroupLayoutEntry, BindingResource, BindingType, CachedComputePipelineId,
            ComputePassDescriptor, ComputePipelineDescriptor, Extent3d, FilterMode, PipelineCache,
            Sampler, SamplerBindingType, SamplerDescriptor, ShaderStages, StorageTextureAccess,
            TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
            TextureUsages, TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        texture::ImageSampler,
        view::{ExtractedView, ViewTarget},
        RenderApp,
    },
};

const WORKGROUP_SIZE: u32 = 8;
const MIP_LEVELS: u32 = 6;

use crate::{
    image_window_auto_size::{auto_resize_image, get_image_bytes_count, FrameData},
    pbr_material::CustomStandardMaterial,
};

pub struct CopyFramePlugin;
impl Plugin for CopyFramePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_image)
            .add_systems(
                Update,
                auto_resize_image::<CustomStandardMaterial, CopyFrameData>,
            )
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

        let copy_frame_pipeline = world.resource::<CopyFramePipeline>();
        let images = world.resource::<RenderAssets<Image>>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(copy_frame_data) = world.get_resource::<CopyFrameData>() else {
            return Ok(());
        };

        let Some(target_image) = images.get(&copy_frame_data.image) else {
            return Ok(());
        };

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(copy_frame_pipeline.pipeline_id) else {
            return Ok(());
        };

        let Some(pipeline2) = pipeline_cache.get_compute_pipeline(copy_frame_pipeline.pipeline_id2) else {
            return Ok(());
        };

        let mut views = Vec::new();
        for i in 0..MIP_LEVELS {
            let view = target_image.texture.create_view(&TextureViewDescriptor {
                label: Some("copy_frame_texture"),
                format: Some(target_image.texture.format()),
                dimension: Some(TextureViewDimension::D2),
                aspect: TextureAspect::All,
                base_mip_level: i,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
            });
            views.push(view);
        }

        for (pass_n, pipeline) in [pipeline, pipeline2].iter().enumerate() {
            let mut entries = vec![
                if pass_n == 0 {
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(view_target.main_texture_view()),
                    }
                } else {
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&views[2]),
                    }
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&copy_frame_pipeline.sampler),
                },
            ];

            for i in 0..3 {
                entries.push(BindGroupEntry {
                    binding: 2 + i as u32,
                    resource: BindingResource::TextureView(
                        &views[if pass_n == 0 { i } else { i + 3 }],
                    ),
                });
            }

            let bind_group =
                render_context
                    .render_device()
                    .create_bind_group(&BindGroupDescriptor {
                        label: Some("copy_frame_bind_group"),
                        layout: &copy_frame_pipeline.layout,

                        entries: &entries,
                    });

            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());

            pass.set_bind_group(0, &bind_group, &[]);

            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(
                // make sure we are >= target_image.size
                (target_image.size.x as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
                (target_image.size.y as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
                1,
            );
        }

        Ok(())
    }
}

#[derive(Resource)]
struct CopyFramePipeline {
    layout: BindGroupLayout,
    sampler: Sampler,
    pipeline_id: CachedComputePipelineId,
    pipeline_id2: CachedComputePipelineId,
}

impl FromWorld for CopyFramePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let mut entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ];
        for i in 0..3 {
            entries.push(BindGroupLayoutEntry {
                binding: 2 + i,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: TextureFormat::Rgba16Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            });
        }

        let layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("copy_frame_bind_group_layout"),
                    entries: &entries,
                });

        let sampler = render_device.create_sampler(&SamplerDescriptor::default());

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/copy_frame_pass.wgsl");

        let pipeline_cache = world.resource::<PipelineCache>();

        let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Cow::from("update"),
        });

        let pipeline_id2 = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Cow::from("update2"),
        });

        Self {
            layout,
            sampler,
            pipeline_id,
            pipeline_id2,
        }
    }
}

#[derive(Resource, Default, Clone, ExtractResource, TypeUuid)]
#[uuid = "e485c7f9-a2c5-4433-ac2b-6e65d1779086"]
pub struct CopyFrameData {
    pub image: Handle<Image>,
}

impl FrameData for CopyFrameData {
    fn image_h(&self) -> Handle<Image> {
        self.image.clone()
    }

    fn size(&self, width: u32, height: u32) -> (u32, u32) {
        (width, height)
    }

    fn resize(&self, width: u32, height: u32, images: &mut Assets<Image>) {
        let image = images.get_mut(&self.image).unwrap();
        image.texture_descriptor.size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        image.data = vec![0; get_image_bytes_count(width, height, MIP_LEVELS, 2, 4)];
    }
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

    let img = Image {
        data: vec![
            0;
            get_image_bytes_count(size.width as u32, size.height as u32, MIP_LEVELS, 2, 4)
        ],
        texture_descriptor: TextureDescriptor {
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[TextureFormat::Rgba16Float],
            label: None,
            size,
            mip_level_count: MIP_LEVELS,
            sample_count: match *msaa {
                Msaa::Off => 1,
                Msaa::Sample2 => 2,
                Msaa::Sample4 => 4,
                Msaa::Sample8 => 8,
            },
        },
        sampler_descriptor: ImageSampler::Descriptor(SamplerDescriptor {
            label: Some("copy_image_sampler_descriptor"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..default()
        }),
        texture_view_descriptor: None,
    };

    commands.insert_resource(CopyFrameData {
        image: images.add(img),
    });
}
