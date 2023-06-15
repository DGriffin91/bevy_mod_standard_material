use std::borrow::Cow;

use bevy::{
    core_pipeline::{core_3d, prepass::ViewPrepassTextures},
    prelude::*,
    reflect::TypeUuid,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
        render_resource::{
            BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindingResource, CachedComputePipelineId, ComputePassDescriptor,
            ComputePipelineDescriptor, Extent3d, FilterMode, PipelineCache, SamplerDescriptor,
            TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
            TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        texture::ImageSampler,
        view::{ExtractedView, ViewTarget},
        RenderApp,
    },
};

const WORKGROUP_SIZE: u32 = 8;
const MIP_LEVELS: u32 = 4;

use crate::{
    bind_group_utils::{
        prepass_get_bind_group_layout_entries, storage_tex_readwrite_layout_entry, tex_view_entry,
    },
    image,
    image_window_auto_size::{get_image_bytes_count, FrameData},
    resource,
};

pub struct PrepassDownsample;
impl Plugin for PrepassDownsample {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_image)
            //.add_systems(
            //    Update,
            //    auto_resize_image::<CustomStandardMaterial, PrepassDownsampleImage>,
            //)
            .add_plugin(ExtractResourcePlugin::<PrepassDownsampleImage>::default());

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
                    return;
                };

        render_app
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
    query: QueryState<(&'static ViewTarget, &'static ViewPrepassTextures), With<ExtractedView>>,
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

        let Ok((view_target, prepass_textures)) = self.query.get_manual(world, view_entity) else {
            return Ok(());
        };

        if !view_target.is_hdr() {
            println!("view_target is not HDR");
            return Ok(());
        }

        let copy_frame_pipeline = world.resource::<PrepassDownsamplePipeline>();
        let images = world.resource::<RenderAssets<Image>>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(copy_frame_pipeline.pipeline_id) else {
            return Ok(());
        };

        let depth_binding = prepass_textures.depth.as_ref().unwrap();
        let normal_binding = prepass_textures.normal.as_ref().unwrap();
        let motion_vectors_binding = prepass_textures.motion_vectors.as_ref().unwrap();

        let target_image = image!(images, &resource!(world, PrepassDownsampleImage).0);

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
        pass.dispatch_workgroups(
            // make sure we are >= target_image.size
            (target_image.size.x as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
            (target_image.size.y as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
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
        entries.extend_from_slice(&prepass_get_bind_group_layout_entries([0, 1, 2], false));

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

#[derive(Resource, Default, Clone, ExtractResource, TypeUuid)]
#[uuid = "8f2f1f50-98e2-43cf-a9b0-2345d38f0a9a"]
pub struct PrepassDownsampleImage(pub Handle<Image>);

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
            get_image_bytes_count(size.width as u32, size.height as u32, MIP_LEVELS, 4, 4)
        ],
        texture_descriptor: TextureDescriptor {
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[TextureFormat::Rgba32Float],
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

    commands.insert_resource(PrepassDownsampleImage(images.add(img)));
}

impl FrameData for PrepassDownsampleImage {
    fn image_h(&self) -> Handle<Image> {
        self.0.clone()
    }

    fn size(&self, width: u32, height: u32) -> (u32, u32) {
        (width, height)
    }

    fn resize(&self, width: u32, height: u32, images: &mut Assets<Image>) {
        let mut image = images.get_mut(&self.0).unwrap();
        image.texture_descriptor.size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        image.data = vec![0; get_image_bytes_count(width, height, MIP_LEVELS, 4, 4)];
        //self.0 = images.add(image);
    }
}
