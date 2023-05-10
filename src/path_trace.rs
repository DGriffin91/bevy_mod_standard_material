use std::borrow::Cow;

use crate::{
    copy_frame::{CopyFrameData, FrameCopyNode},
    image_window_auto_size::{auto_resize_image, get_image_bytes_count, FrameData},
    pbr_material::{BlueNoise, CustomStandardMaterial},
    prepass_downsample::prepass_get_bind_group_layout_entries,
};
use bevy::{
    core_pipeline::{core_3d, prepass::ViewPrepassTextures},
    diagnostic::{Diagnostics, FrameTimeDiagnosticsPlugin},
    math::vec3,
    pbr::get_bind_group_layout_entries,
    prelude::*,
    reflect::TypeUuid,
    render::{
        extract_component::{
            ComponentUniforms, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
        },
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        globals::GlobalsBuffer,
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
        render_resource::{
            BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
            BindingResource, CachedComputePipelineId, CachedRenderPipelineId,
            ComputePassDescriptor, ComputePipelineDescriptor, Extent3d, FilterMode, Operations,
            PipelineCache, RenderPassColorAttachment, RenderPassDescriptor, Sampler,
            SamplerDescriptor, ShaderType, StorageBuffer, TextureDescriptor, TextureDimension,
            TextureFormat, TextureUsages, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::ImageSampler,
        view::{ExtractedView, ViewTarget, ViewUniformOffset, ViewUniforms},
        Extract, RenderApp,
    },
};
use bevy_mod_bvh::pipeline_utils::*;
use bevy_mod_bvh::{
    bind_group_layout_entry,
    gpu_data::{
        extract_gpu_data, new_storage_buffer, DynamicInstanceOrder, GPUBuffers, GPUDataPlugin,
        StaticInstanceOrder,
    },
    BVHPlugin, BVHSet, DynamicTLAS, StaticTLAS,
};

const WORKGROUP_SIZE: u32 = 8;
pub struct PathTracePlugin;
impl Plugin for PathTracePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_image)
            .add_systems(
                Update,
                (set_meshes_tlas, update_settings).before(BVHSet::BlasTlas),
            )
            .add_systems(
                Update,
                auto_resize_image::<CustomStandardMaterial, PathTraceTargetImage>,
            )
            .add_plugin(BVHPlugin)
            .add_plugin(GPUDataPlugin)
            .add_plugin(ExtractComponentPlugin::<TraceSettings>::default())
            .add_plugin(UniformComponentPlugin::<TraceSettings>::default())
            .add_plugin(ExtractResourcePlugin::<BlueNoise>::default())
            .add_plugin(ExtractResourcePlugin::<PathTraceTargetImage>::default());

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(ExtractSchedule, extract_materials.after(extract_gpu_data))
            .init_resource::<GpuMatBuffers>()
            .add_render_graph_node::<PathTraceNode>(core_3d::graph::NAME, PathTraceNode::NAME)
            .add_render_graph_edges(
                core_3d::graph::NAME,
                &[
                    core_3d::graph::node::END_MAIN_PASS,
                    PathTraceNode::NAME,
                    core_3d::graph::node::BLOOM,
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

pub struct PathTraceNode {
    query: QueryState<
        (
            &'static ViewUniformOffset,
            &'static ViewTarget,
            &'static ViewPrepassTextures,
        ),
        With<ExtractedView>,
    >,
}

impl PathTraceNode {
    pub const NAME: &str = "path_trace";
}

impl FromWorld for PathTraceNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl Node for PathTraceNode {
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
        let gpu_buffers = world.resource::<GPUBuffers>();
        let gpu_mat_buffers = world.resource::<GpuMatBuffers>();
        let images = world.resource::<RenderAssets<Image>>();

        let Some(pathtrace_target) = world.get_resource::<PathTraceTargetImage>() else {
            return Ok(());
        };
        let Some(target_image) = images.get(&pathtrace_target.0) else {
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

        let path_trace_pipeline = world.resource::<TracePipeline>();

        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pipeline) = pipeline_cache.get_compute_pipeline(path_trace_pipeline.pipeline_id) else {
            return Ok(());
        };

        let settings_uniforms = world.resource::<ComponentUniforms<TraceSettings>>();
        let Some(settings_binding) = settings_uniforms.uniforms().binding() else {
            return Ok(());
        };

        let Some(gpu_buffer_bind_group_entries) = gpu_buffers
                .bind_group_entries([5, 6, 7, 8, 9, 10, 11]) else {
            return Ok(());
        };

        let Some(static_material_instance_binding) =
            gpu_mat_buffers.static_material_instance_buffer.binding() else
        {
            return Ok(());
        };

        let Some(dynamic_material_instance_binding) =
            gpu_mat_buffers.dynamic_material_instance_buffer.binding() else
        {
            return Ok(());
        };

        let post_process = view_target.post_process_write();

        let depth_binding = prepass_textures.depth.as_ref().unwrap();
        let normal_binding = prepass_textures.normal.as_ref().unwrap();
        let motion_vectors_binding = prepass_textures.motion_vectors.as_ref().unwrap();

        let mut entries = vec![
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
                resource: BindingResource::Sampler(&path_trace_pipeline.sampler),
            },
            BindGroupEntry {
                binding: 4,
                resource: settings_binding.clone(),
            },
            BindGroupEntry {
                binding: 12,
                resource: BindingResource::TextureView(&blue_noise.texture_view),
            },
            BindGroupEntry {
                binding: 13,
                resource: static_material_instance_binding,
            },
            BindGroupEntry {
                binding: 14,
                resource: dynamic_material_instance_binding,
            },
            BindGroupEntry {
                binding: 15,
                resource: BindingResource::TextureView(&depth_binding.default_view),
            },
            BindGroupEntry {
                binding: 16,
                resource: BindingResource::TextureView(&normal_binding.default_view),
            },
            BindGroupEntry {
                binding: 17,
                resource: BindingResource::TextureView(&motion_vectors_binding.default_view),
            },
            BindGroupEntry {
                binding: 18,
                resource: BindingResource::TextureView(&target_image.texture_view),
            },
        ];

        entries.extend(gpu_buffer_bind_group_entries);

        let bind_group = render_context
            .render_device()
            .create_bind_group(&BindGroupDescriptor {
                label: Some("path_trace_bind_group"),
                layout: &path_trace_pipeline.layout,
                entries: &entries,
            });

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        pass.dispatch_workgroups(
            prev_frame.size.x as u32 / WORKGROUP_SIZE,
            prev_frame.size.y as u32 / WORKGROUP_SIZE,
            1,
        );

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
        let render_device = world.resource::<RenderDevice>();

        let mut entries = vec![
            view_entry(0),
            globals_entry(1),
            image_entry(2, TextureViewDimension::D2),
            sampler_entry(3),
            uniform_entry(4, Some(TraceSettings::min_size())),
            image_entry(12, TextureViewDimension::D2Array),
            MaterialData::bind_group_layout_entry(13),
            MaterialData::bind_group_layout_entry(14),
            storage_tex_write(18, TextureFormat::Rgba16Float, TextureViewDimension::D2),
        ];

        entries.append(&mut GPUBuffers::bind_group_layout_entry([5, 6, 7, 8, 9, 10, 11]).to_vec());

        // Prepass
        entries.extend_from_slice(&prepass_get_bind_group_layout_entries([15, 16, 17], false));

        let layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("path_trace_bind_group_layout"),
            entries: &entries,
        });

        let sampler = render_device.create_sampler(&SamplerDescriptor::default());

        let shader = world
            .resource::<AssetServer>()
            .load("shaders/pathtrace/raytrace_example.wgsl");

        let pipeline_id = world
            .resource_mut::<PipelineCache>()
            .queue_compute_pipeline(ComputePipelineDescriptor {
                label: None,
                layout: vec![layout.clone()],
                push_constant_ranges: Vec::new(),
                shader,
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

#[derive(Component, Default, Clone, Copy, ExtractComponent, ShaderType)]
pub struct TraceSettings {
    pub frame: u32,
    pub fps: f32,
}

fn set_meshes_tlas(
    mut commands: Commands,
    query: Query<
        Entity,
        (
            With<Handle<Mesh>>,
            Without<StaticTLAS>,
            Without<DynamicTLAS>,
        ),
    >,
) {
    for entity in &query {
        commands.entity(entity).insert(StaticTLAS);
    }
}

fn update_settings(mut settings: Query<&mut TraceSettings>, diagnostics: Res<Diagnostics>) {
    for mut setting in &mut settings {
        setting.frame = setting.frame.wrapping_add(1);
        if let Some(diag) = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS) {
            let hysteresis = 0.9;
            let fps = hysteresis + diag.value().unwrap_or(0.0) as f32;
            setting.fps = setting.fps * hysteresis + fps * (1.0 - hysteresis);
        }
    }
}

#[derive(ShaderType, Default)]
pub struct MaterialData {
    pub color: Vec3,
    pub perceptual_roughness: f32,
    pub metallic: f32,
    pub reflectance: f32,
}

impl MaterialData {
    bind_group_layout_entry!();
}

#[derive(Resource, Default)]
pub struct GpuMatBuffers {
    pub static_material_instance_buffer: StorageBuffer<Vec<MaterialData>>,
    pub dynamic_material_instance_buffer: StorageBuffer<Vec<MaterialData>>,
}

pub fn extract_materials(
    static_instance_order: Res<StaticInstanceOrder>,
    dynamic_instance_order: Res<DynamicInstanceOrder>,
    custom_materials: Extract<Res<Assets<CustomStandardMaterial>>>,
    entites: Extract<Query<&Handle<CustomStandardMaterial>>>,
    mut gpu_mat_data: ResMut<GpuMatBuffers>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    // TODO figure out better way to detect material changes, just specific to dyn or static
    if static_instance_order.is_changed() || custom_materials.is_changed() {
        gpu_mat_data.static_material_instance_buffer = new_storage_buffer(
            collect_mats(&static_instance_order.0, &entites, &custom_materials),
            "static_material_instance_buffer",
            &render_device,
            &render_queue,
        );
    }
    if dynamic_instance_order.is_changed() || custom_materials.is_changed() {
        gpu_mat_data.dynamic_material_instance_buffer = new_storage_buffer(
            collect_mats(&dynamic_instance_order.0, &entites, &custom_materials),
            "dynamic_material_instance_buffer",
            &render_device,
            &render_queue,
        );
    }
}

fn collect_mats(
    instance_order: &Vec<Entity>,
    entites: &Query<&Handle<CustomStandardMaterial>>,
    custom_materials: &Assets<CustomStandardMaterial>,
) -> Vec<MaterialData> {
    let mut material_data = Vec::new();
    for e in instance_order.iter() {
        if let Ok(mat_h) = entites.get(*e) {
            if let Some(mat) = custom_materials.get(mat_h) {
                let c = mat.base_color.as_linear_rgba_f32();
                material_data.push(MaterialData {
                    color: vec3(c[0], c[1], c[2]),
                    perceptual_roughness: mat.perceptual_roughness,
                    metallic: mat.metallic,
                    reflectance: mat.reflectance,
                })
            } else {
                material_data.push(MaterialData::default())
            }
        } else {
            material_data.push(MaterialData::default())
        }
    }
    material_data
}

#[derive(Resource, Default, Clone, ExtractResource, TypeUuid)]
#[uuid = "c235dff3-905c-4e88-9e0e-fb1c76de1322"]
pub struct PathTraceTargetImage(pub Handle<Image>);

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
        data: vec![0; get_image_bytes_count(size.width as u32, size.height as u32, 1, 4, 4)],
        texture_descriptor: TextureDescriptor {
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
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
        sampler_descriptor: ImageSampler::Descriptor(SamplerDescriptor {
            label: Some("copy_image_sampler_descriptor"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..default()
        }),
        texture_view_descriptor: None,
    };

    commands.insert_resource(PathTraceTargetImage(images.add(img)));
}

impl FrameData for PathTraceTargetImage {
    fn image_h(&self) -> Handle<Image> {
        self.0.clone()
    }

    fn set_image_h(&mut self, image_h: Handle<Image>) {
        self.0 = image_h;
    }

    fn bytes(&self, width: u32, height: u32) -> Vec<u8> {
        vec![0; get_image_bytes_count(width, height, 1, 4, 4)]
    }

    fn size(&self, width: u32, height: u32) -> Extent3d {
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        }
    }
}
