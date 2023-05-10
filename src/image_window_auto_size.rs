use bevy::{
    asset::Asset, math::vec2, prelude::*, reflect::TypeUuid, render::render_resource::Extent3d,
    utils::Uuid,
};

///For automatically resizing an image relative to the viewport size

pub trait ImageUpdate {
    fn update(&mut self, name: &Uuid, image_h: Handle<Image>);
}

pub trait FrameData {
    fn image_h(&self) -> Handle<Image>;
    fn set_image_h(&mut self, image_h: Handle<Image>);
    fn bytes(&self, width: u32, height: u32) -> Vec<u8>;
    fn size(&self, width: u32, height: u32) -> Extent3d;
}

pub fn auto_resize_image<A: Asset + ImageUpdate, T: Resource + FrameData + TypeUuid>(
    mut frame_data: ResMut<T>,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window>,
    mut custom_materials: ResMut<Assets<A>>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let Some(image) = images.get(&frame_data.image_h()) else {
        return;
    };
    let w = window.physical_width();
    let h = window.physical_height();

    if w == 0 || h == 0 {
        return;
    }

    if image.size() != vec2(w as f32, h as f32) {
        let mut image = image.clone();
        image.data = frame_data.bytes(w as u32, h as u32);
        image.texture_descriptor.size = frame_data.size(w as u32, h as u32);
        let image_h = images.add(image);
        for (_, mat) in custom_materials.iter_mut() {
            mat.update(&T::TYPE_UUID, image_h.clone());
        }
        frame_data.set_image_h(image_h.clone());
        // whyyyyyyyyyyyyyyyyyy
    }
}

pub fn get_image_bytes_count(
    w: u32,
    h: u32,
    mip_levels: u32,
    bytes_per_component: u32,
    components: u32,
) -> usize {
    let mut width = w;
    let mut height = h;

    let mut data_size = 0;

    for _ in 0..mip_levels {
        //2 bytes per component, 4 components per pixel
        data_size += width * height * bytes_per_component * components;
        width /= 2;
        height /= 2;
    }

    data_size as usize
}
