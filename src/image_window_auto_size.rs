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
    fn resize(&self, width: u32, height: u32, images: &mut Assets<Image>);
    fn size(&self, width: u32, height: u32) -> (u32, u32);
}

pub fn auto_resize_image<A: Asset + ImageUpdate, T: Resource + FrameData + TypeUuid>(
    frame_data: Res<T>,
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
    let img_size = image.size();
    if (img_size.x as u32, img_size.y as u32) != frame_data.size(w, h) {
        frame_data.resize(w, h, &mut images);
        let image_h = frame_data.image_h();
        for (_, mat) in custom_materials.iter_mut() {
            mat.update(&T::TYPE_UUID, image_h.clone());
        }
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
