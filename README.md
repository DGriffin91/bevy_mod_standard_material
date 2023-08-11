# Minimal PBR material

For use with bevy 0.11

See for reference:
https://github.com/bevyengine/bevy/blob/main/crates/bevy_pbr/src/pbr_material.rs
https://github.com/bevyengine/bevy/blob/main/crates/bevy_pbr/src/render/pbr.wgsl

This also swaps out any instances of the standard material with the custom included material. (see `swap_standard_material()`)

![demo](demo.png)