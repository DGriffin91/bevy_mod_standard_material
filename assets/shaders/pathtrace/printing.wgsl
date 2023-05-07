fn digit_bin(x: i32) -> f32 {
    switch (x) {
        case 0: { return 480599.0; }
        case 1: { return 139810.0; }
        case 2: { return 476951.0; }
        case 3: { return 476999.0; }
        case 4: { return 350020.0; }
        case 5: { return 464711.0; }
        case 6: { return 464727.0; }
        case 7: { return 476228.0; }
        case 8: { return 481111.0; }
        case 9: { return 481095.0; }
        default: { return 0.0; }
    }
}

fn print_value_custom(
    frag_coord: vec2<f32>,
    starting_at: vec2<f32>,
    font_size: vec2<f32>,
    value: f32,
    digits: f32,
    decimals: f32
) -> f32 {
    let char_coord: vec2<f32> = (frag_coord * vec2<f32>(1.0, -1.0) - starting_at * vec2<f32>(1.0, -1.0) + font_size) / font_size;
    if (char_coord.y < 0.0 || char_coord.y >= 1.0) {
        return 0.0;
    }
    var bits: f32 = 0.0;
    let digit_index_1: f32 = digits - floor(char_coord.x) + 1.0;
    if (-digit_index_1 <= decimals) {
        let pow_1: f32 = pow(10.0, digit_index_1);
        let abs_value: f32 = abs(value);
        let pivot: f32 = max(abs_value, 1.5) * 10.0;
        if (pivot < pow_1) {
            if (value < 0.0 && pivot >= pow_1 * 0.1) {
                bits = 1792.0;
            }
        } else if (digit_index_1 == 0.0) {
            if (decimals > 0.0) {
                bits = 2.0;
            }
        } else {
            var value_2: f32;
            if (digit_index_1 < 0.0) {
                value_2 = fract(abs_value);
            } else {
                value_2 = abs_value * 10.0;
            }
            bits = digit_bin(i32((value_2 / pow_1) % 10.0));
        }
    }
    return floor((bits / pow(2.0, floor(fract(char_coord.x) * 4.0) + floor(char_coord.y * 5.0) * 4.0)) % 2.0);
}

fn print_value(
    frag_coord: vec2<f32>,
    color: vec4<f32>,
    row: i32,
    value: f32,
) -> vec4<f32> {
    let row_height = 10.0;
    let is_fps_digit = print_value_custom(
        frag_coord - vec2(0.0, row_height * f32(row) * 2.0),
        vec2(row_height),
        vec2(row_height),
        value,
        8.0,
        3.0
    );
    return select(color, vec4(1.0), is_fps_digit > 0.0);
}

fn print_value_b(
    frag_coord: vec2<f32>,
    color: vec4<f32>,
    row: i32,
    value: f32,
) -> vec4<f32> {
    let row_height = 10.0;
    let is_fps_digit = print_value_custom(
        frag_coord - vec2(0.0, row_height * f32(row) * 2.0),
        vec2(row_height),
        vec2(row_height),
        value,
        8.0,
        3.0
    );
    return select(color, vec4(0.0), is_fps_digit > 0.0);
}