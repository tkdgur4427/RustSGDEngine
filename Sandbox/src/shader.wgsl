// vertex shader

// [[block]] the block decorator indicates this structure type represents the contents of a buffer resource occupying a single binding slot in shader's resource interface
// * any structure used as a 'uniform' must be annotated with [[block]]
[[block]]
struct Uniforms {
    view_proj: mat4x4<f32>;
};

struct InstanceInput {
    [[location(5)]] model_matrix_0: vec4<f32>;
    [[location(6)]] model_matrix_1: vec4<f32>;
    [[location(7)]] model_matrix_2: vec4<f32>;
    [[location(8)]] model_matrix_3: vec4<f32>;
};

// new bind group; we need to specify which one we're using in the shader
// * the number is determined by our 'render_pipeline_layout'
// * the 'texture_bind_group_layout' is listed first, thus it's 'group(0)', and 'uniform_bind_group' is second, so it's 'group(1)'
[[group(1), binding(0)]]
var<uniform> uniforms: Uniforms;

struct VertexInput {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] tex_coords: vec2<f32>;
};

// [[builtin(position)]] bit tells WGPU that this is the value we want to use as vertex's clip positions (= gl_Position)
struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] tex_coords: vec2<f32>;
};

[[stage(vertex)]]
fn main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    var out: VertexOutput;
    out.tex_coords = model.tex_coords;

    // multiplication order is important when it comes to matrices
    // * the vector goes on right, the matrices gone on the left in order of importance
    out.clip_position = uniforms.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}

// fragment shader
[[group(0), binding(0)]]
var t_diffuse: texture_2d<f32>;

[[group(0), binding(1)]]
var s_diffuse: sampler;

// [[location(0)]] bit tells WGPU to store the value the vec4 returned by this function in the first color target
[[stage(fragment)]]
fn main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}