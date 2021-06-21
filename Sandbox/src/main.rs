use std::iter;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use wgpu::util::DeviceExt;

mod texture;

// [repr(C)] : alternative representations: Rust allows you to specify alternative data layout strategies from the default

// [derive(...)]: the compiler is capable of providing basic implementations for some traits via the #[derive]
// * comparison traits: 'Eq', 'ParitalEq', 'Ord', 'PartialOrd'
// * 'Clone', to create 'T' from '&T' via a copy
// * 'Copy', to give a type 'copy semantics' instead of 'move semantics'
// * 'Hash', to compute a hash from &T
// * 'Default', to create an empty instance of a data type
// * 'Debug', to format a value using the '{:?}' formatter

// when you use [derive(bytemuck::Pod, bytemuck::Zeroable)], we do NOT need to implement these:
// * unsafe impl bytemuck::Pod for Vertex {}
// * unsafe impl bytemuck::Zeroable for Vertex {}

// declare array in struct
// * Vec<T>: (vector) dynamically sized; dynamically allocated on the heap
// * [T; n]: (array) statically sized; lives on the stack
// * [T]   : (slice) unsized; usually used from '&[T]'; this is a view into a contiguous set of 'T's in memory somewhere

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

// the 'a reads 'the lifetime a'; technically, every reference has some lifetime associated with it, but the compiler let you elide them in common cases

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        
        // describes how the vertex buffer is interpreted
        wgpu::VertexBufferLayout {
            // the stride in bytes, between elements of this buffer

            // std::mem::sizeof::<Vertex>(): = sizeof(type)
            // 'as' : cast between types or rename an import
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,

            // 'step_mode' tells the pipeline how often it should move to the next vertex
            // * we can specify 'wgpu::InputStepMode::Instance'
            step_mode: wgpu::InputStepMode::Vertex,

            // this list of attributes which comprise a single vertex
            // * vertex attributes describe the individual parts of the vertex; generally 1:1 mapping with struct's fields
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0, 
                    // this tells the shader what location to store this attribute at
                    // * 'layout(location=0) in vec3 x' in the vertex shader would correspond to the position field of struct
                    // * 'layout(location=1) in vec3 x' would be the color field
                    shader_location: 0,

                    // 'format' tells the shader the shape of the attribute
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    // offset of first VertexAttribute
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                }
            ]
        }
    }
}

// we arrange the vertices in Ccw order
const VERTICES: &[Vertex] = &[
    Vertex { position: [-0.0868241, 0.49240386, 0.0], tex_coords: [0.4131759, 0.99240386], },
    Vertex { position: [-0.49513406, 0.06958647, 0.0], tex_coords: [0.0048659444, 0.56958646], },
    Vertex { position: [-0.21918549, -0.44939706, 0.0], tex_coords: [0.28081453, 0.050602943], },
    Vertex { position: [0.35966998, -0.3473291, 0.0], tex_coords: [0.85967, 0.15267089], },
    Vertex { position: [0.44147372, 0.2347359, 0.0], tex_coords: [0.9414737, 0.7347359], },
];

const INDICES: &[u16] = &[
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
    0, // padding
];

struct State {
    // handle to presentable surface
    surface: wgpu::Surface,

    // responsible for creation of most rendering and compute resources
    // these are then used in commands, which are submitted to a ['Queue']
    device: wgpu::Device,

    // handle to a command queue a device
    // * a 'Queue' executes recorded ['CommandBuffer'] objects and provides convenience methods
    //   for writing to [buffers](Queue::write_buffer) and [textures](Queue::write_texture)
    queue: wgpu::Queue,

    // describe a ['SwapChain']
    sc_desc: wgpu::SwapChainDescriptor,

    // a 'SwapChain' represents the image or series of images that will be presented to a ['Surface']
    // a 'SwapChain' may be created with ['Device::create_swap_chain']
    swap_chain: wgpu::SwapChain,

    // a size represented in physical pixels
    size: winit::dpi::PhysicalSize<u32>,

    // handle to rendering (graphics) pipeline
    // a 'render pipeline' object represents a graphics pipeline and its stages, binding, vertex buffers and targets
    // a 'render pipeline' may be created with [Device::create_render_pipeline]
    render_pipeline: wgpu::RenderPipeline,

    // handle to a GPU-accessiable buffer
    // created with ['Device::create_buffer'] or [DeviceExt::create_buffer_init](util::DeviceExt::create_buffer_init)
    vertex_buffer: wgpu::Buffer,

    index_buffer: wgpu::Buffer,
    num_indices: u32,
    diffuse_bind_group: wgpu::BindGroup,

    diffuse_texture: texture::Texture,
}

impl State {
    // creating some of the wgpu types requires async code
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // the instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

        // the surface is used to create the swap_chain
        let surface = unsafe { instance.create_surface(window) };

        // we need the adapter to create the device and queue
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
            },
        ).await.unwrap();

        // create the device and queue with adapter
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                label: None,
            },
            None, // trace path
        ).await.unwrap();

        let sc_desc = wgpu::SwapChainDescriptor {
            // the usage of the swap chain; only supported usage is 'RENDER_ATTACHMENT'
            // RENDER_ATTACHMENT specifies that the textures will be used to write to the screen
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
            // the texture format of the swap chain; the only formats that are guaranteed are 'Bgra8Unorm' and 'Bgra8UnormSrgb'
            format: adapter.get_swap_chain_preferred_format(&surface).unwrap(),
            // width/height of the swap chain
            width: size.width,
            height: size.height,
            // presentation mode of the swap chain; FIFO is the only guaranteed to be supported, through,
            // other formats will automatically fall back to FIFO
            present_mode: wgpu::PresentMode::Fifo,
        };

        let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        // loading an image from a file

        // include_bytes: includes a file as reference to a byte array
        let diffuse_bytes = include_bytes!("happy-tree.png");
        let diffuse_texture = texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree.png").unwrap();

        /*
        let diffuse_rgba = diffuse_image.as_rgba8().unwrap();

        // save image dimensions as actual 'Texture'
        use image::GenericImageView;
        let dimensions = diffuse_image.dimensions();

        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };

        let diffuse_texture = device.create_texture(
            &wgpu::TextureDescriptor {
                // all textures are stored as 3D, we represent our 2D texture by setting depth to 1
                size: texture_size,
                mip_level_count: 1, // we'll talk about this a little later
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                // SAMPLED tells wgpu that we want to use this texture in shaders
                // COPY_DST means that we want to copy data to this texture
                usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
                label: Some("diffuse_texture"),
            }
        );

        // getting data into a Texture
        // * the 'Texture' struct has no methods to interact with the data directly
        // * we can use a method on the 'queue' we created earlier called 'write_texture' to load the texture in

        queue.write_texture(
            // tells wgpu where to copy the pixel data
            wgpu::ImageCopyTexture {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            // the actual pixel data
            diffuse_rgba,
            // the layout of the texture
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * dimensions.0),
                rows_per_image: std::num::NonZeroU32::new(dimensions.1),
            },
            texture_size,
        );

        // TextureViews and Samplers
        // * a 'TextureView' offers us a view into our texture
        // * a 'Sampler' controls how the 'Texture' is sampled

        let diffuse_texture_view = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        */

        // the BindGroup
        // * a 'BindGroup' describes a set of resources and how they can be accessed by a shader
        // * we create a 'BindGroup' using a 'BindGroupLayout'

        // 'texture_bind_group_layout' has two entries: one for a sampled texture at binding 0, and one for a sampler at binding 1.
        // * bindings are visible only to the fragment shader specified by 'FRAGMENT'
        // * any bitwise combination of 'NONE', 'VERTEX', 'FRAGMENT', or 'COMPUTE'
        let texture_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // describes a single binding inside a bind group
                    wgpu::BindGroupLayoutEntry {
                        // binding index; must match shader index and be unique inside a BindGroupLayout
                        // * 'layout(set = 0, binding = 1) uniform' in shaders
                        binding: 0,
                        // which shader stages can see this binding
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        // array size must be 1 or greater
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: true,
                        },
                        count: None,
                    }
                ],
                label: Some("texture_bind_group_layout"),
            }
        );

        let diffuse_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                    }
                ],
                label: Some("diffuse_bind_group"),
            }
        );

        let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            flags: wgpu::ShaderFlags::all(),
            // ShaderSource<'a>; source of a shader module
            // Wgsl module as a string slice
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let render_pipeline_layout = 
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            // the layout of bind groups for this pipeline
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                // the compiled shader module
                module: &shader,
                // the name of entry point in compiled shader
                entry_point: "main",
                // the format of any vertex buffers used with this pipeline
                // the 'buffers' field tells 'wgpu' what type of vertices we want to pass to the vertex shader
                buffers: &[
                    Vertex::desc(),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "main",
                // the color state of the render targets
                targets: &[wgpu::ColorTargetState {
                    format: sc_desc.format,
                    // the blending that is used for this pipeline
                    blend: Some(wgpu::BlendState::REPLACE),
                    // make which enables/disables writes to different color/alpha channel
                    write_mask: wgpu::ColorWrite::ALL,
                }],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // requires Features::DEPTH_CLAMPING
                clamp_depth: false,
                // requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                // this determines how many samples this pipeline will use
                count: 1,
                // samples should be active
                mask: !0,
                // has to do with anti-aliasing
                alpha_to_coverage_enabled: false,
            },
        });

        // 'create_buffer_init' method on 'wgpu::Device' we'll have to import 'DeviceExt' extension trait
        // * we use 'bytemuck' to cast our 'VERTICES' as a '&[u8]'
        // * 'create_buffer_init()' method expects '&[u8]' and 'bytemuck::cast_slice' does that for us
        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsage::VERTEX,
            }
        );

        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsage::INDEX,
            }
        );

        let num_indices = INDICES.len() as u32;

        Self {
            surface,
            device,
            queue,
            sc_desc,
            size, 
            swap_chain,  
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            diffuse_bind_group,
            diffuse_texture,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
    }

    #[allow(unused_variables)]
    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {

    }

    fn render(&mut self) -> Result<(), wgpu::SwapChainError> {
        
        // we need to get a frame to render to; this will include a wgpu::Texture and wgpu::TextureView that will hold the actual image we're drawing to
        let frame = self.swap_chain.get_current_frame()?.output;

        // to create CommandEncoder to create the actual commands to send to the gpu
        // * most modern graphics frameworks expect commands to be stored in a command buffer before being sent to the gpu
        // * the 'encoder' builds a command buffer that we can then send to the gpu
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            // get to clearing the screen
            // * need to use the 'encoder' to create 'RenderPass' to do the actual drawing
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[
                    // this is what [[location(0)]] in the fragment shader targets
                    wgpu::RenderPassColorAttachment {
                        view: &frame.view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: true,
                        }
                    }
                ],
                depth_stencil_attachment: None,
            });

            // set the pipeline on the 'render_pass' using the one we just created
            render_pass.set_pipeline(&self.render_pipeline);

            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);

            // 'set_vertex_buffer' takes two params:
            // * the first is what buffer slot to use for this vertex buffer (you can have multiple buffers)
            // * the second is the slice of the buffer to use; it allows us to specify which portion of the buffer to use
            // * we use '..' to specify the entire buffer
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            // tell 'wgpu' to draw something with 3 vertices and 1 instance where '[[builtin(vertex_index)]]'
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        // submit will accept anything that implements IntoIter
        // std::iter::once - creates an iterator that yields an element exactly once
        self.queue.submit(iter::once(encoder.finish()));

        Ok(())
    }
}

fn main() {
    // env_logger is simple logger that can be configured via environment variables, for use with the logging facade exposed by the 'log' crate
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    use futures::executor::block_on;

    // since main can't be async, we're going to need to block
    let mut state: State = block_on(State::new(&window));

    /*
        Closures are functions that can capture the enclosing environment
        |val| val + x

        * using *move* before vertical pipes force closure to take ownership of captured variables:
        let contains = move |needle| haystack.contains(needle);

        * the underscore ( _ ) is a reserved identifier in Rust and serves different purposes depdning on the context
        usually it means that something is ignored
    */
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit
                        },
                        WindowEvent::KeyboardInput { input, .. } => match input {
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            } => *control_flow = ControlFlow::Exit,
                            _ => {}
                        },
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        },
                        WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                            state.resize(**new_inner_size);
                        },
                        _ => {}
                        // end of 'event' with [if window_id == window.id()]
                    }
                }                
            },
            Event::RedrawRequested(_) => {
                state.update();
                match state.render() {
                    Ok(_) => {},
                    // recreate the swap_chain if lost
                    Err(wgpu::SwapChainError::Lost) => state.resize(state.size),
                    // the system is out of memory, we should probably quit
                    Err(wgpu::SwapChainError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // all other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            },
            Event::MainEventsCleared => {
                // RedrawRequested will only trigger once, unless we manually
                // request it
                window.request_redraw();
            }
            _ => {}
            // end of 'event'
        }
    });
}
