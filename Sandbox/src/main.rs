
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

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
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {

    }

    fn input(&mut self, event: &WindowEvent) -> bool {

    }

    fn update(&mut self) {

    }

    fn render(&mut self) -> Result<(), wgpu::SwapChainError> {

    }
}

fn main() {
    // env_logger is simple logger that can be configured via environment variables, for use with the logging facade exposed by the 'log' crate
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

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
                match event {
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit
                    },
                    _ => {}
                    // end of 'event' with [if window_id == window.id()]
                }
            },
            _ => {}
            // end of 'event'
        }
    });
}
