extern crate glfw;
extern crate gl;

use std::ffi::{CStr, CString};
use std::ptr;
use std::str;

use glfw::{Action, Context, Key};

fn compile_shader(src: &CStr, kind: gl::types::GLenum) -> u32 {
    unsafe {
        let shader = gl::CreateShader(kind);
        gl::ShaderSource(shader, 1, &src.as_ptr(), ptr::null());
        gl::CompileShader(shader);

        // Check for compilation errors
        let mut success = gl::FALSE as gl::types::GLint;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);

        if success != gl::TRUE as i32 {
            let mut len = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);

            let error = CString::new(vec![b' '; len as usize]).unwrap();
            gl::GetShaderInfoLog(shader, len, ptr::null_mut(), error.as_ptr() as *mut _);

            panic!(
                "Shader compilation failed: {}",
                str::from_utf8(error.to_bytes()).unwrap()
            );
        }

        shader
    }
}

fn link_program(vs: u32, fs: u32) -> u32 {
    unsafe {
        let program = gl::CreateProgram();
        gl::AttachShader(program, vs);
        gl::AttachShader(program, fs);
        gl::LinkProgram(program);

        // Check for linking errors
        let mut success = gl::FALSE as gl::types::GLint;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
        if success != gl::TRUE as i32 {
            let mut len = 0;
            gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);

            let error = CString::new(vec![b' '; len as usize]).unwrap();
            gl::GetProgramInfoLog(program, len, ptr::null_mut(), error.as_ptr() as *mut _);

            panic!(
                "Program linking failed: {}",
                str::from_utf8(error.to_bytes()).unwrap()
            );
        }

        gl::DeleteShader(vs);
        gl::DeleteShader(fs);

        program
    }
}

fn main() {
    let mut glfw = glfw::init(glfw::fail_on_errors).expect("Failed to initialize GLFW");

    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));

    let (mut window, events) = glfw
        .create_window(1920, 1080, "Scop", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window");

    window.make_current();
    window.set_key_polling(true);
    window.set_scroll_polling(true);

    gl::load_with(|s| glfw.get_proc_address_raw(s));

    let vertices: [f32; 9] = [
        -0.1, -0.1, 0.0,
        0.1, -0.1, 0.0,
        0.0, 0.1, 0.0,
    ];

    let mut vbo = 0;
    let mut vao = 0;

    unsafe {
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);

        gl::BindVertexArray(vao);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (vertices.len() * std::mem::size_of::<f32>()) as isize,
            vertices.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );

        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 3 * 4, ptr::null());
        gl::EnableVertexAttribArray(0);

        gl::BindBuffer(gl::ARRAY_BUFFER, 0);
        gl::BindVertexArray(0);
    }

    let vertex_shader_src = CString::new(
        "#version 330 core\n\
         layout (location = 0) in vec3 aPos;\n\
         uniform vec3 offset;\n\
         uniform float zoom;\n\
         void main() {\n\
             gl_Position = vec4((aPos * zoom) + offset, 1.0);\n\
         }"
    ).unwrap();

    let fragment_shader_src = CString::new(
        "#version 330 core\n\
         out vec4 FragColor;\n\
         void main() {\n\
             FragColor = vec4(1.0, 1.0, 1.0, 1.0);\n\
         }", // FragColor = Color of triangle
    )
    .unwrap();

    let vertex_shader = compile_shader(&vertex_shader_src, gl::VERTEX_SHADER);
    let fragment_shader = compile_shader(&fragment_shader_src, gl::FRAGMENT_SHADER);
    let shader_program = link_program(vertex_shader, fragment_shader);

    unsafe {
        gl::ClearColor(0.0, 0.0, 0.0, 1.0);
    }

    let mut offset_x = 0.0f32;
    let mut offset_y = 0.0f32;
    let mut zoom: f32 = 1.0;

    while !window.should_close() {
        glfw.poll_events();

        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT);

            gl::UseProgram(shader_program);

            //
            let offset_location = gl::GetUniformLocation(shader_program, CString::new("offset").unwrap().as_ptr());
            let zoom_location = gl::GetUniformLocation(shader_program, CString::new("zoom").unwrap().as_ptr());
            gl::Uniform3f(offset_location, offset_x, offset_y, 0.0);
            gl::Uniform1f(zoom_location, zoom);
            // This section is for moving with arrow_keys and scrolling

            gl::BindVertexArray(vao);
            gl::DrawArrays(gl::TRIANGLES, 0, 3);
        }

        window.swap_buffers();

        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    window.set_should_close(true)
                }
                glfw::WindowEvent::Key(Key::Right, _, action, _) if action == Action::Press || action == Action::Repeat => {
                    offset_x += 0.05;
                }
                glfw::WindowEvent::Key(Key::Left, _, action, _) if action == Action::Press || action == Action::Repeat => {
                    offset_x -= 0.05;
                }
                glfw::WindowEvent::Key(Key::Up, _, action, _) if action == Action::Press || action == Action::Repeat => {
                    offset_y += 0.05;
                }
                glfw::WindowEvent::Key(Key::Down, _, action, _) if action == Action::Press || action == Action::Repeat => {
                    offset_y -= 0.05;
                }
                glfw::WindowEvent::Scroll(_, yoffset) => {
                    zoom += yoffset as f32 * 0.1;
                    if zoom < 0.1 { zoom = 0.1; }
                    if zoom > 7.5 { zoom = 7.5; }
                }
                
                _ => {}
            }
        }
    }
}
