extern crate glfw;
extern crate gl;

use std::ffi::{CStr, CString};
use std::fs;
use std::ptr;
use std::str;
use cgmath::*;

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

fn parse_and_fill_numbers(numbers: &mut Vec<Vector3<f32>>, vectors: &mut Vec<Vector3<f32>>, filename: String) {
    let contents = fs::read_to_string(filename)
        .expect("Failed to read file");

    for line in contents.lines() {
        let split_line: Vec<&str> = line.split_whitespace().collect();

        if split_line.is_empty() || split_line[0].starts_with('#') {
            continue;
        }

        if split_line[0] == "v" && split_line.len() >= 4 {
            let x = split_line[1].parse::<f32>().unwrap_or(0.0);
            let y = split_line[2].parse::<f32>().unwrap_or(0.0);
            let z = split_line[3].parse::<f32>().unwrap_or(0.0);
            numbers.push(Vector3::new(x, y, z));
        }

        if split_line[0] == "f" && split_line.len() == 4 {
            for i in 1..=3
            {
                let parts: Vec<&str> = split_line[i].split('/').collect();
                if let Ok(index) = parts[0].parse::<usize>() {
                    if index > 0 && index <= numbers.len() {
                        vectors.push(numbers[index - 1]);
                    }
                }
            }
        }
        if split_line[0] == "f" && split_line.len() >= 5 {
            for i in 1..=3 {
                let parts: Vec<&str> = split_line[i].split('/').collect();
                if let Ok(index) = parts[0].parse::<usize>() {
                    if index > 0 && index <= numbers.len() {
                        vectors.push(numbers[index - 1]);
                    }
                }
            }
        
            for &i in &[1, 3, 4] {
                let parts: Vec<&str> = split_line[i].split('/').collect();
                if let Ok(index) = parts[0].parse::<usize>() {
                    if index > 0 && index <= numbers.len() {
                        vectors.push(numbers[index - 1]);
                    }
                }
            }
        }
    }
}


fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Matrix4<f32> {
    let f = 1.0 / (fov.to_radians() / 2.0).tan();
    let nf = 1.0 / (near - far);

    Matrix4::new(
        f / aspect, 0.0,  0.0,                          0.0,
        0.0,        f,    0.0,                          0.0,
        0.0,        0.0,  (far + near) * nf,            2.0 * far * near * nf,
        0.0,        0.0, -1.0,                          0.0,
    )
}

fn translation(pos: Vector3<f32>) -> Matrix4<f32> {

    Matrix4::new(
        1.0, 0.0, 0.0, pos.x,
        0.0, 1.0, 0.0, pos.y,
        0.0, 0.0, 1.0, pos.z,
        0.0, 0.0, 0.0, 1.0,
    )
}

fn rotation(angle: f32, vector: Vector3<f32>) -> Matrix4<f32> {
    let c = (angle).to_radians().cos();
    let s = (angle).to_radians().sin();
    vector.normalize();
    let x = vector[0];
    let y = vector[1];
    let z = vector[2];
    let rc = 1.0 - c;
    Matrix4::new(
        x * x * rc + c,     x * y * rc - z * s, x * z * rc + y * s, 0.0,
        y * x * rc + z * s, y * y * rc + c,     y * z * rc - x * s, 0.0,
        z * x * rc - y * s, z * y * rc + x * s, z * z * rc + c,     0.0,
        0.0,                0.0,                0.0,                1.0,
    )
}

fn look_at(from: Vector3<f32>, to: Vector3<f32>, upvector: Vector3<f32>) -> Matrix4<f32> {
    let forward = (from - to).normalize();
    let right = upvector.cross(forward).normalize();
    let up = forward.cross(right);

    Matrix4::new(
        right.x,   right.y,   right.z,   -right.dot(from),
        up.x,      up.y,      up.z,      -up.dot(from),
        forward.x, forward.y, forward.z, -forward.dot(from),
        0.0,       0.0,       0.0,       1.0,
    )
}

fn main()
{
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

    let mut vbo = 0;
    let mut vao = 0;
    let mut numbers = Vec::new();
    let mut vectors = Vec::new();

    if let Some(arg1) = std::env::args().nth(1) {
        parse_and_fill_numbers(&mut numbers, &mut vectors, arg1);
    }
    else {
        print!("No filename given\n");
        std::process::exit(1);
    }

    unsafe {
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);

        gl::BindVertexArray(vao);

        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(
            gl::ARRAY_BUFFER,
            (vectors.len() * std::mem::size_of::<Vector3<f32>>()) as isize,
            vectors.as_ptr() as *const _,
            gl::STATIC_DRAW,
        );

        gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 3 * 4, ptr::null());
        gl::EnableVertexAttribArray(0);

        gl::BindBuffer(gl::ARRAY_BUFFER, 0);
        gl::BindVertexArray(0);

        gl::ClearColor(0.0, 0.0, 0.0, 1.0);
    }

    let vertex_shader_path = "shaders/vs.glsl";

    let vertex_shader_string = fs::read_to_string(vertex_shader_path)
        .expect("Failed to read vertex shader file");

    let vertex_shader_src = CString::new(vertex_shader_string)
        .expect("Vertex shader source contained a null byte");

    let fragment_shader_path = "shaders/fs.glsl";

    let fragment_shader_string = fs::read_to_string(fragment_shader_path)
        .expect("Failed to read fragment shader file");
    
    let fragment_shader_src = CString::new(fragment_shader_string)
        .expect("Fragment shader source contained a null byte");

    let vertex_shader = compile_shader(&vertex_shader_src, gl::VERTEX_SHADER);
    let fragment_shader = compile_shader(&fragment_shader_src, gl::FRAGMENT_SHADER);
    let shader_program = link_program(vertex_shader, fragment_shader);

    let matrice = perspective(45.0, 1920.0 / 1080.0, 0.1, 1000.0);

    let mut position = Vector3::new(0.0, 0.0, -10.0);

    unsafe {
        gl::UseProgram(shader_program);
        let p_location = gl::GetUniformLocation(shader_program, CString::new("p").unwrap().as_ptr());
        gl::UniformMatrix4fv(p_location, 1, gl::TRUE, matrice.as_ptr());
    };

    unsafe {
        gl::Enable(gl::DEPTH_TEST);
    }

    let mut anglex = 90.0;
    let mut angley = 0.0;

    while !window.should_close() {
        glfw.poll_events();

        let matricerotx = rotation(anglex, Vector3::new(0.0, 1.0, 0.0));
        let matriceroty = rotation(angley, Vector3::new(1.0, 0.0, 0.0));
        let translation = translation(position);
        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            gl::UseProgram(shader_program);

            let m_rot = gl::GetUniformLocation(shader_program, CString::new("rot").unwrap().as_ptr());
            gl::UniformMatrix4fv(m_rot, 1, gl::TRUE, (matricerotx * matriceroty * translation).as_ptr());
            // This section is for moving with arrow_keys and scrolling

            gl::BindVertexArray(vao);

            let len = (vectors.len() * 3) as i32;

            gl::DrawArrays(gl::TRIANGLES, 0, len);
        }

        window.swap_buffers();

        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    window.set_should_close(true)
                }
                glfw::WindowEvent::Key(Key::D, _, action, _) if action == Action::Press || action == Action::Repeat => {
                    position.x += 0.05;
                }
                glfw::WindowEvent::Key(Key::A, _, action, _) if action == Action::Press || action == Action::Repeat => {
                    position.x -= 0.05;
                }
                glfw::WindowEvent::Key(Key::W, _, action, _) if action == Action::Press || action == Action::Repeat => {
                    position.y += 0.05;
                }
                glfw::WindowEvent::Key(Key::S, _, action, _) if action == Action::Press || action == Action::Repeat => {
                    position.y -= 0.05;
                }
                glfw::WindowEvent::Key(Key::Right, _, action, _) if action == Action::Press || action == Action::Repeat => {
                    anglex -= 0.5;
                }
                glfw::WindowEvent::Key(Key::Left, _, action, _) if action == Action::Press || action == Action::Repeat => {
                    anglex += 0.5;
                }
                glfw::WindowEvent::Key(Key::Up, _, action, _) if action == Action::Press || action == Action::Repeat => {
                    angley -= 0.5;
                }
                glfw::WindowEvent::Key(Key::Down, _, action, _) if action == Action::Press || action == Action::Repeat => {
                    angley += 0.5;
                }
                glfw::WindowEvent::Scroll(_, yoffset) => {
                    position.z += yoffset as f32 * 0.1;
                }
                
                _ => {}
            }
        }
    }
}
