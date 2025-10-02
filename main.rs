extern crate gl;
extern crate glfw;
extern crate image;

use cgmath::*;
use std::ffi::CString;
use std::fs;
use std::ptr;
use std::panic;
use std::str;
use std::f32::consts::PI;

use glfw::*;

mod calc_matrices;
use calc_matrices::*;
mod shader_n_textures;
use shader_n_textures::*;

fn link_program(vs: u32, fs: u32) -> u32 {
	unsafe {
		let program = gl::CreateProgram();
		gl::AttachShader(program, vs);
		gl::AttachShader(program, fs);
		gl::LinkProgram(program);

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

fn parse_obj(filename: &str) -> Vec<f32> {
	let contents = fs::read_to_string(filename).expect("Failed to read file");

	let mut positions: Vec<Vector3<f32>> = Vec::new();
	let mut texcoords: Vec<Vector2<f32>> = Vec::new();
	let mut vertices: Vec<f32> = Vec::new();
	let mut faces: Vec<Vec<(usize, Option<usize>)>> = Vec::new();

	// Fallback spherical UV mapping for when vt data is missing.
	// u in [0,1] from azimuth, v in [0,1] from elevation.
	fn spherical_uv(p: Vector3<f32>) -> Vector2<f32> {
		let r = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
		if r == 0.0 {
			return Vector2::new(0.5, 0.5);
		}
		let u = 0.5 + p.z.atan2(p.x) / (2.0 * PI);
		let v = 0.5 - (p.y / r).asin() / PI;
		Vector2::new(u.fract(), v.clamp(0.0, 1.0))
	}

	for line in contents.lines() {
		let split: Vec<&str> = line.split_whitespace().collect();
		if split.is_empty() || split[0].starts_with('#') {
			continue;
		}

		match split[0] {
			"v" if split.len() >= 4 => {
				let x = split[1].parse::<f32>().unwrap_or(0.0);
				let y = split[2].parse::<f32>().unwrap_or(0.0);
				let z = split[3].parse::<f32>().unwrap_or(0.0);
				positions.push(Vector3::new(x, y, z));
			}
			"vt" if split.len() >= 3 => {
				let u = split[1].parse::<f32>().unwrap_or(0.0);
				let v = split[2].parse::<f32>().unwrap_or(0.0);
				texcoords.push(Vector2::new(u, v));
			}
			"f" if split.len() >= 4 => {
				let mut face_indices: Vec<(usize, Option<usize>)> = Vec::new();
				for i in 1..split.len() {
					let parts: Vec<&str> = split[i].split('/').collect();
					let pos_idx = parts
						.get(0)
						.and_then(|s| s.parse::<isize>().ok())
						.map(|idx| if idx < 0 { (positions.len() as isize + idx + 1) as usize } else { idx as usize })
						.unwrap_or(0);
					let tex_idx_opt = parts
						.get(1)
						.and_then(|s| if s.is_empty() { None } else { s.parse::<isize>().ok() })
						.map(|idx| if idx < 0 { (texcoords.len() as isize + idx + 1) as usize } else { idx as usize });
					face_indices.push((pos_idx, tex_idx_opt));
				}
				faces.push(face_indices);
			}
			_ => {}
		}
	}

	// Build vertex buffer with triangulation and UV fallback
	for face_indices in faces {
		if face_indices.len() < 3 { continue; }
		for i in 1..face_indices.len() - 1 {
			let tri = [face_indices[0], face_indices[i], face_indices[i + 1]];
			for &(pi, ti_opt) in &tri {
				if pi == 0 || pi > positions.len() { continue; }
				let pos = positions[pi - 1];
				let tex = match ti_opt {
					Some(ti) if ti > 0 && ti <= texcoords.len() => texcoords[ti - 1],
					_ => spherical_uv(pos),
				};
				vertices.extend_from_slice(&[pos.x, pos.y, pos.z, tex.x, tex.y]);
			}
		}
	}

	vertices
}

fn main() {
	panic::set_hook(Box::new(|info| {
        eprintln!("Application panicked: {}", info);
        unsafe {
            glfw::ffi::glfwTerminate();
        }
        std::process::exit(1);
    }));
	let filename = std::env::args().nth(1).expect("No filename given");
	if !filename.ends_with(".obj") {
		panic!("File must be a .obj");
	}
	let texture_filename = std::env::args().nth(2).expect("No filename given");
	if !texture_filename.ends_with(".png") && !texture_filename.ends_with(".jpg") && !texture_filename.ends_with(".jpeg") {
		panic!("Texture file must be a .png, .jpg or .jpeg");
	}
	
	let mut glfw = glfw::init(glfw::fail_on_errors).expect("Failed to initialize GLFW");
	glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
	glfw.window_hint(glfw::WindowHint::OpenGlProfile(
		glfw::OpenGlProfileHint::Core,
	));

	let (mut window, events) = glfw
		.create_window(1920, 1080, "Scop", glfw::WindowMode::Windowed)
		.expect("Failed to create GLFW window");

	window.make_current();
	window.set_key_polling(true);
	window.set_scroll_polling(true);
	window.set_mouse_button_polling(true);
	window.set_cursor_pos_polling(true);

	let mut dragging = false;
	let mut last_cursor = (0.0, 0.0);

	let mut rot_dragging = false;
	let mut rot_last_cursor = (0.0, 0.0);

	gl::load_with(|s| glfw.get_proc_address_raw(s).map_or(ptr::null(), |f| f as *const _));

	let vertices = parse_obj(&filename);

	let mut vao = 0;
	let mut vbo = 0;

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

		// positions
		gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, (5 * 4) as i32, ptr::null());
		gl::EnableVertexAttribArray(0);
		// texcoords
		gl::VertexAttribPointer(
			1,
			2,
			gl::FLOAT,
			gl::FALSE,
			(5 * 4) as i32,
			(3 * 4) as *const _,
		);
		gl::EnableVertexAttribArray(1);
	}

	let vertex_shader_src = CString::new(fs::read_to_string("shaders/vs.glsl").unwrap()).unwrap();
	let fragment_shader_src = CString::new(fs::read_to_string("shaders/fs.glsl").unwrap()).unwrap();
	let vertex_shader = compile_shader(vertex_shader_src.as_c_str(), gl::VERTEX_SHADER);
	let fragment_shader = compile_shader(fragment_shader_src.as_c_str(), gl::FRAGMENT_SHADER);
	let shader_program = link_program(vertex_shader, fragment_shader);

	let texture_id = load_texture(&texture_filename);

	let projection = calc_matrices::perspective(45.0, 1920.0 / 1080.0, 0.1, 1000.0);
	let mut position = Vector3::new(0.0, 0.0, -10.0);
	let mut anglex = 90.0;
	let mut angley = 0.0;
	let mut texture_enabled = false;
	// Transition state for smooth blend between normal and texture
	let mut transitioning = false;
	let mut transition_start = std::time::Instant::now();
	let transition_duration = std::time::Duration::from_millis(300);

	// Cache uniform location for mixFactor
	let mix_loc: i32;

	unsafe {
		gl::UseProgram(shader_program);
		let p_location =
			gl::GetUniformLocation(shader_program, CString::new("p").unwrap().as_ptr());
		gl::UniformMatrix4fv(p_location, 1, gl::TRUE, projection.as_ptr());
		// Bind the sampler to texture unit 0 and make sure unit 0 is active
		let sampler_loc =
			gl::GetUniformLocation(shader_program, CString::new("ourTexture").unwrap().as_ptr());
		if sampler_loc != -1 {
			gl::Uniform1i(sampler_loc, 0);
		}
		// Locate mixFactor and set initial value to 0 (show normals)
		mix_loc = gl::GetUniformLocation(shader_program, CString::new("mixFactor").unwrap().as_ptr());
		if mix_loc != -1 { gl::Uniform1f(mix_loc, 0.0); }
		gl::ActiveTexture(gl::TEXTURE0);
		gl::BindTexture(gl::TEXTURE_2D, texture_id);
		gl::Enable(gl::DEPTH_TEST);
	}

	while !window.should_close() {
		glfw.poll_events();

		let matricerotx = rotation(anglex, Vector3::new(0.0, 1.0, 0.0));
		let matriceroty = rotation(angley, Vector3::new(1.0, 0.0, 0.0));
		let translation = translation(position);

		if window.get_key(Key::D) == Action::Press && !dragging {
		position.x += 0.05;
		}
		if window.get_key(Key::A) == Action::Press && !dragging {
			position.x -= 0.05;
		}
		if window.get_key(Key::W) == Action::Press && !dragging {
			position.y += 0.05;
		}
		if window.get_key(Key::S) == Action::Press && !dragging {
			position.y -= 0.05;
		}
		if window.get_key(Key::Right) == Action::Press && !rot_dragging {
			anglex -= 0.5;
		}
		if window.get_key(Key::Left) == Action::Press && !rot_dragging {
			anglex += 0.5;
		}
		if window.get_key(Key::Up) == Action::Press && !rot_dragging {
			angley -= 0.5;
		}
		if window.get_key(Key::Down) == Action::Press && !rot_dragging {
			angley += 0.5;
		}
		if window.get_key(Key::T) == Action::Press && !transitioning {
			texture_enabled = !texture_enabled;
			transitioning = true;
			transition_start = std::time::Instant::now();
		}

		unsafe {
			gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
			gl::UseProgram(shader_program);

			let m_rot =
				gl::GetUniformLocation(shader_program, CString::new("rot").unwrap().as_ptr());

			let temp = matrix4_mult(&matricerotx, &matriceroty);
			let result = matrix4_mult(&temp, &translation);

			gl::UniformMatrix4fv(
				m_rot,
				1,
				gl::TRUE,
				result.as_ptr() as *const f32,
			);

			// Compute mix factor based on transition state
			let target = if texture_enabled { 1.0f32 } else { 0.0f32 };
			let t = if transitioning {
				let elapsed = transition_start.elapsed();
				(elapsed.as_secs_f32() / transition_duration.as_secs_f32()).min(1.0)
			} else { 1.0 };
			let current = if target > 0.5 { t } else { 1.0 - t };
			if t >= 1.0 { transitioning = false; }
			if mix_loc != -1 { gl::Uniform1f(mix_loc, current); }

			gl::BindVertexArray(vao);
			gl::DrawArrays(gl::TRIANGLES, 0, (vertices.len() / 5) as i32);
		}

		window.swap_buffers();

		for (_, event) in glfw::flush_messages(&events) {
			match event {
				glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
					println!("Escape pressed, exiting.");
					window.set_should_close(true);
				}
				glfw::WindowEvent::Scroll(_, yoffset) => position.z += yoffset as f32 * 0.1,

				glfw::WindowEvent::MouseButton(glfw::MouseButtonLeft, Action::Press, _) => {
					dragging = true;
					last_cursor = window.get_cursor_pos();
				}
				glfw::WindowEvent::MouseButton(glfw::MouseButtonLeft, Action::Release, _) => dragging = false,
				glfw::WindowEvent::CursorPos(x, y) if dragging => {
					let dx = x - last_cursor.0;
					let dy = y - last_cursor.1;

					position.x += dx as f32 * 0.01;
					position.y -= dy as f32 * 0.01;

					last_cursor = (x, y);
				}

				glfw::WindowEvent::MouseButton(glfw::MouseButtonRight, Action::Press, _) => {
					rot_dragging = true;
					rot_last_cursor = window.get_cursor_pos();
				}
				glfw::WindowEvent::MouseButton(glfw::MouseButtonRight, Action::Release, _) => rot_dragging = false,
				glfw::WindowEvent::CursorPos(x, y) if rot_dragging => {
					let dx = x - rot_last_cursor.0;
					let dy = y - rot_last_cursor.1;

					anglex += dx as f32 * 0.5;
					angley += dy as f32 * 0.5;

					rot_last_cursor = (x, y);
				}
				
				_ => {}
			}
		}
	}
	drop(window);
	unsafe {
		gl::DeleteVertexArrays(1, &vao);
		gl::DeleteBuffers(1, &vbo);
		gl::DeleteProgram(shader_program);
		gl::DeleteTextures(1, &texture_id);
		gl::DeleteShader(vertex_shader);
		gl::DeleteShader(fragment_shader);
		glfw::ffi::glfwTerminate();
	}
}
