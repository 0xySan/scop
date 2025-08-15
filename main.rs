extern crate gl;
extern crate glfw;
extern crate image;

use cgmath::*;
use std::ffi::{CStr, CString};
use std::fs;
use std::ptr;
use std::str;
use std::f32::consts::PI;

use glfw::{Action, Context, Key};

fn get_element(m: &Matrix4<f32>, row: usize, col: usize) -> f32 {
	match col {
		0 => m.x[row],
		1 => m.y[row],
		2 => m.z[row],
		3 => m.w[row],
		_ => panic!("Column out of bounds"),
	}
}

fn set_element(m: &mut Matrix4<f32>, row: usize, col: usize, value: f32) {
	match col {
		0 => m.x[row] = value,
		1 => m.y[row] = value,
		2 => m.z[row] = value,
		3 => m.w[row] = value,
		_ => panic!("Column out of bounds"),
	}
}

fn matrix4_mult(a: &Matrix4<f32>, b: &Matrix4<f32>) -> Matrix4<f32> {
	let mut result = Matrix4::new(
	1.0, 0.0, 0.0, 0.0,
	0.0, 1.0, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.0, 0.0, 0.0, 1.0,
	);

	for i in 0..4 {
		for j in 0..4 {
			let mut sum = 0.0;
			for k in 0..4 {
				sum += get_element(a, i, k) * get_element(b, k, j);
			}
			set_element(&mut result, i, j, sum);
		}
	}

	result
}

fn compile_shader(src: &CStr, kind: gl::types::GLenum) -> u32 {
	unsafe {
		let shader = gl::CreateShader(kind);
		gl::ShaderSource(shader, 1, &src.as_ptr(), ptr::null());
		gl::CompileShader(shader);

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

fn load_texture(path: &str) -> u32 {
	let img = image::open(path).expect("Failed to load texture").flipv().to_rgba8();
	let (width, height) = img.dimensions();
	let data = img.as_raw();

	let mut texture = 0;
	unsafe {
		gl::GenTextures(1, &mut texture);
		gl::BindTexture(gl::TEXTURE_2D, texture);

		gl::PixelStorei(gl::UNPACK_ALIGNMENT, 1);

		gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
		gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);
		gl::TexParameteri(
			gl::TEXTURE_2D,
			gl::TEXTURE_MIN_FILTER,
			gl::LINEAR_MIPMAP_LINEAR as i32,
		);
		gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

		gl::TexImage2D(
			gl::TEXTURE_2D,
			0,
			gl::RGBA as i32,
			width as i32,
			height as i32,
			0,
			gl::RGBA,
			gl::UNSIGNED_BYTE,
			data.as_ptr() as *const _,
		);
		gl::GenerateMipmap(gl::TEXTURE_2D);
	}

	texture
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

fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Matrix4<f32> {
	let f = 1.0 / (fov.to_radians() / 2.0).tan();
	let nf = 1.0 / (near - far);

	Matrix4::new(
		f / aspect,
		0.0,
		0.0,
		0.0,
		0.0,
		f,
		0.0,
		0.0,
		0.0,
		0.0,
		(far + near) * nf,
		2.0 * far * near * nf,
		0.0,
		0.0,
		-1.0,
		0.0,
	)
}

fn rotation(angle: f32, vector: Vector3<f32>) -> Matrix4<f32> {
	let c = (angle).to_radians().cos();
	let s = (angle).to_radians().sin();
	let v = vector.normalize();
	let x = v.x;
	let y = v.y;
	let z = v.z;
	let rc = 1.0 - c;
	Matrix4::new(
		x * x * rc + c,
		x * y * rc - z * s,
		x * z * rc + y * s,
		0.0,
		y * x * rc + z * s,
		y * y * rc + c,
		y * z * rc - x * s,
		0.0,
		z * x * rc - y * s,
		z * y * rc + x * s,
		z * z * rc + c,
		0.0,
		0.0,
		0.0,
		0.0,
		1.0,
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

fn main() {
	let filename = std::env::args().nth(1).expect("No filename given");
	let texture_filename = std::env::args().nth(2).expect("No filename given");
	
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

	let projection = perspective(45.0, 1920.0 / 1080.0, 0.1, 1000.0);
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
					window.set_should_close(true);
					drop(glfw);
					unsafe {
						gl::DeleteVertexArrays(1, &vao);
						gl::DeleteBuffers(1, &vbo);
						gl::DeleteProgram(shader_program);
						gl::DeleteTextures(1, &texture_id);
						gl::DeleteShader(vertex_shader);
						gl::DeleteShader(fragment_shader);
					}
					return;
				}
				glfw::WindowEvent::Key(Key::D, _, Action::Press, _)
				| glfw::WindowEvent::Key(Key::D, _, Action::Repeat, _) => position.x += 0.05,
				glfw::WindowEvent::Key(Key::A, _, Action::Press, _)
				| glfw::WindowEvent::Key(Key::A, _, Action::Repeat, _) => position.x -= 0.05,
				glfw::WindowEvent::Key(Key::W, _, Action::Press, _)
				| glfw::WindowEvent::Key(Key::W, _, Action::Repeat, _) => position.y += 0.05,
				glfw::WindowEvent::Key(Key::S, _, Action::Press, _)
				| glfw::WindowEvent::Key(Key::S, _, Action::Repeat, _) => position.y -= 0.05,
				glfw::WindowEvent::Key(Key::Right, _, Action::Press, _)
				| glfw::WindowEvent::Key(Key::Right, _, Action::Repeat, _) => anglex -= 0.5,
				glfw::WindowEvent::Key(Key::Left, _, Action::Press, _)
				| glfw::WindowEvent::Key(Key::Left, _, Action::Repeat, _) => anglex += 0.5,
				glfw::WindowEvent::Key(Key::Up, _, Action::Press, _)
				| glfw::WindowEvent::Key(Key::Up, _, Action::Repeat, _) => angley -= 0.5,
				glfw::WindowEvent::Key(Key::Down, _, Action::Press, _)
				| glfw::WindowEvent::Key(Key::Down, _, Action::Repeat, _) => angley += 0.5,
				glfw::WindowEvent::Scroll(_, yoffset) => position.z += yoffset as f32 * 0.1,
				glfw::WindowEvent::Key(Key::T, _, Action::Press, _) => {
					texture_enabled = !texture_enabled;
					transitioning = true;
					transition_start = std::time::Instant::now();
				}
				_ => {}
			}
		}
	}
}
