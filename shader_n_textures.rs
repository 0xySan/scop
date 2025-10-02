use std::ffi::{CStr, CString};
use std::ptr;
use std::str;

pub fn load_texture(path: &str) -> u32 {
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

pub fn compile_shader(src: &CStr, kind: gl::types::GLenum) -> u32 {
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