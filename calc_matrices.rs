use cgmath::*;

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

pub fn matrix4_mult(a: &Matrix4<f32>, b: &Matrix4<f32>) -> Matrix4<f32> {
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

pub fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Matrix4<f32> {
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

pub fn rotation(angle: f32, vector: Vector3<f32>) -> Matrix4<f32> {
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

pub fn translation(pos: Vector3<f32>) -> Matrix4<f32> {
	Matrix4::new(
		1.0, 0.0, 0.0, pos.x,
		0.0, 1.0, 0.0, pos.y,
		0.0, 0.0, 1.0, pos.z,
		0.0, 0.0, 0.0, 1.0,
	)
}