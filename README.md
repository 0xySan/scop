# Scop - 3D Model Viewer

A modern OpenGL-based 3D model viewer written in Rust that supports `.obj` file rendering with texture mapping, lighting, and interactive controls.

## Features

- **OBJ File Parsing**: Load and render 3D models from Wavefront OBJ files
- **Texture Mapping**: Apply textures from image files (PNG, JPG, JPEG)
- **Automatic UV Mapping**: Spherical UV mapping for models without texture coordinates
- **Dual Rendering Modes**:
  - Smooth grayscale shading with directional lighting
  - Rainbow color mode using vertex normals
  - Texture-mapped rendering
- **Interactive Controls**: 
  - Camera positioning and rotation
  - Mouse-based rotation and panning
  - Smooth transitions between rendering modes
  - Texture cycling through multiple textures
- **Smooth Transitions**: Animated interpolation between color modes and texture states

## Prerequisites

- Rust (2024 edition)
- OpenGL 3.3+ compatible graphics driver
- GLFW library

## Dependencies

- `cgmath` - 3D mathematics library
- `glfw` - OpenGL window management
- `gl` - OpenGL bindings
- `image` - Image loading and processing
- `nalgebra` - Linear algebra library

## Building

Using the provided Makefile:

```bash
# Build release version
make build

# Build and run (debug)
cargo run --release -- <path/to/model.obj>

# Clean build artifacts
make clean

# Full clean (removes binary)
make fclean

# Rebuild from scratch
make re
```

Or using Cargo directly:

```bash
cargo build --release
```

## Usage

```bash
./scop <path/to/model.obj>
```

Example:
```bash
./scop resources/teapot.obj
```

The program will automatically load all textures from the `textures/` directory.

## Controls

### Keyboard

| Key | Action |
|-----|--------|
| `Space` | Toggle auto-rotation on/off |
| `T` | Toggle between color mode and texture mode |
| `R` | Toggle between grayscale and rainbow color modes |
| `N` | Switch to next texture |
| `M` | Switch to previous texture |
| `D` | Move camera right |
| `A` | Move camera left |
| `W` | Move camera up |
| `S` | Move camera down |
| `→` | Rotate model right |
| `←` | Rotate model left |
| `↑` | Rotate model up |
| `↓` | Rotate model down |
| `Esc` | Close application |

### Mouse

- **Left Click + Drag**: Rotate model
- **Right Click + Drag**: Pan camera
- **Scroll Wheel**: Zoom in/out

## Project Structure

```
scop/
├── main.rs                  # Main application and OBJ parser
├── calc_matrices.rs         # Matrix operations and transformations
├── shader_n_textures.rs     # Shader compilation and texture loading
├── Cargo.toml               # Project dependencies
├── Makefile                 # Build automation
├── shaders/
│   ├── vs.glsl             # Vertex shader
│   └── fs.glsl             # Fragment shader
├── resources/              # 3D model files (.obj)
└── textures/               # Texture images (.png, .jpg)
```

## Shaders

### Vertex Shader (`vs.glsl`)
- Transforms vertex positions using model, view, and projection matrices
- Passes texture coordinates and vertex positions to fragment shader

### Fragment Shader (`fs.glsl`)
- Implements three rendering modes:
  - Grayscale shading with directional lighting
  - Rainbow coloring based on normals
  - Texture mapping
- Smooth interpolation between modes using mix factors

## Technical Details

### OBJ Parsing
- Supports vertex positions (`v`), texture coordinates (`vt`), and faces (`f`)
- Handles negative indices and automatic triangulation
- Fallback spherical UV mapping for models without texture coordinates

### Rendering Pipeline
1. Parse OBJ file into vertex buffer
2. Load and bind textures
3. Compile and link GLSL shaders
4. Set up vertex array object (VAO) with position and UV attributes
5. Render loop with dynamic uniform updates

### Coordinate System
- Uses right-handed coordinate system
- Perspective projection with 45° field of view
- Z-axis points toward the viewer

## Example Models

The `resources/` directory includes several sample models:
- `teapot.obj` - Classic Utah teapot
- `42.obj` - Custom model
- `barrel.obj` - Barrel mesh
- `crowbar.obj` - Crowbar model
- And more...

## Example Textures

The `textures/` directory includes various textures to apply to models.