#version 330 core

layout (location = 0) in vec3 aPos;      // position
layout (location = 1) in vec2 aTexCoord; // texture coordinates

uniform vec3 offset;
uniform float zoom;
uniform mat4 p;
uniform mat4 pos;
uniform mat4 rot;

out vec3 coord;     // for color mode
out vec2 TexCoord;  // for texture mode

void main()
{
    mat4 m = mat4(1.0);
    m[3][2] = zoom;
    m[3][1] = -1.0;

    coord = aPos;
    TexCoord = aTexCoord;

    vec4 transformedPos = vec4(aPos, 1.0);
    gl_Position = p * rot * transformedPos;
}