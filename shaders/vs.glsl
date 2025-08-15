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
    coord = aPos;
    TexCoord = aTexCoord;

    vec4 transformedPos = vec4(aPos, 1.0);
    gl_Position = p * rot * transformedPos;
}