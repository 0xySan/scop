#version 330 core

layout (location = 0) in vec3 aPos;

uniform vec3 offset;
uniform float zoom;
uniform mat4 p;
uniform mat4 pos;
uniform mat4 rot;

out vec3 coord;

void main()
{
    mat4 m = mat4(1);
    m[3][2] = zoom;
    m[3][1] = -1;
    vec4 transformedPos = vec4(aPos, 1.0);
    coord = aPos;
    m = m * rot;
    gl_Position = p * rot * transformedPos;
}
