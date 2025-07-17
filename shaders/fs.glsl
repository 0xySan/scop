#version 330 core

in vec3 coord;
out vec4 FragColor;

void main()
{
    FragColor = vec4(coord * 0.5 + 0.5, 1.0);
}