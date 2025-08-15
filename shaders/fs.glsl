#version 330 core

in vec3 coord;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D ourTexture;
uniform float mixFactor; // 0.0 -> normals, 1.0 -> texture

void main()
{
    vec4 normalColor = vec4(abs(normalize(coord)), 1.0);
    vec4 texColor = texture(ourTexture, TexCoord);
    FragColor = mix(normalColor, texColor, clamp(mixFactor, 0.0, 1.0));
}
