#version 330 core

in vec3 coord;
in vec2 TexCoord;

out vec4 FragColor;

uniform sampler2D ourTexture;
uniform float mixFactor;      // 0.0 -> normals, 1.0 -> texture
uniform float colorMixFactor; // 0.0 -> grayscale, 1.0 -> rainbow

void main()
{
    vec3 n = normalize(coord);

    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.3));
    float diff = max(dot(n, lightDir), 0.0);
    float intensity = 0.2 + 0.8 * diff;
    vec4 grayColor = vec4(vec3(intensity), 1.0);

    vec4 rainbowColor = vec4(abs(n), 1.0);

    vec4 normalColor = mix(grayColor, rainbowColor, clamp(colorMixFactor, 0.0, 1.0));

    vec4 texColor = texture(ourTexture, TexCoord);
    FragColor = mix(normalColor, texColor, clamp(mixFactor, 0.0, 1.0));
}
