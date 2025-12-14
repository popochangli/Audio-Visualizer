#version 150 core

in vec3 vNormal;
in float vOnsetPulse;
in vec3 vWorldPos;

uniform float uBandLevel;

out vec4 FragColor;

void main()
{
    vec3 n = normalize(vNormal);
    float nd = max(dot(n, normalize(vec3(0.2, 1.0, 0.4))), 0.0);

    // Base color cycles over time via world position
    float hue = 0.5 + 0.3 * n.y;
    vec3 baseColor = vec3(0.3 + 0.4 * hue, 0.2 + 0.3 * (1.0 - hue), 0.5 + 0.4 * hue);

    // Per-band tint and brightness
    float band = clamp(uBandLevel, 0.0, 1.0);
    float brightness = 0.7 + 0.8 * band;

    // Onset pulse makes the color brighter and more saturated
    float pulse = vOnsetPulse;
    vec3 color = baseColor * (0.4 + 0.6 * nd) * brightness * (1.0 + 1.5 * pulse);

    FragColor = vec4(color, 1.0);
}
