#version 150 core

in vec3 vNormal;
in float vOnsetPulse;
in vec3 vWorldPos;

// Overall energy of the band driving this instance
uniform float uBandLevel;
// Same SH coefficients used for displacement in the vertex shader
uniform float uSHCoeffs[9];

out vec4 FragColor;

void main()
{
    vec3 n = normalize(vNormal);
    vec3 lightDir = normalize(vec3(0.2, 1.0, 0.4));
    float nd = max(dot(n, lightDir), 0.0);

    // Evaluate the same low-order spherical harmonics used in the vertex shader
    float x = n.x;
    float y = n.y;
    float z = n.z;

    float Y[9];
    Y[0] = 0.282095;                      // L0,0
    Y[1] = 0.488603 * y;                  // L1,-1
    Y[2] = 0.488603 * z;                  // L1,0
    Y[3] = 0.488603 * x;                  // L1,1
    Y[4] = 1.092548 * x * y;              // L2,-2
    Y[5] = 1.092548 * y * z;              // L2,-1
    Y[6] = 0.315392 * (3.0 * z * z - 1.0);// L2,0
    Y[7] = 1.092548 * x * z;              // L2,1
    Y[8] = 0.546274 * (x * x - y * y);    // L2,2

    float sh = 0.0;
    for (int i = 0; i < 9; ++i)
    {
        sh += uSHCoeffs[i] * Y[i];
    }

    // Map SH response at this point on the sphere to [0,1]
    float shNorm = clamp(0.5 + 1.2 * sh, 0.0, 1.0);

    float band = clamp(uBandLevel * 1.5, 0.0, 1.0);
    float pulse = vOnsetPulse;

    // Three-color palette blended by SH and band energy
    vec3 lowColor  = vec3(0.10, 0.30, 0.85); // cool
    vec3 midColor  = vec3(0.25, 0.85, 0.45); // greenish
    vec3 highColor = vec3(0.95, 0.35, 0.20); // warm

    vec3 paletteA = mix(lowColor, midColor, band);
    vec3 paletteB = mix(midColor, highColor, band);
    vec3 baseColor = mix(paletteA, paletteB, shNorm);

    // Emphasize outer displaced shell and areas with strong SH response
    float radius = length(vWorldPos);
    float shellMask = smoothstep(0.7, 1.6, radius);

    float brightness = 0.35 + 0.45 * nd + 0.7 * shellMask * band;

    // Onset pulse pushes color towards white and adds extra glow
    baseColor = mix(baseColor, vec3(1.0), pulse * 0.7);

    vec3 color = baseColor * brightness * (0.8 + 0.8 * pulse);

    // Alpha also breathes slightly with band energy
    float alpha = 0.6 + 0.35 * band;
    FragColor = vec4(color, alpha);
}
