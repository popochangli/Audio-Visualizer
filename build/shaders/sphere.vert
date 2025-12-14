#version 150 core

in vec3 aPos;
in vec3 aNormal;

uniform mat4 uProjection;
uniform mat4 uView;
uniform mat4 uModel;
uniform float uSHCoeffs[9];
uniform float uDisplacementScale;
uniform float uOnsetPulse;
uniform float uBandLevel;

out vec3 vNormal;
out float vOnsetPulse;
out vec3 vWorldPos;

vec3 applySH(vec3 worldPos, vec3 normal)
{
    float x = normal.x;
    float y = normal.y;
    float z = normal.z;

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
    for (int i = 0; i < 9; ++i) {
        sh += uSHCoeffs[i] * Y[i];
    }

    // Per-sphere band level scales deformation strength
    float bandFactor = 0.5 + 1.5 * clamp(uBandLevel, 0.0, 1.0);
    // Add a bit of onset pulse to exaggerate movement
    float pulseScale = 1.0 + 0.5 * uOnsetPulse;
    vec3 displaced = worldPos + normal * sh * uDisplacementScale * bandFactor * pulseScale;
    return displaced;
}

void main()
{
    vec3 worldNormal = normalize((uModel * vec4(aNormal, 0.0)).xyz);
    vec3 worldPos    = (uModel * vec4(aPos, 1.0)).xyz;
    vec3 displaced   = applySH(worldPos, worldNormal);

    vNormal     = worldNormal;
    vOnsetPulse = uOnsetPulse;
    vWorldPos   = displaced;

    gl_Position = uProjection * uView * vec4(displaced, 1.0);
}
