#version 150 core

in vec3 aPos; // particle position in model space

uniform mat4 uProjection;
uniform mat4 uView;
uniform mat4 uModel;
uniform float uSHCoeffs[9];
uniform float uDisplacementScale;
uniform float uOnsetPulse;
uniform float uBandLevel;
uniform float uPointSize;

out vec3 vNormal;
out float vOnsetPulse;
out vec3 vWorldPos;

void main()
{
    // Use radial direction as surface normal for particles on/inside the sphere
    vec3 normal = normalize(aPos);
    if (!all(greaterThan(abs(normal), vec3(0.0)))) {
        normal = vec3(0.0, 1.0, 0.0);
    }

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

    float bandFactor = 0.5 + 1.5 * clamp(uBandLevel, 0.0, 1.0);
    float pulseScale = 1.0 + 0.5 * uOnsetPulse;

    vec3 worldPos = (uModel * vec4(aPos, 1.0)).xyz;
    vec3 displaced = worldPos + normal * sh * uDisplacementScale * bandFactor * pulseScale;

    vNormal     = normal;
    vOnsetPulse = uOnsetPulse;
    vWorldPos   = displaced;

    gl_Position = uProjection * uView * vec4(displaced, 1.0);
    gl_PointSize = uPointSize;
}
