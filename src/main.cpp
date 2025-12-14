// OpenGL + GLFW + simple DSP-based audio analysis (WAV + FFT + onset)

#include <GLFW/glfw3.h>
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#else
#include <GL/gl.h>
#endif

// Audio playback (miniaudio). Download official header to external/miniaudio/miniaudio.h
// from https://miniaud.io/ before building.
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <algorithm>

// Simple math types for 3D transforms -------------------------------------------------

struct Vec3
{
    float x, y, z;
};

struct Mat4
{
    float m[16];
};

static Mat4 mat4_identity()
{
    Mat4 r{};
    for (int i = 0; i < 16; ++i)
        r.m[i] = 0.0f;
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

static Mat4 mat4_mul(const Mat4 &a, const Mat4 &b)
{
    Mat4 r{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            r.m[col + row * 4] =
                a.m[0 + row * 4] * b.m[col + 0 * 4] +
                a.m[1 + row * 4] * b.m[col + 1 * 4] +
                a.m[2 + row * 4] * b.m[col + 2 * 4] +
                a.m[3 + row * 4] * b.m[col + 3 * 4];
        }
    }
    return r;
}

static Mat4 mat4_translate(float x, float y, float z)
{
    Mat4 r = mat4_identity();
    r.m[12] = x;
    r.m[13] = y;
    r.m[14] = z;
    return r;
}

static Mat4 mat4_scale(float s)
{
    Mat4 r{};
    r.m[0] = r.m[5] = r.m[10] = s;
    r.m[15] = 1.0f;
    return r;
}

static Mat4 mat4_rotate_y(float angle)
{
    Mat4 r = mat4_identity();
    float c = std::cos(angle);
    float s = std::sin(angle);
    r.m[0] = c;
    r.m[2] = s;
    r.m[8] = -s;
    r.m[10] = c;
    return r;
}

static Mat4 mat4_perspective(float fovRadians, float aspect, float znear, float zfar)
{
    float f = 1.0f / std::tan(fovRadians * 0.5f);
    Mat4 r{};
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (zfar + znear) / (znear - zfar);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * zfar * znear) / (znear - zfar);
    return r;
}

static Mat4 mat4_look_at(const Vec3 &eye, const Vec3 &center, const Vec3 &up)
{
    auto normalize = [](Vec3 v)
    {
        float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        return Vec3{v.x / len, v.y / len, v.z / len};
    };
    auto subtract = [](Vec3 a, Vec3 b)
    {
        return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
    };
    auto cross = [](Vec3 a, Vec3 b)
    {
        return Vec3{a.y * b.z - a.z * b.y,
                    a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x};
    };
    Vec3 f = normalize(subtract(center, eye));
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);

    Mat4 r = mat4_identity();
    r.m[0] = s.x;
    r.m[4] = s.y;
    r.m[8] = s.z;
    r.m[1] = u.x;
    r.m[5] = u.y;
    r.m[9] = u.z;
    r.m[2] = -f.x;
    r.m[6] = -f.y;
    r.m[10] = -f.z;
    r.m[12] = -(s.x * eye.x + s.y * eye.y + s.z * eye.z);
    r.m[13] = -(u.x * eye.x + u.y * eye.y + u.z * eye.z);
    r.m[14] = f.x * eye.x + f.y * eye.y + f.z * eye.z;
    return r;
}

// Simple vec3 helpers for camera movement ---------------------------------------------

static Vec3 vec3_add(const Vec3 &a, const Vec3 &b)
{
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

static Vec3 vec3_sub(const Vec3 &a, const Vec3 &b)
{
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

static Vec3 vec3_scale(const Vec3 &v, float s)
{
    return Vec3{v.x * s, v.y * s, v.z * s};
}

static Vec3 vec3_normalize(const Vec3 &v)
{
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len <= 0.00001f)
        return Vec3{0.0f, 0.0f, 0.0f};
    return Vec3{v.x / len, v.y / len, v.z / len};
}

static Vec3 vec3_cross(const Vec3 &a, const Vec3 &b)
{
    return Vec3{a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x};
}

// Particle system ------------------------------------------------------------------------

struct Particle
{
    Vec3 pos;
    Vec3 vel;
    Vec3 basePos;
    float lifetime = 0.0f;
    float maxLifetime = 0.0f;
    float beatEffect = 0.0f;
};

// Simple hash-based value noise for particle motion
static float hash31(float x, float y, float z)
{
    float f = std::sin(x * 12.9898f + y * 78.233f + z * 37.719f) * 43758.5453f;
    return f - std::floor(f);
}

static float noise3D(float x, float y, float z)
{
    return hash31(x, y, z) * 2.0f - 1.0f; // range [-1, 1]
}

// Shader compilation helpers -------------------------------------------------------------

static std::string loadTextFile(const std::string &path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << path << std::endl;
        return "";
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return content;
}

static GLuint compileShader(GLenum type, const std::string &source)
{
    GLuint shader = glCreateShader(type);
    const char *src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status)
    {
        GLint logLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
        std::string log(logLen, '\0');
        glGetShaderInfoLog(shader, logLen, nullptr, log.data());
        std::cerr << "Shader compile error: " << log << std::endl;
    }
    return shader;
}

static GLuint createProgram(const std::string &vsPath, const std::string &fsPath)
{
    std::string vsSource = loadTextFile(vsPath);
    std::string fsSource = loadTextFile(fsPath);
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSource);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSource);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (!status)
    {
        GLint logLen = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLen);
        std::string log(logLen, '\0');
        glGetProgramInfoLog(program, logLen, nullptr, log.data());
        std::cerr << "Program link error: " << log << std::endl;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

// Simple WAV loader (16-bit PCM, mono or stereo) ---------------------------------------

struct AudioData
{
    std::vector<float> samples; // mono, -1..1
    int sampleRate = 44100;
};

static bool loadWav(const std::string &path, AudioData &out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
    {
        std::cerr << "Failed to open audio file: " << path << std::endl;
        return false;
    }

    auto readU32 = [&]()
    {
        uint32_t v = 0;
        f.read(reinterpret_cast<char *>(&v), 4);
        return v;
    };
    auto readU16 = [&]()
    {
        uint16_t v = 0;
        f.read(reinterpret_cast<char *>(&v), 2);
        return v;
    };

    char riff[4];
    f.read(riff, 4);
    uint32_t chunkSize = readU32();
    (void)chunkSize;
    char wave[4];
    f.read(wave, 4);
    if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE")
    {
        std::cerr << "Not a RIFF/WAVE file" << std::endl;
        return false;
    }

    uint16_t audioFormat = 0;
    uint16_t numChannels = 0;
    uint32_t sampleRate = 0;
    uint16_t bitsPerSample = 0;
    uint32_t dataSize = 0;

    while (f && (!dataSize))
    {
        char id[4];
        f.read(id, 4);
        uint32_t size = readU32();
        std::string sid(id, 4);
        if (sid == "fmt ")
        {
            audioFormat = readU16();
            numChannels = readU16();
            sampleRate = readU32();
            uint32_t byteRate = readU32();
            (void)byteRate;
            uint16_t blockAlign = readU16();
            (void)blockAlign;
            bitsPerSample = readU16();
            if (size > 16)
            {
                f.seekg(size - 16, std::ios::cur);
            }
        }
        else if (sid == "data")
        {
            dataSize = size;
            break;
        }
        else
        {
            f.seekg(size, std::ios::cur);
        }
    }

    if (!dataSize || audioFormat != 1 || (bitsPerSample != 16))
    {
        std::cerr << "Unsupported WAV format (need 16-bit PCM)" << std::endl;
        return false;
    }

    std::vector<int16_t> raw(dataSize / 2);
    f.read(reinterpret_cast<char *>(raw.data()), dataSize);

    out.sampleRate = static_cast<int>(sampleRate);
    out.samples.resize(raw.size() / numChannels);

    for (size_t i = 0; i < out.samples.size(); ++i)
    {
        int32_t sum = 0;
        for (uint16_t ch = 0; ch < numChannels; ++ch)
        {
            sum += raw[i * numChannels + ch];
        }
        float v = static_cast<float>(sum) / (32768.0f * numChannels);
        out.samples[i] = v;
    }
    return true;
}

// FFT and spectral analysis -------------------------------------------------------------

using Complex = std::complex<float>;

static void fft(std::vector<Complex> &a)
{
    const size_t n = a.size();
    if (n <= 1)
        return;

    // bit-reversal permutation
    size_t j = 0;
    for (size_t i = 1; i < n; ++i)
    {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
        {
            j ^= bit;
        }
        j ^= bit;
        if (i < j)
            std::swap(a[i], a[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1)
    {
        float ang = -2.0f * static_cast<float>(M_PI) / static_cast<float>(len);
        Complex wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len)
        {
            Complex w(1.0f, 0.0f);
            for (size_t j2 = 0; j2 < len / 2; ++j2)
            {
                Complex u = a[i + j2];
                Complex v = a[i + j2 + len / 2] * w;
                a[i + j2] = u + v;
                a[i + j2 + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

struct AnalysisFrame
{
    std::vector<float> bandEnergy; // 9 bands
    float flux = 0.0f;             // full-band spectral flux
    float fluxLow = 0.0f;          // low-frequency flux (kick)
    float fluxHigh = 0.0f;         // mid-high flux (snare/hi-hat)
    float time = 0.0f;
};

struct AnalysisResult
{
    std::vector<AnalysisFrame> frames;
    std::vector<float> onsetTimesKick;
    std::vector<float> onsetTimesSnare;
};

// Multiple spheres, each reacting to different frequency bands
// For this version we use a single sphere (1x1 grid)
constexpr int kGridX = 1;
constexpr int kGridZ = 1;
constexpr int kNumSpheres = kGridX * kGridZ; // 1 sphere
std::array<float, kNumSpheres> bandLevels{};
std::array<Vec3, kNumSpheres> sphereOffsets{};

// Map each sphere (by index) to a range of 9 bands
std::array<int, kNumSpheres> sphereBandStart{};
std::array<int, kNumSpheres> sphereBandEnd{};

static void initSphereLayout()
{
    // Arrange spheres in a grid on X-Z plane, centered around origin.
    // With kGridX = kGridZ = 1 this is just a single sphere at the origin.
    float spacing = 6.0f; // distance between spheres
    int index = 0;
    for (int z = 0; z < kGridZ; ++z)
    {
        for (int x = 0; x < kGridX; ++x)
        {
            float fx = (static_cast<float>(x) - (kGridX - 1) * 0.5f) * spacing;
            float fz = (static_cast<float>(z) - (kGridZ - 1) * 0.5f) * spacing;
            sphereOffsets[index] = Vec3{fx, 0.0f, fz};

            // Single sphere: react to the full range of 9 bands
            int start = 0;
            int end = 8;
            sphereBandStart[index] = start;
            sphereBandEnd[index] = end;

            ++index;
        }
    }
}

static AnalysisResult analyzeAudio(const AudioData &audio)
{
    constexpr int frameSize = 1024;
    constexpr int hopSize = 512;
    constexpr int numBands = 9;

    AnalysisResult result;
    const int totalSamples = static_cast<int>(audio.samples.size());
    int numFrames = (totalSamples - frameSize) / hopSize;
    if (numFrames <= 0)
        return result;

    result.frames.resize(numFrames);

    std::vector<float> prevMag(frameSize / 2, 0.0f);
    std::vector<float> prevMagLow(frameSize / 2, 0.0f);
    std::vector<float> prevMagHigh(frameSize / 2, 0.0f);

    for (int fidx = 0; fidx < numFrames; ++fidx)
    {
        int offset = fidx * hopSize;
        std::vector<Complex> buf(frameSize);
        for (int i = 0; i < frameSize; ++i)
        {
            float w = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / (frameSize - 1)));
            float s = audio.samples[offset + i];
            buf[i] = Complex(s * w, 0.0f);
        }
        fft(buf);

        std::vector<float> mag(frameSize / 2);
        for (int k = 0; k < frameSize / 2; ++k)
        {
            mag[k] = std::abs(buf[k]);
        }

        AnalysisFrame frame;
        frame.bandEnergy.assign(numBands, 0.0f);
        int binsPerBand = (frameSize / 2) / numBands;
        for (int b = 0; b < numBands; ++b)
        {
            int start = b * binsPerBand;
            int end = (b == numBands - 1) ? (frameSize / 2) : (start + binsPerBand);
            float sum = 0.0f;
            for (int k = start; k < end; ++k)
                sum += mag[k];
            frame.bandEnergy[b] = sum / static_cast<float>(end - start);
        }

        float flux = 0.0f;
        float fluxLow = 0.0f;
        float fluxHigh = 0.0f;
        if (fidx > 0)
        {
            // approximate frequency per bin
            float binFreq = static_cast<float>(audio.sampleRate) / static_cast<float>(frameSize);
            for (size_t k = 0; k < mag.size(); ++k)
            {
                float diffFull = mag[k] - prevMag[k];
                if (diffFull > 0.0f)
                    flux += diffFull;

                float freq = static_cast<float>(k) * binFreq;
                // Kick region ~ 40-200 Hz
                if (freq >= 40.0f && freq <= 200.0f)
                {
                    float diff = mag[k] - prevMagLow[k];
                    if (diff > 0.0f)
                        fluxLow += diff;
                }
                // Snare / hi-hat region ~ 1-6 kHz
                if (freq >= 1000.0f && freq <= 6000.0f)
                {
                    float diff = mag[k] - prevMagHigh[k];
                    if (diff > 0.0f)
                        fluxHigh += diff;
                }
            }
        }
        frame.flux = flux;
        frame.fluxLow = fluxLow;
        frame.fluxHigh = fluxHigh;
        frame.time = static_cast<float>(offset) / static_cast<float>(audio.sampleRate);

        prevMag = mag;
        prevMagLow = mag;
        prevMagHigh = mag;
        result.frames[fidx] = frame;
    }

    // Onset detection via adaptive thresholds on spectral flux (kick/snare)
    const int window = 10;
    for (int i = 0; i < numFrames; ++i)
    {
        float avgLow = 0.0f;
        float avgHigh = 0.0f;
        int count = 0;
        for (int j = std::max(0, i - window); j < std::min(numFrames, i + window); ++j)
        {
            avgLow += result.frames[j].fluxLow;
            avgHigh += result.frames[j].fluxHigh;
            ++count;
        }
        if (count == 0)
            continue;
        avgLow /= static_cast<float>(count);
        avgHigh /= static_cast<float>(count);

        float thresholdLow = avgLow * 1.7f;
        float thresholdHigh = avgHigh * 1.7f;

        if (result.frames[i].fluxLow > thresholdLow && result.frames[i].fluxLow > 0.0f)
        {
            result.onsetTimesKick.push_back(result.frames[i].time);
        }
        if (result.frames[i].fluxHigh > thresholdHigh && result.frames[i].fluxHigh > 0.0f)
        {
            result.onsetTimesSnare.push_back(result.frames[i].time);
        }
    }
    return result;
}

// GLFW callbacks ------------------------------------------------------------------------

static void framebuffer_size_callback(GLFWwindow * /*window*/, int width, int height)
{
    glViewport(0, 0, width, height);
}

// Main ----------------------------------------------------------------------------------

int main(int argc, char **argv)
{
    int particleCount = 20000;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--particles" && i + 1 < argc)
        {
            int value = std::atoi(argv[i + 1]);
            if (value > 0)
            {
                // Clamp to a reasonable range to avoid extreme GPU/CPU load
                if (value < 1000)
                    value = 1000;
                if (value > 1000000)
                    value = 1000000;
                particleCount = value;
            }
            ++i;
        }
    }

    std::cout << "Particle count: " << particleCount << std::endl;

    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow *window = glfwCreateWindow(1280, 720, "Audio Visualizer SH", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    // Capture mouse for FPS-style camera look
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Start audio playback using miniaudio (plays audio.wav in a loop)
    ma_engine engine;
    if (ma_engine_init(nullptr, &engine) != MA_SUCCESS)
    {
        std::cerr << "Failed to initialize miniaudio engine" << std::endl;
    }
    else
    {
        if (ma_engine_play_sound(&engine, "audio.wav", nullptr) != MA_SUCCESS)
        {
            std::cerr << "Failed to play audio.wav via miniaudio" << std::endl;
        }
    }

    // Load audio and precompute analysis
    AudioData audio;
    if (!loadWav("audio.wav", audio))
    {
        std::cerr << "Place a 16-bit PCM WAV named audio.wav next to the executable." << std::endl;
    }
    AnalysisResult analysis = analyzeAudio(audio);

    // Initialize sphere grid layout and band mapping
    initSphereLayout();

    // Build particle system and OpenGL buffers
    std::vector<Particle> particles(particleCount);

    const float sphereRadius = 1.0f;
    const float innerSphereRadius = 0.25f;
    const float particleLifetimeMin = 1.0f;
    const float particleLifetimeMax = 3.0f;

    for (int i = 0; i < particleCount; ++i)
    {
        float u1 = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        float u2 = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        float u3 = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

        float theta = 2.0f * static_cast<float>(M_PI) * u1;
        float phi = std::acos(2.0f * u2 - 1.0f);
        float r = std::cbrt(u3) * (sphereRadius * innerSphereRadius);

        float x = r * std::sin(phi) * std::cos(theta);
        float y = r * std::sin(phi) * std::sin(theta);
        float z = r * std::cos(phi);

        particles[i].pos = Vec3{x, y, z};
        particles[i].basePos = particles[i].pos;
        particles[i].vel = Vec3{0.0f, 0.0f, 0.0f};

        float lifeRand = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        float life = particleLifetimeMin + (particleLifetimeMax - particleLifetimeMin) * lifeRand;
        particles[i].lifetime = life;
        particles[i].maxLifetime = life;
        particles[i].beatEffect = 0.0f;
    }

    GLuint vao = 0, vbo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(Vec3), nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vec3), (void *)0);

    glBindVertexArray(0);

    // Load shaders (Spherical Harmonics deformation + onset-based color)
    GLuint program = createProgram("shaders/sphere.vert", "shaders/sphere.frag");

    GLint locProj = glGetUniformLocation(program, "uProjection");
    GLint locView = glGetUniformLocation(program, "uView");
    GLint locModel = glGetUniformLocation(program, "uModel");
    GLint locSHCoeffs = glGetUniformLocation(program, "uSHCoeffs");
    GLint locDispScale = glGetUniformLocation(program, "uDisplacementScale");
    GLint locOnsetPulse = glGetUniformLocation(program, "uOnsetPulse");
    GLint locBandLevel = glGetUniformLocation(program, "uBandLevel");
    GLint locPointSize = glGetUniformLocation(program, "uPointSize");

    int width = 1280, height = 720;
    glfwGetFramebufferSize(window, &width, &height);
    float aspect = static_cast<float>(width) / static_cast<float>(height);
    Mat4 proj = mat4_perspective(45.0f * static_cast<float>(M_PI) / 180.0f, aspect, 0.1f, 100.0f);

    // Initial camera state
    Vec3 cameraPos{0.0f, 0.0f, 6.0f};
    float yaw = -90.0f; // looking towards -Z
    float pitch = 0.0f;
    Vec3 worldUp{0.0f, 1.0f, 0.0f};

    double startTime = glfwGetTime();
    double lastFrameTime = startTime;
    float lastKickTime = -10.0f;
    float lastSnareTime = -10.0f;

    std::vector<float> shCoeffs(9, 0.0f);

    const float noiseScale = 4.0f;
    const float noiseSpeed = 0.1f;
    const float turbulenceStrength = 0.005f;
    const float beatStrength = 0.01f;

    while (!glfwWindowShouldClose(window))
    {
        double now = glfwGetTime();
        double tNow = now - startTime;
        double deltaTime = now - lastFrameTime;
        lastFrameTime = now;

        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Camera movement: WASD to move, arrow keys + mouse to look around
        {
            // Compute forward vector from yaw/pitch
            float radYaw = yaw * static_cast<float>(M_PI) / 180.0f;
            float radPitch = pitch * static_cast<float>(M_PI) / 180.0f;
            Vec3 front{
                std::cos(radYaw) * std::cos(radPitch),
                std::sin(radPitch),
                std::sin(radYaw) * std::cos(radPitch)};
            front = vec3_normalize(front);

            Vec3 forwardXZ = vec3_normalize(Vec3{front.x, 0.0f, front.z});
            Vec3 right = vec3_normalize(vec3_cross(forwardXZ, worldUp));

            float moveSpeed = 4.0f * static_cast<float>(deltaTime);
            float rotSpeed = 60.0f * static_cast<float>(deltaTime);

            // Mouse look: accumulate yaw/pitch from cursor delta
            {
                static bool firstMouse = true;
                static double lastX = 0.0;
                static double lastY = 0.0;

                double mouseX, mouseY;
                glfwGetCursorPos(window, &mouseX, &mouseY);

                if (firstMouse)
                {
                    lastX = mouseX;
                    lastY = mouseY;
                    firstMouse = false;
                }

                double offsetX = mouseX - lastX;
                double offsetY = mouseY - lastY;
                lastX = mouseX;
                lastY = mouseY;

                float sensitivity = 0.1f; // adjust to taste
                yaw += static_cast<float>(offsetX) * sensitivity;
                pitch -= static_cast<float>(offsetY) * sensitivity;
            }

            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
                cameraPos = vec3_add(cameraPos, vec3_scale(forwardXZ, moveSpeed));
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
                cameraPos = vec3_sub(cameraPos, vec3_scale(forwardXZ, moveSpeed));
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
                cameraPos = vec3_sub(cameraPos, vec3_scale(right, moveSpeed));
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
                cameraPos = vec3_add(cameraPos, vec3_scale(right, moveSpeed));

            // Optional vertical movement with Q/E
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
                cameraPos.y -= moveSpeed;
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
                cameraPos.y += moveSpeed;

            // Rotate view with arrow keys
            if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
                yaw -= rotSpeed;
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
                yaw += rotSpeed;
            if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
                pitch += rotSpeed;
            if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
                pitch -= rotSpeed;

            pitch = std::clamp(pitch, -89.0f, 89.0f);
        }

        // Map current time to analysis frame
        if (!analysis.frames.empty())
        {
            int frameSize = 1024;
            int hopSize = 512;
            float frameDuration = static_cast<float>(hopSize) / static_cast<float>(audio.sampleRate);
            int idx = static_cast<int>(tNow / frameDuration);
            if (idx >= 0 && idx < static_cast<int>(analysis.frames.size()))
            {
                const AnalysisFrame &af = analysis.frames[idx];
                for (int i = 0; i < 9 && i < static_cast<int>(af.bandEnergy.size()); ++i)
                {
                    // Boost band energy and apply a gentle non-linearity so
                    // SH deformation is more visible and organic.
                    float boosted = af.bandEnergy[i] * 4.0f;
                    float target = std::sqrt(std::max(boosted, 0.0f));
                    // Smooth for organic SH motion
                    shCoeffs[i] = shCoeffs[i] * 0.7f + target * 0.3f;
                }

                // Update per-sphere band levels from configured band ranges
                for (int s = 0; s < kNumSpheres; ++s)
                {
                    int bs = sphereBandStart[s];
                    int be = sphereBandEnd[s];
                    float sum = 0.0f;
                    int countBands = 0;
                    for (int b = bs; b <= be && b < static_cast<int>(af.bandEnergy.size()); ++b)
                    {
                        sum += af.bandEnergy[b];
                        ++countBands;
                    }
                    float energy = (countBands > 0) ? (sum / static_cast<float>(countBands)) : 0.0f;
                    float boosted = energy * 5.0f;
                    float targetLevel = std::sqrt(std::max(boosted, 0.0f));
                    bandLevels[s] = bandLevels[s] * 0.7f + targetLevel * 0.3f;
                }
            }
            // Onset pulses: low (kick) and high (snare)
            for (float ot : analysis.onsetTimesKick)
            {
                if (std::fabs(static_cast<float>(tNow) - ot) < 0.02f)
                {
                    lastKickTime = static_cast<float>(tNow);
                    break;
                }
            }
            for (float ot : analysis.onsetTimesSnare)
            {
                if (std::fabs(static_cast<float>(tNow) - ot) < 0.02f)
                {
                    lastSnareTime = static_cast<float>(tNow);
                    break;
                }
            }
        }

        // Kick: short, strong pulse for scale
        float kickPulse = 0.0f;
        float dtKick = static_cast<float>(tNow) - lastKickTime;
        if (dtKick >= 0.0f && dtKick < 0.25f)
        {
            float x = dtKick / 0.25f;
            kickPulse = 1.0f - x;
        }

        // Snare: slightly longer, softer pulse for color
        float snarePulse = 0.0f;
        float dtSnare = static_cast<float>(tNow) - lastSnareTime;
        if (dtSnare >= 0.0f && dtSnare < 0.35f)
        {
            float x = dtSnare / 0.35f;
            snarePulse = 1.0f - x;
        }

        float onsetPulse = std::max(kickPulse, snarePulse);

        // Recompute view from current camera state
        float radYaw = yaw * static_cast<float>(M_PI) / 180.0f;
        float radPitch = pitch * static_cast<float>(M_PI) / 180.0f;
        Vec3 camFront{
            std::cos(radYaw) * std::cos(radPitch),
            std::sin(radPitch),
            std::sin(radYaw) * std::cos(radPitch)};
        camFront = vec3_normalize(camFront);
        Vec3 center = vec3_add(cameraPos, camFront);
        Mat4 view = mat4_look_at(cameraPos, center, worldUp);

        // Update particle system (CPU side, similar to main.js logic)
        {
            float timeFactor = static_cast<float>(tNow);

            for (int i = 0; i < particleCount; ++i)
            {
                Particle &p = particles[i];

                p.lifetime -= static_cast<float>(deltaTime);

                // Apply simple 3D noise as turbulence
                float nx = noise3D(p.pos.x * noiseScale + timeFactor * noiseSpeed, p.pos.y * noiseScale, p.pos.z * noiseScale);
                float ny = noise3D(p.pos.x * noiseScale, p.pos.y * noiseScale + timeFactor * noiseSpeed, p.pos.z * noiseScale);
                float nz = noise3D(p.pos.x * noiseScale, p.pos.y * noiseScale, p.pos.z * noiseScale + timeFactor * noiseSpeed);

                p.vel.x += nx * turbulenceStrength;
                p.vel.y += ny * turbulenceStrength;
                p.vel.z += nz * turbulenceStrength;

                // Beat effect pushes particles outwards from center
                if (onsetPulse > 0.1f)
                {
                    p.beatEffect = 1.0f;
                }
                p.beatEffect *= 0.95f;

                float dist = std::sqrt(p.pos.x * p.pos.x + p.pos.y * p.pos.y + p.pos.z * p.pos.z);
                if (p.beatEffect > 0.01f && dist > 0.0f)
                {
                    float dx = p.pos.x / dist;
                    float dy = p.pos.y / dist;
                    float dz = p.pos.z / dist;
                    float push = p.beatEffect * beatStrength;
                    p.vel.x += dx * push;
                    p.vel.y += dy * push;
                    p.vel.z += dz * push;
                }

                // Integrate motion
                p.pos.x += p.vel.x * static_cast<float>(deltaTime);
                p.pos.y += p.vel.y * static_cast<float>(deltaTime);
                p.pos.z += p.vel.z * static_cast<float>(deltaTime);

                // Dampen velocity
                p.vel.x *= 0.98f;
                p.vel.y *= 0.98f;
                p.vel.z *= 0.98f;

                // Keep particles within sphere radius
                dist = std::sqrt(p.pos.x * p.pos.x + p.pos.y * p.pos.y + p.pos.z * p.pos.z);
                if (dist > sphereRadius)
                {
                    float overflow = dist - sphereRadius;
                    float pullback = overflow * 0.1f;
                    if (dist > 0.0f)
                    {
                        float dx = p.pos.x / dist;
                        float dy = p.pos.y / dist;
                        float dz = p.pos.z / dist;
                        p.pos.x -= dx * pullback;
                        p.pos.y -= dy * pullback;
                        p.pos.z -= dz * pullback;
                    }
                    p.vel.x *= 0.9f;
                    p.vel.y *= 0.9f;
                    p.vel.z *= 0.9f;
                }

                // Respawn dead particles
                if (p.lifetime <= 0.0f)
                {
                    float u1 = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
                    float u2 = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
                    float u3 = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

                    float theta = 2.0f * static_cast<float>(M_PI) * u1;
                    float phi = std::acos(2.0f * u2 - 1.0f);
                    float r = std::cbrt(u3) * (sphereRadius * innerSphereRadius);

                    float x = r * std::sin(phi) * std::cos(theta);
                    float y = r * std::sin(phi) * std::sin(theta);
                    float z = r * std::cos(phi);

                    p.pos = Vec3{x, y, z};
                    p.basePos = p.pos;
                    p.vel = Vec3{0.0f, 0.0f, 0.0f};

                    float lifeRand = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
                    float life = particleLifetimeMin + (particleLifetimeMax - particleLifetimeMin) * lifeRand;
                    p.lifetime = life;
                    p.maxLifetime = life;
                    p.beatEffect = 0.0f;
                }
            }

            // Upload updated positions to GPU
            std::vector<Vec3> positions(particleCount);
            for (int i = 0; i < particleCount; ++i)
            {
                positions[i] = particles[i].pos;
            }

            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferSubData(GL_ARRAY_BUFFER, 0, particleCount * sizeof(Vec3), positions.data());
        }

        glUseProgram(program);
        glUniformMatrix4fv(locProj, 1, GL_FALSE, proj.m);
        glUniformMatrix4fv(locView, 1, GL_FALSE, view.m);
        glUniform1f(locDispScale, 1.2f);
        glUniform1f(locOnsetPulse, onsetPulse);
        glUniform1fv(locSHCoeffs, 9, shCoeffs.data());
        glUniform1f(locPointSize, 3.0f);

        glBindVertexArray(vao);

        float bandLevel = bandLevels[0];
        Mat4 modelRS = mat4_scale(1.0f + kickPulse * 0.4f + bandLevel * 0.3f);
        Mat4 model = mat4_mul(mat4_translate(0.0f, 0.0f, 0.0f), modelRS);
        glUniformMatrix4fv(locModel, 1, GL_FALSE, model.m);
        glUniform1f(locBandLevel, bandLevel);

        glDrawArrays(GL_POINTS, 0, particleCount);

        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(program);

    ma_engine_uninit(&engine);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
