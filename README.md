# Multi-Sphere Audio Visualizer (OpenGL)

This project implements a 3D audio visualizer using OpenGL with:

- **Spherical Harmonics (SH) deformation** of a 3D sphere driven by audio frequency bands (9 SH coefficients mapped from spectrum bands).
- **Onset detection / beat tracking** using spectral flux to trigger pulses in scale and color, synced to strong rhythm onsets.

## Build (macOS)

1. Install dependencies (GLFW via Homebrew):

```bash
brew install glfw
```

2. Configure and build with CMake:

```bash
cmake -S . -B build
cmake --build build
```

3. Place your audio file:

- Convert a song/loop to **16-bit PCM mono or stereo WAV**.
- Name it `audio.wav` and put it next to the built executable, e.g. `build/audio.wav`.

4. Run:

```bash
cd build
./audio_visualizer
```

The program precomputes FFT-based band energy and spectral flux from `audio.wav`, then:

- Maps 9 bands to 9 real SH coefficients per frame.
- Deforms the sphere surface in the vertex shader using SH basis functions.
- Uses onset times (strong flux peaks) to add pulses in scale and color.

You can adjust SH strength and pulse look in `src/main.cpp` and `shaders/sphere.*`.
