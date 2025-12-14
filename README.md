# Multi-Sphere Audio Visualizer (OpenGL)

This project implements a 3D audio visualizer using OpenGL with:

- **Particle-based sphere visualizer** rendered as point sprites instead of a solid mesh sphere.
- **Spherical Harmonics (SH) deformation** of particle positions driven by audio frequency bands (9 SH coefficients mapped from spectrum bands).
- **Onset detection / beat tracking** using spectral flux to trigger pulses in scale, color, and outward particle motion, synced to strong rhythm onsets.

## Build (macOS)

1. Install dependencies (GLFW and cmake via Homebrew):

```bash
brew install glfw cmake
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
./audio_visualizer          # default ~20k particles

# or specify particle count (clamped between 1000 and 1000000)
./audio_visualizer --particles 50000
```

The program precomputes FFT-based band energy and spectral flux from `audio.wav`, then:

- Maps 9 bands to 9 real SH coefficients per frame.
- Deforms particle positions in the vertex shader using SH basis functions.
- Uses onset times (strong flux peaks) to add pulses in overall scale, color, and particle motion.

You can adjust SH strength, particle motion, and point size in `src/main.cpp` and `shaders/sphere.*`.
