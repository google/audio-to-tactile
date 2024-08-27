workspace(name = "audio_to_tactile")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "benchmark",
    sha256 = "1a6f0678cbcac65a12e2178d77d3c97d050d173389220c9df57e9249a40827ec",
    strip_prefix = "benchmark-1.9.0",
    urls = ["https://github.com/google/benchmark/archive/v1.9.0.zip"],
)

http_archive(
    name = "sdl2",
    build_file = "sdl2.BUILD",
    sha256 = "e6a7c71154c3001e318ba7ed4b98582de72ff970aca05abc9f45f7cbdc9088cb",
    strip_prefix = "SDL2-2.0.8",
    urls = ["https://www.libsdl.org/release/SDL2-2.0.8.zip"],
)
