workspace(name = "audio_to_tactile")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "benchmark",
    urls = ["https://github.com/google/benchmark/archive/v1.5.2.zip"],
    sha256 = "21e6e096c9a9a88076b46bd38c33660f565fa050ca427125f64c4a8bf60f336b",
    strip_prefix = "benchmark-1.5.2",
)

http_archive(
    name = "sdl2",
    build_file = "sdl2.BUILD",
    sha256 = "e6a7c71154c3001e318ba7ed4b98582de72ff970aca05abc9f45f7cbdc9088cb",
    strip_prefix = "SDL2-2.0.8",
    urls = ["https://www.libsdl.org/release/SDL2-2.0.8.zip"],
)
