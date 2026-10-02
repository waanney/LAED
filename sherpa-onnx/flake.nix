{
  description = "Development environment for sherpa-onnx";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              ccache
              clang-tools
              cmake
              curl
              git
              llama-cpp
              ninja
              pkg-config
              (python3.withPackages (ps: with ps; [
                numpy
                sounddevice
              ]))
              unzip
              wget
              zip
            ] ++ lib.optionals stdenv.hostPlatform.isLinux [
              alsa-lib
              alsa-utils
              gdb
              onnxruntime
              portaudio
              pipewire
            ];

            CMAKE_GENERATOR = "Ninja";
            CMAKE_C_COMPILER_LAUNCHER = "ccache";
            CMAKE_CXX_COMPILER_LAUNCHER = "ccache";
            # espeak-ng intentionally builds some files with -Wno-format.
            # Nix's default format hardening turns that into a fatal warning.
            NIX_HARDENING_ENABLE = "fortify pie relro stackprotector";

            shellHook = ''
              export CCACHE_DIR="''${CCACHE_DIR:-$PWD/.ccache}"
            '';
          } // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
            SHERPA_ONNXRUNTIME_INCLUDE_DIR = "${pkgs.onnxruntime}/include/onnxruntime";
            SHERPA_ONNXRUNTIME_LIB_DIR = "${pkgs.onnxruntime}/lib";
          };
        }
      );
    };
}
