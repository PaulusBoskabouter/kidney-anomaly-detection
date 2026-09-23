{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pkgs.python3 pkgs.uv ];
LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
  pkgs.stdenv.cc.cc.lib
  pkgs.zlib
  pkgs.cudaPackages.cudatoolkit
  pkgs.cudaPackages.cuda_cudart
  pkgs.libGL
  pkgs.libxcb
  pkgs.glib
  pkgs.openslide
] + ":/run/opengl-driver/lib";
      };
    };
}
