{
  description = "GPU development shell for widechat/nanochat";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        cuda = pkgs.cudaPackages;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python312
            python312Packages.pip
            uv
            git
            curl
            wget
            pkg-config
            cmake
            ninja
            gnumake
            gcc
            rustc
            cargo
            cuda.cudatoolkit
            cuda.cudnn
            cuda.nccl
            cuda.cuda_nvcc
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            cuda.cudatoolkit
            cuda.cudnn
            cuda.nccl
            pkgs.zlib
            pkgs.stdenv.cc.cc
          ];

          CUDA_PATH = "${cuda.cudatoolkit}";
          CUDA_HOME = "${cuda.cudatoolkit}";
          UV_LINK_MODE = "copy";

          shellHook = ''
            export NANOCHAT_BASE_DIR="''${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
            mkdir -p "$NANOCHAT_BASE_DIR"

            echo "Entered widechat GPU dev shell on ${system}."
            echo "Next steps:"
            echo "  1) uv venv"
            echo "  2) uv sync --extra gpu"
            echo "  3) bash runs/speedrun.sh"
          '';
        };
      });
}
