{
  description = "CO-tip classifier: Burn inference + Python weight conversion";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    rust-overlay,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [rust-overlay.overlays.default];
        };

        # Rust toolchain
        rustTools = {
          stable = pkgs.rust-bin.stable."1.89.0".default.override {
            extensions = ["rust-src"];
          };
          analyzer = pkgs.rust-bin.stable."1.89.0".rust-analyzer;
        };

        # Python environment for weight conversion (H5 → PT)
        pythonEnv = pkgs.python3.withPackages (ps: [
          ps.h5py
          ps.torch
          ps.numpy
        ]);

        devTools = with pkgs; [
          cargo-expand
          pkg-config
        ];

        rustDeps =
          [
            rustTools.stable
            rustTools.analyzer
          ]
          ++ devTools;
      in {
        # Rust development + inference
        devShells.default = pkgs.mkShell {
          name = "co-tip-net";
          buildInputs = rustDeps;
          shellHook = ''
            echo "Using Rust toolchain: $(rustc --version)"
            export CARGO_HOME="$HOME/.cargo"
            export RUSTUP_HOME="$HOME/.rustup"
            mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"
          '';
        };

        # Python environment for H5 → PT weight conversion
        devShells.convert = pkgs.mkShell {
          name = "co-tip-net-convert";
          buildInputs = [pythonEnv pkgs.ruff];
          shellHook = ''
            echo "Python: $(python3 --version)"
            echo "Use: python convert_weights.py"
          '';
        };
      }
    );
}
