{
  description = "CO-tip classifier: Burn inference + Python weight conversion";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {nixpkgs, fenix, ...}: let
    forAllSystems = nixpkgs.lib.genAttrs ["x86_64-linux" "aarch64-linux"];
  in {
    devShells = forAllSystems (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      toolchain = fenix.packages.${system}.stable.withComponents [
        "cargo" "clippy" "rust-src" "rustc" "rustfmt"
      ];
    in {
      default = pkgs.mkShell {
        nativeBuildInputs = [
          toolchain
          fenix.packages.${system}.rust-analyzer
          pkgs.pkg-config
          pkgs.cargo-expand
        ];
      };

      # Python environment for H5 → PT weight conversion
      convert = pkgs.mkShell {
        nativeBuildInputs = [
          (pkgs.python3.withPackages (ps: [ps.h5py ps.torch ps.numpy]))
          pkgs.ruff
        ];
      };
    });
  };
}
