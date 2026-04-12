{
  description = "CO-tip classifier: Burn inference + Python weight conversion";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    forAllSystems = nixpkgs.lib.genAttrs ["x86_64-linux" "aarch64-linux"];
  in {
    devShells = forAllSystems (system: let
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      default = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          cargo clippy rustc rustfmt rust-analyzer
          pkg-config cargo-expand
        ];
        RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
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
