{ pkgs ? import <nixpkgs> {} }:


pkgs.mkShell {
  nativeBuildInputs = [
    pkgs.chromedriver
    pkgs.python310
    (pkgs.poetry.override { python = pkgs.python310; })
  ];
}
