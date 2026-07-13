# Chapter code snapshots

This directory holds **runnable, chapter-exact code** for chapters whose
listings do not exist anywhere else in the repo. The canonical, current
implementation of everything always lives in `lib/`, `tools/`, and `tests/`
— the subdirectories here are *frozen teaching artifacts*, kept so that a
reader can build and run exactly the code a chapter presents, even when
the main tree has since evolved past it.

Conventions:

- One subdirectory per chapter number (`03/`, ...), created only when a
  chapter shows code that has no home in the main tree. Chapters whose
  listings match the current `lib/`/`tests/` code get no directory — the
  repo itself is their code.
- Each subdirectory has its own `BUILD` and `CMakeLists.txt` and builds a
  separately named binary (e.g. `tutorial-opt-03`), because the frozen
  passes register the **same CLI flags** as their modern counterparts in
  `tutorial-opt` — the two cannot live in one binary.
- Each subdirectory's `README.md` maps its files to the chapter sections
  they come from and lists any deliberate deviations (e.g. include paths,
  which necessarily differ from the main tree's `lib/...` paths).
- Scratch `.mlir` examples that chapters create on the fly are *not*
  stored here — the companion scripts in `../script/NN.sh` recreate those
  via heredocs (see `../README.md`).

Build and run (Bazel):

```bash
bazel build //tutorial/code/03:tutorial-opt-03
bazel-bin/tutorial/code/03/tutorial-opt-03 tests/affine_loop_unroll.mlir --affine-full-unroll
```

> **CMake users:** the same targets exist in the CMake build
> (`cmake --build build --target tutorial-opt-03`), subject to the same
> LLVM-submodule requirement as `tutorial-opt` (see the top-level README).
