# Video renderer path validation

## Root cause

`src/video.rs` previously constructed `env!("CARGO_MANIFEST_DIR")/../render_fluid.py`
at runtime. That works only when the source tree is still present at the compile-time
manifest location. The isolated Makefile test uses a temporary crate directory, and an
installed `~/.cargo/bin/md` has no corresponding source tree, so Python failed with:

```text
can't open file '/tmp/.tmp.../md/../render_fluid.py'
```

The compile-time path and the runtime working directory are different concerns: changing
the current directory cannot repair a path captured from the temporary `CARGO_MANIFEST_DIR`.

## Fix

The renderer script is now embedded with `include_bytes!` during compilation. Each video
run writes those bytes into its own existing temporary directory and invokes the configured
Python interpreter on that temporary script. No `/home/...` or random source-tree path is
hard-coded, and the installed binary works from any working directory.

## Evidence

- `f998fd6`: embedded-script regression red; the test failed because the source was empty.
- `4e8fe31` is the prior ramp implementation; this fix follows in the current video-path commit.
- Embedded-script unit regression passes.
- From `week2/`, with the persistent Python environment:

  ```bash
  MD_PYTHON="$HOME/.venvs/amat5315/bin/python" \
  MPLCONFIGDIR=/tmp/amat5315-matplotlib \
  cargo test --release --manifest-path md/Cargo.toml --test video
  ```

  Exit status: 0; the two-frame and below-2MB assertions remain active.

- The full release `--all-targets` suite, including the isolated Makefile test and video
  integration test, passes with the same environment. No generated video or trajectory is
  part of this change.
