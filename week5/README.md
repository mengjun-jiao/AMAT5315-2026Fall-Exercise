# Week 5 — Automatic Differentiation and Checkpointing

This week covers automatic differentiation, acoustic wave propagation,
Born/adjoint differentiation, and Treeverse checkpointing.

## Part 2 — Forward acoustic evidence

The official reflector experiment is read from `inputs/reflector.json`. The
input geometry and Ricker source pulse are plotted in
`artifacts/inputs.png`. The validated forward command is:

```text
seismic --experiment inputs/reflector.json --mode forward --every 3 --out artifacts/forward
```

Forward gathers are generated in `artifacts/forward/gathers.png`. The global
trace L2 norm is `11.574769503614`, with relative error
`4.288517048379e-08` against the reference. The trace maxima occur at index
`83` for all three shots.

Recording uses state step indices `0, 3, ..., 240`; step `150` is frame `50`
at reduced time `30.0`, or physical time `3.00 s`. The recorded
`wavefield.npy` and `echo.npy` files remain local and ignored. The
`wavefield.png` and `echo.png` screenshots were generated manually with the
course viewer. At step 150, the echo is much weaker than the full field: its
maximum absolute value is approximately `0.00622`, compared with
`0.25524` for the background wavefield.

Committed Part 2 evidence:

- `artifacts/inputs.png`
- `artifacts/forward/run.json`
- `artifacts/forward/result.json`
- `artifacts/forward/gathers.png`
- `artifacts/forward/wavefield.png`
- `artifacts/forward/echo.png`
