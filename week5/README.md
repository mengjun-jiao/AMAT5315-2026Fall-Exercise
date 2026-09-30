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

## Part 3 — Born and full-history adjoint evidence

Born mode uses `c0 = background` and the perturbation as the Born tangent
direction `m`, producing `d = Jm`. The official
`sum(born_data**2)` is `3.484789021516375e-02`. The generated
`artifacts/born/born_data.npy` remains local and ignored.

Full-history adjoint mode computes `image = J^T d`, retaining 241 complete
states per shot with peak saved storage of `6481936` bytes. The generated
`artifacts/adjoint/image.npy` remains local and ignored.

The transpose identity is:

```text
left                 = 3.484789021516375e-02
right                = 3.484789021516372e-02
relative difference  = 7.964779343672095e-16
```

The raw signed RTM evidence is `artifacts/adjoint/image.png`. Its row-L2
depth profile localizes the known reflector at `2.1 km`: the true depth is
`2.1 km`, the detected peak is `2.1 km`, and the depth error is `0.0 km`.

Adjoint recording stores reverse frames in decreasing timestep order. Step
132 corresponds to physical time `2.64 s`. The
`artifacts/adjoint/wavefield.png` screenshot was generated manually using
the course viewer.

Committed Part 3 evidence:

- `artifacts/born/run.json`
- `artifacts/born/result.json`
- `artifacts/adjoint/run.json`
- `artifacts/adjoint/result.json`
- `artifacts/adjoint/image.png`
- `artifacts/adjoint/wavefield.png`

The following numerical arrays remain local and ignored:

- `artifacts/born/born_data.npy`
- `artifacts/adjoint/image.npy`
- `artifacts/adjoint/wavefield.npy`
