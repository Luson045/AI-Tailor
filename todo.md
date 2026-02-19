# TODO

## User-Driven Height Correction (Post-Hoc Rescaling)

### Idea
Once the SMPL-X mesh is generated, if the user knows their real height, compute a single
correction factor and apply it uniformly to all mesh vertices:

```
scale_factor = actual_height / predicted_height
# e.g. 165 / 178 = 0.927
```

Since SMPL-X preserves relative body proportions internally, multiplying all vertices by
this scalar keeps the mesh anatomically consistent — height, limb lengths, shoulder width,
and girth measurements all correct proportionally.

---

### What To Do

- [ ] After mesh generation, compute `predicted_height` from the fitted mesh
      (max Y vertex − min Y vertex in real-world metres).
- [ ] Accept `actual_height` as an optional user input (CLI arg or config value).
- [ ] If `actual_height` is provided, compute `scale_factor = actual_height / predicted_height`.
- [ ] Apply `scale_factor` uniformly to all mesh vertices before export.
- [ ] Re-export the corrected `.obj` and `.ply` files.
- [ ] Print a summary showing predicted height, actual height, and correction factor applied.

---

### Notes

- Correction is **post-hoc** — no re-optimization needed, computationally free.
- Linear rescaling is accurate enough for corrections under ~10%. The 165/178 (~7.3%)
  case is well within safe range.
- Circumference measurements (belly, chest, waist) scale linearly on the mesh but are
  volume-derived in reality — for large corrections (>10%) these may be slightly off,
  but acceptable for the target use case.
- This gives users a simple correction hook without touching the optimization pipeline.
- Could later be extended to accept other known measurements (e.g. actual shoulder width)
  as additional correction anchors.