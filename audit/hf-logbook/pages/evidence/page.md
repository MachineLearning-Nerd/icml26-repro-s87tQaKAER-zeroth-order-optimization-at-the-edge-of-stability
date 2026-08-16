# Evidence


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_383bc10d28e9", "created_at": "2026-07-23T04:55:54+00:00", "title": "Verification output (last 40 lines)"}
-->
## Verification output (last 40 lines)

```

==============================================================================
CLAIM 3: frozen ZO-Adam threshold depends on P^{-1}H spectrum (Theorem 3)
==============================================================================
  raw bounds: [0.5063, 1.0256]
  preconditioned bounds: [0.5505, 0.8219]
  bounds differ: True
  -> PASS

==============================================================================
CLAIM 4: CIFAR empirical (deferred)
==============================================================================
  (Paper: CIFAR-10 CNN/ResNet/ViT; not independently reproduced.)
  -> FAIL (deferred)

==============================================================================
CLAIM 5: only Tr(H) + λ_max needed for tracking (Sec 5)
==============================================================================
  bounds use 2 spectral quantities (Tr(H), λ_max)
  full spectrum has 5 eigenvalues
  -> PASS (efficient tracking)

==============================================================================
CLAIM 6: catapult dynamics past mean-square threshold (Sec 6)
==============================================================================
  MS at η=ub*1.5: final=1.66e+29, growing: True
  -> PASS

==============================================================================
VERDICT SUMMARY
==============================================================================
  [PASS] c1_step_size_bound
  [PASS] c2_momentum
  [PASS] c3_preconditioned
  [FAIL] c4_cifar
  [PASS] c5_efficient_tracking
  [PASS] c6_catapult

  5/6 claims verified.
  wrote outputs/verdict.json
```
