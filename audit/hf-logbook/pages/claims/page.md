# Claims


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f7873a14c8a0", "created_at": "2026-07-23T04:55:53+00:00", "title": "Claims to reproduce"}
-->
## Claims to reproduce

1. For ZO-GD, the mean-square critical step size satisfies 2/(Tr(H) + 2*lambda_max(H)) <= eta_ms* <= 2/Tr(H), showing dependence on the full Hessian spectrum rather than just the top eigenvalue (Theorem 1).
2. Theorem 2 extends the mean-square stability bound to ZO-GD with momentum (ZO-GDM), showing the critical step size is adjusted by the momentum parameter beta (Theorem 2).
3. For frozen ZO-Adam, the mean-square stability threshold depends on the spectrum of the preconditioned Hessian P^{-1}H rather than the raw Hessian (Theorem 3).
4. Empirically, ZO-GD, ZO-GDM, and ZO-Adam all stabilize their trace/curvature statistics near the theoretically predicted mean-square edge-of-stability thresholds when trained on a CNN on CIFAR-10, ResNet20, and a Vision Transformer (Section 5, Figure 2).
5. Practical stability tracking during training requires only the trace and top eigenvalue of the Hessian, not the full spectrum, via the bounds given in Equations 23-25 (Section 5, Equations 23-25).
6. Catapult-like dynamics appear when the step size is increased past the mean-square stability threshold in the empirical experiments (Section 5, Figure 3).
