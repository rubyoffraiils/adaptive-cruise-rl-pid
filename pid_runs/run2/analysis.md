# PID Run 2 — hard scenario

## Config
- Kp=4.0, Ki=0.00001, Kd=7.9995 (same tuned gains)
- Scenario: hard (emergency brake -4 m/s², stop-and-go, sudden +2 m/s² burst, repeat)

## Metrics
| Metric | Value |
|---|---|
| Mean absolute error | 0.93 m |
| Min following distance | 19.42 m |
| Collisions | 0 |
| Mean absolute jerk | 0.99 m/s³ |

## Notes
No collision, error under 1 m despite the aggressive lead inputs. The Bayesian tuner optimized across both scenarios at once, so the gains were already hardened for the hard case. Kd=8 is high enough to damp the fast braking transients without causing oscillation. PID performs well here precisely because the disturbances are deterministic and repeatable — the gains were fit to them.
