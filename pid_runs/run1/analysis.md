# PID Run 1 — default scenario

## Config
- Kp=4.0, Ki=0.00001, Kd=7.9995 (Bayesian-tuned across both scenarios)
- Scenario: default (gentle -1 m/s² brake → +1 m/s² accel → -2 m/s² slowdown)

## Metrics
| Metric | Value |
|---|---|
| Mean absolute error | 0.37 m |
| Min following distance | ~19.5 m |
| Collisions | 0 |
| Mean absolute jerk | low |

## Notes
Easy scenario — slow lead-car transitions give the derivative term enough time to react cleanly. Gains were tuned partly on this profile, so it's essentially home turf for PID.
