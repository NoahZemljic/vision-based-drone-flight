# Report fleet results

Final-report evidence

Seeds: 123, 456, 789; 32 environments; delay seed 42.

Distance and measured body-rate variation use the second half of each rollout. Angular error and crash counts use the full rollout. Rates are per-environment population standard deviations, averaged over environments and then seeds. Angular error is relative to the body forward axis.

| Cell | Seeds | Distance error (m) | Angle (deg) | Roll std | Pitch std | Yaw std | Crashes | Envs crashed | Bundle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hover/H0-base | 3 | 0.051891 | — | 0.178077 | 6.402865 | 0.063348 | 0.000000 | 0.000000 | present |
| hover/H1-delay | 3 | 0.188044 | — | 0.006381 | 0.010454 | 0.011822 | 0.000000 | 0.000000 | present |
| hover/H2-K | 3 | 0.956382 | — | 1.431518 | 1.703880 | 0.230766 | 36.333333 | 32.000000 | present |
| state/S0-world | 3 | 0.097841 | 17.238066 | 2.089187 | 1.454457 | 0.674707 | 23.000000 | 16.333333 | present |
| state/S1-body | 3 | 0.089974 | 17.089628 | 4.238447 | 1.733809 | 0.893424 | 11.333333 | 10.666667 | present |
| state/S2-AC | 3 | 0.098973 | 15.670306 | 4.794337 | 1.425429 | 0.959901 | 11.666667 | 10.333333 | present |
| state/S3-PV | 3 | 0.110875 | 17.340495 | 3.098321 | 1.550724 | 0.776087 | 19.666667 | 15.666667 | present |
| state/S4-LPF | 3 | 0.089493 | 14.908159 | 1.326624 | 0.906597 | 0.454773 | 8.333333 | 7.666667 | present |
| state/S5-RS | 3 | 0.108381 | 14.937511 | 0.477462 | 0.413727 | 0.328909 | 6.000000 | 6.000000 | present |
| state/S-K-abl | 3 | 0.103288 | 19.059285 | 2.809709 | 1.464850 | 0.848914 | 27.000000 | 19.666667 | present |
| vision/V0-base | 3 | 0.199309 | 14.728460 | 6.127236 | 1.981754 | 1.172128 | 17.666667 | 14.666667 | present |
| vision/V1-AC | 3 | 0.237190 | 16.868106 | 2.686140 | 1.599164 | 0.721002 | 16.333333 | 12.333333 | present |
| vision/V2-LPF | 3 | 0.209854 | 14.538089 | 1.332883 | 1.006737 | 0.527632 | 7.333333 | 7.000000 | present |
| vision/V3-final | 3 | 0.204939 | 15.248702 | 0.535792 | 0.517164 | 0.395836 | 8.000000 | 6.666667 | present |
| vision/V-K-ablation | 3 | 0.242982 | 18.221022 | 2.518820 | 1.567658 | 0.782209 | 27.000000 | 18.666667 | present |
| state/S5-100m | 3 | 0.080074 | 14.563020 | 0.479361 | 0.437825 | 0.377869 | 7.666667 | 6.666667 | MISSING |
| vision/V3-100m | 3 | 0.212105 | 14.730144 | 0.661813 | 0.669959 | 0.466172 | 14.333333 | 10.666667 | MISSING |

Hover distance is error to (0, 0, 1) m; tracking distance is |d − 1.5|. Rates are in rad/s; crash counts are averages across seeds, not probabilities.

Missing original checkpoint bundles: state/S5-100m, vision/V3-100m.

Rebuilding this summary does not run policies or establish fresh simulation reproducibility.
