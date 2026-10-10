# ACT closed-loop evaluation (gold test set, 1 g)

| Controller | Finished | RMSE mm | p95 mm | max mm | Heading RMSE deg | Corner RMSE mm | Jerk RMS rad/s3 | Draw time s |
|---|---|---|---|---|---|---|---|---|
| ik_real | 0/48 | - | - | - | - | - | - | - |
| act_real_v2 | 48/48 | 0.221 | 0.381 | 0.889 | 9.327 | 0.375 | 310.751 | 81.508 |

## Paired against `ik_real` (drawings both finished)

| Controller | Pairs | Mean RMSE delta mm | Lower RMSE |
|---|---|---|---|
| act_real_v2 | 0 | - | 0/0 |

## Per category (finished / RMSE mm)

| Category | ik_real | act_real_v2 |
|---|---|---|
| Bird | 0/9 · - | 9/9 · 0.213 |
| Cat | 0/12 · - | 12/12 · 0.228 |
| Dog | 0/13 · - | 13/13 · 0.223 |
| Fish | 0/4 · - | 4/4 · 0.225 |
| Frog | 0/5 · - | 5/5 · 0.215 |
| Horse | 0/5 · - | 5/5 · 0.214 |

## Failures

**ik_real**
- 1a5733d2b6fb8c11: Waypoint timeout in DRAW_STROKE_0 at index 272
- 35d8032b6ae21c32: Waypoint timeout in DRAW_STROKE_0 at index 381
- 48f51b30016f8eb1: Waypoint timeout in DRAW_STROKE_0 at index 185
- 91f30c2ab24c76ea: Waypoint timeout in APPROACH_STROKE_0 at index 12
- 99cbe879e4812b0b: Waypoint timeout in DRAW_STROKE_0 at index 263
- bbf4a726747089d9: Waypoint timeout in APPROACH_STROKE_0 at index 26
- c326f229ac98c00a: Waypoint timeout in APPROACH_STROKE_0 at index 42
- c3a528aafbfd7b55: Waypoint timeout in DRAW_STROKE_0 at index 366
- c3b05340ef29131f: Waypoint timeout in DRAW_STROKE_0 at index 1
- 1163bceaf6134cf5: Waypoint timeout in DRAW_STROKE_0 at index 561
- 323bd30d42ee2cf3: Waypoint timeout in DRAW_STROKE_0 at index 44
- 4651e893ac2a26d4: Waypoint timeout in APPROACH_STROKE_0 at index 20
- 4a1ad0b44c7f462c: Waypoint timeout in APPROACH_STROKE_0 at index 13
- 4a933ea53cf400bc: Waypoint timeout in DRAW_STROKE_0 at index 360
- 69fff8521d33da91: Waypoint timeout in APPROACH_STROKE_0 at index 25
- 70bc30f8a5f918eb: Waypoint timeout in APPROACH_STROKE_0 at index 16
- 959e69e03f23a3a3: Waypoint timeout in APPROACH_STROKE_0 at index 22
- ab7beb0372c96b9c: Waypoint timeout in DRAW_STROKE_0 at index 127
- d190d0c5f8107942: Waypoint timeout in APPROACH_STROKE_0 at index 18
- e36c22e2336311bc: Waypoint timeout in APPROACH_STROKE_0 at index 21
- efa93c03483769d5: Waypoint timeout in DRAW_STROKE_0 at index 531
- 07379e952b3f0513: Waypoint timeout in DRAW_STROKE_0 at index 505
- 0b763b6b700d128e: Waypoint timeout in APPROACH_STROKE_0 at index 15
- 14ab42e1e62789e2: Waypoint timeout in DRAW_STROKE_0 at index 334
- 273c3abeece5d4b2: Waypoint timeout in APPROACH_STROKE_0 at index 43
- 4cf9d2757b8a6282: Waypoint timeout in DRAW_STROKE_0 at index 299
- 728ed437fdddba4c: Waypoint timeout in APPROACH_STROKE_0 at index 22
- 96ab6db6d9116afe: Waypoint timeout in DRAW_STROKE_0 at index 403
- 98980e7442f4a93c: Waypoint timeout in APPROACH_STROKE_0 at index 20
- 9c942e2bb316cdb6: Waypoint timeout in DRAW_STROKE_0 at index 297
- cd5042b9731161f9: Waypoint timeout in APPROACH_STROKE_0 at index 10
- e2fecb06cd346683: Waypoint timeout in APPROACH_STROKE_0 at index 22
- ea59d6637578c29d: Waypoint timeout in DRAW_STROKE_0 at index 533
- ed3fc2b8d2e88c0d: Waypoint timeout in APPROACH_STROKE_0 at index 13
- a7cbe5e2b0f491bb: Waypoint timeout in DRAW_STROKE_0 at index 10
- ba3e1657db195fe4: Waypoint timeout in DRAW_STROKE_0 at index 19
- ce7b60c8b362b074: Waypoint timeout in DRAW_STROKE_0 at index 322
- dd305a2886cbf23c: Waypoint timeout in APPROACH_STROKE_0 at index 30
- 3f221bad695edcf8: Waypoint timeout in APPROACH_STROKE_0 at index 14
- 6bbd7fa65eede66e: Waypoint timeout in DRAW_STROKE_0 at index 117
- 8ae3adc42375beba: Waypoint timeout in APPROACH_STROKE_0 at index 13
- a9263f399b2c5dd0: Waypoint timeout in DRAW_STROKE_0 at index 94
- aef94e7551606d10: Waypoint timeout in DRAW_STROKE_0 at index 378
- 165c17090deb8a48: Waypoint timeout in DRAW_STROKE_0 at index 509
- 2b22d42510450af1: Waypoint timeout in APPROACH_STROKE_0 at index 33
- 7c54b4c75832b3b9: Waypoint timeout in DRAW_STROKE_0 at index 432
- a84882576126b982: Waypoint timeout in APPROACH_STROKE_0 at index 20
- e659138f70574103: Waypoint timeout in APPROACH_STROKE_0 at index 20
