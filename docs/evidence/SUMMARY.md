# Evidence summary

Regenerate with `python scripts/evidence.py` and `scripts/time_original.py`.

## Training speed on one computer (CS 370 maze, seed 1671)

| Code | TensorFlow / Keras | Games | Moves | Seconds | ms per move | Cells won | Shortest routes |
|---|---|---|---|---|---|---|---|
| CS 370 notebook | 2.17.0 / 3.5.0 | 384 | 27,978 | 179.9 | 6.43 | 50 of 50 | 50 |
| pirate_agent 2.1.0 | 2.21.0 / 3.15.1 | 296 | 20,026 | 18.0 | 0.9 | 50 of 50 | 50 |
| pirate_agent 2.1.0 | 2.17.0 / 3.5.0 | 366 | 30,380 | 26.9 | 0.88 | 50 of 50 | 50 |

## Ten seeds on the CS 370 maze

| Seed | Stop | All cells won at game | Games | Moves | Seconds | Cells won | Shortest routes (first all-win, kept) |
|---|---|---|---|---|---|---|---|
| 1671 | solved | 286 | 296 | 20,026 | 16.3 | 50 of 50 | 47, 50 |
| 1 | solved | 282 | 417 | 23,327 | 18.6 | 50 of 50 | 44, 50 |
| 2 | solved | 393 | 444 | 31,680 | 28.4 | 50 of 50 | 44, 50 |
| 3 | solved | 247 | 278 | 18,836 | 15.6 | 50 of 50 | 33, 50 |
| 4 | solved | 282 | 282 | 22,282 | 17.8 | 50 of 50 | 50, 50 |
| 5 | solved | 273 | 279 | 20,939 | 16.1 | 50 of 50 | 49, 50 |
| 6 | solved | 267 | 267 | 22,371 | 17.8 | 50 of 50 | 50, 50 |
| 7 | solved | 323 | 323 | 21,396 | 17.9 | 50 of 50 | 50, 50 |
| 8 | solved | 273 | 505 | 27,587 | 23.9 | 50 of 50 | 44, 49 |
| 9 | solved | 317 | 320 | 20,460 | 16.8 | 50 of 50 | 49, 50 |

Solved 10 of 10; every cell won in 10; median 308 games (range 267 to 505); mean 18.9 s per run.

## Early stopping on a run that cannot learn

Learning rate 1e-09, patience 300: stopped as stalled after 374 games (26.3 s) instead of up to 15,000; best win rate 0.344.

## A generated maze

```
#..##.##
......#.
.....#..
..#..#..
#..#.#.#
##.....#
#...##..
..###..T
```

42 open cells, seed 11: solved after 226 games in 6.4 s; 41 of 41 cells won, 41 by a shortest route.

## Replay memory: nanoseconds per insert into a full memory

| Capacity | list with del [0] | deque(maxlen) | Speedup |
|---|---|---|---|
| 512 | 81 | 29 | 2.8x |
| 10,000 | 753 | 29 | 25.6x |
| 100,000 | 8,207 | 30 | 273.1x |

## Dependency audit (pip-audit)

CS 370 pins (numpy==1.26.0, tensorflow==2.17.0, keras==3.5.0): 19 advisories.
- keras 3.5.0: 18 (CVE-2025-12060, PYSEC-2025-121, PYSEC-2025-122, PYSEC-2025-123, PYSEC-2025-75, PYSEC-2025-76, PYSEC-2026-1487, PYSEC-2026-2324, PYSEC-2026-2546, PYSEC-2026-2547, PYSEC-2026-3629, PYSEC-2026-3630, PYSEC-2026-3631, PYSEC-2026-3632, PYSEC-2026-3633, PYSEC-2026-3634, PYSEC-2026-3856, PYSEC-2026-73)
- protobuf 4.25.9: 1 (PYSEC-2026-1805)

Current pins (numpy==2.5.4, tensorflow==2.21.0, keras==3.15.1): 0 advisories.
