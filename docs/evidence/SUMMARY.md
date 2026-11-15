# Evidence summary

Regenerate with `python scripts/evidence.py` and `scripts/time_original.py`.

## Training speed on one computer (CS 370 maze, seed 1671)

| Code | TensorFlow / Keras | Games | Moves | Seconds | ms per move | Cells won | Shortest routes |
|---|---|---|---|---|---|---|---|
| CS 370 notebook | 2.17.0 / 3.5.0 | 384 | 27,978 | 179.9 | 6.43 | 50 of 50 | 50 |
| pirate_agent 2.0.0 | 2.21.0 / 3.15.1 | 286 | 19,843 | 22.4 | 1.13 | 50 of 50 | 47 |
| pirate_agent 2.0.0 | 2.17.0 / 3.5.0 | 359 | 30,259 | 35.2 | 1.16 | 50 of 50 | 49 |

## Ten seeds on the CS 370 maze

| Seed | Stop | Games | Moves | Seconds | Cells won | Shortest routes |
|---|---|---|---|---|---|---|
| 1671 | solved | 286 | 19,843 | 21.0 | 50 of 50 | 47 |
| 1 | solved | 282 | 20,896 | 20.6 | 50 of 50 | 44 |
| 2 | solved | 393 | 30,638 | 31.5 | 50 of 50 | 44 |
| 3 | solved | 247 | 18,148 | 19.2 | 50 of 50 | 33 |
| 4 | solved | 282 | 22,282 | 23.2 | 50 of 50 | 50 |
| 5 | solved | 273 | 20,827 | 19.9 | 50 of 50 | 49 |
| 6 | solved | 267 | 22,371 | 22.7 | 50 of 50 | 50 |
| 7 | solved | 323 | 21,396 | 28.6 | 50 of 50 | 50 |
| 8 | solved | 273 | 23,450 | 21.5 | 50 of 50 | 44 |
| 9 | solved | 317 | 20,418 | 23.8 | 50 of 50 | 49 |

Solved 10 of 10; every cell won in 10; median 282 games (range 247 to 393); mean 23.2 s per run.

## Early stopping on a run that cannot learn

Learning rate 1e-09, patience 300: stopped as stalled after 374 games (30.7 s) instead of up to 15,000; best win rate 0.344.

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

42 open cells, seed 11: solved after 175 games in 8.2 s; 41 of 41 cells won, 39 by a shortest route.

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
