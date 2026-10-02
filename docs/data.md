# Data format

Input is a local pandas DataFrame serialized with `DataFrame.to_pickle()`. Load only files you trust. Required columns are `userID`, `startTime`, `origin`, and `destination`; other columns are ignored.

- Use consistent types within each identifier column. User IDs and station IDs must be sortable.
- Use a shared station encoding for origin and destination. Index 0 is reserved internally for padding; input station IDs are mapped automatically.
- Use nonmissing, timezone-naive departure timestamps in the city's local time. Convert timezone-aware timestamps before serialization.
- Remove invalid station labels and origin-equals-destination records. Departure records are sorted within each user. The strict protocol requires distinct departure timestamps within each user; duplicate times are rejected.
- The strict implementation requires every retained station to occur in training users' history. It raises an error on unseen validation/test stations instead of adding them from future data.

## Windows and users

`history-end` and `future-start` define nonoverlapping history and evaluation windows. An eligible user has at least `min-trips` history records and one future record. Users are partitioned 80%/10%/10% with a fixed split seed. Train targets are history events; validation and test targets are future events. Earlier observed future trips can serve as context for later predictions.

The Guangzhou input used in the manuscript was filtered upstream to users with at least 55 total records. Reproducing that cohort requires the same upstream input filtering in addition to the configured history threshold. Original datasets and their redistribution permissions are not provided by this repository.

## Optional topology

Supply an NPZ containing:

| Array | Shape | Meaning |
| --- | --- | --- |
| `raw_station_id` | `(N,)` | Raw station IDs matching the trip data |
| `shortest_hops` | `(N+1, N+1)` | Undirected shortest-hop distances, with index 0 reserved for padding |

`raw_station_id[i]` corresponds to row/column `i+1` in `shortest_hops`. Distances must be symmetric, diagonal entries zero, and all observed station pairs reachable. Distance 1 identifies adjacent stations. The graph adapter uses those adjacent pairs for one residual normalized-neighbor propagation step on the separate origin and destination embedding tables. It does not constrain destinations by a hop cutoff.
