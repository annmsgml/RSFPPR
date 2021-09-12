# Single Source Personalized PageRank with Loop-erased Walks

## Compile
```sh
sh compile.sh
```

## Parameters
```sh
./RSFPPR --algo [algo] -o [options]
```
- algo:
    - GEN_QUERY: generate query node file
    - GEN_GROUND_TRUTH: generate query node groundtruth by multi-thread power method
    - RSFPPR: personalized Pagerank query
- options
    - -d \<dataset\>
    - -k \<top k\>
    - -n \<query size\>
    - -o \<option\>
        2 query performance of combining forward search and Monte Carlo sampling
        3 compare the empirical variance of two loop-erased walk based estimators
        4 compare the running time of simple walk and loop-erased walk

## Generate queries
Generate query files for the graph data. 

- Example:

```sh
$ sh gen_query.sh
```

## Generate groundtruth of query nodes

- Example:

```sh
$ sh gen_groundtruth.sh
```

## Exact Query

- Example:

```sh
./RSFPPR -d dblp -algo Abs -n 6 -o 3
```
