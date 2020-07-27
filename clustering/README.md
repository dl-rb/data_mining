# Clustering  using Bradley-Fayyad-Reina (BFR) algorithm
Implementation of BFR Clustering using PySpark
### Input Dataset:
- Initializing points from random centroids with random standard deviations.
- Randomly generate some noise data

Example
|data point index   | index of cluster  | data vector
|:------------------|:------------------|:--------------
|0                  |8                  |-127.64433989643463,-112.93438512156577,-123.4960457961025,114.4630547261514,121.64570029890073,-119.54171797733461,109.9719289517553,134.23436237925256,-117.61527240771153,120.42207629196271
|1                  |4                  |-38.191305707322314,-36.739481055180704,-34.47221450208468,33.640148757948026,-53.27570482090691,59.21790911677368,53.15109003438039,36.75210113936672,28.951427009179213,41.41404989722435
|2                  |0                  |194.1751258951049,-214.78572302878496,-199.46759003279055,195.93731866970583,209.634197754483,-192.44259634358372,202.62698763813447,209.16045543699823,197.6554195934683,-202.04341278850256
|3                  |6                  |-36.018560440437376,40.58411243751584,55.96250080682364,47.5720753795009,-56.61561738372609,-54.944502337157715,-42.84314857713225,-28.76477463042852,-29.123766956654677,-59.3528832139923
### Output
- Intermediate results: info in each round: the number of the clusters in the compression set, the number of the compression points, and the
number of the points in the retained set
    - Example: 
        - Round 1: 227, 5236, 20, 258
        - Round 2: 2327, 236, 20, 58
        - ...
- Clustering results: point index, cluster index
    - Example:
        - 0,1
        - 1,1
        - 2,-1
        - ...
