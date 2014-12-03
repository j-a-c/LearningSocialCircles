LearningSocialCircles
=====================

Implementation for the Kaggle competition "Learning Social Circles in
Networks".

# Usage

Run python driver.py -h for help and options.

Flags:

- -s Compute statistics on data.
- --trim Output data common per egonet. This set will be ignored when
  displaying intersection attributes.
- -p Predict social circles using the specified predictor. Supported
  predictors:
  * igraph - Use various community detection algorithms from the igraph
    package. Can be combined with --edge parameter.
  * kmeans - K-means clustering
  * mcl - Markvoc clustering algorithm. Can be combined with --edge parameter.
- -v Visualize data. By default uses original topology to construct graphs.
- --split Split visualizations by circle.
- --save Save output. Graphical output is saved to the folder 'graphs' in the
  current directory.
- --show Show output during visualization calculations.
- --edge Select the edge function to use when visualizing data. Supported
  options are: 
  * top: Uses the original graph topology.
  * top-intersect: Uses original graph topology with a minor weight given to
    attributes that are in common.
  * sim: Creates edges between similar users.
  * tri: Creates edges between users with friends in common.
  * combo: Uses both 'tri' and 'sim' to create edges.
- --prune Select a post-cluster-prediction pruning method. Supported options are:
  * small: Remove small clusters.
