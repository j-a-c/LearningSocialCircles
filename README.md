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
- -p Predict social circles.
- -v Visualize data. By default uses original topology to construct graphs.
- --split Split visualizations by circle.
- --edge Select the edge function to use when visualizing data. Supported
  options are: 
  * top: Uses the original graph topology.
  * sim: Creates edges between similar users.
