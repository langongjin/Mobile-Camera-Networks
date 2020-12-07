# Mobile-Camera-Networks


## config.py
parameters configuration in camera networks.

## display.py
Generating sequence image depending on the situation of cameras and objects. We recommend only process the results of one method once.

## compare.py
First, select the task to be processed (line 47), and then select the method (line 94). The results and images will be printed.


## utils.py: 
- baseline: the function called as permutation, namely, exhaustive algorithm
- round robin : the reciprocating movement with a period of 16 seconds(displacement and rotation）
- cluster_greedy:  calculating the parameters with the combination of mean shift and greedy. When the bandwidth (h=0) in mean shift, it is approximately pure greedy
- herd_greedy_search: calculating the movement for all objects in the coverage. 

## hungarian.py
The configuration with K-means and hungarian algorithm.

## Dependencies
- matplotlib
- scikit-learn
