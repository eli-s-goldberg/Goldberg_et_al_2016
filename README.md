## Goldberg_et_al_2016: brute force determination of the optimal decision tree

## What does this application do?
This program imports a developed database, converts it to dimensionless fluid dynamics and colloidal interaction parameters, and then trains a decision tree based on those features. 

## What should you remember? 
Here it is important to remember that we're not evaluating the generic performance of the decision tree to predict the data outcome. We're using the decision tree as a method to quantitatively investigate and breakdown the data... to see if we can disentangle the relationship between physicochemical parameters and the retention behavior of nanomaterials.

Note that, becuase there are elements of stochasticity in the decision tree growing process (which is wrapped up in the selection of features (max_features = sqrt(nfeatures)), it can be difficult to obtain the optimal decision tree (n.b. for most cases, there is never an optimal decision tree). Here, we cannot promise optimality, but we can iteratively investigate the results of the decision tree and pick the best one.

Here we employ 5-10,000 decision tree runs and report the index of the best one. This index value should be used to declare
the location of the output hierarchical JSON file (flareXX.json), which is output to the figures/decisionTreeVisualization/flare_reports folder. Once the index has been located, modify the appropriate variable in the index.html file contained within teh decisionTreeVisualization folder and run it. This should present the tree in all it's glory:)
