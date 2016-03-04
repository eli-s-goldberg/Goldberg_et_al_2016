## Goldberg_et_al_2016: brute force determination of the optimal decision tree

## What does this application do?
This program imports a developed database, converts it to dimensionless fluid dynamics and colloidal interaction parameters, and then trains a decision tree based on those features. 

## What should you remember? 
Here it is important to remember that we're not evaluating the generic performance of the decision tree to predict the data outcome. We're using the decision tree as a method to quantitatively investigate and breakdown the data... to see if we can disentangle the relationship between physicochemical parameters and the retention behavior of nanomaterials.

Note that, becuase there are elements of stochasticity in the decision tree growing process (which is wrapped up in the selection of features (max_features = sqrt(nfeatures)), it can be difficult to obtain the optimal decision tree (n.b. for most cases, there is never an optimal decision tree). Here, we cannot promise optimality, but we can iteratively investigate the results of the decision tree and pick the best one.

Here we employ 5-10,000 decision tree runs and report the index of the best one. This index value should be used to declare
the location of the output hierarchical JSON file (flareXX.json), which is output to the figures/decisionTreeVisualization/flare_reports folder. Once the index has been located, modify the appropriate variable in the index.html file contained within teh decisionTreeVisualization folder and run it. This should present the tree in all it's glory:)

### Note to Pete
Hey buddy, here's the deal. 
1. optimal_tree.py should be run first. The number of iterations is set at 50 right now. This will create 50 decision trees from the data. The random element in the creation of the trees comes from line 214 within the brute force grid search for the best parameters. When you set the max_features to 'None', the recursive splits choose from among all the features. As such, if you remove the 'sqrt' and 'log2' then all of the trees will be exactly the same. However, this actually does not result in the optimal tree (for a lot of reasons that are not important). Running this script creates several output files.  - targetdata.csv: a csv  of the target data (i.e., a list of exponential or nonexponential responses) 
 - trainingdata.csv: a csv of the training data (i.e., a  explanatory variables as columns, experiments as rows)
 - n # of classification performance reports corresponding to n iterations. These are in            './figures/decisionTreeVisualization/class_reports'
 - n # of .json files in with hierarchical structure sufficient to describe the decision tree. These are in './figures/decisionTreeVisualization/flare_reports. The logic for this is contained in the helper_functions.py 'rules' function (257-301). 
 - A .csv of the f1 scores for the exponential, nonexponential, and weighted average. The name of the .csv corresponds to the number of iterations e.g., DecisiontreeScores50.csv
 - A print in the python terminal of the optimal decision tree (e.g,  33). Note that this corresponds to the flare report for the next flare (i.e., best run is 33 = flare34.json). 
2. figures/index.html should be run next. Within the index.html file, the name of the best performing flare as identified by step 1 should be entered in line 43 (d3.json("./flare_reports/flare34.json", function ( ... 
 - This should pop open a browser window with an svg that can be ripped using SVG crowbar (http://nytimes.github.io/svg-crowbar/). 

### Miscellaneous useful files: 
 - class_test.py: this is a terribly programmed class file that I created to translate the dimensioned measured physicochemical conditions into dimensionless parameters. It works, kinda...there are some bugs. 
 - csv2flare2.py: this is janky script that turns a csv organized by columns into a hierarchical json that can be visualized. This script is quite valuable and, as far as I can tell, is the only available script to do this without a tremendous headache. I've created a generic class function for this because I use it elsewhere (it's not in this git), but it is critical to visualizing the database in several formats. 
 - histogramVisualization.py: this script is used to make histograms that compare the number of experiments with exponential (blue) or nonexponential (orange) retention over each individual physicochemical parameter's observed range. These are saved in the figures/histograms folder. Notice that we do a bit of regex'ing (cheap, poor, poor man's regex'ing), and because I don't care to use the tmp package I manually hold on to the temp files. These temp files are located in  figures/histograms/tmp. The ranges and binning are tuned for this application, but I had some trouble with log bins when the values were low (<1e-3). As a consequence, you'll see that I multiply by 1e11 on line 307 to fix the gravitational number binning (N_g). If you could find out why this happens, that would be awesome. 
 - CV_calculate.py is used to iterate through a csv containing experiments in isolated branches to determine the relative coefficient of variation (std/mean). This is important to understanding the diveristy of experimentation contained within the branch. This also creates a data file 'RadardData.txt' that contains a json string that can be copied and pasted into the ./figures/radarCharts/*RadarCharts.html files -- you'll see the var data = [ COPY ] starting on line 49. If you then run this file, you'll make a live radar chart. 
