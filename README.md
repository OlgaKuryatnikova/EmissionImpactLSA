# EmissionImpactLSA
Python code for Unintended Carbon Impacts of Electricity Storage and Demand Management
There are 3 different codes:
*	Parameters per plant: It gives the modification done from the original database of Dutch generation facilities that is used in our optimization, the output of this code is the file 'Parameters_plants_2019.csv' and 'Parameters_plants_2022.csv' which is already uploaded in the folder source
* Optimisation_sp : It contains a function to optimize the social welfare of a certain market situation, we use the mosek solver
* Graph: uses the results from the code optimization to create results and graphs from the paper.

The folder should be organize as follow in your userpath,

Data :
* result (some results are store from the graph file)
* result_optimisation : (all files from here come from running the function optimisation_sp)
    * All
    * new_file
    * constraint:
      * ramp
 * source (file given)
  
