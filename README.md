# timeseries_analysis
Code for unsupervised clustering of time-series data. Reference to https://arxiv.org/abs/2402.07786 for further details. 

## Input data
A one-dimensional time-series, computed on *N* particles for *T* frames. The input files must contain an array with shape *(N, T)* Supported formats: .npy, .npz, .txt. Also .xyz trajectories are supported, with the fifth column containing the data values. 

## Usage
Install the package using `pip install onion_clustering`. 

The `examples/` folder contains an example of usage. Run `python3 example_script.py`, this will create the following files:
* A text file called `input_parameters.txt` , whose format is explained below;
* A text file called `data_directory.txt` containing one line with the path to the input data file (including the input data file name); 
and run the code. 

## input_parameters.txt 
* `tau_window` (int): the length of the time window (in number of frames). 
* `t_smooth` (int, optional): the length of the smoothing window (in number of frames) for the moving average. A value of `t_smooth = 1` correspond to no smoothing. Default is 1. 
* `t_delay` (int, optional): is for ignoring the first tau_delay frames of the trajectory. Default is 0. 
* `t_conv` (int, optional): converts number of frames in time units. Default is 1. 
* `time_units` (str, optional): a string indicating the time units. Default is `'frames'`.  
* `example_ID` (int, optional): plots the trajectory of the molecule with this ID, colored according to the identified states. Default is 0. 
* `bins` (int, optional): the number of bins used to compute histograms. This should be used only if all the fits fail with the automatic binning. 
* `num_tau_w` (int, optional): the number of different tau_window values tested. Default is 20. 
* `min_tau_w` (int, optional): the smaller tau_window value tested. It has to be larger that 1. Default is 2. 
* `max_tau_w` (int, optional): the larger tau_window value tested. It has to be larger that 2. Default is the largest possible window. 
* `min_t_smooth` (int, optional): the smaller t_smooth value tested. It has to be larger that 0. Default is 1. 
* `max_t_smooth` (int, optional): the larger t_smooth value tested. It has to be larger that 0. Default is 5. 
* `step_t_smooth` (int, optional): the step in the t_smooth values tested. It has to be larger that 0. Default is 1. 

## Output
The algorithm will attempt to perform the clustering on the input data, using different `t_smooth` (from `min_t_smooth` frames to `max_t_smooth` frames, with steps of `step_t_smooth`) and different `tau_window` (logarithmically spaced between 2 frames and the entire trajectory length, unless differently specified in the input parameters). The results are saved in the folowing files:

* `number_of_states.txt` contains the number of clusters for each combination of `tau_window` and `t_smooth` tested. 
* `fraction_0.txt` contains the fraction of unclassified data points for each combination of `tau_window` and `t_smooth` tested. 
* Figures with all the Gaussian fittings are saved in the folder `output_figures` with the format `t_smooth_tau_window_Fig1_iteration.png`. 

Then, the analysis with the values of `tau_window` and `t_smooth` specified in `input_parameters.txt` will be performed. The results are saved in the folowing files:

* `states_output.txt` contains information about the recursive fitting procedure, useful for debugging. 
* `output_figures/Fig1_iteration.png` plot the histograms and best fits for each iteration. 
* `final_states.txt` contains the list of the states, for which central value, width and relevance are listed. 
* `final_tresholds.txt` contains the list of the tresholds between states. 

The analisys returns a `ClusteringObject`, which contains methods for plotting all the results. They are listed in the example scripts. 

## Multivariate time-series version
The `main_2d.py` algorithm works in a similar fashion, taking as input 2D or 3D data. The input file contained in `data_directory.txt` must contain an array of shape `(D, N, T)` where _D_ is the number of components. Only `.npy, .npz` are supported. You can find an example of usage in `examples/example_script_2d.py`

## Required Python 3 packages
`matplotlib`, `numpy`, `plotly`, `scipy`. 

## Gaussian fitting procedure
1. The histogram of the time-series is computed, using the `bins='auto'` numpy option (unless a different `bins` is passed as imput parameter). 
2. The histogram is smoothed with moving average with a window proportional to the number of bins (unless there are less that 50 bins, in wich case no smoothing occurs). 
3. The absolute maximum of the histogram is found. 
4. Two Gaussian fits are performed:
 * The first one inside the interval between the two minima surrounding the maximum. 
 * The second one inside the interval where the peak around the maxima has its half height. 
5. Both fits, if converged, are evaluated according to the coefficinet of determination r^2. 
6. Finally, the fit with the best score is chosen. If only one of the two converged, that one is chosen. If none of the fits converges, the iterative procedure stops, returning a warning message. 

## Aknowledgements
Thanks to Andrew Tarzia for all the help with the code formatting and documentation, and to Domiziano Doria, Chiara Lionello and Simone Martino for the beta-testing. Writing all this code wouldn't have been possible without the help of ChatGPT. 
