# timeseries_analysis
Code for unsupervised clustering of time-correlated data. Reference to INSERT PAPER for further details. 

## Input data
A one-dimensional timeseries, computed on *N* particles for *T* frames. The input files must contain an array with shape *(N, T)* Supported formats: .npy, .npz, .txt.

## Usage
Install the package using `pip install onion_clustering`. 

The `examples/` folder contains an example of usage. Run `python3 example_script.py`, this will create the following files:
* A text file called `input_parameters.txt` , whose format is explained below;
* A text file called `data_directory.txt` containing one line with the path to the input data file (including the input data file name). 
and run the code. 

## input_parameters.txt
The keyword and the value must be separated by one tab. 
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
The algorithm will attempt to perform the clustering on the input data, using different `t_smooth` (from 1 frame, i.e no smoothing, to 5 frames, unless differently specified in the input parameters) and different `tau_window` (logarithmically spaced between 2 frames and the entire trajectory length, unless differently specified in the input parameters). 

* `number_of_states.txt` contains the number of clusters for each combination of `tau_window` and `t_smooth` tested. 
* `fraction_0.txt` contains the fraction of unclassified data points for each combination of `tau_window` and `t_smooth` tested. 
* `output_figures/Time_resolution_analysis.png` plots the previous two data, for the case `t_smooth = 1`. 
* Figures with all the Gaussian fittings are saved in the folder `output_figures` with the format `t_smooth_tau_window_Fig1_iteration.png`. 

Then, the analysis with the values of `tau_window` and `t_smooth`  specified in `input_parameters.txt` will be performed. 

* The file `states_output.txt` contains information about the recursive fitting procedure, useful for debugging. 
* The file `final_states.txt` contains the list of the states, for which central value, width and relevance are listed. 
* The file `final_tresholds.txt` contains the list of the tresholds between states. 
* `output_figures/Fig0.png` plots the raw data. 
* `output_figures/Fig1_iteration.png` plot the histograms and best fits for each iteration. 
* `output_figures/Fig2.png` plots the data with the clustering thresholds and Gaussians. 
* `output_figures/Fig3.png` plots the colored signal for the particle with `example_ID` ID. 
* `output_figures/Fig4.png` shows the mean time sequence inside each state, and it's useful for checking the meaningfulness of the results. 
* The file `all_cluster_IDs_xyz.dat` allows to plot the trajectory using the clustering for the color coding. Altough, they are not super friendly to use. 
* If the trajectory from which the signal was computed is present in the working directory, and called `trajectory.xyz`, a new file, `colored_trj.xyz` will be printed, with the correct typing according to the clustering. But a bit of fine-tuning will be necessary inside the function `print_colored_trj_from_xyz()` in `function.py`. 

## Multivariate time-series version
The `main_2d.py` algorithm works in a similar fashion, taking as input 2D or 3D data. Each component of the signal has to be loaded with its own input data; just add one line with the path to the files to `data_directory.txt`. Signals are normalized between 0 and 1; changing this can change the performance of the algorithm, so you may want to try the clustering with different normalizations. You can find an example of usage in `examples/example_script_2d.py`

## Required Python 3 packages
`matplotlib`, `numpy`, `plotly`, `scipy`. 

## Gaussian fitting procedure
1. The histogram of the timeseries is computed, using the `bins='auto'` numpy option (unless a different `bins` is passed as imput parameter). 
2. The histogram is smoothed with moving average with `window_size=3` (unless there are less that 50 bins, in wich case no smoothing occurs). 
3. The absolute maximum of the histogram is found. 
4. Two Gaussian fits are performed:
 * The first one inside the interval between the two minima surrounding the maximum. 
 * The second one inside the interval where the peak around the maxima has its half height. 
5. Both fits, if converged, are evaluated according to the following points:
 * `mu` is contained inside the fit interval;
 * `sigma` is smaller than the fit interval;
 * the height of the peak is at least half the value of the maximum;
 * the relative uncertanty over the fit parameters is smaller than 0.5.
6. Finally, the fit with the best score is chosen. If only one of the two converged, that one is chosen. If none of the fits converges, the iterative procedure stops, returning a warning message. 

## Aknowledgements
The comments in the code wouldn't have been possible without the help of ChatGPT. 