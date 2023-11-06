# timeseries_analysis
Code for unsupervised clustering of time-correlated data. Reference to INSERT PAPER for further details. 

## Input data
A one-dimensional timeseries, computed on *N* particles for *T* frames. The input files must contain an array with shape *(N, T)* Supported formats: .npy, .npz, .txt.

## Usage
The working directory must contain:
* A text file called `input_parameters.txt` , whose format is explained below;
* A text file called `data_directory.txt` containing one line with the path to the input data file (including the input data file name). 

Examples of this two files are contained in this repository. 

From this directory, the code is run with `python3 ${PATH_TO_CODE}/main.py`. 

## input_parameters.txt
* `tau_window`: the length of the time window (in number of frames). 
* `t_smooth`: the length of the smoothing window (in number of frames) for the moving average. A value of `t_smooth = 1` correspond to no smoothing. 
* `tau_delay`: is for ignoring the first tau_delay frames of the trajectory. By default, use 0. 
* `t_conv`: converts number of frames in time units. By default, use 1. 
* `time_units`: a string indicating the time units (e.g., `ns`). 
* `example_ID`: plots the trajectory of the molecule with this ID, colored according to the identified states. By default, use 0. 
* An **optional** parameter, `bins`, the number of bins used to compute histograms. This should be used only if all the fits fail with the automatic binning. 

## Output
The algorithm will attempt to perform the clustering on the input data, using different `t_smooth` (from 1 frame, i.e no smoothing, to 10 frames) and different `tau_window` (logarithmically spaced between 2 frames and the entire trajectory length). The output of this analysis will be saved in the files `number_of_states.txt` and `fraction_0.txt`, containing respectively the number of states identified and the fraction of unclassified data points for each choice of `tau_window` and `t_smooth`. Finally, the output is summarized in the figure `Time_resolution_analysis.png`. Figures with all the Gaussian fittings are saved in the folder `output_figures` with the format `t_smooth_tau_window_Fig1_iteration.png`. 

Then, the analysis with the values of `tau_window` and `t_smooth`  specified in `input_parameters.txt` will be performed. The output figures will be saved in the folder `output_figures`. The file `final_states.txt` contains the list of the states, for which central value, width and relevance are listed. The file `final_tresholds.txt` contains the list of the tresholds between states. The file `states_output.txt` contains information about the recursive fitting procedure, useful for debugging. The file `all_cluster_IDs_xyz.dat` allows to plot the trajectory using the clustering for the color coding. Altough, they are not easy to use. Maybe this will be improved. Sorry. 

## Required Python 3 packages
`matplotlib`, `numpy`, `os`, `plotly`, `scipy`, `seaborn`, `sys`. 

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