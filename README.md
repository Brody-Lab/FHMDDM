# factorial hidden Markov drift-diffusion model (FHMDDM)

The code for fitting the model and computing quantities to characterizing the model is in the programming language Julia, whereas the code for plotting the said quantities are in MATLAB. 

# table of contents
* [tutorial](#tutorial)
  * [installing the FHMDDM repository](#installing-the-fhmddm-repository)
  * [specifying the model options](#specifying-the-model-options)
  * [fitting the model](#fitting-the-model)
  * [examining the results](#examining-the-results)
    * [PSTH](#plotting-the-peri-stimulus-time-histogram-psth)
    * [parameters](#examining-the-parameters)
* [`Model` type](#model-composite-type)
  * [fixed options](#fixed-hyperparameters-model-options)
  * [data](#data-model-trialsets)
  * [parameters](#parameters)
# tutorial
The following tutorial shows how to fit the model to a recording session on `spock`, the Princeton Neuroscience Institute's (PNI) computation cluster, and the visualizing the results on a Windows machine

In a shell, log into `spock`:
```
> ssh <netID>@spock.princeton.edu
```
After providing credentials, load a version of Julia and run it
```
[<netID>scotty ~]$ module load julia\1.6.0
[<netID>scotty ~]$ julia
julia>
```
## installing the FHMDDM repository
In the Julia read-eval-print loop (REPL), enter the Pkg REPL by pressing `]` from the Julia REPL. To get back to the Julia REPL, press `backspace` or `^C`.
```
julia> ]
(v1.6) pkg> 
```
Add the `FHMDDM` repository
```
(v1.6) pkg> add https://github.com/Brody-Lab/FHMDDM.git
pkg> up
pkg> <backspace>
julia>
```
Check whether the `FHMDDM` repository has been loaded. If information is requested for `Model`, the prompt below should be seen. `Model` is both a composite type as well as the name of a function. 
```
julia> using FHMDDM
julia> ?
help?> Model
Model

  A factorial hidden Markov drift-diffusion model

  ───────────────────────────────────────────────────────────

  Model(datapath; fit_to_choices)

  Load a factorial hidden Markov drift diffusion model from a MATLAB file.

  If the model has already been optimized, a results file is expected.

  ARGUMENT

    •  datapath: full path of the data file

  RETURN

    •  a structure containing information for a factorial hidden Markov drift-diffusion model
```
If an error occurs and requires recompilation, Julia must be restarted.

##  specifying the model options
The fixed hyperparameters of model are documented in a comma-separated values (CSV) file named `options.csv`: An example can be seen [here](https://github.com/Brody-Lab/tzluo/blob/master/analyses/analysis_2023_02_08b_example/options.csv). Each column corresponds to a hyperparameter, and each row corresponds to a separate model. Depending on the goals of the analysis separate models can be fit to the same or different recording sessions. In this example, a single model is fit to the recording session `T176_2018_05_03`. For description of each model option, see [types.jl](/src/types.jl).

## fitting the model
We will prepare two scripts, a Julia script for fitting the model, and a shell script to interact with the computational cluster's job scheduler and to run the Julia script. The typical Julia script is:

```
# load the code
using FHMDDM

# the absolute path of the data is provided as an input from the shell script
datapath = ARGS[1]

# initialize a composite object containing the data, parameters, and hyperparamters
model = Model(datapath)

# perform the initial stage of optimization
learnparameters!(model)

# save a summary of the model, as well as compute and save quantities useful for characterizing the model. The first argument specifies not to compute the Hessian matrix, which can take a long time. The second argument specifies the name of the folder that will be created to contain the summary and the analyses, within the folder that contains the data. The absolute path of the folder is given by `joinpath(dirname(model.options.datapath), results)`
analyzeandsave(false, "results", model)

# simulate data based on the learned parameters
samplepaths = simulateandsave(model,1)
simulation = Model(samplepaths[1])

# recover parameters by fitting to simulated data
learnparameters!(simulation)
analyzeandsave(false, "recovery", simulation)
```

* [example Julia script](https://github.com/Brody-Lab/tzluo/blob/master/analyses/analysis_2023_02_08b_example/optimize.jl)
* [example shell script](https://github.com/Brody-Lab/tzluo/blob/master/analyses/analysis_2023_02_08b_example/optimize.sh) that calls the Julia script. 

Once the scripts are prepared, run the Julia script on the cluster by calling `sbatch`. For the example above, the following would be called:
```
sbatch /usr/people/zhihaol/Documents/tzluo/analyses/analysis_2023_02_08a_choiceLL_scaling_exponent/optimize.sh
```
## examining the results
Code for plotting the results was written in MATLAB 2019a. To use the plotting tools, add the folder `/plotting` into MATLAB's search path. First, let's tabulate the models and their options. Here we are assuming that the MATLAB script is saved in the same folder where the `options.csv` is located:
```
>> analysispath = fileparts(matlab.desktop.editor.getActiveFilename)
>> T = FHMDDM.tabulateoptions(analysispath)
T =

  1×131 table
```
### plotting the peri-stimulus time histogram (PSTH)
First, let's plot the choice-conditioned PSTH of the third neuron in the first [trialset](#data-model-trialsets) (there is only one trialset). Only the first second is plotted:
```
>> PSTH = load(fullfile(T.fitpath{i}, 'results\pethsets_stereoclick.mat'));
>> figure('position', [100 100 275 250])
>> trialset=1;
>> neuron=3;
>> time_s = PSTH.time_s(PSTH.time_s<=1);
>> FHMDDM.plot_peth(PSTH.pethsets{trialset}{neuron}, "leftchoice", time_s)
>> FHMDDM.plot_peth(PSTH.pethsets{trialset}{neuron}, "rightchoice", time_s)
```
<img src="/assets/analysis_2023_02_08b_example_observed_psth_conditioned_on_choice_neuron_3.svg" height="150">

The unconditioned PSTH of the same neuron can be plotted,
```
>> clf
>> FHMDDM.plot_peth(PSTH.pethsets{trialset}{neuron}, "unconditioned", time_s)
```
<img src="/assets/analysis_2023_02_08b_example_observed_psth_unconditioned_neuron_3.svg" height="150">

as well as the neuron's PSTH conditioned on both the choice and the strength of the evidence:
```
>> for condition = ["leftchoice_weak_leftevidence", ...
                 "leftchoice_strong_leftevidence", ...
                 "rightchoice_weak_rightevidence", ...
                 "rightchoice_strong_rightevidence"]
    FHMDDM.plot_peth(PSTH.pethsets{trialset}{neuron}, condition, time_s, 'show_observed_CI', false)
end
```
<img src="/assets/analysis_2023_02_08b_example_observed_psth_conditioned_on_choice_evidence_neuron_3.svg" height="150">

Are these PSTH's recoverable under the model? If simulated data were generated as emissions from the model, and the same optimization scheme were used to fit the simulated data, what the predictions from simulation?
```
>> PSTHrecovered = load(fullfile(T.fitpath{i}, 'sample1\recovery\pethsets_stereoclick.mat'));
>> clf
>> for condition = ["leftchoice", "rightchoice"]
  FHMDDM.plot_peth(PSTH.pethsets{trialset}{neuron}, condition, time_s)
end
```
<img src="/assets/analysis_2023_02_08b_example_recovery_psth_conditioned_on_choice_neuron_3.svg" height="150">

```
>> clf
>> for condition = ["leftchoice_weak_leftevidence", ...
                 "leftchoice_strong_leftevidence", ...
                 "rightchoice_weak_rightevidence", ...
                 "rightchoice_strong_rightevidence"]
  FHMDDM.plot_peth(PSTH.pethsets{trialset}{neuron}, condition, time_s)
end
```

<img src="/assets/analysis_2023_02_08b_example_recovery_psth_conditioned_on_choice_evidence_neuron_3.svg" height="150">

### examining the parameters
```
>> Summary = load(fullfile(T.fitpath{1}, ['results' filesep 'modelsummary.mat']))
```
The value of the per-click noise that was learned is given by
```
>> Summary.thetanative.sigma2_s
ans =

    6.9237
```
 
#  `Model` composite type
The data, parameters, and hyperparameters of an FHMDDM are organized within the fields of in a composite object of the composite type (similar to a MATLAB structure) `Model`:
```
julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2023_02_06d_MAP/B15_scale1/data.mat"
julia> model = Model(datapath)
```

## fixed hyperparameters: `model.options`
The fixed hyperparameters, i.e., settings of the model that the user fixes before hand are and never altered, are stored in the field `options`, which is an object of composite type `Options`:
```
julia> model.options
FHMDDM.Options{Bool, String, Float64, Int64, Vector{Float64}}
  a_latency_s: Float64 0.01
  b_scalefactor: Float64 76.0
  choiceobjective: String "posterior"
  choiceLL_scaling_exponent: Float64 1.0
  datapath: String "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2023_02_06d_MAP/B15_scale1/data.mat"
  Δt: Float64 0.01
  
  ...
  
  Ξ: Int64 53
```
The setting `a_latency_s` is the latency, in seconds, with which auditory clicks influence the accumulated evidence latent variable `a`, and the setting `Δt` is the time step of the model, in seconds. The description of each field of the model is provided in the file [types.jl](/src/types.jl);  all fields are listed alphabetically. The field values cannot be altered:
```
julia> model.options.Δt = 0.1
ERROR: setfield! immutable struct of type Options cannot be changed
```

## data: `model.trialsets`
The data used to fit the model are organized into "trialsets." Each trialset comprises a group of trials and a group of neurons that are recorded on each trial of the trialset: 

A trialset is used to mean something more general than that conveyed by a "session." The trials in a trialset do not need to ordered in time, and they do not have to be consecutive. The only requirement for a trialset is that all the neurons included within a trialset were recorded on each trial of the trialset. 

The data can consist of multiple trialsets, so that the same set of model parameters can be fit to recordings across multiple days. Typically, separately parameters are fit to different recording sessions, and each model has a single trialset.

The behavioral choice, the auditory click timings, and the trial history information are contained in an object of composite type `Trial`:
```
model.trialsets[1].trialsets[1]
``` 

## parameters
The parameters of the model are random. Let's load parameters that were previously fit. After fitting a model, the parameters are saved within a structure named `modelsummary.mat`
```
julia> summarypath = joinpath(model.options.datapath, "final/modelsummary.mat")
julia> sortparameters!(model, summarypath)
```
