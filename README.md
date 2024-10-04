# factorial hidden Markov drift-diffusion model (FHMDDM)

The code for fitting the model and computing quantities to characterizing the model is in the programming language Julia, whereas the code for plotting the said quantities are in MATLAB. 

# table of contents
* [tutorial](#tutorial)
  * [installing the FHMDDM repository](#installing-the-fhmddm-repository)
  * [creating a model with default options](#creating-a-model-with-default-options)
  * [customizing model options](#customizing-model-options)
  * [customizing multiple models](#customizing-multiple-models)
  * [data format](#data-format)
  * [fitting the model](#fitting-the-model)
  * [examining the results](#examining-the-results)
    * [PSTH](#plotting-the-peri-stimulus-time-histogram-psth)
    * [parameters](#examining-the-parameters)
      * [drift-diffusion parameters](#drift-diffusion-parameters)
      * [accumulator encoding weights](#encoding-weight-of-accumulated-evidence)
      * [other GLM parameters](#other-glm-parameters)
  * [developing the codebase](#developing-the-codebase)
* [`Model` type](#model-composite-type)
  * [fixed options](#fixed-hyperparameters-model-options)
  * [data](#data-model-trialsets)
  * [parameters](#parameters)
# tutorial
The following tutorial shows how to fit the model to a recording session on `spock`, the Princeton Neuroscience Institute's (PNI) computation cluster, and the visualizing the results on a Windows machine.

Open a shell (e.g. Windows Terminal) to log into `spock`:
```
> ssh <netID>@spock.princeton.edu
```
After providing credentials, load a version of Julia and run it
```
[<netID>spock ~]$ module load julia\1.6.0
[<netID>spock ~]$ julia
julia>
```
This can also be done locally on a desktop computer. Visual Studio Code provides a nice editor for Julia code. 

## installing the FHMDDM repository

In the Julia read-eval-print loop (REPL), enter the Pkg REPL by pressing `]` from the Julia REPL. To get back to the Julia REPL, press `backspace` or `^C`.
```
julia> ]
(v1.6) pkg> 
```
Add the `FHMDDM` repository
```
(v1.6) pkg> add https://github.com/Brody-Lab/FHMDDM.git
  Updating git-repo `https://github.com/Brody-Lab/FHMDDM.git`
  Username for 'https://github.com':
```
This should take no more than 10 minutes. Having added a package, update your environment:
```
(v1.6) pkg> up
```
If you check the status of your environment, you should see `FHMDDM` as one of your packages:
```
(v1.6) pkg> st
[<some numbers>] FHMDDM v0.1.0 `https://github.com/Brody-Lab/FHMDDM.git#master`
```
Now, return to the Julia REPL by pressing `backspace`
```
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

##   creating a model with default options
To create a model with the default model options, provide the absolute path of a data file ([the data format is discussed below](#data-format), and an example data file is located [here](/assets/T176_2018_05_03.mat))
```
julia> datapath = "/mnt/cup/labs/brody/tzluo/manuscript2023a/recordingsessions/2023_04_12/T176_2018_05_03.mat"
```
and the absolute path of an output folder
```
julia> outputpath = "/mnt/cup/labs/brody/tzluo/miscellaneous/fhmddm_sandbox/"
```
we can now create an object of type `Model` that contains the data, parameters, hyperparameters, and options (fixed hyperparameters):
```
julia> using FHMDDM
julia> model = Model(datapath, outputpath)
```
All the options of this object are indicated in `model.options`
```
julia> model.options
Options{Bool, String, Float64, Int64, Vector{Float64}}
  a_latency_s: Float64 0.01
  choiceobjective: String "posterior"
  choiceLL_scaling_exponent: Float64 0.6
  datapath: String "/mnt/cup/labs/brody/tzluo/manuscript2023a/recordingsessions/2023_04_11/T176_2018_05_03.mat"
  ⋮
```
which is an object of type `Options`. Detailed descriptions of each field can be found in [types.jl](/src/types.jl)

## customizing model options
Suppose we wish not to fit the height of the absorbing bound of the drift-diffusion process, which is optimized by default. To specify this custom option, I create a hash table (a [Dict](https://docs.julialang.org/en/v1/base/collections/#Base.Dict) in Julia):
```
julia> datapath = "/mnt/cup/labs/brody/tzluo/manuscript2023a/recordingsessions/2023_04_12/T176_2018_05_03.mat"
julia> outputpath = "/mnt/cup/labs/brody/tzluo/miscellaneous/fhmddm_sandbox/"
julia> dict = Dict("fit_B"=>false, "datapath"=>datapath, "outputpath"=>outputpath)
```
Using this hash table, the customized model can be created
```
julia> model = Model(Options(dict))
julia> model.options.fit_B
false
```

## customizing multiple models
Typically, multiple models are fit simultaneously, to the same data file under different options, to different data files under the same options, or both. Specifying the options for multiple models can be simplified by using a table. Each row of the table specifies the options of a model.

```
julia> using DataFrames
julia> df = DataFrame(fit_B=[true, false], outputpath = collect(outputpath*s for s = ["fit_B", "fix_B"]), datapath=datapath)
```

Please make sure that the output paths are different. I bolded the differences below

| fit_B   | outputpath                                                       | datapath                                                                                   |
| ------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| TRUE    | /mnt/cup/labs/brody/tzluo/miscellaneous/fhmddm_sandbox/__fit_B__ | /mnt/cup/labs/brody/tzluo/manuscript2023a/recordingsessions/2023_04_12/T176_2018_05_03.mat |
| FALSE   | /mnt/cup/labs/brody/tzluo/miscellaneous/fhmddm_sandbox/__fix_B__ | /mnt/cup/labs/brody/tzluo/manuscript2023a/recordingsessions/2023_04_12/T176_2018_05_03.mat |

```
julia> model_fit_B = Model(df[1,:])
julia> model_fix_B = Model(df[2,:])
```

If a column has a name that is not a field in the type `Options` as indicated in [types.jl](/src/types.jl), the column is ignored.

If you have not added the module [DataFrames.jl](https://dataframes.juliadata.org/stable/) (you should), no worries. A model can also be created by supplying it with the absolute path to a comma-separated values (CSV) file and a row number

```
julia> csvpath = "/mnt/cup/people/zhihaol/Documents/tzl_state_dependent_encoding/src/scripts/analysis_2023_04_12a_cv/options.csv"
julia> model = Model(csvpath, 2)
``` 

## data format
Data are stored in binary MATLAB files (.mat). Below shows a data file of a single recording sessions with 664 complete trials and 76 choice-selective neurons (note that the model is fully compatible at fitting a single model to multiple sessions).  Each trial is an element of a cell array containing a scalar[ structure array](https://www.mathworks.com/help/matlab/ref/struct.html). Let's look at the first trial
```
MATLAB> load('X:\tzluo\manuscript2023a\recordingsessions\2023_04_12\T176_2018_05_03.mat', 'trials')
MATLAB> trials{1}

ans = 

  struct with fields:

                        choice: 0
                    clicktimes: [1×1 struct]
                         gamma: -4
                movementtime_s: 3.8109e+03
                previousanswer: 0
            stereoclick_time_s: 3.8103e+03
                    ntimesteps: 50
    photostimulus_incline_on_s: NaN
    photostimulus_decline_on_s: NaN
                   spiketrains: {76×1 cell}
```
If you want to load this file on your computer, its Unix path is `/mnt/cup/labs/brody/tzluo/manuscript2023a/recordingsessions/2023_04_12/T176_2018_05_03.mat`

The field `choice` is a logical scalar indicating whether the animal chose left (false) or right (true).

The struct `clicktimes` indicates the times of the left and right clicks relative to the stereoclick
```
>> trials{1}.clicktimes

ans = 

  struct with fields:

    L: [0 0.0159 0.0315 0.0409 0.0607 0.1138 0.1498 0.1736 0.1775 0.1881 0.2511 … ]
    R: [0 0.2386]
```
`gamma` is the ratio of the natural logarithm of the generative right click rate to the log of the left click rate.

`movementtime_s` is the inferred time, in seconds, when the animal left the center port. 

`previousanswer` indicates the location where the reward is baited in the previous completed trial (left:-1, right: 1, no previous trial: 0)

`stereoclick_time_s` is the time when the stereoclick occured.

`ntimesteps`: is the number of time steps between the stereoclick and when the animal left the center port (the number of time steps is often capped by a maximum trial duration of 1 second).

`photostimulus_incline_on_s`: onset time of the "on-ramp" of the photostimulus

`photostimulus_decline_on_s`: onset time of the "off-ramp" of the photostimulus

`spiketrains` is a cell array whose each element corresponds to a neuron. The spike train of the first neuron in this trial is a vector length `ntimesteps`

```
MATLAB> trials{1}.spiketrains{1}

ans =

  Columns 1 through 14

     0     0     0     0     0     0     0     0     0     0     0     0     0     1

  Columns 15 through 28

     1     0     0     0     0     0     1     1     0     0     0     0     0     0

  Columns 29 through 42

     0     0     0     0     0     0     0     0     0     0     0     0     0     0

  Columns 43 through 50

     0     1     0     0     0     0     0     0
```

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

The expected run time on a on a node in the PNI cluster and on a "normal" desktop computer is a few hours.
## examining the results
Code for plotting the results was written in MATLAB 2023a. To use the plotting tools, add the local copy of the folder [FHMDDM/src](https://github.com/Brody-Lab/FHMDDM/tree/master/src) into MATLAB's search path. First, let's tabulate the models and their options. Here we are assuming that the MATLAB script is saved in the same folder where the `options.csv` is located:
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
### drift-diffusion parameters
The value of the per-click noise that was learned is given by
```
>> Summary.thetanative.sigma2_s
ans =

    6.9237
```
###  encoding weight of accumulated evidence
The conditional firing rate of a neuron, given the state of the accumulated evidence, at a time step $t$ is given by

$$\lambda_t \equiv \text{softplus}((w\mid a_t)a_t + u_t)$$
where $w \mid a$ is the state-dependent encoding weight and depends on whether the accumulated evidence reached the absorbing bound $B$:

$$w \mid (a > -B \mid a < B) = w_{pre}$$

$$w \mid (a=-B \mid a = B) = w_{post}$$

Let's extract the pre-commitment and post-commitment encoding weights of the accumulated evidence

```
>> scalefactor = Summary.temporal_basis_vectors_accumulator{1}(1);
>> wpre = cellfun(@(x) x.v{1}(1)*scalefactor, Summary.thetaglm{1})
>> wpost = cellfun(@(x) x.beta{1}(1)*scalefactor, Summary.thetaglm{1})
```

The scale factor is a technical thing used to help the optimization converge and des not impact the results. 

Let's compute the engagement index, which quantifies the relative engagement of each neuron in the pre- and post-commitment states

```
>> engagementindices = (abs(wpre)-abs(wpost))./(abs(wpre)+abs(wpost))
>> figure('position', [100 100 250, 200])
>> FHMDDM.prepareaxes
>> histogram(engagementindices, -1:0.2:1)
>> xlabel('engagement index')
>> ylabel('neurons')
```

<img src="/assets/analysis_2023_02_08b_example_engagementindex_histogram.svg" height="150">

It appears that most of the choice-selective neurons used to fit the model are more strongly engaged before, than after, decision commitment. How well can the engagement indices be recovered?

```
>> figure('position', [100 100 400 200])
>> FHMDDM.prepareaxes
>> set(gca, 'DataAspectRatio', [1,1,1])
>> plot(engagementindices, recoveredindices, 'k.')
>> xlabel('inferred')
>> ylabel('recovered')
```

<img src="/assets/analysis_2023_02_08b_example_engagementindex_recovery.svg" height="150">

The goodness-of-recovery, as assessed using the coefficient-of-determination, is given by

```
>> SStotal = sum((mean(engagementindices)-engagementindices).^2);
>> SSresisdual = sum((recoveredindices-engagementindices).^2);
>> 1 - SSresisdual/SStotal
ans =

    0.9616
```

Is the estimation of the engagement index biased by the optimization procedure?
```
>> bootci(1e3, @median, recoveredindices-engagementindices)
ans =

   -0.0313
    0.0129
```
It does not appear so here, because the 95% bootstrap confidence interval of the median difference overlaps with 0.

Are the engagement indices different between dorsomedial frontal cortex and medial prefrontal cortex?

```
>> load(fullfile(T.fitpath{1}, 'data.mat'), 'data')
>> brainareas = cellfun(@(x) x.brainarea, data{1}.units);
>> EI_medial_prefrontal = median(engagementindices(brainareas=="PrL" | brainareas == "MO"))
>> EI_medial_prefrontal =

    0.6535
>> EI_dorsomedial_frontal = median(engagementindices(brainareas=="Cg1" | brainareas == "M2"))
EI_dorsomedial_frontal =

    0.2916
```


#### other GLM parameters

The term $u_t$ is the component of the linear predictor independent of the accumulated evidence and is given by

$$u_t \equiv g + u^{stereo}_t + u^{move}_t + u^{hist}_t$$

where $g$ is a constant specifying the gain, $u^{stereo}_t$ is the input from the post-stereoclick filter, $u^{move}_t$ is the input from the pre-movement filter, and $u^{hist}_t$ is the input from the post-spike filter.

The gain of the third neuron can be found be
```
>> Summary.thetaglm{1}{3}.u_gain
ans =

    8.8468
```

Let's visualize the linear filter defining the time-varying input  of the stereoclick to this neuron
```
>> y = Summary.temporal_basis_vectors_poststereoclick{1}*Summary.thetaglm{1}{3}.u_poststereoclick;
>> figure('position', [100 100 300 250])
>> FHMDDM.prepareaxes
>> plot((1:numel(y))*0.01, y, 'k-')
>> xlabel('time from stereoclick (s)')
```
<img src="/assets/analysis_2023_02_08b_example_poststereoclick_filter.svg" height="150">

Finally, let's also plot the pre-movement and post-spike filters. 

```
>> y = Summary.temporal_basis_vectors_premovement{1}*Summary.thetaglm{1}{3}.u_premovement;
>> figure('position', [100 100 300 250])
>> FHMDDM.prepareaxes
>> plot(0-(1:numel(y))*0.01, y, 'k-')
>> xlabel('time before movement (s)')
```

<img src="/assets/analysis_2023_02_08b_example_premovement_filter.svg" height="150">

```
>> y = Summary.temporal_basis_vectors_postspike{1}*Summary.thetaglm{1}{3}.u_postspike;
>> figure('position', [100 100 300 250])
>> FHMDDM.prepareaxes
>> plot((1:numel(y))*0.01, y, 'k-')
>> xlabel('time after spike (s)')
```

<img src="/assets/analysis_2023_02_08b_example_postspike_filter.svg" height="150">
 
## developing the codebase
First, install the package as [described](#installing-the-fhmddm-repository)

```
julia>
julia> ]
(v1.6) pkg>
(v1.6) pkg> dev FHMDDM
(v1.6) pkg> up
(v1.6) pkg> st
```
You should see the following in your list of packages:

```
[<########>] FHMDDM v0.1.0 `~/.julia/dev/FHMDDM`
```
Now, make sure you can load FHMDDM as expected:
```
(v1.6) pkg> <backspace>
julia>
julia> using FHMDDM
```
Let's really make sure that you are able to use the code before you start modifying it. Let's check out a specific branch, `2023_02_12`:
```
julia>
julia> exit()
[<netID>@spock-login ~]$
[<netID>@spock-login ~]$ cd ~/.julia/dev/FHMDDM
[<netID>@spock-login FHMDDM]$ git fetch
[<netID>@spock-login FHMDDM]$ git checkout 2023_02_12
Branch 2023_02_12 set up to track remote branch 2023_02_12 from origin.
Switched to a new branch '2023_02_12'
```
Returning to the Julia REPL
```
[<netID>@spock-login FHMDDM]$ julia
julia>
julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2023_02_09a_stereoclickonly/stereoclick_only_1s/data.mat"
julia> model = Model(datapath)
Model{FHMDDM.Options{Bool, String, Float64, Int64, Vector{Float64}}, FHMDDM.GaussianPrior{Vector{Int64}, Vector{Float64}, Vector{Float64}, Vector{String}, Matrix{Float64}, Vector{Vector{Int64}}, Vector{Matrix{Float64}}}, FHMDDM.Latentθ{Vector{Float64}}, FHMDDM.Latentθ{Vector{Float64}}, FHMDDM.Latentθ{Vector{Float64}}, Vector{FHMDDM.Trialset{Vector{FHMDDM.MixturePoissonGLM{Int64, Float64, UnitRange{Int64}, Vector{Float64}, Vector{UInt8}, FHMDDM.GLMθ{Bool, FHMDDM.Indices𝐮{ UnitRange{Int64}}, Float64, Vector{Float64}, Vector{Symbol}, Vector{Vector{Float64}}}, Matrix{Float64}}}, Int64, Vector{FHMDDM.Trial{Bool, FHMDDM.Clicks{Vector{Float64}, BitVector, Vector{Int64}, Vector{Vector{Int64}}}, Float64, Int64}}}}}
  options: FHMDDM.Options{Bool, String, Float64, Int64, Vector{Float64}}
  gaussianprior: FHMDDM.GaussianPrior{Vector{Int64}, Vector{Float64}, Vector{Float64}, Vector{String}, Matrix{Float64}, Vector{Vector{Int64}}, Vector{Matrix{Float64}}}
  θnative: FHMDDM.Latentθ{Vector{Float64}}
  θreal: FHMDDM.Latentθ{Vector{Float64}}
  θ₀native: FHMDDM.Latentθ{Vector{Float64}}
  trialsets: Array{FHMDDM.Trialset{Vector{FHMDDM.MixturePoissonGLM{Int64, Float64, UnitRange{Int64}, Vector{Float64}, Vector{UInt8}, FHMDDM.GLMθ{Bool, FHMDDM.Indices𝐮{ UnitRange{Int64}}, Float64, Vector{Float64}, Vector{Symbol}, Vector{Vector{Float64}}}, Matrix{Float64}}}, Int64, Vector{FHMDDM.Trial{Bool, FHMDDM.Clicks{Vector{Float64}, BitVector, Vector{Int64}, Vector{Vector{Int64}}}, Float64, Int64}}}}((1,))

```
If you can load the model, great! You can now switch back to the `master` branch and make a new branch for yourself based on the code in the `master` branch:
```
julia> exit()
[<netID>@spock-login FHMDDM]$ git checkout master
[<netID>@spock-login FHMDDM]$ git checkout -b <name of your new branch>
[<netID>@spock-login FHMDDM]$ git branch

```
You should see your branch listed, and next to it a '*', indicating that is your current branch.

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
