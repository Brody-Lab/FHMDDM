"""
    analyzeandsave(model)

Perform analyses on the model and save the analyses and the model

ARGUMENT
-`model`: a structure containing the data, parameters, and settings

OPTIONAL ARGUMENT
-`prefix`: prefix to the name of the files to be saved
"""
function analyzeandsave(model::Model; folderpath::String=dirname(model.options.datapath), prefix="results")
    savesummary(model; folderpath=folderpath, prefix=prefix)
    savepredictions(model; folderpath=folderpath, prefix=prefix)
	save∇∇loglikelihood(model; folderpath=folderpath, prefix=prefix)
end

"""
    savesummary(model)

Save the results of a model into file compatible with both MATLAB and Julia

Saved as `model.options.datapath/<prefix>.mat`

ARGUMENT
-`model`: a structure containing the data, parameters, and settings
-`prefix`: name of the file to be saved
"""
savesummary(model::Model; folderpath::String=dirname(model.options.datapath), prefix::String="results") = matwrite(joinpath(folderpath, prefix*".mat"), summarize(model))

"""
	summarize(model)

Package the parameters, hyperparameters, and fields useful for analysis into a dictionary
"""
function summarize(model::Model)
	Dict("loglikelihood"=>loglikelihood(model),
		"logposterior"=>logposterior(model),
		"theta_native"=> dictionary(model.θnative),
        "theta_real"=> dictionary(model.θreal),
        "theta0_native" => dictionary(model.θ₀native),
        "thetaglm"=>map(trialset->map(mpGLM->dictionary(mpGLM.θ), trialset.mpGLMs), model.trialsets),
        "Phiaccumulator"=>model.trialsets[1].mpGLMs[1].Φₐ,
        "Phihistory"=>model.trialsets[1].mpGLMs[1].Φₕ,
        "Phiphotostimulus"=>model.trialsets[1].mpGLMs[1].Φₚ,
        "Phipremovement"=>model.trialsets[1].mpGLMs[1].Φₘ,
        "Phitime"=>model.trialsets[1].mpGLMs[1].Φₜ,
        "Phiphotostimulus_timesteps"=>collect(model.trialsets[1].mpGLMs[1].Φₚtimesteps),
        "penaltycoefficients"=>model.gaussianprior.𝛂,
        "penaltymatrices"=>model.gaussianprior.𝐀,
        "penaltymatrixindices"=>model.gaussianprior.index𝐀,
        "precisionmatrix"=>model.gaussianprior.𝚲)
end

"""
    savepredictions(model)

Save predictions of the model:
-accumulator distribution: `folderpath/<prefix>_pa.mat`
-choice-conditioned accumulator distribution: `folderpath/<prefix>_pa_d.mat`
-choice- and spikes-conditioned accumulator distribution: `folderpath/<prefix>_pa_Yd.mat`
-choice- and spikes-conditioned coupling distribution: `folderpath/<prefix>_pc_Yd.mat`
-probability choosing right: `folderpath/<prefix>_pd.mat`
-spike counts: `folderpath/<prefix>_lambdaDeltat.mat`
-choice-conditioned spike counts: `folderpath/<prefix>_lambdaDeltat_d.mat`

ARGUMENT
-`model`: a structure containing the data, parameters, and settings

OPTIONAL ARGUMENT
-'folderpath': full path of the folder
-`prefix`: name of the file to be saved
"""
function savepredictions(model::Model; folderpath::String=dirname(model.options.datapath), prefix::String="results")
	predictions = Predictions(model)
    matwrite(joinpath(folderpath, prefix*"_pa"*".mat"), Dict("pa" => predictions.p𝐚))
    matwrite(joinpath(folderpath, prefix*"_pa_d"*".mat"), Dict("pa_d" => predictions.p𝐚_𝑑))
    matwrite(joinpath(folderpath, prefix*"_pa_Yd"*".mat"), Dict("pa_Yd" => predictions.p𝐚_𝐘𝑑))
    matwrite(joinpath(folderpath, prefix*"_pc_Yd"*".mat"), Dict("pc_Yd" => predictions.p𝐜_𝐘𝑑))
    matwrite(joinpath(folderpath, prefix*"_pd"*".mat"), Dict("pd" => predictions.p𝑑))
    matwrite(joinpath(folderpath, prefix*"_lambdaDeltat"*".mat"), Dict("lambdaDeltat" => predictions.λΔt))
    matwrite(joinpath(folderpath, prefix*"_lambdaDeltat_d"*".mat"), Dict("lambdaDeltat_d" => predictions.λΔt_𝑑))
    return nothing
end

"""
    save∇∇loglikelihood(model;	 folderpath, prefix)

Save the hessian of the log-likelihood as `folderpath/<prefix>_hessian_loglikelihood.mat`

ARGUMENT
-`model`: a structure containing the data, parameters, and settings
-'folderpath': full path of the folder
-`prefix`: name of the file to be saved
"""
function save∇∇loglikelihood(model::Model; folderpath::String=dirname(model.options.datapath), prefix::String="results")
	hessian_loglikelihood = ∇∇loglikelihood(model)[3]
    matwrite(joinpath(folderpath, prefix*"_hessian_loglikelihood"*".mat"), Dict("hessian_loglikelihood"=>hessian_loglikelihood))
    return nothing
end

"""
    save(cvresults,options)

Save the results of crossvalidation

ARGUMENT
-`cvresults`: an instance of `CVResults`, a drift-diffusion linear model
"""
function save(cvresults::CVResults, options::Options)
    path = dirname(options.datapath)*"/cvresults.mat"
    matwrite(path, dictionary(cvresults))
    return nothing
end

"""
    save(options, trialsets)
"""
function save(options::Dict, trialsets::Vector{<:Trialset})
    dict = Dict("options"=> options,
                "trialsets"=> map(trialset->dictionary(trialset), trialsets))
    matwrite(options["datapath"], dict)
    return nothing
end

"""
    savedata(model)

Save the data used to fit a model.

This function is typically used when the data is sampled

ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters used to fit a factorial hidden Markov drift-diffusion mdoel

OPTIONAL ARGUMENT
-`filename`: the data will be saved at the path `joinpath(dirname(model.options.datapath), filename)`
"""
function savedata(model::Model; filename::String=basename(model.options.datapath))
    data =  map(model.trialsets, 1:length(model.trialsets)) do trialset, index
                Dict("trials"=>map(trial->packagedata(trial, model.options.a_latency_s), trialset.trials),
                     "units"=>map(mpGLM->packagedata(mpGLM), trialset.mpGLMs),
                     "index"=>index)
            end
    dict = Dict("data"=>data, "options"=>dictionary(model.options))
    path = joinpath(dirname(model.options.datapath), filename)
    matwrite(path, dict)
end

"""
	packagedata(trial, a_latency_s)

Package the data in one trial into a Dict for saving

ARGUMENT
-`trial`: structure containing the data of one trial
-`a_latency_s`: latency of the accumulator responding to the clicks
"""
function packagedata(trial::Trial, a_latency_s::AbstractFloat)
	Dict("choice" => trial.choice,
         "clicktimes" => packagedata(trial.clicks, a_latency_s),
		 "movementtime_s"=> trial.movementtime_s,
		 "ntimesteps"=> trial.ntimesteps,
		 "previousanswer" => trial.previousanswer)
end

"""
    packagedata(clicks, a_latency_s)

Package the click times in one trial into a Dict for saving

ARGUMENT
-`clicks`: structure containing the data of the click times in one trial
-`a_latency_s`: latency of the accumulator responding to the clicks
"""
function packagedata(clicks::Clicks, a_latency_s::AbstractFloat)
	leftclicktimes = clicks.time[clicks.source .== 0] .- a_latency_s
	rightclicktimes = clicks.time[clicks.source .== 1] .- a_latency_s
    Dict("L" => vcat(0.0, leftclicktimes) , "R" => vcat(0.0, rightclicktimes))
end

"""
	packagedata(mpGLM)

Package the data of one neuron into a Dict for saving
"""
function packagedata(mpGLM::MixturePoissonGLM)
	Dict("y"=>mpGLM.𝐲)
end
