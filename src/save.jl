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
savesummary(model::Model; folderpath::String=dirname(model.options.datapath), prefix::String="results") = matwrite(joinpath(folderpath, prefix*".mat"), dictionary(Summary(model)))

"""
	dictionary(Summary)

Convert an instance of Summary to a dictionary
"""
function dictionary(modelsummary::Summary)
	Dict("loglikelihood"=>modelsummary.loglikelihood,
		"logposterior"=>modelsummary.logposterior,
		"thetanative"=> dictionary(modelsummary.θnative),
        "thetareal"=> dictionary(modelsummary.θreal),
        "theta0native" => dictionary(modelsummary.θ₀native),
        "thetaglm"=>map(θ->map(θ->dictionary(θ), θ), modelsummary.θglm),
        "Phi_accumulator"=>modelsummary.Φₐ,
        "Phi_postspike"=>modelsummary.Φₕ,
        "Phi_postphotostimulus"=>modelsummary.Φₚ,
        "Phi_premovement"=>modelsummary.Φₘ,
        "Phi_poststereoclick"=>modelsummary.Φₜ,
        "Phi_photostimulus_timesteps"=>collect(modelsummary.Φₚtimesteps),
		"parametervalues"=>modelsummary.parametervalues,
		"parameternames"=>modelsummary.parameternames,
        "penaltycoefficients"=>modelsummary.𝛂,
        "penaltymatrices"=>modelsummary.𝐀,
        "penaltymatrixindices"=>modelsummary.index𝐀,
		"penaltynames"=>modelsummary.penaltynames,
        "precisionmatrix"=>modelsummary.𝚲)
end

"""
	Summary(model)

Summarize the model
"""
function Summary(model::Model)
	Summary(loglikelihood=loglikelihood(model),
		 	logposterior=logposterior(model),
			θnative=model.θnative,
			θreal=model.θreal,
			θ₀native=model.θ₀native,
			θglm=map(trialset->map(mpGLM->mpGLM.θ, trialset.mpGLMs), model.trialsets),
			Φₐ=model.trialsets[1].mpGLMs[1].Φₐ,
	        Φₕ=model.trialsets[1].mpGLMs[1].Φₕ,
	        Φₚ=model.trialsets[1].mpGLMs[1].Φₚ,
	        Φₘ=model.trialsets[1].mpGLMs[1].Φₘ,
	        Φₜ=model.trialsets[1].mpGLMs[1].Φₜ,
	        Φₚtimesteps=collect(model.trialsets[1].mpGLMs[1].Φₚtimesteps),
			parametervalues=concatenateparameters(model),
			parameternames=nameparameters(model),
	        𝛂=model.gaussianprior.𝛂,
	        𝐀=model.gaussianprior.𝐀,
	        index𝐀=model.gaussianprior.index𝐀,
			penaltynames=model.gaussianprior.penaltynames,
	        𝚲=model.gaussianprior.𝚲)
end

"""
    savepredictions(model)

Save predictions of the model.

The specific predictions being saved include
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
	save(Predictions(model), model.options; folderpath=folderpath, prefix=prefix)
    return nothing
end

"""
	save(predictions, options)

Save predictions of the model

Not all fields of the structure `Predictions` are saved. The fields not being saved include `p𝐜_𝐘𝑑` and `nsamples`.

ARGUMENT
-`predictions`: a structure containing the predictions
-`options`: a structures containing the fixed hyperparameters of the model

"""
function save(predictions::Predictions, options::Options; folderpath::String=dirname(options.datapath), prefix::String="results")
	dict = FHMDDM.Dict(predictions)
	for key in keys(dict)
		if key != "nsamples"
    		matwrite(joinpath(folderpath, prefix*"_"*key*".mat"), Dict(key =>dict[key]))
		end
	end
    return nothing
end

"""
	dictionary(predictions)

Package an instance `Predictions` as a dictionary
"""
function FHMDDM.Dict(predictions::Predictions)
	Dict("pa" => predictions.p𝐚,
        "pa_d" => predictions.p𝐚_𝑑,
        "pa_Y" => predictions.p𝐚_𝐘,
        "pa_Yd" => predictions.p𝐚_𝐘𝑑,
        "pc_Yd" => predictions.p𝐜_𝐘𝑑,
        "pd" => predictions.p𝑑,
        "pd_Y" => predictions.p𝑑_𝐘,
        "lambdaDeltat" => predictions.λΔt,
        "lambdaDeltat_d" => predictions.λΔt_𝑑,
		"nsamples" => predictions.nsamples)
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
		 "photostimulus_decline_on_s"=> trial.photostimulus_decline_on_s,
		 "photostimulus_incline_on_s"=> trial.photostimulus_incline_on_s,
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
