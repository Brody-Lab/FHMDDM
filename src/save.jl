"""
	Summary(model)

A snapshot of the current state of the model

ARGUMENT
-`model`: a composite containing the data, parameters, and hyperparameters of a factorial hidden Markov drift-diffusion model

OPTIONAL ARGUMENT
-`computehessian`: whether the hessian of the log-likelihood and log-posterior functions are to be computed
"""
function ModelSummary(model::Model; computehessian::Bool=false)
	modelsummary =
	ModelSummary(loglikelihood=loglikelihood(model),
		 	logposterior=logposterior(model),
			thetanative=model.θnative,
			thetareal=model.θreal,
			theta0native=model.θ₀native,
			thetaglm=map(trialset->map(mpGLM->mpGLM.θ, trialset.mpGLMs), model.trialsets),
			temporal_basis_vectors_gain=collect(collect(mpGLM.Φgain for mpGLM in trialset.mpGLMs) for trialset in model.trialsets),
			temporal_basis_vectors_accumulator=collect(trialset.mpGLMs[1].Φaccumulator for trialset in model.trialsets),
	        temporal_basis_vectors_postspike=collect(trialset.mpGLMs[1].Φpostspike for trialset in model.trialsets),
	        temporal_basis_vectors_premovement=collect(trialset.mpGLMs[1].Φpremovement for trialset in model.trialsets),
	        temporal_basis_vectors_poststereoclick=collect(trialset.mpGLMs[1].Φpoststereoclick for trialset in model.trialsets),
			parametervalues=concatenateparameters(model),
			parameternames=nameparameters(model),
	        penaltycoefficients=model.gaussianprior.𝛂,
	        penaltymatrices=model.gaussianprior.𝐀,
	        penaltymatrixindices=model.gaussianprior.index𝐀,
			penaltynames=model.gaussianprior.penaltynames,
	        precisionmatrix=model.gaussianprior.𝚲)
	if computehessian
		modelsummary.hessian_loglikelihood .= ∇∇loglikelihood(model)[3]
		modelsummary.hessian_logposterior .= modelsummary.hessian_loglikelihood - modelsummary.precisionmatrix
	end
	return modelsummary
end

"""
	save(modelsummary, folderpath)

Save the summary of the model

All the fields of the composite `modelsummary` are saved within a file named `modelsummary.mat` within a folder whose absolute path is specified by `folderpath`.
"""
function save(modelsummary::ModelSummary, folderpath::String)
	if !isdir(folderpath)
		mkdir(folderpath)
		@assert isdir(folderpath)
	end
	filepath = joinpath(folderpath, "modelsummary.mat")
	dict = dictionary(modelsummary)
	matwrite(filepath, dict)
end

"""
	dictionary(modelsummary)

Package quantities in the composite `modelsummary` into a `Dict`
"""
function dictionary(modelsummary::ModelSummary)
	entries = 	map(fieldnames(ModelSummary)) do fieldname
					if (fieldname == :thetanative) || (fieldname == :thetareal) || (fieldname == :theta0native)
						String(fieldname)=>dictionary(getfield(modelsummary,fieldname))
					elseif fieldname == :thetaglm
						String(fieldname)=>map(θ->map(θ->dictionary(θ), θ), modelsummary.thetaglm)
					else
						String(fieldname)=>getfield(modelsummary,fieldname)
					end
				end
	Dict(entries...)
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
		 "gamma"=>trial.γ,
		 "movementtime_s"=> trial.movementtime_s,
		 "ntimesteps"=> trial.ntimesteps,
		 "photostimulus_decline_on_s"=> trial.photostimulus_decline_on_s,
		 "photostimulus_incline_on_s"=> trial.photostimulus_incline_on_s,
		 "previousanswer" => trial.previousanswer,
		 "stereoclick_time_s"=>trial.stereoclick_time_s)
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

"""
	analyzeandsave(foldername, model)

Perform routine analyses and save them to a folder

The folder is contained within the parent folder containing the data.
"""
function analyzeandsave(computehessian::Bool, foldername::String, model::Model)
	optimization_folder_path = joinpath(dirname(model.options.datapath), foldername)
	modelsummary = ModelSummary(model; computehessian=computehessian)
	save(modelsummary, optimization_folder_path)
	characterization = Characterization(model)
	save(characterization, optimization_folder_path)
	psthsets = FHMDDM.poststereoclick_time_histogram_sets(characterization.expectedemissions, model)
	save(psthsets, optimization_folder_path)
end
