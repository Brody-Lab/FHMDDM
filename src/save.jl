"""
	save.jl

analyzeandsave(computehessian, foldername, model)
save(options, trialsets)
ModelSummary(model)
save(modelsummary, folderpath)
dictionary(modelsummary)
dictionary(trial, a_latency_s)
dictionary(clicks, a_latency_s)
dictionary(options)
dictionary(x)
"""

"""
	analyzeandsave(computehessian, foldername, model)

Perform routine analyses and save them to a folder

The folder is contained within the parent folder containing the data.
"""
function analyzeandsave(computehessian::Bool, foldername::String, model::Model)
	save(model.options)
	save(model.options, model.trialsets)
	folderpath = joinpath(model.options.outputpath, foldername)
	modelsummary = ModelSummary(model; computehessian=computehessian)
	save(modelsummary, folderpath)
	characterization = Characterization(model)
	save(characterization, folderpath)
	psthsets = poststereoclick_time_histogram_sets(characterization.expectedemissions, model)
	save(psthsets, folderpath)
end

"""
    save(options, trialsets)

Save the hyperparameters and the data used to fit a model

ARGUMENT
-`options`: fixed hyperparameters
-`trialsets`: data
"""
function save(options::Options, trialsets::Vector{<:Trialset})
    trialsets = map(trialsets) do trialset
					trials = map(trial->dictionary(trial, options.a_latency_s), trialset.trials)
                	Dict("trials"=>trials)
            	end
    dict = Dict("trialsets"=>trialsets)
    filepath = joinpath(options.outputpath, "trialsets.mat")
    matwrite(filepath, dict)
end

"""
	save(options)

Save the fixed hyperparameters stored in a struct into a CSV file
"""
function save(options::Options)
    csvpath = joinpath(options.outputpath, "options.csv")
	dict = Dict((string(fieldname)=>getfield(options,fieldname) for fieldname in fieldnames(Options))...)
	dataframe = DataFrames.DataFrame(dict)
	CSV.write(csvpath, dataframe)
end

"""
	ModelSummary(model)

A snapshot of the current state of the model

ARGUMENT
-`model`: a composite containing the data, parameters, and hyperparameters of a factorial hidden Markov drift-diffusion model

OPTIONAL ARGUMENT
-`computehessian`: whether the hessian of the log-likelihood and log-posterior functions are to be computed
"""
function ModelSummary(model::Model; computehessian::Bool=false)
	modelsummary = ModelSummary(designmatrix = collect(collect(mpGLM.𝐗 for mpGLM in trialset.mpGLMs) for trialset in model.trialsets),
			externalinputs = collect(collect(externalinput(mpGLM) for mpGLM in trialset.mpGLMs) for trialset in model.trialsets),
			glm_parameter_scale_factor = model.options.sf_mpGLM[1],
			loglikelihood=loglikelihood(model),
		 	logposterior=logposterior(model),
			parametervalues=concatenateparameters(model),
			parameternames=nameparameters(model),
	        penaltycoefficients=model.gaussianprior.𝛂,
	        penaltymatrices=model.gaussianprior.𝐀,
	        penaltymatrixindices=model.gaussianprior.index𝐀,
			penaltynames=model.gaussianprior.penaltynames,
	        precisionmatrix=model.gaussianprior.𝚲,
			thetanative=model.θnative,
			thetareal=model.θreal,
			theta0native=model.θ₀native,
			thetaglm=map(trialset->map(mpGLM->mpGLM.θ, trialset.mpGLMs), model.trialsets),
			temporal_basis_vectors_gain=collect(collect(mpGLM.Φgain for mpGLM in trialset.mpGLMs) for trialset in model.trialsets),
	        temporal_basis_vectors_postspike=collect(trialset.mpGLMs[1].Φpostspike for trialset in model.trialsets),
	        temporal_basis_vectors_premovement=collect(trialset.mpGLMs[1].Φpremovement for trialset in model.trialsets),
	        temporal_basis_vectors_poststereoclick=collect(trialset.mpGLMs[1].Φpoststereoclick for trialset in model.trialsets))

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
	dictionary(trial, a_latency_s)

Package the data in one trial into a Dict for saving

ARGUMENT
-`trial`: structure containing the data of one trial
-`a_latency_s`: latency of the accumulator responding to the clicks
"""
function dictionary(trial::Trial, a_latency_s::AbstractFloat)
	Dict("choice" => trial.choice,
         "clicktimes" => dictionary(trial.clicks, a_latency_s),
		 "gamma"=>trial.γ,
		 "movementtime_s"=> trial.movementtime_s,
		 "ntimesteps"=> trial.ntimesteps,
		 "photostimulus_decline_on_s"=> trial.photostimulus_decline_on_s,
		 "photostimulus_incline_on_s"=> trial.photostimulus_incline_on_s,
		 "previousanswer" => trial.previousanswer,
		 "spiketrains"=>trial.spiketrains,
		 "stereoclick_time_s"=>trial.stereoclick_time_s)
end

"""
    dictionary(glmθ)

Convert into a dictionary the parameters of a mixture of Poisson generalized linear model
"""
function dictionary(glmθ::GLMθ)
    Dict("b"=>glmθ.b[1],
		 "c"=>glmθ.c[1],
		"u"=>glmθ.𝐮,
		"v"=>glmθ.v[1],
		"beta"=>glmθ.β[1],
		("u_"*string(field)=>glmθ.𝐮[getfield(glmθ.indices𝐮, field)] for field in fieldnames(Indices𝐮))...)
end

"""
    dictionary(clicks, a_latency_s)

Package the click times in one trial into a Dict for saving

ARGUMENT
-`clicks`: structure containing the data of the click times in one trial
-`a_latency_s`: latency of the accumulator responding to the clicks
"""
function dictionary(clicks::Clicks, a_latency_s::AbstractFloat)
	leftclicktimes = clicks.time[clicks.source .== 0] .- a_latency_s
	rightclicktimes = clicks.time[clicks.source .== 1] .- a_latency_s
    Dict("L" => vcat(0.0, leftclicktimes) , "R" => vcat(0.0, rightclicktimes))
end

"""
	dictionary(x)

Convert a struct into a dictionary
"""
dictionary(x) = Dict((String(fieldname)=>getfield(x,fieldname) for fieldname in fieldnames(typeof(x)))...)
