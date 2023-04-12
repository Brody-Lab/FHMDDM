"""
	contents

Model(datapath)
Model(options, trialsets)
Trialset(options, trialset)
Trial(a_latency_s, Δt, index_in_trialset, preceding_timesteps, trial, trialsetindex)
Clicks(a_latency_s, L, R, Δt, ntimesteps)
randomize_latent_parameters(options)
randomize_latent_parameters!(model)
randomizeparameters!(model)
reindex(index_in_trialset, τ₀, trial)
"""

"""
	Model(csvpath, row)

RETURN a struct containing data, parameters, and hyperparameters of a factorial hidden Markov drift-diffusion model

ARGUMENT
-`csvpath`: the absolute path to a comma-separated values (CSV) file
-`row`: the row of the CSV to be considered
"""
Model(csvpath::String, row::Integer) = Model(Options(csvpath, row))
Model(csvpath::String) = Model(csvpath,1)

"""
	Options(csvpath, row)

RETURN a struct containing the fixed hyperparameters of the model

ARGUMENT
-`csvpath`: the absolute path to a comma-separated values (CSV) file
-`row`: the row of the CSV to be considered
"""
function Options(csvpath::String, row::Integer)
	options = DataFrames.DataFrame(CSV.File(csvpath))[row,:]
	options = Dict((name=>options[name] for name in names(options))...)
	Options(options)
end

"""
	Options(options::Dict)

RETURN a struct containing the fixed hyperparameters of the model

ARGUMENT
-`options`: a dictionary
"""
function Options(options::Dict)
	keyset = keys(options)
	defaults = Options()
	entries = 	map(fieldnames(Options)) do fieldname
					if fieldname == :datapath
						if haskey(options, "datapath")
							options["datapath"]
						else
							joinpath(options["datafolder"], options["recording_id"]*".mat")
						end
					elseif fieldname == :outputpath
						if haskey(options, "outputpath")
							options["outputpath"]
						else
							joinpath(options["outputfolder"], options["fitname"])
						end
					elseif fieldname == :sf_tbf
						getfield(defaults,fieldname)
					else
						defaultvalue = getfield(defaults,fieldname)
						if String(fieldname) ∈ keyset
							convert(typeof(defaultvalue), options[String(fieldname)])
						else
							defaultvalue
						end
					end
				end
	options = Options(entries...)
	!isdir(options.outputpath) && mkpath(options.outputpath)
	@assert isdir(options.outputpath)
	return options
end

"""
	Model(options)

RETURN a struct containing data, parameters, and hyperparameters of a factorial hidden Markov drift-diffusion model

ARGUMENT
-`options`: a struct containing the fixed hyperparameters of the model
"""
function Model(options::Options)
	data = read(matopen(options.datapath))
	singletrialset = haskey(data, "trials")
	if singletrialset
		nneurons = length(data["trials"][1]["spiketrains"][1])
	else
		nneurons = 0
		for trialset in data["trialsets"]
			nneurons += length(trialset["trials"][1]["spiketrains"][1])
		end
	end
	options.sf_tbf[1] = nneurons^options.choiceLL_scaling_exponent
	if singletrialset
		trialsets = [Trialset(options, data["trials"], 1)]
	else
		trialsets = map((trialset, trialsetindex)->Trialset(options, trialset["trials"], trialsetindex), vec(data["trialsets"]), 1:length(data["trialsets"]))
	end
	Model(options, trialsets)
end

"""
    Model(options, trialsets)

RETURN a struct containing data, parameters, and hyperparameters of a factorial hidden Markov drift-diffusion model

ARGUMENT
-`options`: a struct containing the fixed hyperparameters of the model
-`trialsets`: data used to constrain the model
"""
function Model(options::Options, trialsets::Vector{<:Trialset})
	gaussianprior=GaussianPrior(options, trialsets)
	θnative = randomize_latent_parameters(options)
	θ₀native = FHMDDM.copy(θnative)
	Model(options=options,
		   gaussianprior=gaussianprior,
		   θnative=θnative,
		   θreal=native2real(options, θnative),
		   θ₀native=θ₀native,
		   trialsets=trialsets)
end

"""
    Trialset(options, trialset)

Create a composite containing the data from one trialset

INPUT
-`nneurons_across_trialsets`: total number of neurons across trialsets
-`options`: user-specified hyperparameters of the model
-`trialset`: a dictionary contain MATLAB-exported data corresponding to a single trial-set

OUTPUT
-a composite containing the stimulus timing, behavioral choice and timing, spike times recorded during the trials of a trialset
"""
function Trialset(options::Options, trials, trialsetindex::Integer)
    trials = vec(trials)
	ntimesteps_each_trial = collect(convert(Int, trial["ntimesteps"]) for trial in trials)
	preceding_timesteps = vcat(0, cumsum(ntimesteps_each_trial[1:end-1]))
	trials = collect(Trial(m, options, preceding_timesteps[m], trials[m], trialsetindex) for m = 1:length(trials))
	mpGLMs = MixturePoissonGLM(options, trials)
    Trialset(mpGLMs=mpGLMs, trials=trials)
end

"""
	Trial(index_in_trialset, options preceding_timesteps, trial, trialsetindex)

RETURN a composite containing the stimulus timing, behavioral choice, and metadata of one trial

ARGUMENT
-`index_in_trialset`: index of this trial among all trials in this trialset
-`options`: fixed hyperparameters of the model
-`preceding_timesteps`: sum of the number of time steps in all trials from the same trialset preceding this trial
-`trial`: a `Dict` containing the data of the trial
-`trialsetindex`: index of the trialset among all trialsets
"""
function Trial(index_in_trialset::Integer, options::Options, preceding_timesteps::Integer, trial::Dict, trialsetindex::Integer)
	leftclicks = trial["clicktimes"]["L"]
	leftclicks = typeof(leftclicks)<:AbstractFloat ? [leftclicks] : vec(leftclicks)
	rightclicks = trial["clicktimes"]["R"]
	rightclicks = typeof(rightclicks)<:AbstractFloat ? [rightclicks] : vec(rightclicks)
	ntimesteps = convert(Int, trial["ntimesteps"])
	clicks = Clicks(options.a_latency_s, options.Δt, leftclicks, ntimesteps, rightclicks)
	spiketrains = collect(convert.(UInt8, vec(spiketrain)) for spiketrain in vec(trial["spiketrains"]))
	Trial(choice=trial["choice"],
		  clicks=clicks,
		  γ=trial["gamma"],
		  index_in_trialset = index_in_trialset,
		  movementtime_s=trial["movementtime_s"],
		  movementtimestep=ceil(Int, (trial["movementtime_s"]-trial["stereoclick_time_s"])/options.Δt),
		  ntimesteps=ntimesteps,
		  photostimulus_incline_on_s=trial["photostimulus_incline_on_s"],
		  photostimulus_decline_on_s=trial["photostimulus_decline_on_s"],
		  previousanswer=convert(Int, trial["previousanswer"]),
		  spiketrains=spiketrains,
		  stereoclick_time_s=trial["stereoclick_time_s"],
		  τ₀ = preceding_timesteps,
		  trialsetindex = trialsetindex)
end

"""
    Clicks(a_latency_s, L, R, Δt, ntimesteps)

Create an instance of `Clicks` to compartmentalize variables related to the times of auditory clicks in one trial

The stereoclick is excluded.

ARGUMENT
-`a_latency_s`: latency of the accumulator with respect to the clicks
-`Δt`: duration, in seconds, of each time step
-`L`: a vector of floating-point numbers specifying the times of left clicks, in seconds. Does not need to be sorted.
-`ntimesteps`: number of time steps in the trial. Time is aligned to the stereoclick. The first time window is `[-Δt, 0.0)`, and the last time window is `[ntimesteps*Δt, (ntimesteps+1)*Δt)`, defined such that `tₘₒᵥₑ - (ntimesteps+1)*Δt < Δt`, where `tₘₒᵥₑ` is the time when movement away from the center port was first detected.
-`R`: a vector of floating-point numbers specifying the times of right clicks, in seconds. Does not need to be sorted.

RETURN
-an instance of the type `Clicks`
"""
function Clicks(a_latency_s::AbstractFloat,
				Δt::AbstractFloat,
                L::Vector{<:AbstractFloat},
                ntimesteps::Integer,
                R::Vector{<:AbstractFloat})
    L = L[.!isapprox.(L, 0.0)] #excluding the stereoclick
    R = R[.!isapprox.(R, 0.0)]
	L .+= a_latency_s
	R .+= a_latency_s
	rightmost_edge_s = (ntimesteps-1)*Δt
	L = L[L.<rightmost_edge_s]
	R = R[R.<rightmost_edge_s]
    clicktimes = [L;R]
    indices = sortperm(clicktimes)
    clicktimes = clicktimes[indices]
    isright = [falses(length(L)); trues(length(R))]
    isright = isright[indices]
    is_in_timestep =
        map(1:ntimesteps) do t
            ((t-2)*Δt .<= clicktimes) .& (clicktimes .< (t-1)*Δt) # the right edge of the first time step is defined as 0.0, the time of the stereoclick
        end
    right = map(is_in_timestep) do I
                findall(I .& isright)
            end
    isleft = .!isright
    left =  map(is_in_timestep) do I
                findall(I .& isleft)
            end
	inputtimesteps=findall(sum.(is_in_timestep).>0)
	inputindex = map(t->findall(inputtimesteps .== t), 1:ntimesteps)
    Clicks(time=clicktimes,
		   inputtimesteps=inputtimesteps,
		   inputindex=inputindex,
           source=isright,
           left=left,
           right=right)
end

"""
	randomizeparameters!(model)

Randomize the parameters of the model
"""
function randomizeparameters!(model::Model)
	randomize_latent_parameters!(model::Model)
	for trialset in model.trialsets
		for mpGLM in trialset.mpGLMs
			randomizeparameters!(mpGLM.θ, model.options)
		end
	end
end
