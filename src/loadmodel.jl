"""
    Model(datapath; fit_to_choices)

Load a factorial hidden Markov drift diffusion model from a MATLAB file.

If the model has already been optimized, a results file is expected.

ARGUMENT
- `datapath`: full path of the data file

RETURN
- a structure containing information for a factorial hidden Markov drift-diffusion model
"""
function Model(datapath::String)
    dataMAT = read(matopen(datapath))
	nunits = 0
	for rawtrialset in dataMAT["data"]
		nunits += length(rawtrialset["units"])
	end
    options = Options(nunits, dataMAT["options"])
	trialsets = map(trialset->Trialset(options, trialset), vec(dataMAT["data"]))
	Model(options, trialsets)
end

"""
    Model(options, trialsets)

Create a factorial hidden Markov drift-diffusion model

ARGUMENT
-`options`: model settings
-`trialsets`: data used to constrain the model

RETURN
- a structure containing information for a factorial hidden Markov drift-diffusion model
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
-`options`: user-specified hyperparameters of the model
-`trialset`: a dictionary contain MATLAB-exported data corresponding to a single trial-set

OUTPUT
-a composite containing the stimulus timing, behavioral choice and timing, spike times recorded during the trials of a trialset
"""
function Trialset(options::Options, trialset::Dict)
	trialsetindex = convert(Int,trialset["index"])
    trials = vec(trialset["trials"])
	𝐓 = map(x->convert(Int, x["ntimesteps"]), trials)
	preceding_timesteps = vcat(0, cumsum(𝐓[1:end-1]))
	𝐘 = map(x->convert.(UInt8, vec(x["y"])), vec(trialset["units"]))
	photostimulus_decline_on_s = collect(trial["photostimulus_decline_on_s"] for trial in trials)
	photostimulus_incline_on_s = collect(trial["photostimulus_incline_on_s"] for trial in trials)
	trials = collect(Trial(options.a_latency_s, options.Δt, m, preceding_timesteps[m], trials[m], trialsetindex) for m = 1:length(trials))
	movementtimesteps = collect(trial.movementtimestep for trial in trials)
	mpGLMs = MixturePoissonGLM(movementtimesteps, options, photostimulus_decline_on_s, photostimulus_incline_on_s, 𝐓, 𝐘)
    Trialset(mpGLMs=mpGLMs, trials=trials)
end

"""
	Trial(a_latency_s, Δt, index_in_trialset, preceding_timesteps, trial, trialsetindex)

Create a composite containing the data of one trial

ARGUMENT
-`a_latency_s`: latency, in seconds, with which the latent variable responds to the auditory clicks
-`Δt`: duration, in seconds, of each time step in the model
-`index_in_trialset`: index of this trial among all trials in this trialset
-`preceding_timesteps`: sum of the number of time steps in all trials from the same trialset preceding this trial
-`trial`: a `Dict` containing the data of the trial
-`trialsetindex`: index of the trialset among all trialsets

RETURN
-a composite containing the stimulus timing, behavioral choice, and relevant metadata for of one trial
"""
function Trial(a_latency_s::AbstractFloat,
				Δt::AbstractFloat,
				index_in_trialset::Integer,
				preceding_timesteps::Integer,
				trial::Dict,
				trialsetindex::Integer)
	leftclicks = trial["clicktimes"]["L"]
	leftclicks = typeof(leftclicks)<:AbstractFloat ? [leftclicks] : vec(leftclicks)
	rightclicks = trial["clicktimes"]["R"]
	rightclicks = typeof(rightclicks)<:AbstractFloat ? [rightclicks] : vec(rightclicks)
	ntimesteps = convert(Int, trial["ntimesteps"])
	clicks = Clicks(a_latency_s, Δt, leftclicks, ntimesteps, rightclicks)
	if haskey(trial, "movementtimestep")
		movementtimestep = convert(Int, trial["movementtimestep"])
	else
		movementtimestep = ceil(Int, trial["movementtime_s"]/Δt)
	end
	Trial(clicks=clicks,
		  choice=trial["choice"],
		  γ=trial["gamma"],
		  movementtimestep=movementtimestep,
		  ntimesteps=ntimesteps,
		  photostimulus_incline_on_s=trial["photostimulus_incline_on_s"],
		  photostimulus_decline_on_s=trial["photostimulus_decline_on_s"],
		  previousanswer=convert(Int, trial["previousanswer"]),
		  index_in_trialset = index_in_trialset,
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
	randomize_latent_parameters(options)

Initialize the value of each model parameters in native space by sampling from a Uniform random variable""

RETURN
-values of model parameter in native space
"""
function randomize_latent_parameters(options::Options)
	θnative = Latentθ()
	randomize_latent_parameters!(θnative, options)
	return θnative
end

"""
	randomize_latent_parameters!(model)

Set the value of each latent-variable parameter as a sample from a Uniform distribution.

Only parameters being fit are randomized

MODIFIED ARGUMENT
-`model`: a structure containing the parameters, hyperparameters, and data of the factorial hidden-Markov drift-diffusion model. Its fields `θnative` and `θreal` are modified.
"""
function randomize_latent_parameters!(model::Model)
	@unpack options, θnative, θreal = model
	randomize_latent_parameters!(θnative, options)
	native2real!(θreal, options, θnative)
end

"""
	randomize_latent_parameters!(θnative, options)

Set the value of each latent-variable parameter as a sample from a Uniform distribution.

Only parameters being fit are randomized

MODIFIED ARGUMENT
-`θnative`: latent variables' parameters in native space

UNMODIFIED ARGUMENT
-`options`: settings of the model
"""
function randomize_latent_parameters!(θnative::Latentθ, options::Options)
	for field in fieldnames(typeof(θnative))
		fit = is_parameter_fit(options, field)
		lqu = getfield(options, Symbol("lqu_"*string(field)))
		l = lqu[1]
		q = lqu[2]
		u = lqu[3]
		getfield(θnative, field)[1] = fit ? l + (u-l)*rand() : q
	end
	return nothing
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

"""
	reindex(index_in_trialset, τ₀, trial)

Instantiate a trial with new indices for subsampling

ARGUMENT
-`index_in_trialset`: index of trial in the subsampled trialset
-`τ₀`: number of time steps summed across all preceding trials in the trialset
-`trial`: structure containing the stimulus and behavioral information of a trial
"""
function reindex(index_in_trialset::Integer, τ₀::Integer, trial::Trial)
	fieldvalues = map(fieldnames(Trial)) do fieldname
		if fieldname == :index_in_trialset
			index_in_trialset
		elseif fieldname == :τ₀
			τ₀
		else
			getfield(trial, fieldname)
		end
	end
	Trial(fieldvalues...)
end
