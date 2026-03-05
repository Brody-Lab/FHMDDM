using Random, Printf, CSV, DataFrames
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
	Model(datapath, outputpath)

RETURN a model under default parameters

ARGUMENT
-`datapath`: absolute path of the binary MATLAB (``.mat`) file containing the data
-`outputpath`: absolute path of the folder where the parameters and predictions of the model are to be stored
"""
# Model(datapath::String, outputpath::String) = Model(Options(Dict("datapath"=>datapath, "outputpath"=>outputpath)))
Model(datapath::String, outputpath::String; do_shuffle::Bool=false, shuffle_seed::Int=0) =
    Model(Options(Dict(
        "datapath"=>datapath,
        "outputpath"=>outputpath,
        "do_shuffle"=>do_shuffle,
        "shuffle_seed"=>shuffle_seed,
    )))
"""
	Options(csvpath, row)

RETURN a struct containing the fixed hyperparameters of the model

ARGUMENT
-`csvpath`: the absolute path to a comma-separated values (CSV) file
-`row`: the row of the CSV to be considered
"""
Options(csvpath::String, row::Integer) = Options(DataFrames.DataFrame(CSV.File(csvpath)), row)
Options(df::DataFrames.DataFrame, row::Integer) = Options(df[row,:])
Options(dfrow::DataFrames.DataFrameRow) = Options(Dict((name=>dfrow[name] for name in names(dfrow))...))

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
Model(options::Options) = Model(options, loadtrialsets(options))

"""
	loadtrialsets(options)

RETURN a vector of objects of the type `Trialset`
"""
function loadtrialsets(options::Options)
	file = matopen(options.datapath)
	data = read(file)
	close(file)
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
		# [Trialset(options, data["trials"], 1)]
        neurons = data["neurons"]   # or whatever key it is in your .mat
        [Trialset(options, data["trials"], neurons, 1)]
	else
		# map((trialset, trialsetindex)->Trialset(options, trialset["trials"], trialsetindex), vec(data["trialsets"]), 1:length(data["trialsets"]))
        map((trialset, trialsetindex) -> begin
            neurons = trialset["neurons"] 
            Trialset(options, trialset["trials"], neurons, trialsetindex)
        end, vec(data["trialsets"]), 1:length(data["trialsets"]))
    end
end

"""
	loadtrials(options)

RETURN a vector of elements of type `Trial`

ARGUMENT
-`options`: a struct containing the fixed hyperparameters of the model
"""
function loadtrials(options::Options)
	data = read(matopen(options.datapath))
	singletrialset = haskey(data, "trials")
	if singletrialset
		trials = data["trials"]
	else
		trials = vcat((trialset["trials"] for trialset in vec(data["trialsets"]))...)
	end
	processtrials(options, trials, 1)
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
    Trialset(options, trials, trialsetindex)

Create a composite containing the data from one trialset

ARGUMENT
-`options`: user-specified hyperparameters of the model
-`trials`: a vector of objects of the type `Dict`
-`trialsetindex`: index of the trialset

OUTPUT
-a composite containing the stimulus timing, behavioral choice and timing, spike times recorded during the trials of a trialset
"""
# function Trialset(options::Options, trials, trialsetindex::Integer)
# 	trials = processtrials(options, trials, trialsetindex)
# 	mpGLMs = MixturePoissonGLM(options, trials)
#     Trialset(mpGLMs=mpGLMs, trials=trials)
# end

# Match MATLAB idea: same click pattern => same "seed"
# We build a stable string key from L/R click times (rounded to microseconds).
function clickpattern_key(trialdict; digits::Int=6)
    L = trialdict["clicktimes"]["L"]; L = (L isa AbstractFloat) ? [L] : vec(L)
    R = trialdict["clicktimes"]["R"]; R = (R isa AbstractFloat) ? [R] : vec(R)

    L = round.(L; digits=digits)
    R = round.(R; digits=digits)

    function vec2str(v)
        isempty(v) && return ""
        return join((@sprintf("%.*f", digits, x) for x in v), ",")
    end
    return "L:" * vec2str(L) * "|R:" * vec2str(R)
end
function Trialset(options::Options, trials, neurons, trialsetindex::Integer)
    raw_trials = vec(trials)  # <-- keep the Dicts for seedkey computation
    trials = processtrials(options, raw_trials, trialsetindex) # original

    if options.do_shuffle
        @warn "Trialset WITH neurons called" do_shuffle=options.do_shuffle outputpath=options.outputpath pwd=pwd() ntrials=length(trials) nneurons=length(neurons)
        rng = MersenneTwister(options.shuffle_seed)

        shuffle_rows = DataFrame(
            trialsetindex = Int[],
            region = String[],
            seedkey = String[],
            choice = Int[],                 # 0/1
            original_trial_idx = Int[],     # NOW: index_in_session
            shuffled_trial_idx = Int[],     # NOW: index_in_session
        )

        # --------- neuron indices grouped by region ----------
        brainareas = [neurons[j]["brainarea"] for j in 1:length(neurons)]
        inds_by_region = Dict{String, Vector{Int}}()
        for (j, ba) in enumerate(brainareas)
            push!(get!(inds_by_region, ba, Int[]), j)
        end

        # --------- trial indices grouped by choice ----------
        left_trials  = findall(t -> t.choice == false, trials)  # choice is Bool in Trial
        right_trials = findall(t -> t.choice == true,  trials)

        # --------- trial indices grouped by click-pattern "seed" ----------
        trial_seedkeys = [clickpattern_key(raw_trials[i]) for i in 1:length(raw_trials)]
        uniq_keys = sort!(unique(trial_seedkeys))

        # helper: shuffle spike trains for a set of trials, but only for selected neurons
        function shuffle_spikes!(trial_inds::Vector{Int}, neuron_inds::Vector{Int},
                                 region::String, seedkey::String, choice::Int)
            if length(trial_inds) <= 1 || isempty(neuron_inds)
                return
            end

            # snapshot original spike trains for these neurons
            orig = Dict{Int, Vector{Vector{UInt8}}}()
            for ti in trial_inds
                orig[ti] = [copy(trials[ti].spiketrains[n]) for n in neuron_inds]
            end

            perm = copy(trial_inds)
            shuffle!(rng, perm)

            # record mapping using index_in_session (stable id)
            for (dst, src) in zip(trial_inds, perm)
                dst_id = Int(trials[dst].index_in_session)
                src_id = Int(trials[src].index_in_session)
                push!(shuffle_rows, (trialsetindex, region, seedkey, choice, dst_id, src_id))
            end

            # apply shuffle (still done by in-memory indices)
            for (dst, src) in zip(trial_inds, perm)
                for (k, n) in enumerate(neuron_inds)
                    trials[dst].spiketrains[n] = orig[src][k]
                end
            end
        end

        # --------- main: shuffle within {region, seed(clickpattern), choice} ----------
        for region in sort!(collect(keys(inds_by_region)))
            neuron_inds = inds_by_region[region]

            for seedkey in uniq_keys
                seed_trials = findall(i -> trial_seedkeys[i] == seedkey, 1:length(trials))

                # intersect with each choice
                lt = intersect(seed_trials, left_trials)
                rt = intersect(seed_trials, right_trials)

                shuffle_spikes!(lt, neuron_inds, region, seedkey, 0)
                shuffle_spikes!(rt, neuron_inds, region, seedkey, 1)
            end
        end

        shuffle_dir = joinpath(options.outputpath, "shuffle_maps")
        isdir(shuffle_dir) || mkpath(shuffle_dir)

        csvpath = joinpath(shuffle_dir, "trialset$(trialsetindex)_region_seed_choice_shuffle.csv")
        CSV.write(csvpath, shuffle_rows)
        @info "Wrote shuffle map" csvpath nrows=nrow(shuffle_rows)
    end

    # Continue as usual
    mpGLMs = MixturePoissonGLM(options, trials) # original
    return Trialset(mpGLMs=mpGLMs, trials=trials) # original
end

"""
	processtrials(options, trials, trialsetindex)

RETURN a vector of objects of type `Trial`

ARGUMENT
-see above `Trialset(options, trials, trialsetindex)`
"""
function processtrials(options::Options, trials, trialsetindex::Integer)
	trials = vec(trials)
	ntimesteps_each_trial = collect(convert(Int, trial["ntimesteps"]) for trial in trials)
	preceding_timesteps = vcat(0, cumsum(ntimesteps_each_trial[1:end-1]))
	collect(Trial(m, options, preceding_timesteps[m], trials[m], trialsetindex) for m = 1:length(trials))
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

    # Eva: trial id from MATLAB
    idx_session = convert(Int, trial["index_in_session"])
    
    Trial(choice=trial["choice"],
		  clicks=clicks,
		  γ=trial["gamma"],
		  index_in_trialset = index_in_trialset,
          index_in_session = idx_session, # <-- EVA NEW FIELD
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
