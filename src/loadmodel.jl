"""
    Model(datapath; fit_to_choices)

Load a factorial hidden Markov drift diffusion model from a MATLAB file.

If the model has already been optimized, a results file is expected.

ARGUMENT
- `datapath`: full path of the data file

OPTIONAL ARGUMENT
-`fit_to_choices`: whether the initial values the model parameters are learned by fitting to the choices alone

RETURN
- a structure containing information for a factorial hidden Markov drift-diffusion model

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat");
```
"""
function Model(datapath::String)
    dataMAT = read(matopen(datapath))
    options = Options(dataMAT["options"])
	if haskey(dataMAT, "data")
	    trialsets = map(trialset->Trialset(options, trialset), vec(dataMAT["data"]))
	else
		trialsets = map(trialset->Trialset(trialset), vec(dataMAT["trialsets"]))
		for trialset in trialsets
			for mpGLM in trialset.mpGLMs
				mpGLM.θ.𝐠[1] = 0
				for k = 2:length(mpGLM.θ.𝐠)
					mpGLM.θ.𝐠[k] = 1-2rand()
				end
				mpGLM.θ.𝐮 .= 1.0 .- 2.0.*rand(length(mpGLM.θ.𝐮))
				for 𝐯ₖ in mpGLM.θ.𝐯
					𝐯ₖ .= 1.0 .- 2.0.*rand(length(𝐯ₖ))
				end
			end
		end
	end
    if isfile(options.resultspath)
        Model(options, options.resultspath, trialsets)
    else
        Model(options, trialsets)
    end
end

"""
    Model(options, resultspath, trialsets)

Load a previously fitted factorial hidden Markov drift-diffusion model

ARGUMENT
-`options`: model settings
-`resultspath`: full path of the file containing the learned values of the parameters
-`trialsets`: data used to constrain the model

RETURN
- a structure containing information for a factorial hidden Markov drift-diffusion model
"""
function Model(options::Options, resultspath::String, trialsets::Vector{<:Trialset})
    resultsMAT = matopen(resultspath)
	glmθ = read(resultsMAT, "thetaglm")
	for i in eachindex(trialsets)
		for n in eachindex(trialsets[i].mpGLMs)
			trialsets[i].mpGLMs[n].θ.b .= glmθ[i][n]["b"]
			trialsets[i].mpGLMs[n].θ.𝐮 .= glmθ[i][n]["u"]
        	for k in eachindex(glmθ[i][n]["g"])
				trialsets[i].mpGLMs[n].θ.𝐠[k] = glmθ[i][n]["g"][k]
			end
			for k in eachindex(glmθ[i][n]["v"])
				trialsets[i].mpGLMs[n].θ.𝐯[k] .= glmθ[i][n]["v"][k]
			end
		end
	end
	gaussianprior = GaussianPrior(options, trialsets)
	gaussianprior.𝛂 .= vec(read(resultsMAT, "penaltycoefficients"))
    precisionmatrix!(gaussianprior)
	Model(options=options,
		   gaussianprior=gaussianprior,
		   θnative=Latentθ(read(resultsMAT, "theta_native")),
		   θreal=Latentθ(read(resultsMAT, "theta_real")),
		   θ₀native=Latentθ(read(resultsMAT, "theta0_native")),
		   trialsets=trialsets)
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
	copy(latentθ)

Make a copy of an instance of `Latentθ`
"""
FHMDDM.copy(latentθ::Latentθ) = Latentθ(([getfield(latentθ, f)...] for f in fieldnames(Latentθ))...)

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
    Trialset(options, trialset)

Parse data exported from MATLAB to create a structure containing data for one trial-set

INPUT
-`trialset`: a dictionary contain MATLAB-exported data corresponding to a single trial-set
-`options`: model settings

OUTPUT
-an instance of `trialsetdata`
"""
function Trialset(options::Options, trialset::Dict)
	inttype = typeof(1)
	floattype = typeof(1.0)
    rawtrials = vec(trialset["trials"])
	movementtimes_s = map(x->x["movementtime_s"], rawtrials)
	@assert all(movementtimes_s.>0)
    𝐓 = map(x->convert(inttype, x["ntimesteps"]), rawtrials)
	units = vec(trialset["units"])
    𝐘 = map(x->convert.(typeof(1), vec(x["y"])), units)
	Ttrialset = sum(𝐓)
    @assert all(length.(𝐘) .== Ttrialset)
	@unpack K, Ξ = options
	d𝛏_dB = (2collect(1:Ξ) .- Ξ .- 1)./(Ξ-1)
	𝐆 = ones(Ttrialset)
	Φₕ = FHMDDM.spikehistorybases(options)
	𝐔ₕ = map(𝐲->FHMDDM.spikehistorybases(Φₕ, 𝐓, 𝐲), 𝐘)
	𝐔ₜ, Φₜ = FHMDDM.timebases(options, 𝐓)
	Φₘ = FHMDDM.premovementbases(options)
	𝐔ₘ = FHMDDM.premovementbases(movementtimes_s, options, Φₘ, 𝐓)
	𝐕, Φₐ = FHMDDM.accumulatorbases(options, 𝐓)
	𝐮indices_hist = 1:size(Φₕ,2)
	𝐮indices_time = 𝐮indices_hist[end] .+ (1:size(Φₜ,2))
	𝐮indices_move = 𝐮indices_time[end] .+ (1:size(Φₘ,2))
	mpGLMs = map(𝐔ₕ, 𝐘) do 𝐔ₕ, 𝐲
				𝐗=hcat(𝐆, 𝐔ₕ, 𝐔ₜ, 𝐔ₘ, 𝐕)
				glmθ = GLMθ(options, 𝐮indices_hist, 𝐮indices_move, 𝐮indices_time, 𝐕)
				MixturePoissonGLM(Δt=options.Δt,
  								d𝛏_dB=d𝛏_dB,
								Φₐ=Φₐ,
								Φₕ=Φₕ,
								Φₘ=Φₘ,
								Φₜ=Φₜ,
								θ=glmθ,
								𝐕=𝐕,
								𝐗=𝐗,
								𝐲=𝐲)
			 end
	rawclicktimes = map(x->x["clicktimes"], rawtrials)
    L = map(rawclicktimes) do x
			leftclicks = x["L"]
			typeof(leftclicks)<:AbstractFloat ? [leftclicks] : vec(leftclicks)
		end
	R = map(rawclicktimes) do x
			rightclicks = x["R"]
			typeof(rightclicks)<:AbstractFloat ? [rightclicks] : vec(rightclicks)
		end
    clicks = map((L,R,T)->Clicks(options.a_latency_s, options.Δt,L,T,R), L, R, 𝐓)
	preceding_timesteps = vcat(0, cumsum(𝐓[1:end-1]))
	indices_in_trialset = 1:length(𝐓)
    trialsetindex = convert(inttype, trialset["index"])
    trials = map(clicks, indices_in_trialset, rawtrials, preceding_timesteps) do clicks, index_in_trialset, rawtrial, preceding_timesteps
                Trial(clicks=clicks,
                      choice=rawtrial["choice"],
					  movementtime_s=rawtrial["movementtime_s"],
                      ntimesteps=convert(inttype, rawtrial["ntimesteps"]),
                      previousanswer=convert(inttype, rawtrial["previousanswer"]),
					  index_in_trialset = index_in_trialset,
					  τ₀ = preceding_timesteps,
					  trialsetindex = trialsetindex)
             end
    Trialset(mpGLMs=mpGLMs, trials=trials)
end

"""
	Trialset(trialset)

Create an instance of `Trialset` from a saved file
"""
function Trialset(trialset::Dict)
	trials = map(trialset["trials"]) do trial
				time = typeof(trial["clicks"]["time"])<:AbstractFloat ? [trial["clicks"]["time"]] : trial["clicks"]["time"]
				inputtimesteps = typeof(trial["clicks"]["inputtimesteps"])<:Integer ? [trial["clicks"]["inputtimesteps"]] : trial["clicks"]["inputtimesteps"]
				inputindex =  map(trial["clicks"]["inputindex"]) do x
						           typeof(x)<:Integer ? [x] : x
						       end
				left =  map(trial["clicks"]["left"]) do x
				           typeof(x)<:Integer ? [x] : x
				       end
				right =  map(trial["clicks"]["right"]) do x
				           typeof(x)<:Integer ? [x] : x
				       	end
				if typeof(trial["clicks"]["source"]) <: Bool
					source = convert(BitArray{1}, [trial["clicks"]["source"]])
				else
					source = convert(BitArray{1}, trial["clicks"]["source"])
				end
				clicks = Clicks(time=time,
								inputtimesteps=inputtimesteps,
								inputindex=inputindex,
								source=source,
								left=left,
								right=right)
				Trial(clicks=clicks,
                      choice=trial["choice"],
                      ntimesteps=trial["ntimesteps"],
                      previousanswer=trial["previousanswer"])
			end
	d𝛏_dB = trialset["mpGLMs"][1]["dxi_dB"]
	Φₐ = trialset["mpGLMs"][1]["Phiaccumulator"]
	Φₕ = trialset["mpGLMs"][1]["Phihistory"]
	Φₘ = trialset["mpGLMs"][1]["Phipremovement"]
	Φₜ = trialset["mpGLMs"][1]["Phitime"]
	𝐕 = trialset["mpGLMs"][1]["V"]
	𝐮indices_hist = min(trialset["mpGLMs"][1]["theta"]["uindices_hist"]):max(trialset["mpGLMs"][1]["theta"]["uindices_hist"])
	𝐮indices_move = min(trialset["mpGLMs"][1]["theta"]["uindices_move"]):max(trialset["mpGLMs"][1]["theta"]["uindices_move"])
	𝐮indices_time = min(trialset["mpGLMs"][1]["theta"]["uindices_time"]):max(trialset["mpGLMs"][1]["theta"]["uindices_time"])
	mpGLMs = map(trialset["mpGLMs"]) do mpGLM
				𝐠 = typeof(mpGLM["theta"]["g"])<:AbstractFloat ? [mpGLM["theta"]["g"]] : mpGLM["theta"]["g"]
				𝐯 = map(mpGLM["theta"]["v"]) do x
			           	typeof(x)<:AbstractFloat ? [x] : x
			        end
				θ = GLMθ(𝐠=𝐠, 𝐮=mpGLM["theta"]["u"], 𝐯=𝐯, 𝐮indices_hist=𝐮indices_hist, 𝐮indices_move=𝐮indices_move, 𝐮indices_time=𝐮indices_time)
				MixturePoissonGLM(Δt=mpGLM["dt"],
									d𝛏_dB=d𝛏_dB,
									Φₐ=Φₐ,
									Φₕ=Φₕ,
									Φₘ=Φₘ,
									Φₜ=Φₜ,
									θ=θ,
									𝐕=𝐕,
									𝐗=mpGLM["X"],
									𝐲=mpGLM["y"])
			end
	Trialset(mpGLMs=mpGLMs, trials=trials)
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
