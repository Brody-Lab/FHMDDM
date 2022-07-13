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
				for 𝐠ₖ in mpGLM.θ.𝐠
					𝐠ₖ .= 1.0 .- 2.0.*rand(length(𝐠ₖ))
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
function Model(options::Options,
		 		resultspath::String,
		 		trialsets::Vector{<:Trialset})
    resultsMAT = matopen(resultspath)
	glmθ = read(resultsMAT, "thetaglm")
	for i in eachindex(trialsets)
		for n in eachindex(trialsets[i].mpGLMs)
			trialsets[i].mpGLMs[n].θ.𝐮 .= glmθ[i][n]["u"]
        	for k in eachindex(glmθ[i][n]["g"])
				trialsets[i].mpGLMs[n].θ.𝐠[k] .= glmθ[i][n]["g"][k]
			end
			for k in eachindex(glmθ[i][n]["v"])
				trialsets[i].mpGLMs[n].θ.𝐯[k] .= glmθ[i][n]["v"][k]
			end
		end
	end
	𝛂 = vec(read(resultsMAT, "shrinkagecoefficients"))
	𝐬 = vec(read(resultsMAT, "smoothingcoefficients"))
	gaussianprior = GaussianPrior(options, trialsets, vcat(𝛂,𝐬))
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
	θnative = initializeparameters(options)
	θ₀native = Latentθ(([getfield(θnative, f)...] for f in fieldnames(typeof(θnative)))...) # making a deep copy
	Model(options=options,
		   gaussianprior=gaussianprior,
		   θnative=θnative,
		   θreal=native2real(options, θnative),
		   θ₀native=θ₀native,
		   trialsets=trialsets)
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
    Trialset(options, trialset)

Parse data exported from MATLAB to create a structure containing data for one trial-set

INPUT
-`trialset`: a dictionary contain MATLAB-exported data corresponding to a single trial-set
-`options`: model settings

OUTPUT
-an instance of `trialsetdata`
"""
function Trialset(options::Options, trialset::Dict)
    rawtrials = vec(trialset["trials"])
    ntimesteps = map(x->convert(Int64, x["ntimesteps"]), rawtrials)
	units = vec(trialset["units"])
    𝐘 = map(x->convert.(Int64, vec(x["y"])), units)
    @assert sum(ntimesteps) == length(𝐘[1])
	@unpack K, Ξ = options
	d𝛏_dB = (2collect(1:Ξ) .- Ξ .- 1)./(Ξ-2)
	𝐕, Φ = temporal_bases_values(options, ntimesteps)
	𝐆 = ones(size(trialset["Xtiming"],1))
	mpGLMs = map(units, 𝐘) do unit, 𝐲
				𝐗=hcat(𝐆, unit["Xautoreg"], trialset["Xtiming"], 𝐕)
				MixturePoissonGLM(Δt=options.Δt,
  								d𝛏_dB=d𝛏_dB,
								max_spikehistory_lag = size(unit["Xautoreg"],2),
								Φ=Φ,
								θ=GLMθ(options, 𝐗, 𝐕),
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
	@assert typeof(trialset["lagged"]["lag"])==Float64  && trialset["lagged"]["lag"] == -1.0
    previousanswer = vec(convert.(Int64, trialset["lagged"]["answer"]))
    clicks = map((L,R,ntimesteps)->Clicks(options.a_latency_s, options.Δt,L,ntimesteps,R), L, R, ntimesteps)
    trials = map(clicks, rawtrials, ntimesteps, previousanswer) do clicks, rawtrial, ntimesteps, previousanswer
                Trial(clicks=clicks,
                      choice=rawtrial["choice"],
                      ntimesteps=ntimesteps,
                      previousanswer=previousanswer)
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
	Φ = trialset["mpGLMs"][1]["Phi"]
	𝐕 = trialset["mpGLMs"][1]["V"]
	mpGLMs = map(trialset["mpGLMs"]) do mpGLM
				𝐠 = map(mpGLM["theta"]["g"]) do x
			           	typeof(x)<:AbstractFloat ? [x] : x
			        end
				𝐯 = map(mpGLM["theta"]["v"]) do x
			           	typeof(x)<:AbstractFloat ? [x] : x
			        end
				θ = GLMθ(𝐠=𝐠, 𝐮=mpGLM["theta"]["u"], 𝐯=𝐯)
				MixturePoissonGLM(Δt=mpGLM["dt"],
									d𝛏_dB=d𝛏_dB,
									max_spikehistory_lag=mpGLM["max_spikehistory_lag"],
									Φ=Φ,
									θ=θ,
									𝐕=𝐕,
									𝐗=mpGLM["X"],
									𝐲=mpGLM["y"])
			end
	Trialset(mpGLMs=mpGLMs, trials=trials)
end

"""
	initializeparameters(options)

Initialize the value of each model parameters in native space by sampling from a Uniform random variable""

RETURN
-values of model parameter in native space
"""
function initializeparameters(options::Options)
	θnative = Latentθ()
	for field in fieldnames(Latentθ)
		fit = is_parameter_fit(options, field)
		lqu = getfield(options, Symbol("lqu_"*string(field)))
		l = lqu[1]
		q = lqu[2]
		u = lqu[3]
		getfield(θnative, field)[1] = fit ? l + (u-l)*rand() : q
	end
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
	for field in fieldnames(Latentθ)
		fit = is_parameter_fit(options, field)
		lqu = getfield(options, Symbol("lqu_"*string(field)))
		l = lqu[1]
		q = lqu[2]
		u = lqu[3]
		getfield(θnative, field)[1] = fit ? l + (u-l)*rand() : q
	end
	native2real!(θreal, options, θnative)
end
