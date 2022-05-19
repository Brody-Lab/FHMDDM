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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true);
```
"""
function Model(datapath::String; randomize::Bool=true)
    dataMAT = matopen(datapath);
    options = Options(read(dataMAT, "options"))
    trialsets = vec(map(trialset->Trialset(options, trialset), read(dataMAT, "data")))
    if isfile(options.resultspath)
        Model(options, options.resultspath, trialsets)
    else
        Model(options, trialsets; randomize=randomize)
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
			for k in eachindex(glmθ[i][n]["v"])
				trialsets[i].mpGLMs[n].θ.𝐯[k] .= glmθ[i][n]["v"][k]
			end
		end
	end
	Model(options=options,
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
function Model(options::Options,
				trialsets::Vector{<:Trialset}; randomize::Bool=true)
	θnative = randomize ? randomlyinitialize(options) : initializeparameters(options)
	θ₀native = Latentθ(([getfield(θnative, f)...] for f in fieldnames(typeof(θnative)))...) # just making a deep copy
	Model(options=options,
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
	𝐔₀ = ones(size(trialset["Xtiming"],1))
	mpGLMs = map(units, 𝐘) do unit, 𝐲
				𝐗=hcat(𝐔₀, unit["Xautoreg"], trialset["Xtiming"], 𝐕)
				MixturePoissonGLM(Δt=options.Δt,
  								d𝛏_dB=d𝛏_dB,
								max_spikehistory_lag = size(unit["Xautoreg"],2),
								Φ=Φ,
								θ=GLMθ(K, 𝐗, 𝐕),
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
	initializeparameters(options)

Initialize the value of each model parameters in native space with defaults

RETURN
-values of model parameter in native space
"""
function initializeparameters(options::Options)
	Latentθ(Aᶜ₁₁=[options.q_Aᶜ₁₁],
			Aᶜ₂₂=[options.q_Aᶜ₂₂],
			B=[options.q_B],
			k=[options.q_k],
			λ=options.fit_λ ? [-1e-2] : zeros(1),
			μ₀=zeros(1),
			ϕ=[options.q_ϕ],
			πᶜ₁=[options.q_πᶜ₁],
			ψ=[options.q_ψ],
			σ²ₐ=[options.q_σ²ₐ],
			σ²ᵢ=[options.q_σ²ᵢ],
			σ²ₛ=[options.q_σ²ₛ],
			wₕ=zeros(1))
end

"""
	initializeparameters(options)

Initialize the value of each model parameters in native space by sampling from a Uniform random variable

RETURN
-values of model parameter in native space
"""
function randomlyinitialize(options::Options)
	Latentθ(Aᶜ₁₁=options.K==2 ? [options.bound_z + rand()*(1-2*options.bound_z)] : [options.q_Aᶜ₁₁],
			Aᶜ₂₂=options.K==2 ? [options.bound_z + rand()*(1-2*options.bound_z)] : [options.q_Aᶜ₂₂],
			B=options.fit_B ? 2options.q_B*rand(1) : [options.q_B],
			k=options.fit_k ? rand(1) : [options.q_k],
			λ=options.fit_λ ? [1-2rand()] : zeros(1),
			μ₀=options.fit_μ₀ ? [1-2rand()] : zeros(1),
			ϕ=options.fit_ϕ ? rand(1) : [options.q_ϕ],
			πᶜ₁=options.K==2 ? [options.bound_z + rand()*(1-2*options.bound_z)] : [options.q_πᶜ₁],
			ψ=options.fit_ψ ? [options.bound_ψ + rand()*(1-2*options.bound_ψ)] : [options.q_ψ],
			σ²ₐ=options.fit_σ²ₐ ? [options.bounds_σ²ₐ[1]+rand()*diff(options.bounds_σ²ₐ)[1]] : [options.q_σ²ₐ],
			σ²ᵢ=options.fit_σ²ᵢ ? [options.bounds_σ²ᵢ[1]+rand()*diff(options.bounds_σ²ᵢ)[1]] : [options.q_σ²ᵢ],
			σ²ₛ=options.fit_σ²ₛ ? [options.bounds_σ²ₛ[1]+rand()*diff(options.bounds_σ²ₛ)[1]] : [options.q_σ²ₛ],
			wₕ=options.fit_wₕ ? [1-2rand()] : zeros(1))
end

"""
	initializeparameters!(model)

Initialize the values of a subset of the parameters by maximizing the likelihood of only the choices.

The parameters specifying the transition probability of the coupling variable are not modified. The weights of the GLM are computed by maximizing the expectation of complete-data log-likelihood across accumulator states.

MODIFIED ARGUMENT
-`model`: an instance of the factorial hidden Markov drift-diffusion model

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"
julia> model = Model(datapath; randomize=true)
julia> FHMDDM.initializeparameters!(model)
```
"""
function initializeparameters!(model::Model)
	@unpack options, θnative, θ₀native, θreal, trialsets = model
	@unpack K = model.options
	FHMDDM.maximize_choice_posterior!(model)
	memory = Memoryforgradient(model)
	choiceposteriors!(memory, model)
	for i in eachindex(model.trialsets)
	    for mpGLM in model.trialsets[i].mpGLMs
	        maximize_expectation_of_loglikelihood!(mpGLM, memory.γ[i])
	    end
	end
	return nothing
end

"""
	initialize_for_stochastic_transition!(model)

Initialize the parameters of the model such that the state transitions over time

MODIFIED ARGUMENT
-`model`: a structure containing the data, parameters, and the hyperparameters of the model. The prior and transition probabilities of the coupling variable are modified. The drift-diffusion parameters are optimized to maximize the posterior likelihood of the choices. The state-independent filters are optimized by maximizing the likelihood of each neuron's GLM as though it does not dependent on the accumulator. The filters of the accumulator dependent input are optimized by performing a single M-step using the posterior probabilities of the latent variable conditioned on only the choices. The data are split such that a different subset of the time steps in each trial used to optimize the filters in each state. The first subset of the time steps are used to optimize the filters of the accumulator in the first state, and the last subset of time steps are used to optimize the filters in the last state.

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_11_test/T176_2018_05_03_001/data.mat"
julia> model = Model(datapath; randomize=true)
julia> ℓs = FHMDDM.initialize_for_stochastic_transition!(model;EMiterations=100)
```
"""
function initialize_for_stochastic_transition!(model::Model; EMiterations::Integer=1, relativeΔℓ::Real=1e-6)
	@unpack options, θnative, θ₀native, θreal, trialsets = model
	@unpack K = model.options
	θ₀native.πᶜ₁[1] = θnative.πᶜ₁[1] = options.q_πᶜ₁
	θ₀native.Aᶜ₁₁[1] = θnative.Aᶜ₁₁[1] = min(0.95, options.q_Aᶜ₁₁)
	θ₀native.Aᶜ₂₂[1] = θnative.Aᶜ₂₂[1] = 1.0 - options.q_Aᶜ₂₂
	native2real!(θreal, options, θnative)
	FHMDDM.maximize_choice_posterior!(model)
	memory = FHMDDM.Memoryforgradient(model)
	FHMDDM.choiceposteriors!(memory, model)
	for i in eachindex(model.trialsets)
	    for mpGLM in model.trialsets[i].mpGLMs
	        FHMDDM.maximize_expectation_of_loglikelihood!(mpGLM, memory.γ[i])
	    end
	end
	FHMDDM.posteriors!(memory, model)
	for i in eachindex(model.trialsets)
	    for mpGLM in model.trialsets[i].mpGLMs
	        FHMDDM.maximize_expectation_of_loglikelihood!(mpGLM, memory.γ[i])
	    end
	end
	ℓs = fill(NaN, EMiterations)
	∑γ = fill(NaN, model.options.K)
	∑χ = fill(NaN, model.options.K, model.options.K)
	for i = 1:EMiterations
		FHMDDM.joint_posteriors_of_coupling!(memory, model, ∑χ, ∑γ)
		ℓs[i] = memory.ℓ[1]
		if (i > 1) && (ℓs[i]-ℓs[i-1] >= 0.0) && ((ℓs[i]-ℓs[i-1])/abs(ℓs[i-1]) < relativeΔℓ)
			break
		end
		for s in eachindex(model.trialsets)
		    for mpGLM in model.trialsets[s].mpGLMs
		        FHMDDM.maximize_expectation_of_loglikelihood!(mpGLM, memory.γ[s])
		    end
		end
		FHMDDM.maximizeECDLL!(model, ∑χ, ∑γ)
	end
	return ℓs
end

"""
	maximizeECDLL!(model, ∑χ)

Optimize the parameters of the coupling variable by maximizing the expectation of the complete-data log-likelihood

MODIFIED ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters

UNMODIFIED ARGUMENT
-`∑χ`: sum of the joint posterior probability of the coupling variable across consecutive time steps
-`∑γ`: sum of the posterior probability of the coupling variable across consecutive time steps
"""
function maximizeECDLL!(model::Model, ∑χ::Matrix{<:Real}, ∑γ::Vector{<:Real})
	@unpack options, θnative, θreal = model
	θnative.Aᶜ₁₁[1] = max(min(∑χ[1,1]/(∑χ[1,1]+∑χ[2,1]), 1-options.bound_z), options.bound_z)
	θnative.Aᶜ₂₂[1] = max(min(∑χ[2,2]/(∑χ[2,2]+∑χ[1,2]), 1-options.bound_z), options.bound_z)
	θnative.πᶜ₁[1] = max(min(∑γ[1]/(∑γ[1]+∑γ[2]), 1-options.bound_z), options.bound_z)
	native2real!(θreal, options, θnative)
end
