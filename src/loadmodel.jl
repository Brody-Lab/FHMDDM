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
				trialsets::Vector{<:Trialset}; randomize::Bool=false)
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
			σ²ₐ=options.fit_σ²ₐ ? rand(1) : [options.q_σ²ₐ],
			σ²ᵢ=options.fit_σ²ᵢ ? rand(1) : [options.q_σ²ᵢ],
			σ²ₛ=options.fit_σ²ₛ ? rand(1)/10 : [options.q_σ²ₛ],
			wₕ=options.fit_wₕ ? [1-2rand()] : zeros(1))
end

"""
	initializeparameters!(model)

Initialize the values of a subset of the parameters by maximizing the likelihood of only the choices.

The parameters specifying the transition probability of the coupling variable are not modified. The weights of the GLM are computed by maximizing the expectation of complete-data log-likelihood across accumulator states, assuming a coupled state.

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
	maximize_choice_posterior!(model)
	learn_state_independent_filters!(model)
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
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"
julia> model = Model(datapath; randomize=true)
julia> FHMDDM.initialize_for_stochastic_transition!(model)
```
"""
function initialize_for_stochastic_transition!(model::Model)
	@unpack options, θnative, θ₀native, θreal, trialsets = model
	@unpack K = model.options
	θ₀native.πᶜ₁[1] = θnative.πᶜ₁[1] = options.q_πᶜ₁
	θ₀native.Aᶜ₁₁[1] = θnative.Aᶜ₁₁[1] = min(0.95, options.q_Aᶜ₁₁)
	θ₀native.Aᶜ₂₂[1] = θnative.Aᶜ₂₂[1] = 1.0 - options.q_Aᶜ₂₂
	native2real!(θreal, options, θnative)
	learn_state_independent_filters!(model)
	maximize_choice_posterior!(model)
	γ = choiceposteriors(model)
	γ = map(γᵢ->dropdims(sum(γᵢ,dims=2),dims=2), γ)
	max_ntimesteps = maximum_number_of_time_steps(model)
	ntimesteps = collect(trial.ntimesteps for trialset in trialsets for trial in trialset.trials)
	𝐭 = round.(Int, collect(1:max_ntimesteps/K:max_ntimesteps+1))
	for k = 1:K
		t0 = 𝐭[k]
		t1 = 𝐭[k+1]-1
		indices = indices_for_subselection(model, t0, t1)
		γsub = subselect(indices, γ)
		for i in eachindex(model.trialsets)
			for n in eachindex(model.trialsets[i].mpGLMs)
				mpGLM = subselect(indices[i], model.trialsets[i].mpGLMs[n])
				learn_state_dependent_filters!(mpGLM, γsub[i], k)
			end
		end
	end
end

"""
	indices_for_subselection(model, t0, t1)

ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters
-`t0`: first timestep in each trial to be included
-`t1`: last timestep to be included

RETURN
-`indices`: a vector of BitVectors indicating which time steps in the trialset are used. Element `indices[i][τ]` corresponds to the τ-th time step in the i-th trialset
"""
function indices_for_subselection(model::Model, t0::Integer, t1::Integer)
	indices = map(model.trialsets) do trialset
				falses(sum(collect(trial.ntimesteps for trial in trialset.trials)))
			end
	offset = 0
	for i in eachindex(model.trialsets)
		for j in eachindex(model.trialsets[i].trials)
			indices[i][offset.+(t0:min(t1, model.trialsets[i].trials[j].ntimesteps))] .= true
		end
	end
	return indices
end

"""
	subselect(indices, γ)

Select the poster probabilities corresponding to a subset of time steps

ARGUMENT
-`indices`: a vector of BitVectors indicating which time steps in the trialset are used. Element `indices[i][τ]` corresponds to the τ-th time step in the i-th trialset
-`γ`: posterior probabilities of each accumulator and coupling state. Element `γ[i][j][τ]` corresponds to the posterior probability of the j-th accumulator state  in the τ-th time step in the i-th trialset.
-`k`: coupling state index

RETURN
-`γ`: sub-selected posterior probabilities
"""
function subselect(indices::Vector{<:BitVector}, γ::Vector{<:Vector{<:Vector{<:Real}}})
	map(indices, γ) do indices, γ
		map(γ) do γ
			γ[indices]
		end
	end
end

"""
	subselect(indices, mpGLM)

Select the input and observations corresponding to a subset of time steps

ARGUMENT
-`indices`: a BitVector indicating which time steps in the trialset are used
-`mpGLM`: a mixture of Poisson generalized linear model

RETURN
-`mpGLM`: a new structure in which the input and the observations have been sub-selected. Note that other fields reference the same memory as the corresponding fields in the original structure.
"""
function subselect(indices::BitVector, mpGLM::MixturePoissonGLM)
	MixturePoissonGLM(Δt = mpGLM.Δt,
					d𝛏_dB = mpGLM.d𝛏_dB,
					max_spikehistory_lag = mpGLM.max_spikehistory_lag,
					Φ = mpGLM.Φ,
					θ = mpGLM.θ,
					𝐕 = mpGLM.𝐕[indices,:],
					𝐗 = mpGLM.𝐗[indices,:],
					𝐲 = mpGLM.𝐲[indices])
end

"""
	learn_state_dependent_filters!(mpGLM, γ, k)

Learn the filters in the k-th state

MODIFIED ARGUMENT
-`mpGLM`: a structure containing the data and parameters of the mixture of Poisson GLM of one neuron

UNMODIFIED ARGUMENT
-`γ`: posterior probability of the latent variables. Element `γ[j][τ]` corresponds to the posterior probability of the j-th accumulator state  in the τ-th time step
-`k`: index of the coupling state
"""
function learn_state_dependent_filters!(mpGLM::MixturePoissonGLM, γ::Vector{<:Vector{<:Real}}, k::Integer; show_trace::Bool=true, iterations::Integer=20)
	Q = fill(NaN,1)
	n𝐯 = length(mpGLM.θ.𝐯[k])
	∇Q = fill(NaN, n𝐯)
	∇∇Q = fill(NaN, n𝐯, n𝐯)
	f(𝐯ₖ) = -expectation_of_loglikelihood!(mpGLM,Q,∇Q,∇∇Q,γ,k,𝐯ₖ)
	∇f!(∇, 𝐯ₖ) = ∇negexpectation_of_loglikelihood!(∇,mpGLM,Q,∇Q,∇∇Q,γ,k,𝐯ₖ)
	∇∇f!(∇∇, 𝐯ₖ) = ∇∇negexpectation_of_loglikelihood!(∇∇,mpGLM,Q,∇Q,∇∇Q,γ,k,𝐯ₖ)
    results = Optim.optimize(f, ∇f!, ∇∇f!, copy(mpGLM.θ.𝐯[k]), NewtonTrustRegion(), Optim.Options(show_trace=show_trace, iterations=iterations))
	mpGLM.θ.𝐯[k] .= Optim.minimizer(results)
	return nothing
end

"""
	expectation_of_loglikelihood!(mpGLM,Q,∇Q,∇∇Q,γ,k,𝐯ₖ)

Expectation of the log-likelihood under the posterior probability of the latent variables

Only the component that depend on the state-dependent filters k-th state are included

MODIFIED ARGUMENT
-`mpGLM`: a structure containing the data and parameters of the mixture of Poisson GLM of one neuron
-`Q`: an one-element vector that quantifies the expectation
-`∇Q`: gradient of the expectation with respect to the filters in the k-th state
-`∇∇Q`: Hessian of the expectation with respect to the filters in the k-th state

UNMODIFIED ARGUMENT
-`γ`: posterior probability of the latent variables. Element `γ[j][τ]` corresponds to the posterior probability of the j-th accumulator state  in the τ-th time step
-`k`: index of the coupling state
-`𝐯ₖ`: values of the filters of the accumulator-dependent input in the k-th state
"""
function expectation_of_loglikelihood!(mpGLM::MixturePoissonGLM, Q::Vector{<:Real}, ∇Q::Vector{<:Real}, ∇∇Q::Matrix{<:Real}, γ::Vector{<:Vector{<:Real}}, k::Integer, 𝐯ₖ::Vector{<:Real})
	if (mpGLM.θ.𝐯[k] != 𝐯ₖ) || isnan(Q[1])
		mpGLM.θ.𝐯[k] .= 𝐯ₖ
		∇∇expectation_of_loglikelihood!(Q,∇Q,∇∇Q,γ,k,mpGLM)
	end
	Q[1]
end

"""
	∇negexpectation_of_loglikelihood!(∇,mpGLM,Q,∇Q,∇∇Q,γ,k,𝐯ₖ)

Gradient of the negative of the expectation of the log-likelihood under the posterior probability of the latent variables

MODIFIED ARGUMENT
-`∇`: gradient of the negative of the expectation
-`mpGLM`: a structure containing the data and parameters of the mixture of Poisson GLM of one neuron
-`Q`: an one-element vector that quantifies the expectation
-`∇Q`: gradient of the expectation with respect to the filters in the k-th state
-`∇∇Q`: Hessian of the expectation with respect to the filters in the k-th state

UNMODIFIED ARGUMENT
-`γ`: posterior probability of the latent variables. Element `γ[j][τ]` corresponds to the posterior probability of the j-th accumulator state  in the τ-th time step
-`k`: index of the coupling state
-`𝐯ₖ`: values of the filters of the accumulator-dependent input in the k-th state
"""
function ∇negexpectation_of_loglikelihood!(∇::Vector{<:Real}, mpGLM::MixturePoissonGLM, Q::Vector{<:Real}, ∇Q::Vector{<:Real}, ∇∇Q::Matrix{<:Real}, γ::Vector{<:Vector{<:Real}}, k::Integer, 𝐯ₖ::Vector{<:Real})
	if (mpGLM.θ.𝐯[k] != 𝐯ₖ) || isnan(Q[1])
		mpGLM.θ.𝐯[k] .= 𝐯ₖ
		∇∇expectation_of_loglikelihood!(Q,∇Q,∇∇Q,γ,k,mpGLM)
	end
	for i in eachindex(∇)
		∇[i] = -∇Q[i]
	end
	return nothing
end

"""
	∇∇negexpectation_of_loglikelihood!(∇∇,mpGLM,Q,∇Q,∇∇Q,γ,k,𝐯ₖ)

Hessian of the negative of the expectation of the log-likelihood under the posterior probability of the latent variables

MODIFIED ARGUMENT
-`∇∇`: Hessian of the negative of the expectation
-`mpGLM`: a structure containing the data and parameters of the mixture of Poisson GLM of one neuron
-`Q`: an one-element vector that quantifies the expectation
-`∇Q`: gradient of the expectation with respect to the filters in the k-th state
-`∇∇Q`: Hessian of the expectation with respect to the filters in the k-th state

UNMODIFIED ARGUMENT
-`γ`: posterior probability of the latent variables. Element `γ[j][τ]` corresponds to the posterior probability of the j-th accumulator state  in the τ-th time step
-`k`: index of the coupling state
-`𝐯ₖ`: values of the filters of the accumulator-dependent input in the k-th state
"""
function ∇∇negexpectation_of_loglikelihood!(∇∇::Matrix{<:Real}, mpGLM::MixturePoissonGLM, Q::Vector{<:Real}, ∇Q::Vector{<:Real}, ∇∇Q::Matrix{<:Real}, γ::Vector{<:Vector{<:Real}}, k::Integer, 𝐯ₖ::Vector{<:Real})
	if (mpGLM.θ.𝐯[k] != 𝐯ₖ) || isnan(Q[1])
		mpGLM.θ.𝐯[k] .= 𝐯ₖ
		∇∇expectation_of_loglikelihood!(Q,∇Q,∇∇Q,γ,k,mpGLM)
	end
	n𝐯 = length(𝐯ₖ)
	for i =1:n𝐯
		for j=i:n𝐯
			∇∇[i,j] = ∇∇[j,i] = -∇∇Q[i,j]
		end
	end
	return nothing
end

"""
	∇∇expectation_of_loglikelihood!(Q,∇Q,∇∇Q,γ,k,mpGLM)

Compute the expectation of the log-likelihood and its gradient and Hessian

ARGUMENT
-`Q`: expectation of the log-likelihood under the posterior probability of the latent variables. Only the component in the coupling state `k` is included
-`∇Q`: first-order derivatives of the expectation
-`∇∇Q`: second-order derivatives of the expectation

UNMODIFIED ARGUMENT
-`γ`: posterior probabilities of the latent variables
-`k`: index of the coupling state
-`mpGLM`: a structure containing the data and parameters of the mixture of Poisson GLM of one neuron
"""
function ∇∇expectation_of_loglikelihood!(Q::Vector{<:Real},
										∇Q::Vector{<:Real},
										∇∇Q::Matrix{<:Real},
										γ::Vector{<:Vector{<:Real}},
										k::Integer,
										mpGLM::MixturePoissonGLM)
    @unpack Δt, 𝐕, d𝛏_dB, 𝐲 = mpGLM
	d𝛏_dB² = d𝛏_dB.^2
	Ξ = size(γ,1)
	T = length(𝐲)
	∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB = zeros(T)
	∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB² = zeros(T)
	Q[1] = 0.0
	@inbounds for i = 1:Ξ
		𝐋 = linearpredictor(mpGLM,i,k)
		for t=1:T
			d²ℓ_dL², dℓ_dL, ℓ = differentiate_loglikelihood_twice_wrt_linearpredictor(Δt, 𝐋[t], 𝐲[t])
			Q[1] += γ[i][t]*ℓ
			dQᵢₖ_dLᵢₖ = γ[i][t] * dℓ_dL
			∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB[t] += γ[i][t] * dℓ_dL * d𝛏_dB[i]
			∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB²[t] += γ[i][t] * d²ℓ_dL² * d𝛏_dB²[i]
		end
	end
	𝐕ᵀ = transpose(𝐕)
	∇Q .= 𝐕ᵀ*∑ᵢ_dQᵢₖ_dLᵢₖ⨀dξᵢ_dB
	∇∇Q .= 𝐕ᵀ*(∑ᵢ_d²Qᵢₖ_dLᵢₖ²⨀dξᵢ_dB².*𝐕)
	return nothing
end
