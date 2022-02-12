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
"""
function Model(datapath::String)
    dataMAT = matopen(datapath);
    options = Options(read(dataMAT, "options"))
    trialsets = vec(map(trialset->Trialset(options, trialset), read(dataMAT, "data")))
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
	glmθ = read(resultsMAT, "glmtheta")
	for i in eachindex(trialsets)
		for n in eachindex(trialsets[i].mpGLMs)
			trialsets[i].mpGLMs[n].𝐮 .= glmθ[i][n]["u"]
			trialsets[i].mpGLMs[n].𝐯 .= glmθ[i][n]["v"]
			trialsets[i].mpGLMs[n].a .= glmθ[i][n]["a"]
			trialsets[i].mpGLMs[n].b .= glmθ[i][n]["b"]
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
				trialsets::Vector{<:Trialset})
	θnative = initializeparameters(options)
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
    rawclicktimes = map(x->x["clicktimes"], rawtrials)

    L = map(x->vec(x["L"]), rawclicktimes)
    isscalar = map(x->typeof(x),L).==Float64
    L[isscalar] = map(x->[x], L[isscalar])
    L = convert(Array{Array{Float64,1},1},L)

    R = map(x->vec(x["R"]), rawclicktimes)
    isscalar = map(x->typeof(x),R).==Float64
    R[isscalar] = map(x->[x], R[isscalar])
    R = convert(Array{Array{Float64,1},1},R)

    ntimesteps = map(x->convert(Int64, x["ntimesteps"]), rawtrials)
    choice = map(x->x["choice"], rawtrials)
	@assert typeof(trialset["lagged"]["lag"])==Float64  && trialset["lagged"]["lag"] == -1.0
    previousanswer = vec(convert.(Int64, trialset["lagged"]["answer"]))
    clicks = map((L,R,ntimesteps)->Clicks(options.a_latency_s, options.Δt,L,ntimesteps,R), L, R, ntimesteps)
    trials = map(clicks, choice, ntimesteps, previousanswer) do clicks, choice, ntimesteps, previousanswer
                Trial(clicks=clicks,
                      choice=choice,
                      ntimesteps=ntimesteps,
                      previousanswer=previousanswer)
             end

    units = vec(trialset["units"])
    𝐘 = map(x->vec(x["y"]), units)
    @assert sum(ntimesteps) == length(𝐘[1])
    𝐔ₕ = map(x->x["Xautoreg"], units)
    𝐔ₑ = trialset["Xtiming"]
	@unpack Ξ = options
	𝛏normalized = (2collect(1:Ξ) .- Ξ .- 1)./(Ξ-1) # if not normalized, the denominator is `Ξ-2`
	𝚽, Φ = temporal_bases_values(options, ntimesteps)
	if all(isempty.(𝐔ₕ))
		𝐗 = hcat(𝐔ₑ,𝚽)
		mpGLMs = map(𝐘) do 𝐲
					θ = GLMθ(𝐮 = 1.0 .- 2.0.*rand(size(𝐔ₑ,2)),
							 𝐯 = 1.0 .- 2.0.*rand(size(𝚽,2)))
					MixturePoissonGLM(Δt=options.Δt, K=options.K, 𝚽=𝚽, Φ=Φ, θ=θ, 𝐔=𝐔ₑ, 𝛏=𝛏normalized, 𝐗=𝐗, 𝐲=𝐲)
				 end
	else
		mpGLMs = map(𝐔ₕ, 𝐘) do 𝐔ₕ, 𝐲
					𝐔 = hcat(𝐔ₕ, 𝐔ₑ)
					𝐗 = hcat(𝐔, 𝚽)
					θ = GLMθ(𝐮 = 1.0 .- 2.0.*rand(size(𝐔,2)),
							 𝐯 = 1.0 .- 2.0.*rand(size(𝚽,2)))
					MixturePoissonGLM(Δt=options.Δt, K=options.K, 𝚽=𝚽, Φ=Φ, θ=θ, 𝐔=𝐔, 𝐗=𝐗, 𝛏=𝛏normalized, 𝐲=𝐲)
	             end
	end
    Trialset(mpGLMs=mpGLMs, trials=trials)
end

"""
	initializeparameters(options)

Initialize the value of each model parameters in native space by sampling from a Uniform random variable

RETURN
-values of model parameter in native space
"""
function initializeparameters(options::Options)
	Latentθ(Aᶜ₁₁=options.K==1 ? [1] : [1-rand()/10],
			Aᶜ₂₂=options.K==1 ? [1] : [1-rand()/10],
			B=options.fit_B ? 2options.q_B*rand(1) : [options.q_B],
			k=options.fit_k ? rand(1) : [options.q_k],
			λ=options.fit_λ ? [1-2rand()] : zeros(1),
			μ₀=options.fit_μ₀ ? [1-2rand()] : zeros(1),
			ϕ=options.fit_ϕ ? rand(1) : [options.q_ϕ],
			πᶜ₁=options.K==1 ? [1] : [1-rand()/10],
			ψ=options.fit_ψ ? rand(1)/10 : [options.q_ψ],
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
"""
function initializeparameters!(model::Model)
	γ = choiceposteriors(model)
	estimatefilters!(model.trialsets, γ, model.options)
	return nothing
end
