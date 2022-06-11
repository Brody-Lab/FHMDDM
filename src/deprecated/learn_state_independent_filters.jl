
"""
	learn_state_independent_filters!(model)

Learn the filters of the state-independent inputs of each neuron's GLM

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"; randomize=true);
julia> FHMDDM.learn_state_independent_filters!(model)
```
"""
function learn_state_independent_filters!(model::Model)
	q = length(model.trialsets[1].mpGLMs[1].θ.𝐮)
	Opt = PoissonGLMOptimization(𝐮 = fill(NaN, q))
	for trialset in model.trialsets
		for mpGLM in trialset.mpGLMs
			learn_state_independent_filters!(mpGLM, Opt)
		end
	end
	return nothing
end

"""
    learn_state_independent_filters!(mpGLM, Opt)

Learn the filters of the state-independent inputs

ARGUMENT
-`mpGLM`: the Poisson mixture GLM of one neuron

OPTIONAL ARGUMENT
-`show_trace`: whether to show information about each step of the optimization

RETURN
-weights concatenated into a single vector

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"; randomize=true);
julia> mpGLM = model.trialsets[1].mpGLMs[1]
julia> Opt = FHMDDM.PoissonGLMOptimization(𝐮 = fill(NaN, length(mpGLM.θ.𝐮)))
julia> FHMDDM.estimatefilters!(mpGLM, Opt)
```
"""
function learn_state_independent_filters!(mpGLM::MixturePoissonGLM,
										Opt::PoissonGLMOptimization,
										iterations::Integer=20,
										show_trace::Bool=false)
    f(𝐮) = -loglikelihood!(mpGLM,Opt,𝐮)
	g!(∇, 𝐮) = ∇negloglikelihood!(∇,mpGLM,Opt,𝐮)
	h!(∇∇, 𝐮) = ∇∇negloglikelihood!(∇∇,mpGLM,Opt,𝐮)
    results = Optim.optimize(f, g!, h!, copy(mpGLM.θ.𝐮), NewtonTrustRegion(), Optim.Options(show_trace=show_trace, iterations=iterations))
	mpGLM.θ.𝐮 .= Optim.minimizer(results)
	return nothing
end

"""
	loglikelihood!(mpGLM, Opt, concatenatedθ)

Compute the log-likelihood of Poisson GLM

ARGUMENT
-`mpGLM`: a structure containing the parameters, input, and observations a mixture of Poisson GLM
-`Opt`: a structure for maximizing the expectation of the log-likelihood of a mixture of Poisson GLM
-`concatenatedθ`: parameters of a GLM concatenated into a vector

RETURN
-log-likelihood of a Poisson GLM
"""
function loglikelihood!(mpGLM::MixturePoissonGLM,
						Opt::PoissonGLMOptimization,
						𝐮::Vector{<:AbstractFloat})
	update!(mpGLM, Opt, 𝐮)
	return Opt.ℓ[1]
end

"""
	∇negloglikelihood!(g, mpGLM, Opt, γ, concatenatedθ)

Compute the gradient of the negative of the log-likelihood of a Poisson GLM

MODIFIED ARGUMENT
-`g`: gradient

ARGUMENT
-`mpGLM`: a structure containing the parameters, input, and observations a mixture of Poisson GLM
-`Opt`: a structure for maximizing the expectation of the log-likelihood of a mixture of Poisson GLM
-`concatenatedθ`: parameters of a GLM concatenated into a vector
"""
function ∇negloglikelihood!(g::Vector{<:AbstractFloat},
							mpGLM::MixturePoissonGLM,
							Opt::PoissonGLMOptimization,
							𝐮::Vector{<:AbstractFloat})
	update!(mpGLM, Opt, 𝐮)
	for i in eachindex(g)
		g[i] = -Opt.∇ℓ[i]
	end
	return nothing
end

"""
	∇∇negloglikelihood!(h, mpGLM, Opt, γ, concatenatedθ)

Compute the hessian of the negative of the log-likelihood of a Poisson GLM

MODIFIED ARGUMENT
-`h`: hessian

ARGUMENT
-`mpGLM`: a structure containing the parameters, input, and observations a mixture of Poisson GLM
-`Opt`: a structure for maximizing the expectation of the log-likelihood of a mixture of Poisson GLM
-`concatenatedθ`: parameters of a GLM concatenated into a vector
"""
function ∇∇negloglikelihood!(h::Matrix{<:AbstractFloat},
							mpGLM::MixturePoissonGLM,
							Opt::PoissonGLMOptimization,
							𝐮::Vector{<:AbstractFloat})
	update!(mpGLM, Opt, 𝐮)
	for i in eachindex(h)
		h[i] = -Opt.∇∇ℓ[i]
	end
	return nothing
end

"""
	update!(mpGLM, Opt, concatenatedθ)

Update quantities for computing the log-likelihood of a Poisson GLM and its gradient and hessian

MODIFIED ARGUMENT
-`mpGLM`: a structure containing the parameters, input, and observations a mixture of Poisson GLM
-`Opt`: a structure for maximizing the expectation of the log-likelihood of a mixture of Poisson GLM

ARGUMENT
-`concatenatedθ`: parameters of a GLM concatenated into a vector

EXAMPLE
```julia-rep
julia> using FHMDDM, Random
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"; randomize=true);
julia> mpGLM = model.trialsets[1].mpGLMs[1]
julia> Opt = FHMDDM.PoissonGLMOptimization(𝐮 = fill(NaN, length(mpGLM.θ.𝐮)))
julia> rand𝐮 = copy(mpGLM.θ.𝐮)
julia> FHMDDM.update!(mpGLM, Opt, rand𝐮)
julia> using ForwardDiff
julia> f(𝐮) = FHMDDM.loglikelihood(mpGLM, 𝐮)
julia> fauto = f(rand𝐮)
julia> gauto = ForwardDiff.gradient(f, rand𝐮)
julia> hauto = ForwardDiff.hessian(f, rand𝐮)
julia> abs(fauto - Opt.ℓ[1])
julia> maximum(abs.(gauto .- Opt.∇ℓ))
julia> maximum(abs.(hauto .- Opt.∇∇ℓ))
```
"""
function update!(mpGLM::MixturePoissonGLM,
				Opt::PoissonGLMOptimization,
				𝐮::Vector{<:AbstractFloat})
	if 𝐮 != Opt.𝐮
		Opt.𝐮 .= 𝐮
		mpGLM.θ.𝐮 .= 𝐮
		∇∇loglikelihood!(Opt, mpGLM)
	end
end

"""
    ∇∇loglikelihood!(Opt, mpGLM)

Compute the log-likelihood of a Poisson mixture GLM and its first and second derivatives

MODIFIED ARGUMENT
-`Opt`: a structure for maximizing the expectation of the log-likelihood of a mixture of Poisson GLM

UNMODIFIED ARGUMENT
-`mpGLM`: the Poisson mixture GLM of one neuron
```
"""
function ∇∇loglikelihood!(Opt::PoissonGLMOptimization, mpGLM::MixturePoissonGLM)
	@unpack ℓ, ∇ℓ, ∇∇ℓ = Opt
    @unpack Δt, 𝐗, 𝐲 = mpGLM
	@unpack 𝐮 = mpGLM.θ
	𝐔 = @view 𝐗[:,1:length(𝐮)]
	𝐔ᵀ = transpose(𝐔)
	𝐋 = 𝐔*𝐮
	T = length(𝐲)
	d²ℓ_dL², dℓ_dL = zeros(T), zeros(T)
	ℓ[1] = 0.0
	for t = 1:T
		d²ℓ_dL²[t], dℓ_dL[t], ℓₜ = differentiate_loglikelihood_twice_wrt_linearpredictor(Δt, 𝐋[t], 𝐲[t])
		ℓ[1] += ℓₜ
	end
	∇ℓ .= 𝐔ᵀ*dℓ_dL
	∇∇ℓ .= 𝐔ᵀ*(d²ℓ_dL².*𝐔)
	return nothing
end

"""
	loglikelihood(mpGLM, concatenatedθ)

Compute the log-likelihood of Poisson GLM

ARGUMENT
-`mpGLM`: a structure containing the parameters, input, and observations a mixture of Poisson GLM
-`𝐮`: filters of the state-independent inputs of the GLM

RETURN
-log-likelihood of a Poisson GLM
"""
function loglikelihood(mpGLM::MixturePoissonGLM, 𝐮::Vector{type}) where {type<:Real}
	@unpack Δt, 𝐗, 𝐲 = mpGLM
	𝐔 = @view 𝐗[:,1:length(𝐮)]
	𝐔ᵀ = transpose(𝐔)
	𝐋 = 𝐔*𝐮
	T = length(𝐲)
	d²ℓ_dL², dℓ_dL = zeros(type, T), zeros(type, T)
	ℓ = 0.0
	for t = 1:T
		ℓ += poissonloglikelihood(Δt, 𝐋[t], 𝐲[t])
	end
	ℓ
end


"""
    expectation_loglikelihood(γ, mpGLM, x)

ForwardDiff-compatible computation of the expectation of the log-likelihood of the mixture of Poisson generalized model of one neuron

Ignores the log(y!) term, which does not depend on the parameters

ARGUMENT
-`γ`: posterior probability of the latent variable
-`mpGLM`: the GLM of one neuron
-`x`: weights of the linear filters of the GLM concatenated as a vector of floating-point numbers

RETURN
-expectation of the log-likelihood of the spike train of one neuron
"""
function expectation_loglikelihood(concatenatedθ::Vector{<:Real},
								   γ::Matrix{<:Vector{<:AbstractFloat}},
								   mpGLM::MixturePoissonGLM)
	mpGLM = MixturePoissonGLM(concatenatedθ, mpGLM)
    @unpack Δt, 𝐲 = mpGLM
    T = length(𝐲)
    Ξ,K = size(γ)
    Q = 0.0
    @inbounds for i = 1:Ξ
	    for k = 1:K
			𝐋 = linearpredictor(mpGLM,i,k)
            for t = 1:T
				Q += γ[i,k][t]*poissonloglikelihood(Δt, 𝐋[t], 𝐲[t])
            end
        end
    end
    return Q
end
