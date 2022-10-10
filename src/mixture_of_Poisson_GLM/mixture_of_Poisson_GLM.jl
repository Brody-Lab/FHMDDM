"""
	MixturePoissonGLM(concatenatedθ, glmθindex, mpGLM)

Create a structure for a mixture of Poisson GLM with updated parameters

ARGUMENT
-`concatenatedθ`: a vector of new parameter values
-`glmθindex`: index of each parameter in the vector of values
-`mpGLM`: a structure containing information on the mixture of Poisson GLM for one neuron

OUTPUT
-a new structure for the mixture of Poisson GLM of a neuron with new parameter values
"""
function MixturePoissonGLM(concatenatedθ::Vector{T},
						   mpGLM::MixturePoissonGLM;
						   offset::Integer=0,
						   omitb::Bool=false) where {T<:Real}
	mpGLM = MixturePoissonGLM(Δt=mpGLM.Δt,
							d𝛏_dB=mpGLM.d𝛏_dB,
							Φₐ=mpGLM.Φₐ,
                        	Φₕ=mpGLM.Φₕ,
							Φₘ=mpGLM.Φₘ,
							Φₜ=mpGLM.Φₜ,
							θ=GLMθ(mpGLM.θ, concatenatedθ; offset=offset, omitb=omitb),
							𝐕=mpGLM.𝐕,
							𝐗=mpGLM.𝐗,
							𝐲=mpGLM.𝐲)
	return mpGLM
end

"""
    linearpredictor(mpGLM, j, k)

Linear combination of the weights in the j-th accumulator state and k-th coupling state

ARGUMENT
-`mpGLM`: the mixture of Poisson generalized linear model of one neuron
-`j`: state of the accumulator variable
-`k`: state of the coupling variable

RETURN
-`𝐋`: a vector whose element 𝐋[t] corresponds to the t-th time bin in the trialset
"""
function linearpredictor(mpGLM::MixturePoissonGLM, j::Integer, k::Integer)
    @unpack 𝐗, d𝛏_dB = mpGLM
    @unpack b, b_scalefactor, 𝐠, 𝐮, 𝐯 = mpGLM.θ
	gₖ = 𝐠[min(length(𝐠), k)]
	𝐯ₖ = 𝐯[min(length(𝐯), k)]
	transformedξ = transformaccumulator(b[1]*b_scalefactor, d𝛏_dB[j])
	𝐗*vcat(gₖ, 𝐮, 𝐯ₖ.*transformedξ)
end

"""
    scaledlikelihood(mpGLM, j, k, s)

Conditional likelihood of the spike train, given the index of the state of the accumulator `j` and the state of the coupling `k`, and also by the prior likelihood of the regression weights

UNMODIFIED ARGUMENT
-`mpGLM`: the mixture of Poisson generalized linear model for one neuron
-`j`: state of accumulator variable
-`k`: state of the coupling variable

RETURN
-`𝐩`: a vector by which the conditional likelihood of the spike train and the prior likelihood of the regression weights are multiplied against
"""
function scaledlikelihood(mpGLM::MixturePoissonGLM, j::Integer, k::Integer, s::Real)
    @unpack Δt, 𝐲 = mpGLM
    𝐋 = linearpredictor(mpGLM, j, k)
    𝐩 = 𝐋 # reuse memory
    @inbounds for i=1:length(𝐩)
        𝐩[i] = scaledpoissonlikelihood(Δt, 𝐋[i], s, 𝐲[i])
    end
    return 𝐩
end

"""
    scaledlikelihood!(𝐩, mpGLM, j, k, s)

In-place multiplication of `𝐩` by the conditional likelihood of the spike train, given the index of the state of the accumulator `j` and the state of the coupling `k`, and also by the prior likelihood of the regression weights

MODIFIED ARGUMENT
-`𝐩`: a vector by which the conditional likelihood of the spike train and the prior likelihood of the regression weights are multiplied against

UNMODIFIED ARGUMENT
-`mpGLM`: the mixture of Poisson generalized linear model for one neuron
-`j`: state of accumulator variable
-`k`: state of the coupling variable

RETURN
-`nothing`
"""
function scaledlikelihood!(𝐩::Vector{<:Real}, mpGLM::MixturePoissonGLM, j::Integer, k::Integer, s::Real)
    @unpack Δt, 𝐲 = mpGLM
    𝐋 = linearpredictor(mpGLM, j, k)
    @inbounds for i=1:length(𝐩)
		𝐩[i] *= scaledpoissonlikelihood(Δt, 𝐋[i], s, 𝐲[i])
    end
    return nothing
end

"""
	expectation_∇loglikelihood!(∇Q, γ, mpGLM)

Expectation under the posterior probability of the gradient of the log-likelihood.

This function is used for computing the gradient of the log-likelihood of the entire model

MODIFIED ARGUMENT
-`∇`: The gradient

UNMODIFIED ARGUMENT
-`γ`: Joint posterior probability of the accumulator and coupling variable. γ[i,k][t] corresponds to the i-th accumulator state and the k-th coupling state in the t-th time bin in the trialset.
-`mpGLM`: structure containing information for the mixture of Poisson GLM for one neuron

EXAMPLE
```julia-rep
julia> using FHMDDM, Random
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"; randomize=true);
julia> mpGLM = model.trialsets[1].mpGLMs[1]
julia> γ = FHMDDM.randomposterior(mpGLM; rng=MersenneTwister(1234))
julia> ∇Q = FHMDDM.GLMθ(mpGLM.θ, eltype(mpGLM.θ.𝐮))
julia> FHMDDM.expectation_∇loglikelihood!(∇Q, γ, mpGLM)
julia> ghand = FHMDDM.concatenateparameters(∇Q)[1]
julia> using ForwardDiff
julia> concatenatedθ = FHMDDM.concatenateparameters(mpGLM.θ)[1]
julia> f(x) = FHMDDM.expectation_loglikelihood(x, γ, mpGLM)
julia> gauto = ForwardDiff.gradient(f, concatenatedθ)
julia> maximum(abs.(gauto .- ghand))
```
"""
function expectation_∇loglikelihood!(∇Q::GLMθ, γ::Matrix{<:Vector{<:Real}}, mpGLM::MixturePoissonGLM)
	@unpack Δt, 𝐕, 𝐗, 𝐲 = mpGLM
	@unpack 𝐯 = mpGLM.θ
	𝛚 = transformaccumulator(mpGLM)
	d𝛚_db = dtransformaccumulator(mpGLM)
	Ξ, K = size(γ)
	T = length(𝐲)
	∑ᵢ_dQᵢₖ_dLᵢₖ = collect(zeros(T) for k=1:K)
	∑ᵢ_dQᵢₖ_dLᵢₖ⨀ωᵢ = collect(zeros(T) for k=1:K)
	if ∇Q.fit_b
		∑ᵢ_dQᵢₖ_dLᵢₖ⨀dωᵢ_db = collect(zeros(T) for k=1:K)
	end
	@inbounds for i = 1:Ξ
		for k = 1:K
			𝐋 = linearpredictor(mpGLM,i,k)
			for t=1:T
				dQᵢₖ_dLᵢₖ = γ[i,k][t] * differentiate_loglikelihood_wrt_linearpredictor(Δt, 𝐋[t], 𝐲[t])
				∑ᵢ_dQᵢₖ_dLᵢₖ[k][t] += dQᵢₖ_dLᵢₖ
				∑ᵢ_dQᵢₖ_dLᵢₖ⨀ωᵢ[k][t] += dQᵢₖ_dLᵢₖ*𝛚[i]
				if ∇Q.fit_b
					∑ᵢ_dQᵢₖ_dLᵢₖ⨀dωᵢ_db[k][t] += dQᵢₖ_dLᵢₖ*d𝛚_db[i]
				end
			end
		end
	end
	𝐔 = @view 𝐗[:, 2:1+length(∇Q.𝐮)]
	∑ᵢₖ_dQᵢₖ_dLᵢₖ = sum(∑ᵢ_dQᵢₖ_dLᵢₖ)
    ∇Q.𝐮 .= 𝐔' * ∑ᵢₖ_dQᵢₖ_dLᵢₖ
	@inbounds for k = 2:length(∇Q.𝐠)
		∇Q.𝐠[k] = sum(∑ᵢ_dQᵢₖ_dLᵢₖ[k])
	end
	if length(∇Q.𝐯) == K
		@inbounds for k = 1:K
			mul!(∇Q.𝐯[k], 𝐕', ∑ᵢ_dQᵢₖ_dLᵢₖ⨀ωᵢ[k])
		end
	else
		mul!(∇Q.𝐯[1], 𝐕', sum(∑ᵢ_dQᵢₖ_dLᵢₖ⨀ωᵢ))
	end
	if ∇Q.fit_b
		if length(∇Q.𝐯) == K
			∇Q.b[1] = 0.0
			@inbounds for k = 1:K
				∇Q.b[1] += dot(∑ᵢ_dQᵢₖ_dLᵢₖ⨀dωᵢ_db[k], 𝐕, 𝐯[k])
			end
		else
			∇Q.b[1] = dot(sum(∑ᵢ_dQᵢₖ_dLᵢₖ⨀dωᵢ_db), 𝐕, 𝐯[k])
		end
	end
	return nothing
end

"""
	scale_expectation_∇loglikelihood(∇Q, s)

Multiply the expectation of the gradient of the log-likelihood of a mixture of Poisson GLM with the scale factor

MODIFIED ARGUMENT
-`∇Q`: expectation of the gradient of the log-likelihood of a mixture of Poisson GLM
-`s`: scale factor
"""
function scale_expectation_∇loglikelihood!(∇Q::GLMθ, s::Real)
	@inbounds for k = 2:length(∇Q.𝐠)
		∇Q.𝐠[k] *= s
	end
	for i in eachindex(∇Q.𝐮)
		∇Q.𝐮[i] *= s
	end
	for 𝐯ₖ in ∇Q.𝐯
		for i in eachindex(𝐯ₖ)
			𝐯ₖ[i] *= s
		end
	end
	if ∇Q.fit_b
		∇Q.b[1] *= s
	end
	return nothing
end

"""
    expectation_of_loglikelihood(γ, mpGLM, x)

ForwardDiff-compatible computation of the expectation of the log-likelihood of the mixture of Poisson generalized model of one neuron

Ignores the log(y!) term, which does not depend on the parameters

ARGUMENT
-`γ`: posterior probability of the latent variable
-`mpGLM`: the GLM of one neuron
-`x`: weights of the linear filters of the GLM concatenated as a vector of floating-point numbers

RETURN
-expectation of the log-likelihood of the spike train of one neuron
"""
function expectation_of_loglikelihood(γ::Matrix{<:Vector{<:AbstractFloat}},
									   mpGLM::MixturePoissonGLM,
									   x::Vector{<:Real};
										omitb::Bool=false)
	mpGLM = MixturePoissonGLM(x, mpGLM; omitb=omitb)
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

"""
	∇negativeloglikelihood!(∇nℓ, ∇ℓglms, offset)

Concatenate the first-order partial derivatives of the model's log-likelihood w.r.t. to the parameters in each neuron's GLM

MODIFIED ARGUMENT
-`∇nℓ`: a vector representing the gradient of the model's log-likelihood

UNMODIFIED ARGUMENT
-`∇ℓglm`: a nested vector of the partial derivatives of the model's log-likelihood w.r.t. to the  parameter of each neuron's mixture of Poisson GLM. Element `∇ℓglms[i][n]` corresponds to the n-th neuron in the i-th trialset
-`offset`: the number of elements at beginning of `∇nℓ` that are unrelated to the GLM's
"""
function ∇negativeloglikelihood!(∇nℓ::Vector{<:Real}, ∇ℓglm::Vector{<:Vector{<:GLMθ}}, offset::Integer)
	counter = offset
	for ∇ℓglm in ∇ℓglm
		for ∇ℓglm in ∇ℓglm
			if ∇ℓglm.fit_b
				counter+=1
				∇nℓ[counter] = -∇ℓglm.b[1]
			end
			for k = 2:length(∇ℓglm.𝐠)
				counter+=1
				∇nℓ[counter] = -∇ℓglm.𝐠[k]
			end
			for u in ∇ℓglm.𝐮
				counter+=1
				∇nℓ[counter] = -u
			end
			for 𝐯ₖ in ∇ℓglm.𝐯
				for v in 𝐯ₖ
					counter+=1
					∇nℓ[counter] = -v
				end
			end
		end
	end
	return nothing
end
