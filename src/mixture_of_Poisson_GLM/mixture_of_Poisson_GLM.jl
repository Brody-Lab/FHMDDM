"""
	MixturePoissonGLM(concatenatedθ, mpGLM)

Create a structure for a mixture of Poisson GLM with updated parameters

ARGUMENT
-`concatenatedθ`: a vector of new parameter values
-`mpGLM`: a structure containing information on the mixture of Poisson GLM for one neuron

OUTPUT
-a new structure for the mixture of Poisson GLM of a neuron with new parameter values
"""
function MixturePoissonGLM(concatenatedθ::Vector{<:Real}, mpGLM::MixturePoissonGLM; offset::Integer=0, initialization::Bool=false)
	values = map(fieldnames(MixturePoissonGLM)) do fieldname
				if fieldname == :θ
					GLMθ(mpGLM.θ, concatenatedθ; offset=offset, initialization=initialization)
				else
					getfield(mpGLM, fieldname)
				end
			end
	return MixturePoissonGLM(values...)
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
    @unpack 𝐗, d𝛏_dB, Ξ = mpGLM
    @unpack b, b_scalefactor, 𝐠, 𝐮, 𝐯, 𝛃, fit_𝛃 = mpGLM.θ
	gₖ = 𝐠[min(length(𝐠), k)]
	if j == 1 || j == Ξ
		if fit_𝛃
			𝐰ₖ = 𝛃[min(length(𝛃), k)].*d𝛏_dB[j]
		else
			𝐰ₖ = 𝐯[min(length(𝐯), k)].*d𝛏_dB[j]
		end
	else
		𝐯ₖ = 𝐯[min(length(𝐯), k)]
		transformedξ = transformaccumulator(b[1]*b_scalefactor, d𝛏_dB[j])
		𝐰ₖ = 𝐯ₖ.*transformedξ
	end
	𝐗*vcat(gₖ, 𝐮, 𝐰ₖ)
end

"""
	linearpredictor_without_transformation(mpGLM, j, k)

Linear combination without transforming the accumulated evidence

ARGUMENT
-see above

RETURN
-see above
"""
function linearpredictor_without_transformation(mpGLM::MixturePoissonGLM, j::Integer, k::Integer)
	@unpack 𝐗, d𝛏_dB, Ξ = mpGLM
	@unpack 𝐠, 𝐮, 𝐯, 𝛃, fit_𝛃 = mpGLM.θ
	gₖ = 𝐠[min(length(𝐠), k)]
	if (j == 1 || j == Ξ) && fit_𝛃
		𝐰ₖ = 𝛃[min(length(𝛃), k)].*d𝛏_dB[j]
	else
		𝐰ₖ = 𝐯[min(length(𝐯), k)].*d𝛏_dB[j]
	end
	𝐗*vcat(gₖ, 𝐮, 𝐰ₖ)
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
    𝐩 = 𝐋
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
"""
function expectation_∇loglikelihood!(∇Q::GLMθ, γ::Matrix{<:Vector{<:Real}}, mpGLM::MixturePoissonGLM)
	@unpack Δt, 𝐕, 𝐗, Ξ, 𝐲 = mpGLM
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
	if ∇Q.fit_𝛃
		∑_bounds_dQᵢₖ_dLᵢₖ⨀ωᵢ = collect(zeros(T) for k=1:K)
	end
	@inbounds for k = 1:K
		for i = (1,Ξ)
			𝐋 = linearpredictor(mpGLM,i,k)
			for t=1:T
				dQᵢₖ_dLᵢₖ = γ[i,k][t] * differentiate_loglikelihood_wrt_linearpredictor(Δt, 𝐋[t], 𝐲[t])
				∑ᵢ_dQᵢₖ_dLᵢₖ[k][t] += dQᵢₖ_dLᵢₖ
				if ∇Q.fit_𝛃
					∑_bounds_dQᵢₖ_dLᵢₖ⨀ωᵢ[k][t] += dQᵢₖ_dLᵢₖ*𝛚[i]
				else
					∑ᵢ_dQᵢₖ_dLᵢₖ⨀ωᵢ[k][t] += dQᵢₖ_dLᵢₖ*𝛚[i]
				end
			end
		end
		for i = 2:Ξ-1
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
	if ∇Q.fit_𝛃
		if length(∇Q.𝛃) == K
			@inbounds for k = 1:K
				mul!(∇Q.𝛃[k], 𝐕', ∑_bounds_dQᵢₖ_dLᵢₖ⨀ωᵢ[k])
			end
		else
			mul!(∇Q.𝛃[1], 𝐕', sum(∑_bounds_dQᵢₖ_dLᵢₖ⨀ωᵢ))
		end
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
function expectation_of_loglikelihood(γ::Matrix{<:Vector{<:AbstractFloat}}, mpGLM::MixturePoissonGLM, x::Vector{<:Real}; initialization::Bool=false)
	mpGLM = FHMDDM.MixturePoissonGLM(x, mpGLM; initialization=initialization)
    @unpack Δt, 𝐲 = mpGLM
    T = length(𝐲)
    Ξ,K = size(γ)
    Q = 0.0
    @inbounds for i = 1:Ξ
	    for k = 1:K
			if initialization
				𝐋 = linearpredictor_without_transformation(mpGLM,i,k)
			else
				𝐋 = linearpredictor(mpGLM,i,k)
			end
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
			if ∇ℓglm.fit_𝛃
				for 𝛃ₖ in ∇ℓglm.𝛃
					for β in 𝛃ₖ
						counter+=1
						∇nℓ[counter] = -β
					end
				end
			end
		end
	end
	return nothing
end

"""
	postspikefilter(mpGLM)

Return a vector representing the post-spike filter of a Poisson mixture GLM.

The first element of the vector corresponds to the first time step after the spike.
"""
function postspikefilter(mpGLM::MixturePoissonGLM)
	@unpack Φₕ, θ = mpGLM
	@unpack 𝐮, 𝐮indices_hist = θ
	return Φₕ*𝐮[𝐮indices_hist]
end

"""
	externalinput(mpGLM)

Sum the input from extern events for each time step in a trialset.

The external events typically consist of the stereoclick, departure from the center port, and the photostimulus.

RETURN
-a vector whose τ-th element corresponds to the τ-th time step in the trialset
"""
function externalinput(mpGLM::MixturePoissonGLM)
	@unpack 𝐗, 𝐗columns_time, 𝐗columns_move, 𝐗columns_phot, θ = mpGLM
	@unpack 𝐮, 𝐮indices_time, 𝐮indices_move, 𝐮indices_phot = θ
	𝐄 = @view 𝐗[:,vcat(𝐗columns_time, 𝐗columns_move, 𝐗columns_phot)]
	𝐞 = 𝐮[vcat(𝐮indices_time, 𝐮indices_move, 𝐮indices_phot)]
	return 𝐄*𝐞
end
