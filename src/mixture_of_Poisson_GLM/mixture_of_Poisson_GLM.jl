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
    @unpack θ, 𝐗 = mpGLM
    @unpack 𝐠, 𝐮 = θ
	gₖ = 𝐠[min(length(𝐠), k)]
	𝐰 = evidenceweight(j, k, mpGLM)
	𝐗*vcat(gₖ, 𝐮, 𝐰)
end

"""
	evidenceweight(j,k,mpGLM)

Encoding weight of the accumulated evidence conditioned on the states of the latent variables

ARGUMENT
-`j`: state of the accumulator variable
-`k`: state of the coupling variable
-`mpGLM`: the mixture of Poisson generalized linear model of one neuron

RETURN
-`𝐰`: vector representing the encoding weight of accumulated evidence
"""
function evidenceweight(j::Integer, k::Integer, mpGLM::MixturePoissonGLM)
	@unpack d𝛏_dB, Ξ = mpGLM
    @unpack b, b_scalefactor, 𝐯, Δ𝐯, fit_Δ𝐯 = mpGLM.θ
	kᵥ = min(length(𝐯), k)
	if (j == 1 || j == Ξ) && fit_Δ𝐯
		𝐯ₖ = 𝐯[kᵥ] .+ Δ𝐯[kᵥ]
	else
		𝐯ₖ = 𝐯[kᵥ]
	end
	if b != 0.0
		𝐯ₖ.*transformaccumulator(b[1]*b_scalefactor, d𝛏_dB[j])
	else
		𝐯ₖ.*d𝛏_dB[j]
	end
end

"""
	conditionallikelihood!(p, mpGLM, τ)

Conditional likelihood the spike train response at a single timestep

MODIFIED ARGUMENT
-`p`: a matrix whose element `p[i,j]` represents the likelihood conditioned on the accumulator in the i-th state and the coupling in the j-th state

UNMODIFIED ARGUMENT
-`mpGLM`:a structure with information on the mixture of Poisson GLM of a neuron
-`τ`: timestep among time steps concatenated across all trials in a trialset
"""
function conditionallikelihood!(p::Matrix{<:Real}, mpGLM::MixturePoissonGLM, τ::Integer)
	@unpack Δt, θ, 𝐕, 𝐗, 𝐗columns_gain, 𝐲 = mpGLM
	@unpack 𝐠, 𝐮 = θ
	Gₜ = 𝐗[τ,𝐗columns_gain]
	𝐔ₜ𝐮 = 0
	offset𝐔 = maximum(𝐗columns_gain)
	for i in eachindex(𝐮)
		q = offset𝐔 + i
		𝐔ₜ𝐮 += 𝐗[τ,q]*𝐮[i]
	end
	Ξ, K = size(p)
	K𝐠 = length(𝐠)
	for k=1:K
		gₖ = 𝐠[min(k,K𝐠)]
		Gₜgₖ = Gₜ⋅gₖ
		for j=1:Ξ
			𝐰 = evidenceweight(j,k,mpGLM)
			L = Gₜgₖ + 𝐔ₜ𝐮 + 𝐕[τ,:]⋅𝐰
			p[j,k] = poissonlikelihood(Δt, L, 𝐲[τ])
		end
	end
	return nothing
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
	if ∇Q.fit_Δ𝐯
		∑_bounds_dQᵢₖ_dLᵢₖ⨀ωᵢ = collect(zeros(T) for k=1:K)
	end
	@inbounds for k = 1:K
		for i = 1:Ξ
			𝐋 = linearpredictor(mpGLM,i,k)
			for t=1:T
				dQᵢₖ_dLᵢₖ = γ[i,k][t] * differentiate_loglikelihood_wrt_linearpredictor(Δt, 𝐋[t], 𝐲[t])
				∑ᵢ_dQᵢₖ_dLᵢₖ[k][t] += dQᵢₖ_dLᵢₖ
				∑ᵢ_dQᵢₖ_dLᵢₖ⨀ωᵢ[k][t] += dQᵢₖ_dLᵢₖ*𝛚[i]
				if ∇Q.fit_Δ𝐯 && (i==1 || i==Ξ)
					∑_bounds_dQᵢₖ_dLᵢₖ⨀ωᵢ[k][t] += dQᵢₖ_dLᵢₖ*𝛚[i]
				end
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
	if ∇Q.fit_Δ𝐯
		if length(∇Q.Δ𝐯) == K
			@inbounds for k = 1:K
				mul!(∇Q.Δ𝐯[k], 𝐕', ∑_bounds_dQᵢₖ_dLᵢₖ⨀ωᵢ[k])
			end
		else
			mul!(∇Q.Δ𝐯[1], 𝐕', sum(∑_bounds_dQᵢₖ_dLᵢₖ⨀ωᵢ))
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
			if ∇ℓglm.fit_Δ𝐯
				for Δ𝐯ₖ in ∇ℓglm.Δ𝐯
					for Δv in Δ𝐯ₖ
						counter+=1
						∇nℓ[counter] = -Δv
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

"""
    subsample(mpGLM, timesteps)

Create a mixture of Poisson GLM by subsampling the spike train of a neuron

ARGUMENT
-`mpGLM`: a structure with information on the mixture of Poisson GLM of a neuron
-`timesteps`: a vector of integers indexing the timesteps to include

OUTPUT
-an instance of `MixturePoissonGLM`
"""
function subsample(mpGLM::MixturePoissonGLM, timesteps::Vector{<:Integer})
    MixturePoissonGLM(Δt = mpGLM.Δt,
                        d𝛏_dB = mpGLM.d𝛏_dB,
						Φₐ = mpGLM.Φₐ,
						Φₕ = mpGLM.Φₕ,
						Φₘ = mpGLM.Φₘ,
						Φₚ = mpGLM.Φₚ,
						Φₚtimesteps = mpGLM.Φₚtimesteps,
						Φₜ = mpGLM.Φₜ,
						θ = FHMDDM.copy(mpGLM.θ),
                        𝐕 = mpGLM.𝐕[timesteps, :],
                        𝐗 = mpGLM.𝐗[timesteps, :],
                        𝐲 =mpGLM.𝐲[timesteps])
end
