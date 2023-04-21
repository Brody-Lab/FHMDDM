"""
	contents

linearpredictor(mpGLM, i)
evidenceinput(i, mpGLM)
conditionallikelihood!(p, mpGLM, τ)
scaledlikelihood(mpGLM, i)
scaledlikelihood!(𝐩, mpGLM, i)
expectation_∇loglikelihood!(∇Q, D, γ, mpGLM)
expectation_of_loglikelihood(γ, mpGLM, x)
postspikefilter(mpGLM)
externalinput(mpGLM)
subsample(mpGLM, timesteps)
samplespiketrain(a, c, 𝐄𝐞, 𝐡, mpGLM, 𝛚, 𝛕)
"""

"""
    linearpredictor(mpGLM, i)

Linear combination of the weights in the i-th accumulator state and the coupled state

ARGUMENT
-`mpGLM`: the mixture of Poisson generalized linear model of one neuron
-`i`: state of the accumulator variable

RETURN
-`𝐋`: a vector whose element 𝐋[t] corresponds to the t-th time bin in the trialset
"""
linearpredictor(mpGLM::MixturePoissonGLM, i::Integer) = mpGLM.𝐗 * vcat(mpGLM.θ.𝐮, evidenceinput(i, mpGLM))

"""
	evidenceinput(i,mpGLM)

Weighted input of the accumulated evidence

ARGUMENT
-`i`: state of the accumulator variable
-`mpGLM`: the mixture of Poisson generalized linear model of one neuron

RETURN
-vector representing the weighted input of the accumulated evidence
"""
function evidenceinput(i::Integer, mpGLM::MixturePoissonGLM)
	@unpack d𝛏_dB, Ξ = mpGLM
    @unpack b, v, β, fit_β = mpGLM.θ
	ω = transformaccumulator(b[1]*mpGLM.sf_mpGLM, d𝛏_dB[i])
	if (i == 1 || i == Ξ) && fit_β
		ω*β[1]
	else
		ω*v[1]
	end
end

"""
	scaledlikelihood!(p𝐘, mpGLMs)

Scaled conditional likelihood of the population spike train

MODIFIED ARGUMENT
-`pₛ𝐘`: a nested array whose element `pₛ𝐘[m][t][i]` corresponds to the conditional scaled likelihood of the population response at the t-th time step of the m-th trial, given that the accumulator is in the i-th state.

UNMODIFIED ARGUMENT
-`mpGLMs`: a vector whose each element corresponds to the Poisson mixture GLM of a neuron
"""
function scaledlikelihood!(p𝐘::Vector{<:Vector{<:Vector{type}}}, mpGLMs::Vector{<:MixturePoissonGLM}) where {type<:Real}
	@unpack Δt, index0, sf_y, Ξ, 𝐲 = mpGLMs[1]
    𝐟₀ = collect(conditionallikelihood(index0, mpGLM) for mpGLM in mpGLMs)
	ntimesteps = length(𝐟₀[1])
	for i = 1:Ξ
		𝐩 = ones(type, ntimesteps)
		for (𝐟₀, mpGLM) in zip(𝐟₀, mpGLMs)
			if i == index0
				for τ = 1:ntimesteps
					𝐩[τ] *= sf_y*𝐟₀[τ]
				end
			else
				π₁ = couplingprobability(mpGLM)
				π₀ = 1-π₁
			    𝐟₁ = conditionallikelihood(i, mpGLM)
			    for τ = 1:ntimesteps
					𝐩[τ] *= sf_y*(π₁*𝐟₁[τ] + π₀*𝐟₀[τ])
			    end
			end
		end
		τ = 0
		for m in eachindex(p𝐘)
			for t in eachindex(p𝐘[m])
				τ += 1
				p𝐘[m][t][i] = 𝐩[τ]
			end
		end
	end
end

"""
	conditionallikelihood(mpGLM)

Likelihood of the spike train given the i-th accumulator state and a coupled state

ARGUMENT
-`mpGLM`: an object containing the parameters, hyperparameters, and data of Poisson mixture model

RETURN
-`𝐟`: a vector whose each element corresponds to the likelihood of a spike train response
"""
function conditionallikelihood(i::Integer, mpGLM::MixturePoissonGLM)
    @unpack Δt, 𝐲 = mpGLM
    𝐋₁ = linearpredictor(mpGLM, i)
    𝐟₁ = 𝐋₁
    for τ in eachindex(𝐟₁)
        𝐟₁[τ] = poissonlikelihood(Δt, 𝐋₁[τ], 𝐲[τ])
    end
    return 𝐟₁
end

"""
	conditionallikelihood!(p, mpGLM, τ)

Conditional likelihood the spike train response at a single timestep

MODIFIED ARGUMENT
-`p`: a vector whose element `p[i]` represents the likelihood conditioned on the accumulator in the i-th state

UNMODIFIED ARGUMENT
-`mpGLM`:a structure with information on the mixture of Poisson GLM of a neuron
-`τ`: timestep among time steps concatenated across all trials in a trialset
"""
function conditionallikelihood!(p::Vector{<:Real}, mpGLM::MixturePoissonGLM, τ::Integer)
	@unpack Δt, index0, 𝐗, 𝐲, Ξ = mpGLM
	@unpack 𝐮 = mpGLM.θ
	L₀ = 0
	for i in eachindex(𝐮)
		L₀ += 𝐗[τ,i]*𝐮[i]
	end
	f₀ = poissonlikelihood(Δt, L₀, 𝐲[τ])
	π = couplingprobability(mpGLM)
	for i=1:Ξ
		if i == index0
			p[i] = f₀
		else
			L₁ = L₀ + 𝐗[τ,end]*evidenceinput(i,mpGLM)
			f₁ = poissonlikelihood(Δt, L₁, 𝐲[τ])
			p[i] = π*f₁ + (1-π)*f₀
		end
	end
	return nothing
end

"""
	couplingprobability(glmθ)

Probability of being coupled to the accumulator

ARGUMENT
-`glmθ`: parameters of a Poisson mixture GLM

RETURN
-a scalar indicating the probability fo coupling
"""
function couplingprobability(mpGLM::MixturePoissonGLM)
	r = mpGLM.θ.c[1]*mpGLM.sf_mpGLM
	real2native(r, mpGLM.θ.c_q, mpGLM.θ.c_l, mpGLM.θ.c_u)
end
function differentiate_π_wrt_c(mpGLM::MixturePoissonGLM)
	r = mpGLM.θ.c[1]*mpGLM.sf_mpGLM
	mpGLM.sf_mpGLM*differentiate_native_wrt_real(r, mpGLM.θ.c_q, mpGLM.θ.c_l, mpGLM.θ.c_u)
end

"""
	expectation_∇loglikelihood!(∇Q, D, γ, mpGLM)

Expectation under the posterior probability of the gradient of the log-likelihood.

This function is used for computing the gradient of the log-likelihood of the entire model

MODIFIED ARGUMENT
-`∇`: The gradient

UNMODIFIED ARGUMENT
-`γ`: Joint posterior probability of the accumulator and coupling variable. γ[i][t] corresponds to the i-th accumulator state in the t-th time bin in the trialset.
-`D`: a struct containing quantities used for computing the derivatives of GLM parameters
-`mpGLM`: structure containing information for the mixture of Poisson GLM for one neuron
"""
function expectation_∇loglikelihood!(∇Q::GLMθ, γ::Vector{<:Vector{<:Real}}, mpGLM::MixturePoissonGLM)
	@unpack Δt, 𝐗, Ξ, 𝐲, index0 = mpGLM
	@unpack fit_b, fit_β, fit_c, v, β = mpGLM.θ
	𝛚 = transformaccumulator(mpGLM)
	d𝛚_db = dtransformaccumulator(mpGLM)
	for parameter in mpGLM.θ.concatenationorder
		getfield(∇Q, parameter) .= 0
	end
	𝐋₀ = linearpredictor(mpGLM,index0)
	𝛌₀ = inverselink.(𝐋₀)
	𝐟₀ = collect(poissonlikelihood(λ₀*Δt, y) for (λ₀,y) in zip(𝛌₀,𝐲))
	dℓ₀_d𝐋₀ = collect(differentiate_loglikelihood_wrt_linearpredictor(Δt, L₀, λ₀, y) for (L₀, λ₀, y) in zip(𝐋₀,𝛌₀,𝐲))
	π₁ = couplingprobability(mpGLM)
	π₀ = 1-π₁
	for i = 1:Ξ
		𝐋₁ = (i == index0) ? 𝐋₀ : linearpredictor(mpGLM,i)
		for t=1:length(𝐲)
			if i == index0
				dℓ₁_dL₁ = dℓ₀_d𝐋₀[t]
				f₁ = 𝐟₀[t]
			else
				λ₁ = inverselink(𝐋₁[t])
				dℓ₁_dL₁ = differentiate_loglikelihood_wrt_linearpredictor(Δt, 𝐋₁[t], λ₁, 𝐲[t])
				f₁ = poissonlikelihood(λ₁*Δt, 𝐲[t])
			end
			f₁π₁ = f₁*π₁
			f₀π₀ = 𝐟₀[t]*π₀
			f = f₁π₁ + f₀π₀
			x = (f₁π₁*dℓ₁_dL₁ + f₀π₀*dℓ₀_d𝐋₀[t])/f
			for j in eachindex(∇Q.𝐮)
				∇Q.𝐮[j] += γ[i][t]*𝐗[t,j]*x
			end
			useβ = fit_β && ((i==1) || (i==Ξ))
			if i != index0
				γdℓ_dwₐ = γ[i][t]*𝛚[i]*𝐗[t,end]*f₁π₁*dℓ₁_dL₁/f
				if useβ
					∇Q.β[1] += γdℓ_dwₐ
				else
					∇Q.v[1] += γdℓ_dwₐ
				end
				if fit_b
					dL₁_db = useβ ? d𝛚_db[i]*𝐗[t,end]*β[1] : d𝛚_db[i]*𝐗[t,end]*v[1]
					∇Q.b[1] += γ[i][t]*f₁π₁*dℓ₁_dL₁*dL₁_db/f
				end
			end
			if fit_c
				∇Q.c[1] += γ[i][t]*(f₁-𝐟₀[t])/f
			end
		end
	end
	if fit_c
		∇Q.c[1] *= differentiate_π_wrt_c(mpGLM)
	end
	return nothing
end

"""
    expectation_of_loglikelihood(γ, mpGLM, x)

ForwardDiff-compatible computation of the expectation of the log-likelihood of the mixture of Poisson generalized linear model (mpGLM) of one neuron

Terms that do not depend on the parameters, such as the `log(y!)` term in the non-overdispersed Poisson model, are ignored

ARGUMENT
-`γ`: posterior probability of the latent variable
-`mpGLM`: the GLM of one neuron
-`x`: weights of the linear filters of the GLM concatenated as a vector of floating-point numbers

RETURN
-expectation of the log-likelihood of the spike train of one neuron
"""
function expectation_of_loglikelihood(γ::Vector{<:Vector{<:AbstractFloat}}, mpGLM::MixturePoissonGLM, x::Vector{<:Real}; initialization::Bool=false)
	mpGLM = MixturePoissonGLM(x, mpGLM; initialization=initialization)
    @unpack Δt, index0, 𝐲, Ξ = mpGLM
    Q = 0.0
	𝐋₀ = linearpredictor(mpGLM,index0)
	𝐟₀ = collect(poissonlikelihood(Δt, L, y) for (L,y) in zip(𝐋₀,𝐲))
	π₁ = couplingprobability(mpGLM)
	π₀ = 1-π₁
    for i = 1:Ξ
		𝐋₁ = (i == index0) ? 𝐋₀ : linearpredictor(mpGLM,i)
        for t = 1:length(𝐲)
			f₁ = (i == index0) ? 𝐟₀[t] : poissonlikelihood(Δt, 𝐋₁[t], 𝐲[t])
			Q += γ[i][t]*log(π₁*f₁ + π₀*𝐟₀[t])
        end
    end
    return Q
end

"""
	postspikefilter(mpGLM)

Return a vector representing the post-spike filter of a Poisson mixture GLM.

The first element of the vector corresponds to the first time step after the spike.
"""
postspikefilter(mpGLM::MixturePoissonGLM) = mpGLM.Φpostspike*mpGLM.θ.𝐮[mpGLM.θ.indices𝐮.postspike]

"""
	externalinput(mpGLM)

Weighted inputs, except for those from the latent variables and the spike history, to a neuron on each time step in a trialset

The inputs include gain, time after stereoclick (i.e., the start of each trial), time before movement (i.e., the rat removing its nose from the center port), and the photostimulus (if any).

The conditional firing rate of a neuron at each time step `t`, given the accumulator `a` is given by

	`λₜ ∣ aₜ ≡ softplus{𝐰_gain ⋅ 𝐱ₜ_gain + 𝐰_stereoclick ⋅ 𝐱ₜ_stereoclick + 𝐰_move ⋅ 𝐱ₜ_move + 𝐰_move ⋅ 𝐱ₜ_move + 𝐰_phostostimulus ⋅ 𝐱ₜ_photostimulus + 𝐰_hist ⋅ 𝐱ₜ_hist + (w ∣ aₜ)aₜ}`

Each element of the vector returned by this function corresponds to following linear combination

	`𝐰_gain ⋅ 𝐱ₜ_gain + 𝐰_stereoclick ⋅ 𝐱ₜ_stereoclick + 𝐰_move ⋅ 𝐱ₜ_move + 𝐰_move ⋅ 𝐱ₜ_move + 𝐰_phostostimulus ⋅ 𝐱ₜ_photostimulus`

RETURN
-a vector whose τ-th element corresponds to the τ-th time step in the trialset
"""
function externalinput(mpGLM::MixturePoissonGLM)
	@unpack 𝐗, θ = mpGLM
	@unpack 𝐮, indices𝐮 = θ
	indices = vcat(indices𝐮.gain, indices𝐮.poststereoclick, indices𝐮.premovement, indices𝐮.postphotostimulus)
	𝐄 = @view 𝐗[:,indices]
	return 𝐄*𝐮[indices]
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
function subsample(mpGLM::MixturePoissonGLM, timesteps::Vector{<:Integer}, trialindices::Vector{<:Integer})
    MixturePoissonGLM(Δt = mpGLM.Δt,
                        d𝛏_dB = mpGLM.d𝛏_dB,
						sf_y=mpGLM.sf_y,
						sf_mpGLM=mpGLM.sf_mpGLM,
						Φgain = mpGLM.Φgain[trialindices, :],
						Φpostspike = mpGLM.Φpostspike,
						Φpremovement = mpGLM.Φpremovement,
						Φpostphotostimulus = mpGLM.Φpostphotostimulus,
						Φpostphotostimulus_timesteps = mpGLM.Φpostphotostimulus_timesteps,
						Φpoststereoclick = mpGLM.Φpoststereoclick,
						θ = FHMDDM.copy(mpGLM.θ),
                        𝐗 = mpGLM.𝐗[timesteps, :],
                        𝐲 =mpGLM.𝐲[timesteps])
end

"""
	samplespiketrain(a, 𝐄𝐞, 𝐡, mpGLM, 𝛚, 𝛕)

Generate a sample of spiking response on each time step of one trial

ARGUMENT
-`a`: a vector representing the state of the accumulator at each time step of a trial. Note that length(a) >= length(𝛕).
-`𝐄𝐞`: input from external events
-`𝐡`: value of the post-spikefilter at each time lag
-`mpGLM`: a structure containing information on the mixture of Poisson GLM of a neuron
-`𝛚`: transformed values of the accumulator
-`𝛕`: time steps in the trialset. The number of time steps in the trial corresponds to the length of 𝛕.

RETURN
-`𝛌`: a vector of floats representing the spikes per second at each time step
-`𝐲̂`: a vector of integers representing the sampled spiking response at each time step
"""
function samplespiketrain(a::Vector{<:Integer}, 𝐄𝐞::Vector{<:AbstractFloat}, 𝐡::Vector{<:AbstractFloat}, mpGLM::MixturePoissonGLM, 𝛚::Vector{<:AbstractFloat}, 𝛕::UnitRange{<:Integer})
	@unpack Δt, 𝐗, 𝐲, Ξ = mpGLM
	@unpack 𝐮, v, β, fit_β = mpGLM.θ
	max_spikehistory_lag = length(𝐡)
	max_spikes_per_step = floor(1000Δt)
    𝐲̂ = zeros(eltype(𝐲), length(𝛕))
	𝛌 = zeros(length(𝛕))
	π₁ = couplingprobability(mpGLM)
    for t = 1:length(𝛕)
        τ = 𝛕[t]
		L = 𝐄𝐞[τ]
		for lag = 1:min(max_spikehistory_lag, t-1)
			if 𝐲̂[t-lag] > 0
				L += 𝐡[lag]*𝐲̂[t-lag]
			end
		end
		if rand() < π₁
			wₐ = (fit_β && (a[t]==1 || a[t]==Ξ)) ? β[1] : v[1]
			L += 𝛚[a[t]]*𝐗[τ,end]*wₐ
		end
        𝛌[t] = inverselink(L)
		𝐲̂[t] = min(rand(Poisson(𝛌[t]*Δt)), max_spikes_per_step)
    end
	return 𝛌, 𝐲̂
end
