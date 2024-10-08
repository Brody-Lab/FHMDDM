"""
	contents

MixturePoissonGLM(options, trials)
linearpredictor(mpGLM, j, k)
evidenceinput(j, k, mpGLM)
conditionallikelihood!(p, mpGLM, τ)
scaledlikelihood(mpGLM, j, k)
scaledlikelihood!(𝐩, mpGLM, j, k)
expectation_∇loglikelihood!(∇Q, D, γ, mpGLM)
expectation_of_loglikelihood(γ, mpGLM, x)
postspikefilter(mpGLM)
externalinput(mpGLM)
subsample(mpGLM, timesteps)
samplespiketrain(a, c, 𝐄𝐞, 𝐡, mpGLM, 𝛚, 𝛕)
"""

"""
	MixturePoissonGLM(options, trials)

Initialize the Poisson mixture generalized linear model for each neuron in a trialset

ARGUMENT
-`options`: a composite containing the fixed hyperparameters of the model
-`trials`: a vector of structs each of which containing the data of one trial

RETURN
-a vector of structs each of which corresponds to the Poisson mixture generalized linear model of a neuron
"""
function MixturePoissonGLM(options::Options, trials::Vector{<:Trial})
	Φpoststereoclick = temporal_basis_functions("poststereoclick", options)
	Φpostspike = temporal_basis_functions("postspike", options)
	Φpremovement = temporal_basis_functions("premovement", options)
	movementtimesteps = collect(trial.movementtimestep for trial in trials)
	trialdurations = collect(trial.ntimesteps for trial in trials)
	𝐔poststereoclick = poststereoclickbasis(Φpoststereoclick, trialdurations)
	𝐔premovement = premovementbasis(movementtimesteps, Φpremovement, trialdurations)
	photostimulus_incline_on_s = collect(trial.photostimulus_incline_on_s for trial in trials)
	photostimulus_decline_on_s = collect(trial.photostimulus_decline_on_s for trial in trials)
	Φpostphotostimulus, Φpostphotostimulus_timesteps, 𝐔postphotostimulus = photostimulusbasis(options, photostimulus_incline_on_s, photostimulus_decline_on_s, trialdurations)
	Φaccumulator = accumulatorbasis(maximum(trialdurations), options)
	𝐕 = temporal_basis_functions(Φaccumulator, trialdurations)
	d𝛏_dB=(2collect(1:options.Ξ) .- options.Ξ .- 1)./(options.Ξ-1)
	stereoclick_times_s = collect(trial.stereoclick_time_s for trial in trials)
	nneurons = length(trials[1].spiketrains)
	file = matopen(options.datapath)
	neuronsinfo = read(file, "neurons")
	close(file)
	map(1:nneurons) do n
		𝐲 = vcat((trial.spiketrains[n] for trial in trials)...)
		@assert length(𝐲)==sum(trialdurations)
		Φgain, 𝐔gain = drift_design_matrix(options, stereoclick_times_s, trialdurations, 𝐲)
		𝐔postspike = spikehistorybasis(Φpostspike, trialdurations, 𝐲)
		𝐗=hcat(𝐔gain, 𝐔postspike, 𝐔poststereoclick, 𝐔premovement, 𝐔postphotostimulus, 𝐕)
		indices𝐮 = Indices𝐮(size(𝐔gain,2), size(Φpostspike,2), size(Φpoststereoclick,2), size(Φpremovement,2), size(Φpostphotostimulus,2))
		glmθ = GLMθ(indices𝐮, size(𝐕,2), options)
		MixturePoissonGLM(brainarea = neuronsinfo[n]["brainarea"],
						Δt=options.Δt,
						d𝛏_dB=d𝛏_dB,
						likelihoodscalefactor=options.sf_y,
						Φaccumulator=Φaccumulator,
						Φgain=Φgain,
						Φpostphotostimulus=Φpostphotostimulus,
						Φpostphotostimulus_timesteps=Φpostphotostimulus_timesteps,
						Φpostspike=Φpostspike,
						Φpoststereoclick=Φpoststereoclick,
						Φpremovement=Φpremovement,
						θ=glmθ,
						𝐕=𝐕,
						𝐗=𝐗,
						𝐲=𝐲)
	end
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
	ωⱼ𝐯ₖ = evidenceinput(j, k, mpGLM)
	𝐗*vcat(θ.𝐮, ωⱼ𝐯ₖ)
end

"""
	evidenceinput(j,k,mpGLM)

Weighted input of the accumulated evidence

ARGUMENT
-`j`: state of the accumulator variable
-`k`: state of the coupling variable
-`mpGLM`: the mixture of Poisson generalized linear model of one neuron

RETURN
-vector representing the weighted input of the accumulated evidence
"""
function evidenceinput(j::Integer, k::Integer, mpGLM::MixturePoissonGLM)
	@unpack d𝛏_dB, Ξ = mpGLM
    @unpack b, b_scalefactor, 𝐯, 𝛃, fit_𝛃 = mpGLM.θ
	ωⱼ = (b == 0.0) ? d𝛏_dB[j] : transformaccumulator(b[1]*b_scalefactor, d𝛏_dB[j])
	if (j == 1 || j == Ξ) && fit_𝛃
		ωⱼ.*𝛃[k]
	else
		ωⱼ.*𝐯[k]
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
	@unpack Δt, θ, 𝐗, 𝐕, 𝐲 = mpGLM
	@unpack a, fit_overdispersion, 𝐮 = θ
	α = inverselink(a[1])
	overdispersed = α > 0
	L = 0
	for i in eachindex(𝐮)
		L += 𝐗[τ,i]*𝐮[i]
	end
	Ξ, K = size(p)
	for k=1:K
		for j=1:Ξ
			Lⱼₖ = L
			ωⱼ𝐯ₖ = evidenceinput(j,k,mpGLM)
			for q in eachindex(ωⱼ𝐯ₖ)
				Lⱼₖ += 𝐕[τ,q]*ωⱼ𝐯ₖ[q]
			end
			p[j,k] = overdispersed ? negbinlikelihood(α, Δt, inverselink(Lⱼₖ), 𝐲[τ]) : poissonlikelihood(Δt, Lⱼₖ, 𝐲[τ])
		end
	end
	return nothing
end

"""
    scaledlikelihood(mpGLM, j, k)

Conditional likelihood of the spike train, given the index of the state of the accumulator `j` and the state of the coupling `k`, and also by the prior likelihood of the regression weights

UNMODIFIED ARGUMENT
-`mpGLM`: the mixture of Poisson generalized linear model for one neuron
-`j`: state of accumulator variable
-`k`: state of the coupling variable

RETURN
-`𝐩`: a vector by which the conditional likelihood of the spike train and the prior likelihood of the regression weights are multiplied against
"""
function scaledlikelihood(mpGLM::MixturePoissonGLM, j::Integer, k::Integer)
    @unpack Δt, likelihoodscalefactor, 𝐲 = mpGLM
	@unpack a, fit_overdispersion = mpGLM.θ
	α = inverselink(a[1])
	overdispersed = α > 0
    𝐋 = linearpredictor(mpGLM, j, k)
    𝐩 = 𝐋
    @inbounds for i=1:length(𝐩)
		μ = inverselink(𝐋[i])
		p = overdispersed ? negbinlikelihood(α, Δt, μ, 𝐲[i]) : poissonlikelihood(μ*Δt, 𝐲[i])
        𝐩[i] = p*likelihoodscalefactor
    end
    return 𝐩
end

"""
    scaledlikelihood!(𝐩, mpGLM, j, k)

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
function scaledlikelihood!(𝐩::Vector{<:Real}, mpGLM::MixturePoissonGLM, j::Integer, k::Integer)
    @unpack Δt, likelihoodscalefactor, 𝐲 = mpGLM
	@unpack a, fit_overdispersion = mpGLM.θ
	α = inverselink(a[1])
	overdispersed = α > 0
    𝐋 = linearpredictor(mpGLM, j, k)
    @inbounds for i=1:length(𝐩)
		μ = inverselink(𝐋[i])
		p = overdispersed ? negbinlikelihood(α, Δt, μ, 𝐲[i]) : poissonlikelihood(μ*Δt, 𝐲[i])
		𝐩[i] *= p*likelihoodscalefactor
    end
    return nothing
end

"""
	expectation_∇loglikelihood!(∇Q, D, γ, mpGLM)

Expectation under the posterior probability of the gradient of the log-likelihood.

This function is used for computing the gradient of the log-likelihood of the entire model

MODIFIED ARGUMENT
-`∇`: The gradient

UNMODIFIED ARGUMENT
-`γ`: Joint posterior probability of the accumulator and coupling variable. γ[i,k][t] corresponds to the i-th accumulator state and the k-th coupling state in the t-th time bin in the trialset.
-`D`: a struct containing quantities used for computing the derivatives of GLM parameters
-`mpGLM`: structure containing information for the mixture of Poisson GLM for one neuron
"""
function expectation_∇loglikelihood!(∇Q::GLMθ, D::GLMDerivatives, γ::Matrix{<:Vector{<:Real}}, mpGLM::MixturePoissonGLM)
	@unpack Δt, 𝐕, 𝐗, Ξ, 𝐲 = mpGLM
	@unpack a, fit_b, fit_𝛃, fit_overdispersion, 𝐯 = mpGLM.θ
	𝛚 = transformaccumulator(mpGLM)
	d𝛚_db = dtransformaccumulator(mpGLM)
	Ξ, K = size(γ)
	T = length(𝐲)
	∑ᵢ_dQᵢₖ_dLᵢₖ = collect(zeros(T) for k=1:K)
	if ∇Q.fit_𝛃
		∑_post_dQᵢₖ_dLᵢₖ⨀ωᵢ = collect(zeros(T) for k=1:K)
		∑_pre_dQᵢₖ_dLᵢₖ⨀ωᵢ = collect(zeros(T) for k=1:K)
	else
		∑ᵢ_dQᵢₖ_dLᵢₖ⨀ωᵢ = collect(zeros(T) for k=1:K)
	end
	differentiate_overdispersion!(D,a[1])
	∇Q.a[1] = 0
	if ∇Q.fit_b
		∑ᵢ_dQᵢₖ_dLᵢₖ⨀dωᵢ_db = collect(zeros(T) for k=1:K)
	end
	@inbounds for k = 1:K
		for i = 1:Ξ
			𝐋 = linearpredictor(mpGLM,i,k)
			for t=1:T
				differentiate_loglikelihood!(D, 𝐋[t], 𝐲[t])
				if fit_overdispersion
					∇Q.a[1] += γ[i,k][t] * D.dℓ_da[1]
				end
				dQᵢₖ_dLᵢₖ = γ[i,k][t] * D.dℓ_dL[1]
				∑ᵢ_dQᵢₖ_dLᵢₖ[k][t] += dQᵢₖ_dLᵢₖ
				if fit_𝛃
					if (i==1) || (i==Ξ)
						∑_post_dQᵢₖ_dLᵢₖ⨀ωᵢ[k][t] += dQᵢₖ_dLᵢₖ*𝛚[i]
					else
						∑_pre_dQᵢₖ_dLᵢₖ⨀ωᵢ[k][t] += dQᵢₖ_dLᵢₖ*𝛚[i]
					end
				else
					∑ᵢ_dQᵢₖ_dLᵢₖ⨀ωᵢ[k][t] += dQᵢₖ_dLᵢₖ*𝛚[i]
				end
				if fit_b
					∑ᵢ_dQᵢₖ_dLᵢₖ⨀dωᵢ_db[k][t] += dQᵢₖ_dLᵢₖ*d𝛚_db[i]
				end
			end
		end
	end
	𝐔 = @view 𝐗[:, mpGLM.𝐗columns_𝐮]
	mul!(∇Q.𝐮, 𝐔', sum(∑ᵢ_dQᵢₖ_dLᵢₖ))
	𝐕ᵀ = 𝐕'
	if fit_𝛃
		@inbounds for k = 1:K
			mul!(∇Q.𝐯[k], 𝐕ᵀ, ∑_pre_dQᵢₖ_dLᵢₖ⨀ωᵢ[k])
			mul!(∇Q.𝛃[k], 𝐕ᵀ, ∑_post_dQᵢₖ_dLᵢₖ⨀ωᵢ[k])
		end
	else
		@inbounds for k = 1:K
			mul!(∇Q.𝐯[k], 𝐕ᵀ, ∑ᵢ_dQᵢₖ_dLᵢₖ⨀ωᵢ[k])
		end
	end
	if ∇Q.fit_b
		∇Q.b[1] = 0.0
		@inbounds for k = 1:K
			∇Q.b[1] += dot(∑ᵢ_dQᵢₖ_dLᵢₖ⨀dωᵢ_db[k], 𝐕, 𝐯[k])
		end
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
function expectation_of_loglikelihood(γ::Matrix{<:Vector{<:AbstractFloat}}, mpGLM::MixturePoissonGLM, x::Vector{<:Real}; initialization::Bool=false)
	mpGLM = MixturePoissonGLM(x, mpGLM; initialization=initialization)
    @unpack Δt, 𝐲 = mpGLM
	@unpack a, fit_overdispersion = mpGLM.θ
	α = inverselink(a[1])
	overdispersed = α > 0
    T = length(𝐲)
    Ξ,K = size(γ)
    Q = 0.0
    @inbounds for i = 1:Ξ
	    for k = 1:K
			𝐋 = linearpredictor(mpGLM,i,k)
            for t = 1:T
				ℓ = overdispersed ? negbinloglikelihood(α, Δt, inverselink(𝐋[t]), 𝐲[t]) : poissonloglikelihood(Δt, 𝐋[t], 𝐲[t])
				Q += γ[i,k][t]*ℓ
            end
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
    MixturePoissonGLM(brainarea = mpGLM.brainarea,
    					Δt = mpGLM.Δt,
                        d𝛏_dB = mpGLM.d𝛏_dB,
						likelihoodscalefactor=mpGLM.likelihoodscalefactor,
						Φaccumulator = mpGLM.Φaccumulator,
						Φgain = mpGLM.Φgain[trialindices, :],
						Φpostspike = mpGLM.Φpostspike,
						Φpremovement = mpGLM.Φpremovement,
						Φpostphotostimulus = mpGLM.Φpostphotostimulus,
						Φpostphotostimulus_timesteps = mpGLM.Φpostphotostimulus_timesteps,
						Φpoststereoclick = mpGLM.Φpoststereoclick,
						θ = FHMDDM.copy(mpGLM.θ),
                        𝐕 = mpGLM.𝐕[timesteps, :],
                        𝐗 = mpGLM.𝐗[timesteps, :],
                        𝐲 =mpGLM.𝐲[timesteps])
end

"""
	samplespiketrain(a, c, 𝐄𝐞, 𝐡, mpGLM, 𝛚, 𝛕)

Generate a sample of spiking response on each time step of one trial

ARGUMENT
-`a`: a vector representing the state of the accumulator at each time step of a trial. Note that length(a) >= length(𝛕).
-`c`: a vector representing the state of the coupling variable at each time step. Note that length(c) >= length(𝛕).
-`𝐄𝐞`: input from external events
-`𝐡`: value of the post-spikefilter at each time lag
-`mpGLM`: a structure containing information on the mixture of Poisson GLM of a neuron
-`𝛚`: transformed values of the accumulator
-`𝛕`: time steps in the trialset. The number of time steps in the trial corresponds to the length of 𝛕.

RETURN
-`𝛌`: a vector of floats representing the spikes per second at each time step
-`𝐲̂`: a vector of integers representing the sampled spiking response at each time step
"""
function samplespiketrain(a::Vector{<:Integer}, c::Vector{<:Integer}, 𝐄𝐞::Vector{<:AbstractFloat}, 𝐡::Vector{<:AbstractFloat}, mpGLM::MixturePoissonGLM, 𝛚::Vector{<:AbstractFloat}, 𝛕::UnitRange{<:Integer})
	@unpack Δt, 𝐕, 𝐲, Ξ = mpGLM
	@unpack 𝐮, 𝐯, 𝛃, fit_𝛃, fit_overdispersion = mpGLM.θ
	α = inverselink(mpGLM.θ.a[1])
	overdispersed = α > 0
	max_spikehistory_lag = length(𝐡)
	K = length(𝐯)
	max_spikes_per_step = floor(1000Δt)
    𝐲̂ = zeros(eltype(𝐲), length(𝛕))
	𝛌 = zeros(length(𝛕))
    for t = 1:length(𝛕)
        τ = 𝛕[t]
        j = a[t]
        k = c[t]
		L = 𝐄𝐞[τ]
		for i in eachindex(𝐯[k])
			if fit_𝛃 && (j==1 || j==Ξ)
				L += 𝛚[j]*𝐕[τ,i]*𝛃[k][i]
			else
				L += 𝛚[j]*𝐕[τ,i]*𝐯[k][i]
			end
		end
		for lag = 1:min(max_spikehistory_lag, t-1)
			if 𝐲̂[t-lag] > 0
				L += 𝐡[lag]*𝐲̂[t-lag]
			end
		end
        𝛌[t] = softplus(L)
		if overdispersed
			μ = 𝛌[t]
			p = probabilitysuccess(α,Δt,μ)
			r = 1/α
			𝐲̂[t] = min(rand(NegativeBinomial(r,p)), max_spikes_per_step)
		else
			𝐲̂[t] = min(rand(Poisson(𝛌[t]*Δt)), max_spikes_per_step)
		end
    end
	return 𝛌, 𝐲̂
end
