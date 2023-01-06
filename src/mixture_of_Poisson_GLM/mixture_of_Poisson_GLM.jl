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
-`ωⱼ𝐯ₖ`: vector representing the weighted input of the accumulated evidence
"""
function evidenceinput(j::Integer, k::Integer, mpGLM::MixturePoissonGLM)
	@unpack d𝛏_dB, Ξ = mpGLM
    @unpack b, b_scalefactor, 𝐯, Δ𝐯, fit_Δ𝐯 = mpGLM.θ
	if (j == 1 || j == Ξ) && fit_Δ𝐯
		𝐯ₖ = 𝐯[k] .+ Δ𝐯[k]
	else
		𝐯ₖ = 𝐯[k]
	end
	ωⱼ = (b == 0.0) ? d𝛏_dB[j] : transformaccumulator(b[1]*b_scalefactor, d𝛏_dB[j])
	𝐯ₖ.*ωⱼ
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
	@unpack 𝐮 = θ
	L = 0
	for i in eachindex(𝐮)
		L += 𝐗[τ,i]*𝐮[i]
	end
	Ξ, K = size(p)
	for k=1:K
		for j=1:Ξ
			ωⱼ𝐯ₖ = evidenceinput(j,k,mpGLM)
			for q in eachindex(ωⱼ𝐯ₖ)
				L += 𝐕[τ,q]*ωⱼ𝐯ₖ[q]
			end
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
	@unpack Δt, 𝐕, 𝐗, 𝐗columns_𝐮, Ξ, 𝐲 = mpGLM
	@unpack 𝐯 = mpGLM.θ
	𝛚 = transformaccumulator(mpGLM)
	d𝛚_db = dtransformaccumulator(mpGLM)
	Ξ, K = size(γ)
	T = length(𝐲)
	∑ᵢ_dQᵢₖ_dLᵢₖ = collect(zeros(T) for k=1:K)
	∑ᵢ_dQᵢₖ_dLᵢₖ⨀ωᵢ = collect(zeros(T) for k=1:K)
	if ∇Q.fit_Δ𝐯
		∑_bounds_dQᵢₖ_dLᵢₖ⨀ωᵢ = collect(zeros(T) for k=1:K)
	end
	if ∇Q.fit_b
		∑ᵢ_dQᵢₖ_dLᵢₖ⨀dωᵢ_db = collect(zeros(T) for k=1:K)
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
	𝐔 = @view 𝐗[:, 𝐗columns_𝐮]
	mul!(∇Q.𝐮, 𝐔', sum(∑ᵢ_dQᵢₖ_dLᵢₖ))
	@inbounds for k = 1:K
		mul!(∇Q.𝐯[k], 𝐕', ∑ᵢ_dQᵢₖ_dLᵢₖ⨀ωᵢ[k])
	end
	if ∇Q.fit_Δ𝐯
		@inbounds for k = 1:K
			mul!(∇Q.Δ𝐯[k], 𝐕', ∑_bounds_dQᵢₖ_dLᵢₖ⨀ωᵢ[k])
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
	mpGLM = MixturePoissonGLM(x, mpGLM; initialization=initialization)
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
	postspikefilter(mpGLM)

Return a vector representing the post-spike filter of a Poisson mixture GLM.

The first element of the vector corresponds to the first time step after the spike.
"""
postspikefilter(mpGLM::MixturePoissonGLM) = mpGLM.Φₕ*mpGLM.θ.𝐮[mpGLM.θ.indices𝐮.postspike]

"""
	externalinput(mpGLM)

Sum the input from extern events for each time step in a trialset.

The external events typically consist of the stereoclick, departure from the center port, and the photostimulus.

RETURN
-a vector whose τ-th element corresponds to the τ-th time step in the trialset
"""
function externalinput(mpGLM::MixturePoissonGLM)
	@unpack 𝐗, θ = mpGLM
	@unpack 𝐮, indices𝐮 = θ
	𝐗columns = vcat(mpGLM.𝐗columns_gain, mpGLM.𝐗columns_time, mpGLM.𝐗columns_move, mpGLM.𝐗columns_phot)
	𝐮indices = vcat(indices𝐮.gain, indices𝐮.poststereoclick, indices𝐮.premovement, indices𝐮.postphotostimulus)
	𝐄 = @view 𝐗[:,𝐗columns]
	return 𝐄*𝐮[𝐮indices]
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

"""
	MixturePoissonGLM(movementtimes_s, options, photostimulus_decline_on_s, photostimulus_incline_on_s, 𝐓, 𝐘)

Initialize the Poisson mixture generalized linear model for each neuron in a trialset

ARGUMENT
-`movementtimes_s`: a vector of floats indicating the time, in seconds, when the animal left the center port, relative to the time of the stereoclick
-`options`: a composite containing the fixed hyperparameters of the model
-`photostimulus_decline_on_s`: a vector of floats indicating the time, in seconds, when the photostimulus, if any, began to taper off (i.e., the time of the onset of the offset ramp), relative to the timing of the stereoclick
-`photostimulus_incline_on_s`: the time in each trial when the photostimulus began to ramp up in intensity
-`𝐓`: a vector of integers indicating the number of time steps in each trial
-`𝐘`: a nested vector whose each element corresponds to a neuron in the trialset and represents the spike train response of that neuron.

RETURN
-a vector whose each element is a composite containing the data and parameters of the Poisson mixture generalized linear model of a neuron
"""
function MixturePoissonGLM(movementtimes_s::Vector{<:AbstractFloat},
							options::Options,
							photostimulus_decline_on_s::Vector{<:AbstractFloat},
 							photostimulus_incline_on_s::Vector{<:AbstractFloat},
							𝐓::Vector{<:Integer},
							𝐘::Vector{<:Vector{<:Integer}})
	@unpack Ξ = options
	sum𝐓 = sum(𝐓)
	maximum𝐓 = maximum(𝐓)
    @assert all(length.(𝐘) .== sum𝐓)
	𝐆 = ones(sum𝐓).*(options.tbf_gain_scalefactor/sqrt(maximum𝐓))
	Φₕ = spikehistorybasis(options)
	𝐔ₕ = map(𝐲->spikehistorybasis(Φₕ, 𝐓, 𝐲), 𝐘)
	Φₘ = premovementbasis(options)
	𝐔ₘ = premovementbasis(movementtimes_s, options, Φₘ, 𝐓)
	Φₜ = timebasis(options)
	𝐔ₜ = timebasis(Φₜ, 𝐓)
	Φₚ, Φₚtimesteps, 𝐔ₚ = photostimulusbasis(options, photostimulus_incline_on_s, photostimulus_decline_on_s, 𝐓)
	Φₐ = accumulatorbasis(maximum𝐓, options)
	𝐕 = temporal_basis_functions(Φₐ, 𝐓)
	indices𝐮 = Indices𝐮(size(Φₕ,2), size(Φₜ,2), size(Φₘ,2), size(Φₚ,2))
	map(𝐔ₕ, 𝐘) do 𝐔ₕ, 𝐲
		𝐗=hcat(𝐆, 𝐔ₕ, 𝐔ₜ, 𝐔ₘ, 𝐔ₚ, 𝐕)
		glmθ = GLMθ(indices𝐮, size(𝐕,2), options)
		MixturePoissonGLM(Δt=options.Δt,
						d𝛏_dB=(2collect(1:Ξ) .- Ξ .- 1)./(Ξ-1),
						Φₐ=Φₐ,
						Φₕ=Φₕ,
						Φₘ=Φₘ,
						Φₚ=Φₚ,
						Φₚtimesteps=Φₚtimesteps,
						Φₜ=Φₜ,
						θ=glmθ,
						𝐕=𝐕,
						𝐗=𝐗,
						𝐲=𝐲)
	 end
end

"""
	GLMθ(indices𝐮, options, n𝐯)

Randomly initiate the parameters for a mixture of Poisson generalized linear model

ARGUMENT
-`indices𝐮`: indices of the encoding weights of the temporal basis vectors of each filter that is independent of the accumulator
-`n𝐯`: number of temporal basis vectors specifying the time-varying weight of the accumulator
-`options`: settings of the model

OUTPUT
-an instance of `GLMθ`
"""
function GLMθ(indices𝐮::Indices𝐮, n𝐯::Integer, options::Options)
	n𝐮 = maximum(vcat((getfield(indices𝐮, field) for field in fieldnames(Indices𝐮))...))
	θ = GLMθ(b = fill(NaN,1),
			b_scalefactor = options.b_scalefactor,
			fit_b = options.fit_b,
			fit_Δ𝐯 = options.fit_Δ𝐯,
			𝐮 = fill(NaN, n𝐮),
			indices𝐮=indices𝐮,
			𝐯 = collect(fill(NaN,n𝐯) for k=1:options.K))
	randomizeparameters!(θ, options)
	return θ
end

"""
	randomizeparameters!(θ, options)

Randomly initialize parameters of a mixture of Poisson GLM

MODIFIED ARGUMENT
-`θ`: structure containing parameters of a mixture of Poisson GLM

UNMODIFIED ARGUMENT
-`options`: hyperparameters of the model
"""
function randomizeparameters!(θ::GLMθ, options::Options)
	θ.b[1] = 0.0
	for i in eachindex(θ.𝐮)
		θ.𝐮[i] = 1.0 .- 2rand()
	end
	θ.𝐮[θ.indices𝐮.gain] ./= options.tbf_gain_scalefactor
	θ.𝐮[θ.indices𝐮.postspike] ./= options.tbf_hist_scalefactor
	θ.𝐮[θ.indices𝐮.poststereoclick] ./= options.tbf_time_scalefactor
	θ.𝐮[θ.indices𝐮.premovement] ./= options.tbf_move_scalefactor
	θ.𝐮[θ.indices𝐮.postphotostimulus] ./= options.tbf_phot_scalefactor
	if length(θ.𝐯) > 1
		K = length(θ.𝐯)
		𝐯₀ = -1.0:2.0/(K-1):1.0
		for k = 1:K
			θ.𝐯[k] .= 𝐯₀[k]
			θ.Δ𝐯[k] .= 0.0
		end
	else
		θ.𝐯[1] .= 1.0 .- 2rand(length(θ.𝐯[1]))
		if θ.fit_Δ𝐯
			θ.Δ𝐯[1] .= -θ.𝐯[1]
		else
			θ.Δ𝐯[1] .= 0.0
		end
	end
	for k = 1:length(θ.𝐯)
		θ.𝐯[k] ./= options.tbf_accu_scalefactor
		θ.Δ𝐯[k] ./= options.tbf_accu_scalefactor
	end
end

"""
	sample(a, c, 𝐄𝐞, 𝐡, mpGLM, 𝛚, 𝛕)

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
-`𝐲̂`: a vector representing the sampled spiking response at each time step
"""
function sample(a::Vector{<:Integer}, c::Vector{<:Integer}, 𝐄𝐞::Vector{<:AbstractFloat}, 𝐡::Vector{<:AbstractFloat}, mpGLM::MixturePoissonGLM, 𝛚::Vector{<:AbstractFloat}, 𝛕::UnitRange{<:Integer})
	@unpack Δt, 𝐕, 𝐲, Ξ = mpGLM
	@unpack 𝐮, 𝐯, Δ𝐯, fit_Δ𝐯 = mpGLM.θ
	max_spikehistory_lag = length(𝐡)
	K = length(𝐯)
	max_spikes_per_step = floor(1000Δt)
    𝐲̂ = zeros(Int, length(𝛕))
    for t = 1:length(𝛕)
        τ = 𝛕[t]
        j = a[t]
        k = c[t]
		L = 𝐄𝐞[τ]
		for i in eachindex(𝐯[k])
			L += 𝛚[j]*𝐕[τ,i]*𝐯[k][i]
			if fit_Δ𝐯 && (j==1 || j==Ξ)
				L += 𝛚[j]*𝐕[τ,i]*Δ𝐯[k][i]
			end
		end
		for lag = 1:min(max_spikehistory_lag, t-1)
			if 𝐲̂[t-lag] > 0
				L += 𝐡[lag]*𝐲̂[t-lag]
			end
		end
        λ = softplus(L)
        𝐲̂[t] = min(rand(Poisson(λ*Δt)), max_spikes_per_step)
    end
	return 𝐲̂
end
