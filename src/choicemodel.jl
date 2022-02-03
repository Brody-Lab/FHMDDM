"""
	maximizechoiceLL!(model)

Learn the parameters that maximize the log-likelihood of the behavioral choices

OPTIONAL ARGUMENT
-`extended_trace`: save additional information
-`f_tol`: threshold for determining convergence in the objective value
-`g_tol`: threshold for determining convergence in the gradient
-`iterations`: number of inner iterations that will be run before the algorithm gives up
-`outer_iterations`: number of outer iterations that will be run before the algorithm gives up
-`show_every`: trace output is printed every `show_every`th iteration.
-`show_trace`: should a trace of the optimization algorithm's state be shown?
-`x_tol`: threshold for determining convergence in the input vector

"""
function maximizechoiceLL!(model::Model;
		                 extended_trace::Bool=true,
		                 f_tol::AbstractFloat=1e-9,
		                 g_tol::AbstractFloat=1e-8,
		                 iterations::Integer=1000,
						 outer_iterations::Integer=10,
		                 show_every::Integer=1,
		                 show_trace::Bool=true,
		                 x_tol::AbstractFloat=1e-5)
	concatenatedθ, indexθ = concatenate_choice_related_parameters(model)
    f(concatenatedθ) = -loglikelihood!(model, concatenatedθ, indexθ)
    g!(∇, concatenatedθ) = ∇negativeloglikelihood!(∇, model, concatenatedθ, indexθ)
	# lowerbounds, upperbounds = concatenatebounds(indexθ, model.options)
    Optim_options = Optim.Options(extended_trace=extended_trace,
								  f_tol=f_tol,
                                  g_tol=g_tol,
                                  iterations=iterations,
								  outer_iterations=outer_iterations,
                                  show_every=show_every,
                                  show_trace=show_trace,
                                  x_tol=x_tol)
	# algorithm = Fminbox(LBFGS(linesearch = LineSearches.BackTracking()))
	algorithm = LBFGS(linesearch = LineSearches.BackTracking())
	θ₀ = deepcopy(concatenatedθ)
	# optimizationresults = Optim.optimize(f, g!, lowerbounds, upperbounds, θ₀, algorithm, Optim_options)
	optimizationresults = Optim.optimize(f, g!, θ₀, algorithm, Optim_options)
    println(optimizationresults)
    maximumlikelihoodθ = Optim.minimizer(optimizationresults)
	sortparameters!(model, maximumlikelihoodθ, indexθ)
end

"""
    concatenate_choice_related_parameters(model)

Concatenate values of parameters being fitted into a vector of floating point numbers

ARGUMENT
-`model`: the factorial hidden Markov drift-diffusion model

RETURN
-`concatenatedθ`: a vector of the concatenated values of the parameters being fitted
-`indexθ`: a structure indicating the index of each model parameter in the vector of concatenated values
"""
function concatenate_choice_related_parameters(model::Model)
    @unpack options, θreal, trialsets = model
	concatenatedθ = zeros(0)
    counter = 0
	latentθ = Latentθ(collect(zeros(Int64,1) for i in fieldnames(Latentθ))...)
	tofit = true
	for field in fieldnames(Latentθ)
		if field == :Aᶜ₁₁ || field == :Aᶜ₂₂ || field == :πᶜ₁
			tofit = false
		else
			options_field = Symbol("fit_"*String(field))
			if hasfield(typeof(options), options_field)
				tofit = getfield(options, options_field)
			else
				error("Unrecognized field: "*String(field))
			end
		end
		if tofit
			counter += 1
			getfield(latentθ, field)[1] = counter
			concatenatedθ = vcat(concatenatedθ, getfield(θreal, field)[1])
		else
			getfield(latentθ, field)[1] = 0
		end
	end
	emptyindex = map(trialset->map(mpGLM->zeros(Int, 0), trialset.mpGLMs), model.trialsets)
    indexθ = Indexθ(latentθ=latentθ,
					𝐮 = emptyindex,
					𝐥 = emptyindex,
					𝐫 = emptyindex)
    return concatenatedθ, indexθ
end


"""
    loglikelihood!(model, concatenatedθ)

Compute the log-likelihood of the choices

ARGUMENT
-`model`: an instance of FHM-DDM

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: index of each parameter after if all parameters being fitted are concatenated

RETURN
-log-likelihood
"""
function loglikelihood!(model::Model,
					    concatenatedθ::Vector{<:Real},
						indexθ::Indexθ)
	sortparameters!(model, concatenatedθ, indexθ)
	trialinvariant = Trialinvariant(model; purpose="loglikelihood")
	ℓ = map(model.trialsets) do trialset
			map(trialset.trials) do trial
				loglikelihood(model.θnative, trial, trialinvariant)
			end
		end
	return sum(sum(ℓ))
end

"""
	loglikelihood(p𝐘𝑑, trialinvariant, θnative, trial)

Compute the log-likelihood of the choice from one trial

ARGUMENT
-`θnative`: model parameters in their native space
-`trial`: stimulus and behavioral information of one trial
-`trialinvariant`: a structure containing quantities that are used in each trial

RETURN
-`ℓ`: log-likelihood of the data from one trial
"""
function loglikelihood(θnative::Latentθ,
					   trial::Trial,
					   trialinvariant::Trialinvariant)
	@unpack clicks = trial
	@unpack Aᵃsilent, Δt, 𝛏, Ξ = trialinvariant
	C = adapt(clicks, θnative.k[1], θnative.ϕ[1])
	μ = θnative.μ₀[1] + trial.previousanswer*θnative.wₕ[1]
	σ = √θnative.σ²ᵢ[1]
	f = probabilityvector(μ, σ, 𝛏)
	D = sum(f)
	f /= D
	ℓ = log(D)
	T = eltype(θnative.λ[1])
	Aᵃ = zeros(T, Ξ, Ξ)
	p𝑑 = conditional_probability_of_choice(trial.choice, θnative.ψ[1], Ξ)
	@inbounds for t = 2:trial.ntimesteps
		if isempty(clicks.inputindex[t])
			f = Aᵃsilent * f
		else
			cL = sum(C[clicks.left[t]])
			cR = sum(C[clicks.right[t]])
			stochasticmatrix!(Aᵃ, cL, cR, trialinvariant, θnative)
			f = Aᵃ * f
		end
		if t == trial.ntimesteps
			f .*= p𝑑
		end
		D = sum(f)
		f /= D
		ℓ += log(D)
	end
	return ℓ
end

"""
    ∇negativeloglikelihood!(∇, γ, model, concatenatedθ, indexθ)

Gradient of the negative log-likelihood of the factorial hidden Markov drift-diffusion model (FHMDDM)

MODIFIED INPUT
-`∇`: a vector of partial derivatives
-`model`: a structure containing information of the factorial hidden Markov drift-diffusion model

UNMODIFIED ARGUMENT
-`concatenatedθ`: parameter values concatenated into a vector
-`indexθ`: index of each parameter after if all parameters being fitted are concatenated

"""
function ∇negativeloglikelihood!(∇::Vector{<:AbstractFloat},
								 model::Model,
								 concatenatedθ::Vector{<:AbstractFloat},
								 indexθ::Indexθ)
	sortparameters!(model, concatenatedθ, indexθ)
	@unpack options, θnative, θreal, trialsets = model
	@unpack K = options
	trialinvariant = Trialinvariant(model; purpose="gradient")
	gradients=map(trialsets) do trialset
				pmap(trialset.trials) do trial
					∇loglikelihood(θnative, trial, trialinvariant)
				end
			end
	g = gradients[1][1] # reuse this memory
	for field in fieldnames(Latentθ)
		latent∂ = getfield(g, field)
		for i in eachindex(gradients)
			start = i==1 ? 2 : 1
			for m in start:length(gradients[i])
				latent∂[1] += getfield(gradients[i][m], field)[1]
			end
		end
	end
	g.B[1] *= θnative.B[1]*logistic(-θreal.B[1])
	g.k[1] *= θnative.k[1]
	g.ϕ[1] *= θnative.ϕ[1]*(1.0 - θnative.ϕ[1])
	tmpψ = logistic(θreal.ψ[1] + logit(options.q_ψ))
	g.ψ[1] *= (1.0-options.bound_ψ)*tmpψ*(1.0 - tmpψ)
	g.σ²ₐ[1] *= θnative.σ²ₐ[1]
	g.σ²ᵢ[1] *= options.q_σ²ᵢ*exp(θreal.σ²ᵢ[1])
	g.σ²ₛ[1] *= θnative.σ²ₛ[1]
	for field in fieldnames(Latentθ)
		index = getfield(indexθ.latentθ, field)[1]
		if index != 0
			∇[index] = -getfield(g,field)[1] # note the negative sign
		end
	end
end

"""
	∇loglikelihood(trialinvariant, θnative, trial)

Compute quantities needed for the gradient of the log-likelihood of the data observed in one trial

ARGUMENT
-`θnative`: parameters for the latent variables in their native space
-`trial`: information on the stimuli and behavioral choice of one trial
-`trialinvariant`: structure containing quantities used across trials

RETURN
-`latent∇`: gradient of the log-likelihood of the data observed in one trial with respect to the parameters specifying the latent variables
"""
function ∇loglikelihood(θnative::Latentθ,
						trial::Trial,
						trialinvariant::Trialinvariant)
	@unpack choice, clicks = trial
	@unpack inputtimesteps, inputindex = clicks
	@unpack Aᵃsilent, dAᵃsilentdμ, dAᵃsilentdσ², dAᵃsilentdB, Aᶜ, Δt, K, 𝛚, Ξ, 𝛏 = trialinvariant
	dℓdk, dℓdλ, dℓdϕ, dℓdσ²ₐ, dℓdσ²ₛ, dℓdB = 0., 0., 0., 0., 0., 0.
	μ = θnative.μ₀[1] + trial.previousanswer*θnative.wₕ[1]
	σ = √θnative.σ²ᵢ[1]
	πᵃ, dπᵃdμ, dπᵃdσ², dπᵃdB = zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ)
	probabilityvector!(πᵃ, dπᵃdμ, dπᵃdσ², dπᵃdB, μ, 𝛚, σ, 𝛏)
	n_steps_with_input = length(clicks.inputtimesteps)
	Aᵃ = map(x->zeros(Ξ,Ξ), clicks.inputtimesteps)
	dAᵃdμ = map(x->zeros(Ξ,Ξ), clicks.inputtimesteps)
	dAᵃdσ² = map(x->zeros(Ξ,Ξ), clicks.inputtimesteps)
	dAᵃdB = map(x->zeros(Ξ,Ξ), clicks.inputtimesteps)
	Δc = zeros(n_steps_with_input)
	∑c = zeros(n_steps_with_input)
	C, dCdk, dCdϕ = ∇adapt(clicks, θnative.k[1], θnative.ϕ[1])
	for i in 1:n_steps_with_input
		t = clicks.inputtimesteps[i]
		cL = sum(C[clicks.left[t]])
		cR = sum(C[clicks.right[t]])
		stochasticmatrix!(Aᵃ[i], dAᵃdμ[i], dAᵃdσ²[i], dAᵃdB[i], cL, cR, trialinvariant, θnative)
		Δc[i] = cR-cL
		∑c[i] = cL+cR
	end
	D, f = forward(Aᵃ, trial.choice, inputindex, πᵃ, θnative.ψ[1], trialinvariant)
	b = ones(Ξ)
	λΔt = θnative.λ[1]*Δt
	expλΔt = exp(λΔt)
	dμdΔc = (expλΔt - 1.0)/λΔt
	η = (expλΔt - dμdΔc)/θnative.λ[1]
	𝛏ᵀΔtexpλΔt = transpose(𝛏)*Δt*expλΔt
	p𝑑 = conditional_probability_of_choice(choice, θnative.ψ[1], Ξ)
	@inbounds for t = trial.ntimesteps:-1:1
		if t == trial.ntimesteps-1
			b .*= p𝑑
		end
		if t < trial.ntimesteps # backward step
			Aᵃₜ₊₁ = isempty(inputindex[t+1]) ? Aᵃsilent : Aᵃ[inputindex[t+1][1]]
			b = transpose(Aᵃₜ₊₁) * b / D[t+1]
		end
		if t > 1 # joint posterior over consecutive time bins, computations involving the transition matrix
			if isempty(inputindex[t])
				Aᵃₜ = Aᵃsilent
				dAᵃₜdμ = dAᵃsilentdμ
				dAᵃₜdσ² = dAᵃsilentdσ²
				dAᵃₜdB = dAᵃsilentdB
			else
				i = inputindex[t][1]
				Aᵃₜ = Aᵃ[i]
				dAᵃₜdμ = dAᵃdμ[i]
				dAᵃₜdσ² = dAᵃdσ²[i]
				dAᵃₜdB = dAᵃdB[i]
			end
			if t == trial.ntimesteps
				χᵃ_Aᵃ = p𝑑.*b .* transpose(f[t-1]) ./ D[t]
			else
				χᵃ_Aᵃ = b .* transpose(f[t-1]) ./ D[t]
			end
			χᵃ_dlogAᵃdμ = χᵃ_Aᵃ .* dAᵃₜdμ # χᵃ⊙ d/dμ{log(Aᵃ)} = χᵃ⊘ Aᵃ⊙ d/dμ{Aᵃ}
			∑_χᵃ_dlogAᵃdμ = sum(χᵃ_dlogAᵃdμ)
			∑_χᵃ_dlogAᵃdσ² = sum(χᵃ_Aᵃ .* dAᵃₜdσ²) # similarly, χᵃ⊙ d/dσ²{log(Aᵃ)} = χᵃ⊘ Aᵃ⊙ d/dσ²{Aᵃ}
			dℓdσ²ₐ += ∑_χᵃ_dlogAᵃdσ² # the Δt is multiplied after summing across time steps
			dℓdB += sum(χᵃ_Aᵃ .* dAᵃₜdB)
			if isempty(inputindex[t])
				dμdλ = 𝛏ᵀΔtexpλΔt
			else
				dμdλ = 𝛏ᵀΔtexpλΔt .+ Δc[i].*η
				dℓdσ²ₛ += ∑_χᵃ_dlogAᵃdσ²*∑c[i]
				dcLdϕ = sum(dCdϕ[clicks.left[t]])
				dcRdϕ = sum(dCdϕ[clicks.right[t]])
				dcLdk = sum(dCdk[clicks.left[t]])
				dcRdk = sum(dCdk[clicks.right[t]])
				dσ²dϕ = θnative.σ²ₛ[1]*(dcLdϕ + dcRdϕ)
				dσ²dk = θnative.σ²ₛ[1]*(dcLdk + dcRdk)
				dℓdϕ += ∑_χᵃ_dlogAᵃdμ*dμdΔc*(dcRdϕ - dcLdϕ) + ∑_χᵃ_dlogAᵃdσ²*dσ²dϕ
				dℓdk += ∑_χᵃ_dlogAᵃdμ*dμdΔc*(dcRdk - dcLdk) + ∑_χᵃ_dlogAᵃdσ²*dσ²dk
			end
			dℓdλ += sum(χᵃ_dlogAᵃdμ.*dμdλ)
		end
	end
	dℓdσ²ₐ *= Δt
	γᵃ₁_oslash_πᵃ = b # reuse memory
	γᵃ₁_oslash_πᵃ ./= D[1]
	∑_γᵃ₁_dlogπᵃdμ = γᵃ₁_oslash_πᵃ ⋅ dπᵃdμ # similar to above, γᵃ₁⊙ d/dμ{log(πᵃ)} = γᵃ₁⊘ πᵃ⊙ d/dμ{πᵃ}
	dℓdμ₀ = ∑_γᵃ₁_dlogπᵃdμ
	dℓdwₕ = ∑_γᵃ₁_dlogπᵃdμ * trial.previousanswer
	dℓdσ²ᵢ = γᵃ₁_oslash_πᵃ ⋅ dπᵃdσ²
	dℓdB += γᵃ₁_oslash_πᵃ ⋅ dπᵃdB
	dℓdψ = differentiateℓ_wrt_ψ(trial.choice, f[end], θnative.ψ[1])
	Latentθ(B	= [dℓdB],
			k	= [dℓdk],
			λ	= [dℓdλ],
			μ₀	= [dℓdμ₀],
			ϕ	= [dℓdϕ],
			ψ	= [dℓdψ],
			σ²ₐ	= [dℓdσ²ₐ],
			σ²ᵢ	= [dℓdσ²ᵢ],
			σ²ₛ	 = [dℓdσ²ₛ],
			wₕ	 = [dℓdwₕ])
end

"""
	forward(Aᵃ, inputindex, πᵃ, p𝐘d, trialinvariant)

Forward pass of the forward-backward algorithm

ARGUMENT
-`Aᵃ`: transition probabilities of the accumulator variable. Aᵃ[t][j,k] ≡ p(aₜ=ξⱼ ∣ aₜ₋₁=ξₖ)
`inputindex`: index of the time steps with auditory input. For time step `t`, if the element `inputindex[t]` is nonempty, then `Aᵃ[inputindex[t][1]]` is the transition matrix for that time step. If `inputindex[t]` is empty, then the corresponding transition matrix is `Aᵃsilent`.
-`πᵃ`: a vector of floating-point numbers specifying the prior probability of each accumulator state
-`trialinvariant`: a structure containing quantities that are used in each trial

RETURN
-`D`: scaling factors with element `D[t]` ≡ p(𝐘ₜ ∣ 𝐘₁, ... 𝐘ₜ₋₁)
-`f`: Forward recursion terms. `f[t][j,k]` ≡ p(aₜ=ξⱼ, zₜ=k ∣ 𝐘₁, ... 𝐘ₜ) where 𝐘 refers to all the spike trains

"""
function forward(Aᵃ::Vector{<:Matrix{<:AbstractFloat}},
				 choice::Bool,
 				 inputindex::Vector{<:Vector{<:Integer}},
				 πᵃ::Vector{<:AbstractFloat},
				 ψ::Real,
				 trialinvariant::Trialinvariant)
	@unpack Aᵃsilent, Ξ, 𝛏 = trialinvariant
	ntimesteps = length(inputindex)
	f = map(x->zeros(Ξ), 1:ntimesteps)
	D = zeros(ntimesteps)
	f[1] = πᵃ
	D[1] = sum(f[1])
	f[1] /= D[1]
	p𝑑 = conditional_probability_of_choice(choice, ψ, Ξ)
	@inbounds for t = 2:ntimesteps
		if isempty(inputindex[t])
			Aᵃₜ = Aᵃsilent
		else
			i = inputindex[t][1]
			Aᵃₜ = Aᵃ[i]
		end
		f[t] = Aᵃₜ * f[t-1]
		if t == ntimesteps
			f[t] .*= p𝑑
		end
		D[t] = sum(f[t])
		f[t] /= D[t]
	end
	return D,f
end

"""
    conditional_probability_of_choice(choice, ψ, Ξ)

Probability of a choice conditioned on the accumulator state

ARGUMENT
-`choice`: the observed choice, either right (`choice`=true) or left.
-`ψ`: the prior probability of a lapse state
-`Ξ`: number of accumulator states

RETURN
-a vector whose length is equal to the number of accumulator states
"""
function conditional_probability_of_choice(choice::Bool, ψ::Real, Ξ::Integer)
	p = zeros(typeof(ψ), Ξ)
	zeroindex = cld(Ξ,2)
    p[zeroindex] = 0.5
    if choice
        p[1:zeroindex-1]   .= ψ/2
        p[zeroindex+1:end] .= 1-ψ/2
    else
        p[1:zeroindex-1]   .= 1-ψ/2
        p[zeroindex+1:end] .= ψ/2
    end
    return p
end
