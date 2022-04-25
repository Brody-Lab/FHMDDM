"""
	∇∇negativeloglikelihood!(h,concatenatedθ, fgh, indexθ)

Hessian of the negative log-likelihood

MODIFIED ARGUMENT
-`h`: hessian matrix, with respect to only the parameters being fitted and their values in real space

UNMODIFIED ARGUMENT
-`concatenatedθ`: values of the parameters in real space concatenated as a vector
-`fgh`: structure containing the negative log-likelihood, its gradient, and its hessian, as well as the parameters values used to compute each of these quantities
-`indexθ`: a structure for sorting (i.e., un-concatenating) the parameters
-`model`: a structure containing the data and hyperparameters of the factorial hidden drift-diffusion model

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_09_test/data.mat"; randomize=true)
julia> concatenatedθ, indexθ = concatenateparameters(model)
julia> fgh = FHMDDM.FGH(model)
julia> h = similar(fgh.h)
julia> FHMDDM.∇∇negativeloglikelihood!(h, concatenatedθ, fgh, model)
```
"""
function ∇∇negativeloglikelihood!(h::Matrix{<:Real}, concatenatedθ::Vector{<:Real}, fgh::FGH, model::Model)
	if concatenatedθ != fgh.x_h
		ℓ, ∇ℓ, ∇∇ℓ = ∇∇loglikelihood!(model, concatenatedθ, indexθ)
		fgh.f[1] = -ℓ
		for i in eachindex(∇ℓ)
			fgh.g[i] = -∇ℓ[i]
		end
		for ij in eachindex(∇∇ℓ)
			fgh.h[ij] = -∇∇ℓ[ij]
		end
		copyto!(fgh.x_f, concatenatedθ)
		copyto!(fgh.x_g, concatenatedθ)
		copyto!(fgh.x_h, concatenatedθ)
	end
	if h != fgh.h
		copyto!(h, fgh.h)
	end
	return nothing
end

"""
	FGH(model)

Makes a structure storing the negative log-likelihood, its gradient, its Hessian, the parameter values used to compute each of these quantities, and the index of the parameters.
"""
function FGH(model::Model)
	concatenatedθ, indexθ = concatenateparameters(model)
	ℓ, ∇ℓ, ∇∇ℓ = ∇∇loglikelihood!(model, concatenatedθ, indexθ)
	FGH(x_f=concatenatedθ, x_g=concatenatedθ, x_h=concatenatedθ, f=[ℓ], g=∇ℓ, h=∇∇ℓ, indexθ=indexθ)
end

"""
	∇∇loglikelihood!(model, concatenatedθ, indexθ)

Sort a vector of parameters and compute the log-likelihood, its gradient, and its hessian

ARGUMENT
-`model`: a structure containing the data and hyperparameters of the factorial hidden drift-diffusion model
-`concatenatedθ`: values of the parameters in real space concatenated as a vector
-`indexθ`: a structure for sorting (i.e., un-concatenating) the parameters

RETURN
-`ℓ`: log-likelihood
-`∇ℓ`: gradient of the log-likelihood with respect to fitted parameters in real space
-`∇∇ℓ`: Hessian matrix of the log-likelihood with respect to fitted parameters in real space

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true)
julia> concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
julia> ℓ, ∇ℓ, ∇∇ℓ = FHMDDM.∇∇loglikelihood!(model, concatenatedθ, indexθ)
```
"""
function ∇∇loglikelihood!(model::Model, concatenatedθ::Vector{<:Real}, indexθ::Indexθ)
	sortparameters!(model, concatenatedθ, indexθ)
	ℓ, ∇ℓ, ∇∇ℓ = ∇∇loglikelihood(model)
	native2real!(∇ℓ, ∇∇ℓ, indexθ.latentθ, model)
	∇ℓ = sortparameters(indexθ.latentθ, ∇ℓ)
	∇∇ℓ = sortparameters(indexθ.latentθ, ∇∇ℓ)
	return ℓ, ∇ℓ, ∇∇ℓ
end

"""
	∇∇loglikelihood(model)

Hessian of the log-likelihood of data under a factorial hidden Markov drift-diffusion model (FHMDDM).

The gradient and the log-likelihood are also returned

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of an FHMDDM

RETURN
-`ℓ`: log-likelihood
-`∇ℓ`: gradient of the log-likelihood with respect to all parameters in native space
-`∇∇ℓ`: Hessian matrix of the log-likelihood with respect to all parameters in native space

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true)
julia> ℓ, ∇ℓ, ∇∇ℓ = FHMDDM.∇∇loglikelihood(model)
```
"""
function ∇∇loglikelihood(model::Model)
	sameacrosstrials = Sameacrosstrials(model)
	@unpack trialsets = model
	output =map(trialsets, eachindex(trialsets)) do trialset, s
				glmθs = collect(trialset.mpGLMs[n].θ for n = 1:length(trialset.mpGLMs))
		 		map(trialset.trials) do trial #pmap
					∇∇loglikelihood(glmθs, model.θnative, s, sameacrosstrials, trial)
				end
			end
	ℓ = output[1][1][1]
	∇ℓ = output[1][1][2]
	∇∇ℓ = output[1][1][3]
	for i in eachindex(output)
		for m = 2:length(output[i])
			ℓ += output[i][m][1]
			∇ℓ .+= output[i][m][2]
			∇∇ℓ .+= output[i][m][3]
		end
	end
	for i = 1:size(∇∇ℓ,1)
		for j = i+1:size(∇∇ℓ,2)
			∇∇ℓ[j,i] = ∇∇ℓ[i,j]
		end
	end
	return ℓ, ∇ℓ, ∇∇ℓ
end

"""
	∇∇loglikelihood(glmθs, θnative, s, sameacrosstrials, trial)

Hessian of the log-likelihood of the observations from one trial

The gradient and the log-likelihood are also returned

ARGUMENT
-`glmθs`: a vector whose each element is a structure containing the parameters of of the generalized linear model of a neuron
-`θnative`: a structure containing parameters specifying the latent variables in their native space
-`s`: index of the trialset
-`sameacrosstrials`: a structure containing quantities used in each trial
-`trial`: a structure containing information on the sensory stimuli, spike trains, input to each neuron's GLM, and behavioral choice

RETURN
-`ℓ`: log-likelihood
-`∇ℓ`: gradient of the log-likelihood
-`∇∇ℓ`: Hessian matrix of the log-likelihood
"""
function ∇∇loglikelihood(glmθs::Vector{<:GLMθ},
						 θnative::Latentθ,
						 s::Integer,
						 sameacrosstrials::Sameacrosstrials,
						 trial::Trial)
	@unpack clicks = trial
	@unpack Aᵃsilent, ∇Aᵃsilent, ∇∇Aᵃsilent, Aᶜᵀ, ∇Aᶜᵀ, Δt, indexθ_pa₁, indexθ_paₜaₜ₋₁, indexθ_pc₁, indexθ_pcₜcₜ₋₁, indexθ_ψ, K, πᶜᵀ, ∇πᶜᵀ, Ξ, nθ_pa₁, nθ_paₜaₜ₋₁, nθ_pc₁, nθ_pcₜcₜ₋₁, nθ_ψ, index_pa₁_in_θ, index_paₜaₜ₋₁_in_θ, index_pc₁_in_θ, index_pcₜcₜ₋₁_in_θ, index_ψ_in_θ = sameacrosstrials
	indexθ_pY = sameacrosstrials.indexθ_pY[s]
	nθ_pY = sameacrosstrials.nθ_pY[s]
	index_pY_in_θ = sameacrosstrials.index_pY_in_θ[s]
	indexθ_trialset = sameacrosstrials.indexθ_trialset[s]
	nθ_trialset = sameacrosstrials.nθ_trialset[s]
	∇f = map(i->zeros(Ξ,K), 1:nθ_trialset)
	∇∇f = map(i->zeros(Ξ,K), CartesianIndices((nθ_trialset,nθ_trialset)))
	P = Probabilityvector(Δt, θnative, Ξ)
	∇∇pa₁ = map(i->zeros(Ξ), CartesianIndices((nθ_pa₁,nθ_pa₁)))
 	∇pa₁ = map(i->zeros(Ξ), 1:nθ_pa₁)
	∇∇priorprobability!(∇∇pa₁, ∇pa₁, P, trial.previousanswer)
	pa₁ = P.𝛑
	pY = zeros(Ξ,K)
	∇pY = collect(zeros(Ξ,K) for n=1:nθ_pY)
	∇∇pY = map(i->zeros(Ξ,K), CartesianIndices((nθ_pY,nθ_pY)))
	∇∇conditionallikelihood!(∇∇pY, ∇pY, pY, glmθs, 1, trial.spiketrainmodels, sameacrosstrials)
	pY₁⨀pc₁ = pY .* πᶜᵀ
	for i = 1:nθ_pa₁
		q = indexθ_pa₁[i]
		for j = i:nθ_pa₁
			r = indexθ_pa₁[j]
			∇∇f[q,r] = ∇∇pa₁[i,j] .* pY₁⨀pc₁
		end
	end
	pa₁⨀pc₁ = pa₁ .* πᶜᵀ
	for i = 1:nθ_pY
		q = indexθ_pY[i]
		for j = i:nθ_pY
			r = indexθ_pY[j]
			∇∇f[q,r] = ∇∇pY[i,j] .* pa₁⨀pc₁
		end
	end
	for q = 1:nθ_trialset
		for r = q:nθ_trialset
			i = index_pc₁_in_θ[q]
			j = index_pY_in_θ[r]
			if i > 0 && j > 0
				∇∇f[q,r] .+= ∇pY[j] .* pa₁ .* ∇πᶜᵀ[i]
			end
			i = index_pa₁_in_θ[q]
			j = index_pY_in_θ[r]
			if i > 0 && j > 0
				∇∇f[q,r] .+= ∇pY[j] .* ∇pa₁[i] .* πᶜᵀ
			end
			i = index_pa₁_in_θ[q]
			j = index_pc₁_in_θ[r]
			if i > 0 && j > 0
				∇∇f[q,r] .+= pY .* ∇pa₁[i] .* ∇πᶜᵀ[j]
			end
			i = index_pc₁_in_θ[q]
			j = index_pa₁_in_θ[r]
			if i > 0 && j > 0
				∇∇f[q,r] .+= pY .* ∇pa₁[j] .* ∇πᶜᵀ[i]
			end
		end
	end
	for i = 1:nθ_pY
		q = indexθ_pY[i]
		∇f[q] = ∇pY[i] .* pa₁⨀pc₁
	end
	for i = 1:nθ_pa₁
		q = indexθ_pa₁[i]
		∇f[q] = ∇pa₁[i] .* pY₁⨀pc₁
	end
	for i = 1:nθ_pc₁
		q = indexθ_pc₁[i]
		∇f[q] .= pa₁ .* pY .* ∇πᶜᵀ[i]
	end
	f = pa₁⨀pc₁ # reuse memory
	f .*= pY
	∇∇ℓ = zeros(nθ_trialset, nθ_trialset)
	∇ℓ = zeros(nθ_trialset)
	ℓ = zeros(1)
	forward!(∇∇f, ∇f, f, ∇∇ℓ, ∇ℓ, ℓ)
	if !isempty(clicks.inputtimesteps)
		adaptedclicks = ∇∇adapt(clicks, θnative.k[1], θnative.ϕ[1])
		∇∇Aᵃinput = map(i->zeros(Ξ,Ξ), CartesianIndices((nθ_paₜaₜ₋₁,nθ_paₜaₜ₋₁)))
		∇Aᵃinput = map(i->zeros(Ξ,Ξ), 1:nθ_paₜaₜ₋₁)
		Aᵃinput = zeros(Ξ,Ξ)
		Aᵃinput[1,1] = Aᵃinput[Ξ, Ξ] = 1.0
	end
	for t=2:trial.ntimesteps
		if t ∈ clicks.inputtimesteps
			update_for_∇∇transition_probabilities!(P, adaptedclicks, clicks, t)
			∇∇transitionmatrix!(∇∇Aᵃinput, ∇Aᵃinput, Aᵃinput, P)
			∇∇Aᵃ = ∇∇Aᵃinput
			∇Aᵃ = ∇Aᵃinput
			Aᵃ = Aᵃinput
		else
			∇∇Aᵃ = ∇∇Aᵃsilent
			∇Aᵃ = ∇Aᵃsilent
			Aᵃ = Aᵃsilent
		end
		∇∇conditionallikelihood!(∇∇pY, ∇pY, pY, glmθs, t, trial.spiketrainmodels, sameacrosstrials)
		if t==trial.ntimesteps
			∂pY𝑑_∂ψ = pY₁⨀pc₁ # reuse memory
			differentiate_pY𝑑_wrt_ψ!(∂pY𝑑_∂ψ, pY, trial.choice)
			p𝑑 = P.𝛑 #reuse memory
			conditionallikelihood!(p𝑑, trial.choice, θnative.ψ[1])
			pY .*= p𝑑
		end
		f⨉Aᶜᵀ = f * Aᶜᵀ
		Aᵃ⨉f⨉Aᶜᵀ = Aᵃ * f⨉Aᶜᵀ
		for q = 1:nθ_trialset
			for r = q:nθ_trialset
				∇∇f[q,r] = pY .* (Aᵃ * ∇∇f[q,r] * Aᶜᵀ)
			end
		end
		if t==trial.ntimesteps
			dp𝑑_dψ = differentiate_p𝑑_wrt_ψ(trial.choice, K, Ξ)
			for q = 1:nθ_trialset
				for r = q:nθ_trialset
					i = index_ψ_in_θ[q]
					if i > 0
						∇∇f[q,r] .+= ∂pY𝑑_∂ψ .* (Aᵃ * ∇f[r] * Aᶜᵀ)
					end
					j = index_ψ_in_θ[r]
					if j > 0
						∇∇f[q,r] .+= ∂pY𝑑_∂ψ .* (Aᵃ * ∇f[q] * Aᶜᵀ)
					end
					i = index_pcₜcₜ₋₁_in_θ[q]
					# j = index_ψ_in_θ[r]
					if i > 0 && j > 0
						∇∇f[q,r] .+= ∂pY𝑑_∂ψ .* (Aᵃ * f * ∇Aᶜᵀ[i])
					end
					i = index_paₜaₜ₋₁_in_θ[q]
					# j = index_ψ_in_θ[r]
					if i > 0 && j > 0
						∇∇f[q,r] .+= ∂pY𝑑_∂ψ .* (∇Aᵃ[i] * f * Aᶜᵀ)
					end
					i = index_ψ_in_θ[q]
					j = index_paₜaₜ₋₁_in_θ[r]
					if i > 0 && j > 0
						∇∇f[q,r] .+= ∂pY𝑑_∂ψ .* (∇Aᵃ[j] * f * Aᶜᵀ)
					end
					# i = index_ψ_in_θ[q]
					j = index_pY_in_θ[r]
					if i > 0 && j > 0
						∇∇f[q,r] .+= dp𝑑_dψ .* ∇pY[j] .* Aᵃ⨉f⨉Aᶜᵀ
					end
				end
			end
			for i = 1:nθ_pY
				∇pY[i] .*= p𝑑
				for j = i:nθ_pY
					∇∇pY[i,j] .*= p𝑑
				end
			end
		end
		for i = 1:nθ_pY
			q = indexθ_pY[i]
			for j = i:nθ_pY
				r = indexθ_pY[j]
				∇∇f[q,r] .+= ∇∇pY[i,j] .* Aᵃ⨉f⨉Aᶜᵀ
			end
		end
		for i = 1:nθ_paₜaₜ₋₁
			q = indexθ_paₜaₜ₋₁[i]
			for j = i:nθ_paₜaₜ₋₁
				r = indexθ_paₜaₜ₋₁[j]
				∇∇f[q,r] .+= pY .* (∇∇Aᵃ[i,j] * f⨉Aᶜᵀ)
			end
		end
		for q = 1:nθ_trialset
			for r = q:nθ_trialset
				i = index_pY_in_θ[q]
				if i > 0
					∇∇f[q,r] .+= ∇pY[i] .* (Aᵃ * ∇f[r] * Aᶜᵀ)
				end
				j = index_pY_in_θ[r]
				if j > 0
					∇∇f[q,r] .+= ∇pY[j] .* (Aᵃ * ∇f[q] * Aᶜᵀ)
				end
				i = index_paₜaₜ₋₁_in_θ[q]
				if i > 0
					∇∇f[q,r] .+= pY .* (∇Aᵃ[i] * ∇f[r] * Aᶜᵀ)
				end
				j = index_paₜaₜ₋₁_in_θ[r]
				if j > 0
					∇∇f[q,r] .+= pY .* (∇Aᵃ[j] * ∇f[q] * Aᶜᵀ)
				end
				i = index_pcₜcₜ₋₁_in_θ[q]
				if i > 0
					∇∇f[q,r] .+= pY .* (Aᵃ * ∇f[r] * ∇Aᶜᵀ[i])
				end
				j = index_pcₜcₜ₋₁_in_θ[r]
				if j > 0
					∇∇f[q,r] .+= pY .* (Aᵃ * ∇f[q] * ∇Aᶜᵀ[j])
				end
				i = index_paₜaₜ₋₁_in_θ[q]
				j = index_pY_in_θ[r]
				if i > 0 && j > 0
					∇∇f[q,r] .+= ∇pY[j] .* (∇Aᵃ[i] * f⨉Aᶜᵀ)
				end
				i = index_pcₜcₜ₋₁_in_θ[q]
				j = index_pY_in_θ[r]
				if i > 0 && j > 0
					∇∇f[q,r] .+= ∇pY[j] .* (Aᵃ * f * ∇Aᶜᵀ[i])
				end
				i = index_pcₜcₜ₋₁_in_θ[q]
				j = index_paₜaₜ₋₁_in_θ[r]
				if i > 0 && j > 0
					∇∇f[q,r] .+= pY .* (∇Aᵃ[j] * f * ∇Aᶜᵀ[i])
				end
			end
		end
		for q = 1:nθ_trialset
			∇f[q] = pY .* (Aᵃ * ∇f[q] * Aᶜᵀ)
		end
		for i = 1:nθ_paₜaₜ₋₁
			q = indexθ_paₜaₜ₋₁[i]
			∇f[q] .+= pY .* (∇Aᵃ[i] * f⨉Aᶜᵀ)
		end
		for i = 1:nθ_pY
			q = indexθ_pY[i]
			∇f[q] .+= ∇pY[i] .* Aᵃ⨉f⨉Aᶜᵀ
		end
		Aᵃ⨉f = Aᵃ * f
		for i = 1:nθ_pcₜcₜ₋₁
			q = indexθ_pcₜcₜ₋₁[i]
			∇f[q] .+= pY .* (Aᵃ⨉f * ∇Aᶜᵀ[i])
		end
		if t==trial.ntimesteps
			q = indexθ_ψ[1]
			∇f[q] = ∂pY𝑑_∂ψ .* Aᵃ⨉f⨉Aᶜᵀ
		end
		f = Aᵃ⨉f⨉Aᶜᵀ # reuse memory
		f .*= pY
		forward!(∇∇f, ∇f, f, ∇∇ℓ, ∇ℓ, ℓ)
	end
	return ℓ[1], ∇ℓ, ∇∇ℓ
end

"""
	∇loglikelihood!(model, concatenatedθ, indexθ)

Sort a vector of parameters and compute the log-likelihood and its gradient

ARGUMENT
-`model`: a structure containing the data and hyperparameters of the factorial hidden drift-diffusion model
-`concatenatedθ`: values of the parameters in real space concatenated as a vector
-`indexθ`: a structure for sorting (i.e., un-concatenating) the parameters

RETURN
-`ℓ`: log-likelihood
-`∇ℓ`: gradient of the log-likelihood with respect to fitted parameters in real space

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true)
julia> concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
julia> ℓ, ∇ℓ = FHMDDM.∇loglikelihood!(model, concatenatedθ, indexθ)
```
"""
function ∇loglikelihood!(model::Model, concatenatedθ::Vector{<:Real}, indexθ::Indexθ)
	sortparameters!(model, concatenatedθ, indexθ)
	ℓ, ∇ℓ = ∇loglikelihood(model)
	native2real!(∇ℓ, indexθ.latentθ, model)
	∇ℓ = sortparameters(indexθ.latentθ, ∇ℓ)
	return ℓ, ∇ℓ
end

"""
	∇loglikelihood(model)

Gradient of the log-likelihood of data under a factorial hidden Markov drift-diffusion model (FHMDDM).

The log-likelihood is also returned

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of an FHMDDM

RETURN
-`ℓ`: log-likelihood
-`∇ℓ`: gradient of the log-likelihood with respect to all parameters in native space

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true)
julia> ℓ, ∇ℓ = FHMDDM.∇loglikelihood(model)
```
"""
function ∇loglikelihood(model::Model)
	sameacrosstrials = FHMDDM.Sameacrosstrials(model)
	@unpack θnative, trialsets = model
	@unpack Δt, K, Ξ = model.options
	Aᵃinput = zeros(Ξ,Ξ)
	Aᵃinput[1,1] = 1.0
	Aᵃinput[Ξ,Ξ] = 1.0
	∇Aᵃinput = map(i->zeros(Ξ,Ξ), 1:sameacrosstrials.nθ_paₜaₜ₋₁)
	ℓ = zeros(1)
	∇ℓ = zeros(sameacrosstrials.nθ_alltrialsets)
	∇f = map(i->zeros(Ξ,K), 1:sameacrosstrials.nθ_alltrialsets)
 	∇pa₁ = map(i->zeros(Ξ), 1:sameacrosstrials.nθ_pa₁)
	pY = zeros(Ξ,K)
	∇pY = collect(zeros(Ξ,K) for n=1:sameacrosstrials.nθ_pY[1])
	P = FHMDDM.Probabilityvector(Δt, θnative, Ξ)
	for s in eachindex(trialsets)
		glmθs = collect(trialsets[s].mpGLMs[n].θ for n = 1:length(trialsets[s].mpGLMs))
		for m in eachindex(trialsets[s].trials)
			FHMDDM.∇loglikelihood!(Aᵃinput, ∇Aᵃinput, ℓ, ∇ℓ, ∇f, ∇pa₁, pY, ∇pY, P, glmθs, θnative, s, sameacrosstrials, trialsets[s].trials[m])
		end
	end
	return ℓ[1], ∇ℓ
end

"""
	∇loglikelihood(glmθs, θnative, s, sameacrosstrials, trial)

Gradient of the log-likelihood of the observations from one trial

The log-likelihood is also returned

ARGUMENT
-`glmθs`: a vector whose each element is a structure containing the parameters of of the generalized linear model of a neuron
-`θnative`: a structure containing parameters specifying the latent variables in their native space
-`s`: index of the trialset
-`sameacrosstrials`: a structure containing quantities used in each trial
-`trial`: a structure containing information on the sensory stimuli, spike trains, input to each neuron's GLM, and behavioral choice

RETURN
-`ℓ`: log-likelihood
-`∇ℓ`: gradient of the log-likelihood
"""
function ∇loglikelihood!(Aᵃinput::Matrix{<:Real},
						∇Aᵃinput::Vector{<:Matrix{<:Real}},
						ℓ::Vector{<:Real},
						∇ℓ::Vector{<:Real},
						∇f::Vector{<:Matrix{<:Real}},
						∇pa₁::Vector{<:Vector{<:Real}},
						pY::Matrix{<:Real},
						∇pY::Vector{<:Matrix{<:Real}},
						P::Probabilityvector,
						glmθs::Vector{<:GLMθ},
						θnative::Latentθ,
						s::Integer,
						sameacrosstrials::Sameacrosstrials,
						trial::Trial)
	@unpack clicks = trial
	@unpack Aᵃsilent, ∇Aᵃsilent, Aᶜᵀ, ∇Aᶜᵀ, Δt, indexθ_pa₁, indexθ_paₜaₜ₋₁, indexθ_pc₁, indexθ_pcₜcₜ₋₁, indexθ_ψ, K, πᶜᵀ, ∇πᶜᵀ, Ξ, nθ_pa₁, nθ_paₜaₜ₋₁, nθ_pc₁, nθ_pcₜcₜ₋₁, nθ_ψ, d𝛏_dB = sameacrosstrials
	indexθ_pY = sameacrosstrials.indexθ_pY[s]
	indexθ_trialset = sameacrosstrials.indexθ_trialset[s]
	nθ_pY = sameacrosstrials.nθ_pY[s]
	nθ_trialset = sameacrosstrials.nθ_trialset[s]
	∇priorprobability!(∇pa₁, P, trial.previousanswer)
	pa₁ = P.𝛑
	∇conditionallikelihood!(∇pY, pY, glmθs, 1, trial.spiketrainmodels, sameacrosstrials)
	pa₁⨀pc₁ = pa₁ .* πᶜᵀ
	for i = 1:nθ_pY
		q = indexθ_pY[i]
		for jk in eachindex(pa₁⨀pc₁)
			∇f[q][jk] = ∇pY[i][jk] * pa₁⨀pc₁[jk]
		end
	end
	pY₁⨀pc₁ = pY .* πᶜᵀ
	for i = 1:nθ_pa₁
		q = indexθ_pa₁[i]
		for j=1:Ξ
			for k = 1:K
				∇f[q][j,k] = ∇pa₁[i][j] * pY₁⨀pc₁[j,k]
			end
		end
	end
	for i = 1:nθ_pc₁
		q = indexθ_pc₁[1]
		for j=1:Ξ
			for k = 1:K
				∇f[q][j,k] = pY[j,k] * pa₁[j] * ∇πᶜᵀ[i][k]
			end
		end
	end
	f = pa₁⨀pc₁ # reuse memory
	f .*= pY
	forward!(∇f, f, ∇ℓ, ℓ)
	if !isempty(clicks.inputtimesteps)
		adaptedclicks = ∇adapt(clicks, θnative.k[1], θnative.ϕ[1])
	end
	indexθ_trialset_but_ψ = indexθ_trialset[vcat(1:(indexθ_ψ[1]-1), indexθ_ψ[1]+1:end)]
	for t=2:trial.ntimesteps
		if t ∈ clicks.inputtimesteps
			update_for_∇transition_probabilities!(P, adaptedclicks, clicks, t)
			∇transitionmatrix!(∇Aᵃinput, Aᵃinput, P)
			∇Aᵃ = ∇Aᵃinput
			Aᵃ = Aᵃinput
		else
			∇Aᵃ = ∇Aᵃsilent
			Aᵃ = Aᵃsilent
		end
		∇conditionallikelihood!(∇pY, pY, glmθs, t, trial.spiketrainmodels, sameacrosstrials)
		conditionallikelihood!(pY, Δt, d𝛏_dB, glmθs, K, t, trial.spiketrainmodels)
		if t==trial.ntimesteps
			∂pY𝑑_∂ψ = pY₁⨀pc₁ # reuse memory
			differentiate_pY𝑑_wrt_ψ!(∂pY𝑑_∂ψ, pY, trial.choice)
			p𝑑 = P.𝛑 #reuse memory
			conditionallikelihood!(p𝑑, trial.choice, θnative.ψ[1])
			pY .*= p𝑑
			for i = 1:nθ_pY
				∇pY[i] .*= p𝑑
			end
		end
		f⨉Aᶜᵀ = f * Aᶜᵀ
		Aᵃ⨉f⨉Aᶜᵀ = Aᵃ * f⨉Aᶜᵀ
		Aᵃ⨉f = Aᵃ * f
		if t == 2
			for q in indexθ_trialset
				if q∈indexθ_paₜaₜ₋₁[2:end]
					∇f[q] .= 0
				elseif q ∈ indexθ_pcₜcₜ₋₁
					∇f[q] .= 0
				elseif q ∈ indexθ_ψ
					∇f[q] .= 0
				else
					∇f[q] = pY .* (Aᵃ * ∇f[q] * Aᶜᵀ)
				end
			end
		else
			for q in indexθ_trialset_but_ψ
				∇f[q] = pY .* (Aᵃ * ∇f[q] * Aᶜᵀ)
			end
		end
		for i = 1:nθ_paₜaₜ₋₁
			q = indexθ_paₜaₜ₋₁[i]
			∇Aᵃ⨉f⨉Aᶜᵀ = ∇Aᵃ[i] * f⨉Aᶜᵀ
			for jk in eachindex(pY)
				∇f[q][jk] += pY[jk] * ∇Aᵃ⨉f⨉Aᶜᵀ[jk]
			end
		end
		for i = 1:nθ_pY
			q = indexθ_pY[i]
			for jk in eachindex(Aᵃ⨉f⨉Aᶜᵀ)
				∇f[q][jk] += ∇pY[i][jk] * Aᵃ⨉f⨉Aᶜᵀ[jk]
			end
		end
		for i = 1:nθ_pcₜcₜ₋₁
			q = indexθ_pcₜcₜ₋₁[i]
			Aᵃ⨉f⨉∇Aᶜᵀ = Aᵃ⨉f * ∇Aᶜᵀ[i]
			for jk in eachindex(pY)
				∇f[q][jk] += pY[jk] * Aᵃ⨉f⨉∇Aᶜᵀ[jk]
			end
		end
		if t==trial.ntimesteps
			for jk in eachindex(∂pY𝑑_∂ψ)
				∇f[indexθ_ψ[1]][jk] = ∂pY𝑑_∂ψ[jk] * Aᵃ⨉f⨉Aᶜᵀ[jk]
			end
		end
		f = Aᵃ⨉f⨉Aᶜᵀ # reuse memory
		f .*= pY
		forward!(∇f, f, ∇ℓ, ℓ)
	end
	return nothing
end

"""
	loglikelihood(model)

Log-likelihood of data under a factorial hidden Markov drift-diffusion model (FHMDDM).

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of an FHMDDM

RETURN
-`ℓ`: log-likelihood

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true)
julia> ℓ = loglikelihood(model)
```
"""
function loglikelihood(model::Model)
	sameacrosstrials = Sameacrosstrials(model)
	@unpack options, θnative, trialsets = model
	@unpack Δt, K, Ξ = options
	P = Probabilityvector(Δt, θnative, Ξ)
	pY = zeros(Ξ,K)
	Aᵃinput = zeros(Ξ,Ξ)
	Aᵃinput[1,1] = Aᵃinput[Ξ, Ξ] = 1.0
	ℓ = 0.0
	for s in eachindex(trialsets)
		glmθs = collect(trialsets[s].mpGLMs[n].θ for n = 1:length(trialsets[s].mpGLMs))
		for m in eachindex(trialsets[s].trials)
			ℓ += loglikelihood!(Aᵃinput, P, pY, glmθs, model.θnative, sameacrosstrials, trialsets[s].trials[m])
		end
	end
	return ℓ
end

"""
	loglikelihood(glmθs, θnative, s, sameacrosstrials, trial)

Log-likelihood of the observations from one trial, not meant for ForwardDiff

ARGUMENT
-`glmθs`: a vector whose each element is a structure containing the parameters of of the generalized linear model of a neuron
-`θnative`: a structure containing parameters specifying the latent variables in their native space
-`s`: index of the trialset
-`sameacrosstrials`: a structure containing quantities used in each trial
-`trial`: a structure containing information on the sensory stimuli, spike trains, input to each neuron's GLM, and behavioral choice

RETURN
-`ℓ`: log-likelihood
"""
function loglikelihood!(Aᵃinput::Matrix{<:Real},
					   P::Probabilityvector,
						pY::Matrix{<:Real},
						glmθs::Vector{<:GLMθ},
					   	θnative::Latentθ,
						sameacrosstrials::Sameacrosstrials,
						trial::Trial)
	@unpack clicks = trial
	@unpack Aᵃsilent, Aᶜᵀ, Δt, K, πᶜᵀ, Ξ, d𝛏_dB = sameacrosstrials
	priorprobability!(P, trial.previousanswer)
	pa₁ = P.𝛑
	conditionallikelihood!(pY, Δt, d𝛏_dB, glmθs, K, 1, trial.spiketrainmodels)
	f = pY .* pa₁ .* πᶜᵀ
	ℓ = zeros(1)
	forward!(f, ℓ)
	if !isempty(clicks.inputtimesteps)
		adaptedclicks = adapt(clicks, θnative.k[1], θnative.ϕ[1])
	end
	for t=2:trial.ntimesteps
		if t ∈ clicks.inputtimesteps
			update_for_transition_probabilities!(P, adaptedclicks, clicks, t)
			transitionmatrix!(Aᵃinput, P)
			Aᵃ = Aᵃinput
		else
			Aᵃ = Aᵃsilent
		end
		conditionallikelihood!(pY, Δt, d𝛏_dB, glmθs, K, t, trial.spiketrainmodels)
		if t==trial.ntimesteps
			p𝑑 = P.𝛑 #reuse memory
			conditionallikelihood!(p𝑑, trial.choice, θnative.ψ[1])
			pY .*= p𝑑
		end
		f = pY .* (Aᵃ * f * Aᶜᵀ)
		forward!(f, ℓ)
	end
	return ℓ[1]
end

"""
EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true)
julia> p𝐘𝑑 = FHMDDM.likelihood(model)
julia> ℓ = FHMDDM.loglikelihood!(p𝐘𝑑, model)
```
"""
function loglikelihood!(p𝐘𝑑::Vector{<:Vector{<:Vector{<:Matrix{<:Real}}}}, model::Model)
	sameacrosstrials = Sameacrosstrials(model)
	@unpack options, θnative, trialsets = model
	@unpack Δt, K, Ξ = options
	likelihood!(p𝐘𝑑, model.trialsets, model.θnative.ψ[1])
	P = Probabilityvector(Δt, θnative, Ξ)
	Aᵃinput = zeros(Ξ,Ξ)
	Aᵃinput[1,1] = Aᵃinput[Ξ, Ξ] = 1.0
	ℓ = 0.0
	for s in eachindex(model.trialsets)
		for m in eachindex(model.trialsets[s].trials)
			ℓ += loglikelihood!(Aᵃinput, P, p𝐘𝑑[s][m], model.θnative, sameacrosstrials, model.trialsets[s].trials[m])
		end
	end
	return ℓ
end

"""
	loglikelihood(glmθs, θnative, s, sameacrosstrials, trial)

Log-likelihood of the observations from one trial, not meant for ForwardDiff

ARGUMENT
-`glmθs`: a vector whose each element is a structure containing the parameters of of the generalized linear model of a neuron
-`θnative`: a structure containing parameters specifying the latent variables in their native space
-`s`: index of the trialset
-`sameacrosstrials`: a structure containing quantities used in each trial
-`trial`: a structure containing information on the sensory stimuli, spike trains, input to each neuron's GLM, and behavioral choice

RETURN
-`ℓ`: log-likelihood
"""
function loglikelihood!(Aᵃinput::Matrix{<:Real},
					    P::Probabilityvector,
						p𝐘𝑑::Vector{<:Matrix{<:Real}},
					   	θnative::Latentθ,
						sameacrosstrials::Sameacrosstrials,
						trial::Trial)
	@unpack clicks = trial
	@unpack Aᵃsilent, Aᶜᵀ, Δt, K, πᶜᵀ, Ξ = sameacrosstrials
	priorprobability!(P, trial.previousanswer)
	pa₁ = P.𝛑
	f = p𝐘𝑑[1] .* pa₁ .* πᶜᵀ
	ℓ = zeros(1)
	forward!(f, ℓ)
	if !isempty(clicks.inputtimesteps)
		adaptedclicks = adapt(clicks, θnative.k[1], θnative.ϕ[1])
	end
	for t=2:trial.ntimesteps
		if t ∈ clicks.inputtimesteps
			update_for_transition_probabilities!(P, adaptedclicks, clicks, t)
			transitionmatrix!(Aᵃinput, P)
			Aᵃ = Aᵃinput
		else
			Aᵃ = Aᵃsilent
		end
		f = p𝐘𝑑[t] .* (Aᵃ * f * Aᶜᵀ)
		forward!(f, ℓ)
	end
	return ℓ[1]
end

"""
	loglikelihood_θnative(concatenatedθnative, model)

Log-likelihood of the data given the complete set of parameters in their native space

ARGUMENT
-`x`: vector of concatenated parameters in their native space. This vector needs to include all parameters, regardless whether they are being fitted
-`model`: a structure containing the

RETURN
-log-likelihood of the data

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_09_test/data.mat"; randomize=true)
julia> x = FHMDDM.concatenate_native_parameters(model)
julia> ℓ = FHMDDM.loglikelihood_θnative(x, model)
```
"""
function loglikelihood_θnative(x::Vector{<:Real}, model::Model)
	model = sort_native_parameters(x, model)
	@unpack options, trialsets = model
	output =map(trialsets) do trialset
				glmθs = collect(trialset.mpGLMs[n].θ for n = 1:length(trialset.mpGLMs))
		 		omap(trialset.trials) do trial
					loglikelihood(glmθs, options, model.θnative, trial)
				end
			end
	ℓ = output[1][1]
	for i in eachindex(output)
		for m = 2:length(output[i])
			ℓ += output[i][m]
		end
	end
	return ℓ
end

"""
	loglikelihood(glmθs, θnative, sameacrosstrials, trial)

Log-likelihood of the data in one trial

ARGUMENT
-`glmθs`: a vector whose each element is a structure containing the parameters of of the generalized linear model of a neuron
-`θnative`: a structure containing parameters specifying the latent variables in their native space
-`sameacrosstrials`: a structure containing quantities used in each trial
-`trial`: a structure containing information on the sensory stimuli, spike trains, input to each neuron's GLM, and behavioral choice

RETURN
-`ℓ`: log-likelihood
"""
function loglikelihood(glmθs::Vector{<:GLMθ},
   					   options::Options,
					   θnative::Latentθ,
					   trial::Trial)
	@unpack clicks = trial
	@unpack Δt, K, Ξ = options
	type = eltype(glmθs[1].𝐮[1])
	adaptedclicks = adapt(clicks, θnative.k[1], θnative.ϕ[1])
	Aᵃinput, Aᵃsilent = zeros(type,Ξ,Ξ), zeros(type,Ξ,Ξ)
	expλΔt = exp(θnative.λ[1]*Δt)
	dμ_dΔc = differentiate_μ_wrt_Δc(Δt, θnative.λ[1])
	d𝛏_dB = (2 .*collect(1:Ξ) .- Ξ .- 1)./(Ξ-2)
	𝛏 = θnative.B[1].*d𝛏_dB
	transitionmatrix!(Aᵃsilent, expλΔt.*𝛏, √(Δt*θnative.σ²ₐ[1]), 𝛏)
	Aᶜ₁₁ = θnative.Aᶜ₁₁[1]
	Aᶜ₂₂ = θnative.Aᶜ₂₂[1]
	πᶜ₁ = θnative.πᶜ₁[1]
	if K == 2
		Aᶜᵀ = [Aᶜ₁₁ 1-Aᶜ₁₁; 1-Aᶜ₂₂ Aᶜ₂₂]
		πᶜᵀ = [πᶜ₁ 1-πᶜ₁]
	else
		Aᶜᵀ = ones(1,1)
		πᶜᵀ = ones(1,1)
	end
	pa₁ = probabilityvector(θnative.μ₀[1]+θnative.wₕ[1]*trial.previousanswer, √θnative.σ²ᵢ[1], 𝛏)
	pY = zeros(type, Ξ, K)
	conditionallikelihood!(pY, Δt, d𝛏_dB, glmθs, K, 1, trial.spiketrainmodels)
	f = pY .* pa₁ .* πᶜᵀ
	ℓ = zeros(type, 1)
	forward!(f, ℓ)
	for t=2:trial.ntimesteps
		if t ∈ clicks.inputtimesteps
			cL = sum(adaptedclicks.C[clicks.left[t]])
			cR = sum(adaptedclicks.C[clicks.right[t]])
			𝛍 = expλΔt.*𝛏 .+ (cR-cL)*dμ_dΔc
			σ = √((cR+cL)*θnative.σ²ₛ[1] + Δt*θnative.σ²ₐ[1])
			transitionmatrix!(Aᵃinput, 𝛍, σ, 𝛏)
			Aᵃ = Aᵃinput
		else
			Aᵃ = Aᵃsilent
		end
		conditionallikelihood!(pY, Δt, d𝛏_dB, glmθs, K, t, trial.spiketrainmodels)
		if t==trial.ntimesteps
			p𝑑 = pa₁ #reuse memory
			conditionallikelihood!(p𝑑, trial.choice, θnative.ψ[1])
			pY .*= p𝑑
		end
		f = pY .* (Aᵃ * f * Aᶜᵀ)
		forward!(f, ℓ)
	end
	return ℓ[1]
end

"""
	compare_gradients_hessians_θnative(model)

Compare the automatically computed and hand-coded gradients and hessians with respect to the parameters in their native space

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of a factorial hidden-Markov drift-diffusion model

RETURN
-`absdiffℓ`: absolute difference in the log-likelihood evaluted using the algorithm bein automatically differentiated and the hand-coded algorithm
-`absdiff∇`: absolute difference in the gradients
-`absdiff∇∇`: absolute difference in the hessians

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_09_test/data.mat"; randomize=true)
julia> absdiffℓ, absdiff∇, absdiff∇∇ = FHMDDM.compare_gradients_hessians_θnative(model)
julia> maximum(absdiff∇∇)
```
"""
function compare_gradients_hessians_θnative(model::Model)
	x0 = concatenate_native_parameters(model)
	f(x) = loglikelihood_θnative(x, model)
	ℓauto = f(x0)
	∇auto = ForwardDiff.gradient(f, x0)
	∇∇auto = ForwardDiff.hessian(f, x0)
	ℓhand, ∇hand, ∇∇hand = FHMDDM.∇∇loglikelihood(model)
	return abs(ℓauto-ℓhand), abs.(∇auto .- ∇hand), abs.(∇∇auto .- ∇∇hand)
end

"""
    loglikelihood(concatenatedθ, indexθ, model)

Compute the log-likelihood in a way that is compatible with ForwardDiff

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values
-`model`: an instance of FHM-DDM

RETURN
-log-likelihood
"""
# function loglikelihood(concatenatedθ::Vector{<:Real},
# 					   indexθ::Indexθ,
# 					   model::Model)
# 	model = sortparameters(concatenatedθ, indexθ, model)
# 	@unpack options, trialsets = model
# 	output =map(trialsets) do trialset
# 				glmθs = collect(trialset.mpGLMs[n].θ for n = 1:length(trialset.mpGLMs))
# 		 		map(trialset.trials) do trial #pmap
# 					loglikelihood(glmθs, options, model.θnative, trial)
# 				end
# 			end
# 	ℓ = output[1][1]
# 	for i in eachindex(output)
# 		for m = 2:length(output[i])
# 			ℓ += output[i][m]
# 		end
# 	end
# 	return ℓ
# end

"""
	compare_gradients_hessians(model)

Compare the automatically computed and hand-coded gradients and hessians with respect to the parameters being fitted in their real space

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of a factorial hidden-Markov drift-diffusion model

RETURN
-`absdiffℓ`: absolute difference in the log-likelihood evaluted using the algorithm bein automatically differentiated and the hand-coded algorithm
-`absdiff∇`: absolute difference in the gradients
-`absdiff∇∇`: absolute difference in the hessians

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true)
julia> absdiffℓ, absdiff∇, absdiff∇∇ = FHMDDM.compare_gradients_hessians(model)
```
"""
function compare_gradients_hessians(model::Model)
	concatenatedθ, indexθ = concatenateparameters(model)
	f(x) = loglikelihood(x, indexθ, model)
	ℓauto = f(concatenatedθ)
	∇auto = ForwardDiff.gradient(f, concatenatedθ)
	∇∇auto = ForwardDiff.hessian(f, concatenatedθ)
	ℓhand, ∇hand, ∇∇hand = FHMDDM.∇∇loglikelihood!(model,concatenatedθ,indexθ)
	return abs(ℓauto-ℓhand), abs.(∇auto .- ∇hand), abs.(∇∇auto .- ∇∇hand)
end

"""
	compare_gradients(model)

Compare the automatically computed and hand-coded gradients with respect to the parameters being fitted in their real space

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of a factorial hidden-Markov drift-diffusion model

RETURN
-`absdiffℓ`: absolute difference in the log-likelihood evaluted using the algorithm bein automatically differentiated and the hand-coded algorithm
-`absdiff∇`: absolute difference in the gradients

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true)
julia> absdiffℓ, absdiff∇ = FHMDDM.compare_gradients(model)
```
"""
function compare_gradients(model::Model)
	concatenatedθ, indexθ = concatenateparameters(model)
	f(x) = loglikelihood(x, indexθ, model)
	ℓauto = f(concatenatedθ)
	∇auto = ForwardDiff.gradient(f, concatenatedθ)
	ℓhand, ∇hand = FHMDDM.∇loglikelihood!(model,concatenatedθ,indexθ)
	return abs(ℓauto-ℓhand), abs.(∇auto .- ∇hand)
end

"""
	forward!(∇∇f,∇f,f,∇∇ℓ,∇ℓ,ℓ)

Normalize the forward term and its first- and second-order partial derivatives and update the log-likelihood and its first- and second-order partial derivatives

For computational efficiency, no entry below the main diagonal of the Hessian matrix of the log-likelihood is updated

ARGUMENT
-`∇∇f`: Second-order partial derivatives of the un-normalized forward term. Element `∇∇f[q,r][i,j]` corresponds to the derivative of the un-normalized forward term in the i-th accumulator state and j-th coupling state with respect to the q-th and r-th parameter.
-`∇f`: first-order partial derivatives of the un-normalized forward term. Element `∇f[q][i,j]` corresponds to the derivative of the un-normalized forward term in the i-th accumulator state and j-th coupling state with respect to the q-th parameter.
-`f`: un-normalized forward term. Element `f[i,j]` corresponds to the un-normalized forward term in the i-th accumulator state and j-th coupling state
-`∇∇ℓ`: Hessian matrix of the log-likelihood for the data before the current time-step
-`∇ℓ`: gradient of the log-likelihood for the data before the current time-step
-`ℓ`: log-likelihood for the data before the current time-step

POST-MODIFICATION ARGUMENT
-`∇∇f`: Second-order partial derivatives of the normalized forward term. Element `∇∇f[q,r][i,j]` corresponds to `∂²p{a(t)=ξ(i), c(t)=j ∣ 𝐘(1:t)} / ∂θ(q)∂θ(r)`
-`∇f`: first-order partial derivatives of the normalized forward term. Element `∇f[q][i,j]` corresponds to `∂p{a(t)=ξ(i), c(t)=j ∣ 𝐘(1:t)} / ∂θ(q)`
-`f`: normalized forward term, which represents the joint conditional probability of the latent variables given the elapsed emissions. Element `f[i,j]` corresponds to `p{a(t)=ξ(i), c(t)=j ∣ 𝐘(1:t)}`
-`∇∇ℓ`: Hessian matrix of the log-likelihood for the data up to the current time-step
-`∇ℓ`: gradient of the log-likelihood for the data up to the current time-step
-`ℓ`: log-likelihood for the data before the up to the current time-step

"""
function forward!(∇∇f::Matrix{<:Matrix{<:Real}},
				  ∇f::Vector{<:Matrix{<:Real}},
				  f::Matrix{<:Real},
				  ∇∇ℓ::Matrix{<:Real},
 				  ∇ℓ::Vector{<:Real},
				  ℓ::Vector{<:Real})
	∇D, D = forward!(∇f,f,∇ℓ,ℓ)
	nparameters = length(∇f)
	for i = 1:nparameters
		for j = i:nparameters
			∇∇Dᵢⱼ = sum(∇∇f[i,j])
			∇∇ℓ[i,j] += (∇∇Dᵢⱼ - ∇D[i]*∇D[j]/D)/D
			∇∇f[i,j] = (∇∇f[i,j] .- ∇f[i].*∇D[j] .- ∇f[j].*∇D[i] .- f.*∇∇Dᵢⱼ)./D
		end
	end
	return nothing
end

"""
	forward!(∇f,f,∇ℓ,ℓ)

Normalize the forward term and its first-order partial derivatives and update the log-likelihood and its first-partial derivatives

ARGUMENT
-`∇f`: first-order partial derivatives of the un-normalized forward term. Element `∇f[q][i,j]` corresponds to the derivative of the un-normalized forward term in the i-th accumulator state and j-th coupling state with respect to the q-th parameter.
-`f`: un-normalized forward term. Element `f[i,j]` corresponds to the un-normalized forward term in the i-th accumulator state and j-th coupling state
-`∇ℓ`: gradient of the log-likelihood for the data before the current time-step
-`ℓ`: log-likelihood for the data before the current time-step

POST-MODIFICATION ARGUMENT
-`∇f`: first-order partial derivatives of the normalized forward term. Element `∇f[q][i,j]` corresponds to `∂p{a(t)=ξ(i), c(t)=j ∣ 𝐘(1:t)} / ∂θ(q)`
-`f`: normalized forward term, which represents the joint conditional probability of the latent variables given the elapsed emissions. Element `f[i,j]` corresponds to `p{a(t)=ξ(i), c(t)=j ∣ 𝐘(1:t)}`
-`∇ℓ`: gradient of the log-likelihood for the data up to the current time-step
-`ℓ`: log-likelihood for the data before the up to the current time-step

RETURN
-`∇D`: gradient of the past-conditioned emissions likelihood for the current time step
-`D`: past-conditioned emissions likelihood: `p{𝐘(t) ∣ 𝐘(1:t-1))`
"""
function forward!(∇f::Vector{<:Matrix{<:Real}},
				  f::Matrix{<:Real},
 				  ∇ℓ::Vector{<:Real},
				  ℓ::Vector{<:Real})
	D = forward!(f,ℓ)
	∇D = map(sum, ∇f)
	∇ℓ .+= ∇D./D
	for i in eachindex(∇f)
		∇f[i] = (∇f[i] .- f.*∇D[i])./D
	end
	return ∇D, D
end

"""
	forward!(f,ℓ)

Normalize the forward term and update the log-likelihood to include the emissions from the current time step

ARGUMENT
-`f`: un-normalized forward term. Element `f[i,j]` corresponds to the un-normalized forward term in the i-th accumulator state and j-th coupling state
-`ℓ`: log-likelihood for the data before the current time-step

POST-MODIFICATION ARGUMENT
-`f`: normalized forward term, which represents the joint conditional probability of the latent variables given the elapsed emissions. Element `f[i,j]` corresponds to `p{a(t)=ξ(i), c(t)=j ∣ 𝐘(1:t)}`
-`ℓ`: log-likelihood for the data before the up to the current time-step

RETURN
-`D`: past-conditioned emissions likelihood: `p{𝐘(t) ∣ 𝐘(1:t-1))`
"""
function forward!(f::Matrix{<:Real},
				  ℓ::Vector{<:Real})
	D = sum(f)
	ℓ[1] += log(D)
	f ./= D
	return D
end

"""
	differentiate_pY𝑑_wrt_ψ!(∂pY𝑑_∂ψ, pY, 𝑑, ψ)

Partial derivatives of the conditional likelihood of the emissions at the last time step with respect to the lapse rate

ARGUMENT
-`pY`: conditional likelihood of the population spiking at the last time step `T`. Element `pY[i,j]` represents p{Y(T) ∣ a(T)=ξ(i), c(T)=j}
-`𝑑`: left (false) or right (true) choice of the animal

MODIFIED ARGUMENT
-`∂pY𝑑_∂ψ`: partial derivative of the emissions at the last time step (population spiking and the choice) with respect to the lapse rate. Element `∂pY𝑑_∂ψ[i,j]` represents:
	∂p{Y(T), 𝑑 ∣ a(T)=ξ(i), c(T)=j}}/∂ψ
"""
function differentiate_pY𝑑_wrt_ψ!(∂pY𝑑_∂ψ::Matrix{<:Real}, pY::Matrix{<:Real}, 𝑑::Bool)
	if 𝑑
		∂p𝑑_ξ⁻_∂ψ = 0.5
		∂p𝑑_ξ⁺_∂ψ = -0.5
	else
		∂p𝑑_ξ⁻_∂ψ = -0.5
		∂p𝑑_ξ⁺_∂ψ = 0.5
	end
	Ξ,K = size(pY)
	zeroindex = cld(Ξ,2)
	for j = 1:K
		for i = 1:zeroindex-1
			∂pY𝑑_∂ψ[i,j] = pY[i,j]*∂p𝑑_ξ⁻_∂ψ
		end
		∂pY𝑑_∂ψ[zeroindex,j] = 0.0
		for i = zeroindex+1:Ξ
			∂pY𝑑_∂ψ[i,j] = pY[i,j]*∂p𝑑_ξ⁺_∂ψ
		end
	end
end

"""
	differentiate_p𝑑_wrt_ψ(𝑑,K,Ξ)

Derivative of the conditional likelihood of the choice with respect to ψ

ARGUMENT
-`𝑑`: choice: left(false) or right(true)
-`K`: number of coupling states
-`Ξ`: number of accumulator states

RETURN
-`dp𝑑_dψ`: a matrix whose element `dp𝑑_dψ[i,j]` represents ∂p(𝑑 ∣ a=i, c=j)/∂ψ
"""
function differentiate_p𝑑_wrt_ψ(𝑑::Bool, K::Integer, Ξ::Integer)
	dp𝑑_dψ = zeros(Ξ,K)
	if 𝑑
		∂p𝑑_ξ⁻_∂ψ = 0.5
		∂p𝑑_ξ⁺_∂ψ = -0.5
	else
		∂p𝑑_ξ⁻_∂ψ = -0.5
		∂p𝑑_ξ⁺_∂ψ = 0.5
	end
	zeroindex = cld(Ξ,2)
	for j = 1:K
		for i = 1:zeroindex-1
			dp𝑑_dψ[i,j] = ∂p𝑑_ξ⁻_∂ψ
		end
		for i = zeroindex+1:Ξ
			dp𝑑_dψ[i,j] = ∂p𝑑_ξ⁺_∂ψ
		end
	end
	return dp𝑑_dψ
end

"""
	conditionallikelihood!(p𝑑, 𝑑, ψ)

conditional likelihood of the choice

ARGUMENT
-`P`: a matrix whose element `P[i,j]` corresponds to the i-th accumulator state and j-th coupling state
-`𝑑`: left (false) or right (true) choice of the animal
-`ψ`: lapse rate

MODIFIED ARGUMENT
-`p𝑑`: p{𝑑 ∣ a(T)=ξ(i), c(T)=j}
"""
function conditionallikelihood!(p𝑑::Vector{<:Real}, 𝑑::Bool, ψ::Real)
	if 𝑑
		p𝑑_ξ⁻ = ψ/2
		p𝑑_ξ⁺ = 1-ψ/2
	else
		p𝑑_ξ⁻ = 1-ψ/2
		p𝑑_ξ⁺ = ψ/2
	end
	Ξ = length(p𝑑)
	zeroindex = cld(Ξ,2)
	for i = 1:zeroindex-1
		p𝑑[i] = p𝑑_ξ⁻
	end
	p𝑑[zeroindex] = 0.5
	for i = zeroindex+1:Ξ
		p𝑑[i] = p𝑑_ξ⁺
	end
	return nothing
end

"""
	native2real!(∇ℓ, ∇∇ℓ, latentθindex, model)

Convert the gradient and hessian from being with respect to the parameters in native space to parameters in real space

ARGUMENT
-`∇ℓ`: gradient of the log-likelihood with respect to all parameters in native space
-`∇∇ℓ`: Hessian matrix of the log-likelihood with respect to all parameters in native space
-`latentθindex`: index of each latent parameter in the gradient and Hessian
-`model`: a structure containing the data, parameters, and hyperparameters of an FHMDDM

MODIFIED ARGUMENT
-`∇ℓ`: gradient of the log-likelihood with respect to all parameters in real space
-`∇∇ℓ`: Hessian matrix of the log-likelihood with respect to all parameters in real space
"""
function native2real!(∇ℓ::Vector{<:Real}, ∇∇ℓ::Matrix{<:Real}, latentθindex::Latentθ, model::Model)
	firstderivatives = differentiate_native_wrt_real(model)
	secondderivatives = differentiate_twice_native_wrt_real(model)
	for parametername in fieldnames(Latentθ)
		d1 = getfield(firstderivatives, parametername)[1]
		d2 = getfield(secondderivatives, parametername)[1]
		if d1 != 1.0
			i = getfield(latentθindex, parametername)[1]
			∇∇ℓ[i,:] .*= d1
			∇∇ℓ[:,i] .*= d1
			∇∇ℓ[i,i] += d2*∇ℓ[i]
			∇ℓ[i] *= d1
		end
	end
	return nothing
end

"""
	native2real!(∇ℓ, latentθindex, model)

Convert the gradient from being with respect to the parameters in native space to parameters in real space

ARGUMENT
-`∇ℓ`: gradient of the log-likelihood with respect to all parameters in native space
-`latentθindex`: index of each latent parameter in the gradient and Hessian
-`model`: a structure containing the data, parameters, and hyperparameters of an FHMDDM

MODIFIED ARGUMENT
-`∇ℓ`: gradient of the log-likelihood with respect to all parameters in real space
"""
function native2real!(∇ℓ::Vector{<:Real}, latentθindex::Latentθ, model::Model)
	firstderivatives = differentiate_native_wrt_real(model)
	for parametername in fieldnames(Latentθ)
		d1 = getfield(firstderivatives, parametername)[1]
		if d1 != 1.0
			i = getfield(latentθindex, parametername)[1]
			∇ℓ[i] *= d1
		end
	end
	return nothing
end

"""
	differentiate_native_wrt_real(model)

Derivative of each latent-variable-related parameter in its native space with respect to its value in real space

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of an FHMDDM

RETURN
-`derivatives`: an instance of `Latentθ` containing the derivative of each parameter in its native space with respect to its value in real space
"""
function differentiate_native_wrt_real(model::Model)
	@unpack options, θreal, θnative = model
	tmpAᶜ₁₁ = logistic(θreal.Aᶜ₁₁[1] + logit(options.q_Aᶜ₁₁))
	tmpAᶜ₂₂ = logistic(θreal.Aᶜ₂₂[1] + logit(options.q_Aᶜ₂₂))
	tmpπᶜ₁ 	= logistic(θreal.πᶜ₁[1] + logit(options.q_πᶜ₁))
	tmpψ 	= logistic(θreal.ψ[1] + logit(options.q_ψ))
	f_bound_z = 1.0-2.0*options.bound_z
	f_bound_ψ = 1.0-2.0*options.bound_ψ
	d = Latentθ()
	d.Aᶜ₁₁[1] = f_bound_z*tmpAᶜ₁₁*(1.0 - tmpAᶜ₁₁)
	d.Aᶜ₂₂[1] = f_bound_z*tmpAᶜ₂₂*(1.0 - tmpAᶜ₂₂)
	d.B[1] = θnative.B[1]*logistic(-θreal.B[1])
	d.k[1] = θnative.k[1]
	d.λ[1] = 1.0
	d.μ₀[1] = 1.0
	d.ϕ[1] = θnative.ϕ[1]*(1.0 - θnative.ϕ[1])
	d.πᶜ₁[1] = f_bound_z*tmpπᶜ₁*(1.0 - tmpπᶜ₁)
	d.ψ[1] = f_bound_ψ*tmpψ*(1.0 - tmpψ)
	d.σ²ₐ[1] = options.q_σ²ₐ*exp(θreal.σ²ₐ[1])
	d.σ²ᵢ[1] = options.q_σ²ᵢ*exp(θreal.σ²ᵢ[1])
	d.σ²ₛ[1] = options.q_σ²ₛ*exp(θreal.σ²ₛ[1])
	d.wₕ[1] = 1.0
	return d
end

"""
	differentiate_twice_native_wrt_real(model)

Second derivative of each latent-variable-related parameter in its native space with respect to its value in real space

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of an FHMDDM

RETURN
-`derivatives`: an instance of `Latentθ` containing the derivative of each parameter in its native space with respect to its value in real space
"""
function differentiate_twice_native_wrt_real(model::Model)
	@unpack options, θreal, θnative = model
	tmpAᶜ₁₁ = logistic(θreal.Aᶜ₁₁[1] + logit(options.q_Aᶜ₁₁))
	tmpAᶜ₂₂ = logistic(θreal.Aᶜ₂₂[1] + logit(options.q_Aᶜ₂₂))
	tmpπᶜ₁ 	= logistic(θreal.πᶜ₁[1] + logit(options.q_πᶜ₁))
	tmpψ 	= logistic(θreal.ψ[1] + logit(options.q_ψ))
	f_bound_z = 1.0-2.0*options.bound_z
	f_bound_ψ = 1.0-2.0*options.bound_ψ
	d = Latentθ()
	d.Aᶜ₁₁[1] = f_bound_z*(tmpAᶜ₁₁*(1-tmpAᶜ₁₁)^2 - tmpAᶜ₁₁^2*(1-tmpAᶜ₁₁))
	d.Aᶜ₂₂[1] = f_bound_z*(tmpAᶜ₂₂*(1-tmpAᶜ₂₂)^2 - tmpAᶜ₂₂^2*(1-tmpAᶜ₂₂))
	fB = logistic(θreal.B[1])
	d.B[1] = 2options.q_B*(fB*(1-fB)^2 - fB^2*(1-fB))
	d.k[1] = θnative.k[1]
	d.λ[1] = 0.0
	d.μ₀[1] = 0.0
	d.ϕ[1] = θnative.ϕ[1]*(1.0 - θnative.ϕ[1])^2 - θnative.ϕ[1]^2*(1.0 - θnative.ϕ[1])
	d.πᶜ₁[1] = f_bound_z*(tmpπᶜ₁*(1-tmpπᶜ₁)^2 - tmpπᶜ₁^2*(1-tmpπᶜ₁))
	d.ψ[1] = f_bound_ψ*(tmpψ*(1-tmpψ)^2 - tmpψ^2*(1-tmpψ))
	d.σ²ₐ[1] = options.q_σ²ₐ*exp(θreal.σ²ₐ[1])
	d.σ²ᵢ[1] = options.q_σ²ᵢ*exp(θreal.σ²ᵢ[1])
	d.σ²ₛ[1] = options.q_σ²ₛ*exp(θreal.σ²ₛ[1])
	d.wₕ[1] = 0.0
	return d
end

"""
	Sameacrosstrials(model)

Make a structure containing quantities that are used in each trial

ARGUMENT
-`model`: data, parameters, and hyperparameters of the factorial hidden-Markov drift-diffusion model

RETURN
-a structure containing quantities that are used in each trial

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true)
julia> S = FHMDDM.Sameacrosstrials(model)
```
"""
function Sameacrosstrials(model::Model)
	@unpack options, θnative, θreal, trialsets = model
	@unpack Δt, K, Ξ = options
	Aᶜ₁₁ = θnative.Aᶜ₁₁[1]
	Aᶜ₂₂ = θnative.Aᶜ₂₂[1]
	πᶜ₁ = θnative.πᶜ₁[1]
	if K == 2
		Aᶜ = [Aᶜ₁₁ 1-Aᶜ₂₂; 1-Aᶜ₁₁ Aᶜ₂₂]
		∇Aᶜ = [[1.0 0.0; -1.0 0.0], [0.0 -1.0; 0.0 1.0]]
		πᶜ = [πᶜ₁, 1-πᶜ₁]
		∇πᶜ = [[1.0, -1.0]]
	else
		Aᶜ = ones(1,1)
		∇Aᶜ = [zeros(1,1), zeros(1,1)]
		πᶜ = ones(1)
		∇πᶜ = [zeros(1)]
	end
	indexθ_pa₁ = [3,6,11,13]
	indexθ_paₜaₜ₋₁ = [3,4,5,7,10,12]
	indexθ_pc₁ = [8]
	indexθ_pcₜcₜ₋₁ = [1,2]
	indexθ_ψ = [9]
	counter = 13
	indexθ_py = map(trialset->map(mpGLM->zeros(Int,length(mpGLM.θ.𝐮)+length(mpGLM.θ.𝐯)), trialset.mpGLMs), trialsets)
	for s in eachindex(indexθ_py)
		for n in eachindex(indexθ_py[s])
			for q in eachindex(indexθ_py[s][n])
				counter += 1
				indexθ_py[s][n][q] = counter
			end
		end
	end
	indexθ_pY = map(x->vcat(x...), indexθ_py)
	nθ_trialset = indexθ_pY[end][end]
	index_pa₁_in_θ, index_paₜaₜ₋₁_in_θ, index_pc₁_in_θ, index_pcₜcₜ₋₁_in_θ, index_ψ_in_θ = zeros(Int, nθ_trialset), zeros(Int, nθ_trialset), zeros(Int, nθ_trialset), zeros(Int, nθ_trialset), zeros(Int, nθ_trialset)
	index_pa₁_in_θ[indexθ_pa₁] .= 1:length(indexθ_pa₁)
	index_paₜaₜ₋₁_in_θ[indexθ_paₜaₜ₋₁] .= 1:length(indexθ_paₜaₜ₋₁)
	index_pc₁_in_θ[indexθ_pc₁] .= 1:length(indexθ_pc₁)
	index_pcₜcₜ₋₁_in_θ[indexθ_pcₜcₜ₋₁] .= 1:length(indexθ_pcₜcₜ₋₁)
	index_ψ_in_θ[indexθ_ψ] .= 1:length(indexθ_ψ)
	index_pY_in_θ = map(x->zeros(Int, nθ_trialset), indexθ_pY)
	for i = 1:length(index_pY_in_θ)
		index_pY_in_θ[i][indexθ_pY[i]] = 1:length(indexθ_pY[i])
	end
	nθ_paₜaₜ₋₁ = length(indexθ_paₜaₜ₋₁)
	P = Probabilityvector(Δt, θnative, Ξ)
	update_for_∇∇transition_probabilities!(P)
	∇∇Aᵃsilent = map(i->zeros(Ξ,Ξ), CartesianIndices((nθ_paₜaₜ₋₁,nθ_paₜaₜ₋₁)))
	∇Aᵃsilent = map(i->zeros(Ξ,Ξ), 1:nθ_paₜaₜ₋₁)
	Aᵃsilent = zeros(typeof(θnative.B[1]), Ξ, Ξ)
	Aᵃsilent[1,1] = Aᵃsilent[Ξ, Ξ] = 1.0
	∇∇transitionmatrix!(∇∇Aᵃsilent, ∇Aᵃsilent, Aᵃsilent, P)
	Sameacrosstrials(Aᵃsilent=Aᵃsilent,
					∇Aᵃsilent=∇Aᵃsilent,
					∇∇Aᵃsilent=∇∇Aᵃsilent,
					Aᶜ=Aᶜ,
					∇Aᶜ=∇Aᶜ,
					Δt=options.Δt,
					indexθ_pa₁=indexθ_pa₁,
					indexθ_paₜaₜ₋₁=indexθ_paₜaₜ₋₁,
					indexθ_pc₁=indexθ_pc₁,
					indexθ_pcₜcₜ₋₁=indexθ_pcₜcₜ₋₁,
					indexθ_ψ=indexθ_ψ,
					indexθ_py=indexθ_py,
					indexθ_pY=indexθ_pY,
					K=K,
					πᶜ=πᶜ,
					∇πᶜ=∇πᶜ,
					Ξ=Ξ)
end
