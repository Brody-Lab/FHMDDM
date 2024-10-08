"""
    ∇∇loglikelihood(model)

Hessian of the log-likelihood of the data

ARGUMENT
-`model`: a structure containing the parameters, data, and hyperparameters of a factorial hidden Markov drift-diffusion model

RETURN
-`ℓ`: log-likelihood, as a float
-`∇ℓ`: gradient of the log-likelihood with respect to the fitted parameters in real space
-`∇∇ℓ`: Hessian matrix of the log-likelihood with respect to the fitted parameters in real space
"""
function ∇∇loglikelihood(model::Model)
	ℓ, ∇ℓ, ∇∇ℓ = twopasshessian(model)
	native2real!(∇ℓ, ∇∇ℓ, model)
	if scale_factor_choiceLL(model) != 1.0
		∇∇scalechoiceLL!(ℓ, ∇ℓ, ∇∇ℓ, model)
	end
	indexθ = indexparameters(model)
	concatenatedindices = concatenate(indexθ; includeunfit=true)
	isfitted = concatenatedindices .> 0
	if !all(isfitted)
		∇ℓ = ∇ℓ[isfitted]
		∇∇ℓ = ∇∇ℓ[isfitted, isfitted]
	end
	return ℓ[1], ∇ℓ, ∇∇ℓ
end

"""
	twopasshessian(model)

Compute the hessian as the Jacobian of the expectation conjugate gradient

ARGUMENT
-`model`: a structure containing the parameters, data, and hyperparameters of a factorial hidden Markov drift-diffusion model

RETURN
-`ℓ`: log-likelihood, as an one element vector
-`∇ℓ`: gradient of the log-likelihood with respect to parameters in native space. Parameters not being fitted have not been sorted out.
-`∇∇ℓ`: Hessian matrix of the log-likelihood with respect to parameters in native space. Parameters not being fitted have not been sorted out.
```
"""
function twopasshessian(model::Model)
	sameacrosstrials = Sameacrosstrials(model)
	memoryforhessian = Memoryforhessian(model, sameacrosstrials)
	@inbounds for trialset in model.trialsets
		for trial in trialset.trials
			twopasshessian!(memoryforhessian, model, sameacrosstrials, trial)
		end
	end
	@unpack ℓ, ∇ℓ, ∇∇ℓ = memoryforhessian
	@inbounds for i = 1:size(∇∇ℓ,1)
		for j = i+1:size(∇∇ℓ,2)
			∇∇ℓ[j,i] = ∇∇ℓ[i,j]
		end
	end
	return ℓ, ∇ℓ, ∇∇ℓ
end

"""
	twopasshessian!(memoryforhessian, model, sameacrosstrials, trial)

Compute the hessian for one trial as the Jacobian of the expectation conjugate gradient

MODIFIED ARGUMENT
-`memoryforhessian`: structure containing intermediate quantities that are modified for each trial

UNMODIFIED ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters
-`sameacrosstrials`: structure containing intermediate quantities that are fixed across trials
-`trial`: structure containing the data of the trial being used for computation
"""
function twopasshessian!(memoryforhessian::Memoryforhessian, model::Model, sameacrosstrials::Sameacrosstrials, trial::Trial)
	@unpack θnative = model
	@unpack K, Ξ, sf_y = model.options
	@unpack clicks, trialsetindex = trial
	@unpack mpGLMs = model.trialsets[trialsetindex]
	@unpack ℓ, ∇ℓ, ∇∇ℓ, f, ∇f, D, ∇D, ∇b = memoryforhessian
	@unpack P, ∇pa₁, ∇∇pa₁, Aᵃinput, ∇Aᵃinput, ∇∇Aᵃinput = memoryforhessian
	@unpack ∇logpy, ∇∇logpy, pY, ∇pY, ∂pY𝑑_∂ψ = memoryforhessian
	@unpack Aᵃsilent, ∇Aᵃsilent, ∇∇Aᵃsilent = sameacrosstrials
	@unpack Aᶜ, Aᶜᵀ, ∇Aᶜ, ∇Aᶜᵀ, πᶜ, πᶜᵀ, ∇πᶜ, ∇πᶜᵀ = sameacrosstrials
	@unpack indexθ_pa₁, indexθ_paₜaₜ₋₁, indexθ_paₜaₜ₋₁only, indexθ_pc₁, indexθ_pcₜcₜ₋₁, indexθ_ψ = sameacrosstrials
	@unpack nθ_pa₁, nθ_paₜaₜ₋₁, nθ_pc₁, nθ_pcₜcₜ₋₁, nθ_ψ = sameacrosstrials
	@unpack index_pa₁_in_θ, index_paₜaₜ₋₁_in_θ, index_pc₁_in_θ, index_pcₜcₜ₋₁_in_θ, index_ψ_in_θ = sameacrosstrials
	indexθ_py = sameacrosstrials.indexθ_py[trialsetindex]
	nθ_py = sameacrosstrials.nθ_py[trialsetindex]
	indexθ_pY = sameacrosstrials.indexθ_pY[trialsetindex]
	nθ_pY = sameacrosstrials.nθ_pY[trialsetindex]
	index_pY_in_θ = sameacrosstrials.index_pY_in_θ[trialsetindex]
	indexθ_trialset = sameacrosstrials.indexθ_trialset[trialsetindex]
	nθ_trialset = sameacrosstrials.nθ_trialset[trialsetindex]
	adaptedclicks = FHMDDM.∇∇adapt(clicks, θnative.k[1], θnative.ϕ[1])
	spikcountderivatives!(memoryforhessian, mpGLMs, sameacrosstrials, trial)
	update_emissions!(∂pY𝑑_∂ψ, pY[trial.ntimesteps], ∇pY[trial.ntimesteps], trial.choice, θnative.ψ[1])
	@inbounds for q in eachindex(∇f[1])
		∇f[1][q] .= 0
	end
	∇∇priorprobability!(∇∇pa₁, ∇pa₁, P, trial.previousanswer)
	pa₁ = copy(P.𝛑)
	pY₁⨀pc₁ = pY[1] .* πᶜᵀ
	pa₁⨀pc₁ = pa₁ .* πᶜᵀ
	@inbounds for i = 1:nθ_pY
		for j=1:Ξ
			for k = 1:K
				∇f[1][indexθ_pY[i]][j,k] = ∇pY[1][i][j,k]*pa₁⨀pc₁[j,k]
			end
		end
	end
	@inbounds for i = 1:nθ_pa₁
		for j=1:Ξ
			for k = 1:K
				∇f[1][indexθ_pa₁[i]][j,k] = pY₁⨀pc₁[j,k]*∇pa₁[i][j]
			end
		end
	end
	@inbounds for i = 1:nθ_pc₁
		for j=1:Ξ
			for k = 1:K
				∇f[1][indexθ_pc₁[1]][j,k] = pY[1][j,k]*pa₁[j]*∇πᶜ[i][k]
			end
		end
	end
	@inbounds for j=1:Ξ
		for k = 1:K
			f[1][j,k] = pY[1][j,k] * pa₁⨀pc₁[j,k]
		end
	end
	D[1] = sum(f[1])
	forward!(∇D[1], f[1], ∇f[1], ℓ, D[1])
	@inbounds for t=2:trial.ntimesteps
		if t ∈ clicks.inputtimesteps
			update_for_∇∇transition_probabilities!(P, adaptedclicks, clicks, t)
			clickindex = clicks.inputindex[t][1]
			Aᵃ = Aᵃinput[clickindex]
			∇Aᵃ = ∇Aᵃinput[clickindex]
			∇∇Aᵃ = ∇∇Aᵃinput[clickindex]
			∇∇transitionmatrix!(∇∇Aᵃ, ∇Aᵃ, Aᵃ, P)
		else
			Aᵃ = Aᵃsilent
			∇Aᵃ = ∇Aᵃsilent
			∇∇Aᵃ = ∇∇Aᵃsilent
		end
		f⨉Aᶜᵀ = f[t-1] * Aᶜᵀ
		Aᵃ⨉f⨉Aᶜᵀ = Aᵃ * f⨉Aᶜᵀ
		Aᵃ⨉f = Aᵃ * f[t-1]
		for q in indexθ_trialset
			i_aₜ = index_paₜaₜ₋₁_in_θ[q]
			i_cₜ = index_pcₜcₜ₋₁_in_θ[q]
			i_y = index_pY_in_θ[q]
			i_ψ = index_ψ_in_θ[q]
			if i_aₜ > 0
				∇f[t][q] = pY[t] .* (∇Aᵃ[i_aₜ] * f⨉Aᶜᵀ .+ Aᵃ * ∇f[t-1][q] * Aᶜᵀ)
			elseif i_cₜ > 0
				∇f[t][q] = pY[t] .* (Aᵃ⨉f * ∇Aᶜᵀ[i_cₜ] .+ Aᵃ * ∇f[t-1][q] * Aᶜᵀ)
			elseif i_y > 0
				∇f[t][q] = ∇pY[t][i_y] .* Aᵃ⨉f⨉Aᶜᵀ .+ pY[t] .* (Aᵃ * ∇f[t-1][q] * Aᶜᵀ)
			elseif i_ψ > 0
				if t==trial.ntimesteps
					∇f[t][q] = ∂pY𝑑_∂ψ .* Aᵃ⨉f⨉Aᶜᵀ # questionable that it does not have the `pY[t] .* (Aᵃ * ∇f[t-1][q] * Aᶜᵀ)` term
				else
					∇f[t][q] .= 0.0
				end
			else
				∇f[t][q] = pY[t] .* (Aᵃ * ∇f[t-1][q] * Aᶜᵀ)
			end
		end
		f[t] = pY[t] .* Aᵃ⨉f⨉Aᶜᵀ
		D[t] = sum(f[t])
		forward!(∇D[t], f[t], ∇f[t], ℓ, D[t])
	end
	ℓ[1] -= length(mpGLMs)*trial.ntimesteps*log(sf_y)
	bₜ = ones(Ξ,K)
	@inbounds for t = trial.ntimesteps:-1:1
		γ = f[t] # resuse memory
		∇γ = ∇f[trial.ntimesteps] # resuse memory
		if t == trial.ntimesteps
			for q in indexθ_trialset
				∇b[q] .= 0
			end
			# the p(𝑑 ∣ aₜ, cₜ) term
			q = indexθ_ψ[1]
			∇ℓ[q] += expectation_derivative_logp𝑑_wrt_ψ(trial.choice, γ, θnative.ψ[1])
			∇∇ℓ[q,q] += expectation_second_derivative_logp𝑑_wrt_ψ(trial.choice, γ, θnative.ψ[1])
			for r in indexθ_trialset
				if r < q
					∇∇ℓ[r,q] += expectation_derivative_logp𝑑_wrt_ψ(trial.choice, ∇γ[r], θnative.ψ[1])
				else
					∇∇ℓ[q,r] += expectation_derivative_logp𝑑_wrt_ψ(trial.choice, ∇γ[r], θnative.ψ[1])
				end
			end
		else
			if t+1 ∈ clicks.inputtimesteps
				clickindex = clicks.inputindex[t+1][1]
				Aᵃₜ₊₁ = Aᵃinput[clickindex]
				∇Aᵃₜ₊₁ = ∇Aᵃinput[clickindex]
			else
				Aᵃₜ₊₁ = Aᵃsilent
				∇Aᵃₜ₊₁ = ∇Aᵃsilent
			end
			Aᵃₜ₊₁ᵀ = transpose(Aᵃₜ₊₁)
			bₜ₊₁ = bₜ # rename
			bₜ₊₁⨀pYₜ₊₁ = bₜ₊₁ .* pY[t+1]
			bₜ = transpose(Aᵃₜ₊₁) * (bₜ₊₁⨀pYₜ₊₁./D[t+1]) * Aᶜ
			for q in indexθ_trialset
				i_aₜ = index_paₜaₜ₋₁_in_θ[q]
				i_cₜ = index_pcₜcₜ₋₁_in_θ[q]
				i_y = index_pY_in_θ[q]
				i_ψ = index_ψ_in_θ[q]
				if i_aₜ > 0
					dAᵃₜ₊₁ᵀ_dθ = transpose(∇Aᵃₜ₊₁[i_aₜ])
					∇b[q] = ((dAᵃₜ₊₁ᵀ_dθ*bₜ₊₁⨀pYₜ₊₁ .+ Aᵃₜ₊₁ᵀ*(∇b[q].*pY[t+1])) * Aᶜ .- bₜ.*∇D[t+1][q])./D[t+1]
				elseif i_cₜ > 0
					∇b[q] = (Aᵃₜ₊₁ᵀ * (bₜ₊₁⨀pYₜ₊₁*∇Aᶜ[i_cₜ] .+ (∇b[q].*pY[t+1]) * Aᶜ) .- bₜ.*∇D[t+1][q])./D[t+1]
				elseif i_y > 0
					∇b[q] = (Aᵃₜ₊₁ᵀ * (bₜ₊₁.*∇pY[t+1][i_y] .+ (∇b[q].*pY[t+1])) * Aᶜ .- bₜ.*∇D[t+1][q])./D[t+1]
				elseif i_ψ > 0
					if t == trial.ntimesteps-1
						∇b[q] = (Aᵃₜ₊₁ᵀ * (bₜ₊₁.*∂pY𝑑_∂ψ) * Aᶜ .- bₜ.*∇D[t+1][q])./D[t+1]
					end
				else
					∇b[q] = (Aᵃₜ₊₁ᵀ*(∇b[q].*pY[t+1])*Aᶜ .- bₜ.*∇D[t+1][q])./D[t+1]
				end
			end
			for q in indexθ_trialset
				for ij in eachindex(∇γ[q])
					∇γ[q][ij] = ∇f[t][q][ij]*bₜ[ij] + f[t][ij]*∇b[q][ij]
				end
			end
			γ .*= bₜ # modify γ only after modifying ∇γ because γ shares memory with f[t]
		end
		if t > 1
			if t ∈ clicks.inputtimesteps
				clickindex = clicks.inputindex[t][1]
				Aᵃ = Aᵃinput[clickindex]
				∇Aᵃ = ∇Aᵃinput[clickindex]
				∇∇Aᵃ = ∇∇Aᵃinput[clickindex]
			else
				Aᵃ = Aᵃsilent
				∇Aᵃ = ∇Aᵃsilent
				∇∇Aᵃ = ∇∇Aᵃsilent
			end
			# the p(aₜ ∣ aₜ₋₁) term
			for i = 1:nθ_paₜaₜ₋₁
				q = indexθ_paₜaₜ₋₁[i]
				η = sum_product_over_states(D[t], f[t-1], bₜ, pY[t], ∇Aᵃ[i], Aᶜ)
				∇ℓ[q] += η
				for j=i:nθ_paₜaₜ₋₁
					r = indexθ_paₜaₜ₋₁[j]
					∇∇ℓ[q,r] += sum_product_over_states(D[t], ∇f[t-1][r], bₜ, pY[t], ∇Aᵃ[i], Aᶜ)
					∇∇ℓ[q,r] += sum_product_over_states(D[t], f[t-1], ∇b[r], pY[t], ∇Aᵃ[i], Aᶜ)
					∇∇ℓ[q,r] -= η*∇D[t][r]/D[t]
					∇∇ℓ[q,r] += sum_product_over_states(D[t], f[t-1], bₜ, pY[t], ∇∇Aᵃ[i,j], Aᶜ)
				end
				for j = 1:nθ_pcₜcₜ₋₁
					r = indexθ_pcₜcₜ₋₁[j]
					indices = (r > q) ? CartesianIndex(q,r) : CartesianIndex(r,q)
					∇∇ℓ[indices] += sum_product_over_states(D[t], ∇f[t-1][r], bₜ, pY[t], ∇Aᵃ[i], Aᶜ)
					∇∇ℓ[indices] += sum_product_over_states(D[t], f[t-1], ∇b[r], pY[t], ∇Aᵃ[i], Aᶜ)
					∇∇ℓ[indices] -= η*∇D[t][r]/D[t]
					∇∇ℓ[indices] += sum_product_over_states(D[t], f[t-1], bₜ, pY[t], ∇Aᵃ[i], ∇Aᶜ[j])
				end
			end
			# the p(cₜ ∣ cₜ₋₁) term
			for i = 1:nθ_pcₜcₜ₋₁
				q = indexθ_pcₜcₜ₋₁[i]
				η = sum_product_over_states(D[t], f[t-1], bₜ, pY[t], Aᵃ, ∇Aᶜ[i])
				∇ℓ[q] += η
				for j = i:nθ_pcₜcₜ₋₁
					r = indexθ_pcₜcₜ₋₁[j]
					∇∇ℓ[q,r] += sum_product_over_states(D[t], ∇f[t-1][r], bₜ, pY[t], Aᵃ, ∇Aᶜ[i])
					∇∇ℓ[q,r] += sum_product_over_states(D[t], f[t-1], ∇b[r], pY[t], Aᵃ, ∇Aᶜ[i])
					∇∇ℓ[q,r] -= η*∇D[t][r]/D[t]
				end
			end
		end
		# the p(yₙₜ ∣ aₜ, cₜ) term
		for n = 1:length(indexθ_py)
			for i = 1:nθ_py[n]
				q = indexθ_py[n][i]
				∇ℓ[q] += dot(γ, ∇logpy[t][n][i])
				for j = i:nθ_py[n]
					r = indexθ_py[n][j]
					∇∇ℓ[q,r] += dot(∇γ[r], ∇logpy[t][n][i]) + dot(γ, ∇∇logpy[t][n][i,j])
				end
				for m = n+1:length(indexθ_py)
					for j = 1:nθ_py[m]
						r = indexθ_py[m][j]
						∇∇ℓ[q,r] += dot(∇γ[r], ∇logpy[t][n][i])
					end
				end
				for r in indexθ_paₜaₜ₋₁
					if r > q
						∇∇ℓ[q,r] += dot(∇γ[r], ∇logpy[t][n][i])
					else
						∇∇ℓ[r,q] += dot(∇γ[r], ∇logpy[t][n][i])
					end
				end
				for r in indexθ_pcₜcₜ₋₁
					if r > q
						∇∇ℓ[q,r] += dot(∇γ[r], ∇logpy[t][n][i])
					else
						∇∇ℓ[r,q] += dot(∇γ[r], ∇logpy[t][n][i])
					end
				end
			end
		end
	end
	#last backward step
	t = 1
	# the p(a₁) term
	@inbounds for i = nθ_pa₁:-1:1 # because B comes first
		q = indexθ_pa₁[i]
		η = sum_product_over_states(D[t], bₜ, pY[t], ∇pa₁[i], πᶜ)
		∇ℓ[q] += η
		for j = i:-1:1
			r = indexθ_pa₁[j]
			∇∇ℓ[r,q] += sum_product_over_states(D[t], ∇b[r], pY[t], ∇pa₁[i], πᶜ)
			∇∇ℓ[r,q] -= η*∇D[t][r]/D[t]
			∇∇ℓ[r,q] += sum_product_over_states(D[t], bₜ, pY[t], ∇∇pa₁[j,i], πᶜ)
		end
		for index in (indexθ_paₜaₜ₋₁only, indexθ_pcₜcₜ₋₁)
			for r in index
				indices = r < q ? CartesianIndex(r,q) : CartesianIndex(q,r)
				∇∇ℓ[indices] += sum_product_over_states(D[t], ∇b[r], pY[t], ∇pa₁[i], πᶜ)
				∇∇ℓ[indices] -= η*∇D[t][r]/D[t]
			end
		end
		if q ∉indexθ_paₜaₜ₋₁
			for r in indexθ_pY
				∇∇ℓ[q,r] += sum_product_over_states(D[t], ∇b[r], pY[t], ∇pa₁[i], πᶜ)
				∇∇ℓ[q,r] -= η*∇D[t][r]/D[t]
				j = index_pY_in_θ[r]
				∇∇ℓ[q,r] += sum_product_over_states(D[t], bₜ, ∇pY[t][j], ∇pa₁[i], πᶜ)
			end
		end
	end
	# the p(c₁) term
	@inbounds for i = 1:nθ_pc₁
		q = indexθ_pc₁[i]
		η = sum_product_over_states(D[t], bₜ, pY[t], pa₁, ∇πᶜ[i])
		∇ℓ[q] += η
		for j = i:nθ_pc₁
			r = indexθ_pc₁[j]
			∇∇ℓ[q,r] += sum_product_over_states(D[t], ∇b[r], pY[t], pa₁, ∇πᶜ[i])
			∇∇ℓ[q,r] -= η*∇D[t][r]/D[t]
		end
		for index in (indexθ_pY, indexθ_paₜaₜ₋₁only, indexθ_pcₜcₜ₋₁, indexθ_pa₁)
			for r in index
				indices = r < q ? CartesianIndex(r,q) : CartesianIndex(q,r)
				∇∇ℓ[indices] += sum_product_over_states(D[t], ∇b[r], pY[t], pa₁, ∇πᶜ[i])
				∇∇ℓ[indices] -= η*∇D[t][r]/D[t]
				j = index_pY_in_θ[r]
				if j > 0
					∇∇ℓ[indices] += sum_product_over_states(D[t], bₜ, ∇pY[t][j], pa₁, ∇πᶜ[i])
				end
				j = index_pa₁_in_θ[r]
				if j > 0
					∇∇ℓ[indices] += sum_product_over_states(D[t], bₜ, pY[t], ∇pa₁[j], ∇πᶜ[i])
				end
			end
		end
	end
	return nothing
end

"""
	forward!(D,∇D,f,∇f,ℓ,∇ℓ,t)

Normalize the forward term and its first-order partial derivatives and update the log-likelihood and its first-partial derivatives

PRE-MODIFIED ARGUMENT
-`f`: un-normalized forward term. Element `f[i,j]` corresponds to the un-normalized forward term in the i-th accumulator state and j-th coupling state
-`∇f`: first-order partial derivatives of the un-normalized forward term. Element `∇f[q][i,j]` corresponds to the derivative of the un-normalized forward term in the i-th accumulator state and j-th coupling state with respect to the q-th parameter.
-`ℓ`: log-likelihood for the data before the current time-step
-`∇ℓ`: gradient of the log-likelihood for the data before the current time-step

POST-MODIFICATION ARGUMENT
-`∇D`: gradient of the past-conditioned emissions likelihood for the current time step
-`∇f`: first-order partial derivatives of the normalized forward term. Element `∇f[q][i,j]` corresponds to `∂p{a(t)=ξ(i), c(t)=j ∣ 𝐘(1:t)} / ∂θ(q)`
-`f`: normalized forward term, which represents the joint conditional probability of the latent variables given the elapsed emissions. Element `f[i,j]` corresponds to `p{a(t)=ξ(i), c(t)=j ∣ 𝐘(1:t)}`
-`ℓ`: log-likelihood for the data before the up to the current time-step

UNMODIFIED ARGUMENT
-`D`: past-conditioned emissions likelihood: `p{𝐘(t) ∣ 𝐘(1:t-1))`

"""
function forward!(∇D::Vector{<:Real},
				f::Matrix{<:Real},
				∇f::Vector{<:Matrix{<:Real}},
				ℓ::Vector{<:Real},
				D::Real)
	f ./= D
	@inbounds for i in eachindex(∇D)
		∇D[i] = sum(∇f[i])
	end
	@inbounds for i in eachindex(∇f)
		for jk in eachindex(∇f[i])
			∇f[i][jk] = (∇f[i][jk] - f[jk]*∇D[i])/D
		end
	end
	ℓ[1] += log(D)
end

"""
	update_emissions!(∂pY𝑑_∂ψ, pY, ∇pY, choice, ψ)

Update the conditional likelihood of the emissions as well as its gradient

MODIFIED ARGUMENT
-`∂pY𝑑_∂ψ`: derivative of the conditional likelihood of the emissions at the last time step of a trial with respect to the lapse parameter ψ. Element `∂pY𝑑_∂ψ[i,j]` corresponds to the i-th accumulator state and j-th coupling state
-`pY`: conditional likelihood of the emissions at the last time step of the trial. Element `pY[i,j]` corresponds to the i-th accumulator state and j-th coupling state
-`∇pY`: gradient of the conditional likelihood of the emissions at the last time step of the trial. Element `∇pY[q][i,j]` corresponds to the q-th parameter among all GLMs, i-th accumulator state and j-th coupling state

UNMODIFIED ARGUMENT
-`choice`: a Bool indicating left (false) or right (true)
-`ψ`: lapse rate
"""
function update_emissions!(∂pY𝑑_∂ψ::Matrix{<:Real}, pY::Matrix{<:Real}, ∇pY::Vector{<:Matrix{<:Real}}, choice::Bool, ψ::Real)
	differentiate_pY𝑑_wrt_ψ!(∂pY𝑑_∂ψ, pY, choice)
	conditionallikelihood!(pY, choice, ψ)
	@inbounds for q in eachindex(∇pY)
		conditionallikelihood!(∇pY[q], choice, ψ)
	end
	return nothing
end

"""
	conditionallikelihood!(pY, 𝑑, ψ)

Multiply the conditional likelihood of the choice to the conditional likelihood of spiking

ARGUMENT
-`pY`: a matrix whose element `pY[i,j]` corresponds to the i-th accumulator state and j-th coupling state and represents p{Y ∣ a(T)=ξ(i), c(T)=j}
-`𝑑`: left (false) or right (true) choice of the animal
-`ψ`: lapse rate

MODIFIED ARGUMENT
-`pY`: p{Y, 𝑑 ∣ a(T)=ξ(i), c(T)=j}
"""
function conditionallikelihood!(pY::Matrix{<:Real}, 𝑑::Bool, ψ::Real)
	if 𝑑
		p𝑑_ξ⁻ = ψ/2
		p𝑑_ξ⁺ = 1-ψ/2
	else
		p𝑑_ξ⁻ = 1-ψ/2
		p𝑑_ξ⁺ = ψ/2
	end
	Ξ,K = size(pY)
	zeroindex = cld(Ξ,2)
	@inbounds for j = 1:K
		for i = 1:zeroindex-1
			pY[i,j] *= p𝑑_ξ⁻
		end
		pY[zeroindex,j] *= 0.5
		for i = zeroindex+1:Ξ
			pY[i,j] *= p𝑑_ξ⁺
		end
	end
	return nothing
end

"""
	sum_product_over_states(D,fₜ₋₁,bₜ,Y,A,C)

Multiply terms across different states of the latent variables at consecutive time step and sum

ARGUMENT
-`Y`: similar to η, element Y[i,j] corresponds to i-th state of the accumulator at time t and the j-th state of the coupling at time t
-`A`: element A[i,j] corresponds to i-th state of the accumulator at time t and the j-th state of the accumulator at time t-1
-`C`: element C[i,j] corresponds to i-th state of the coupling at time t and the j-th state of the coupling at time t-1

RETURN
-`s`: sum of the product across all states of the two latent variables at two consecutive time steps
"""
function sum_product_over_states(D::Real, fₜ₋₁::Matrix{<:Real}, bₜ::Matrix{<:Real}, Y::Matrix{<:Real}, A::Matrix{<:Real}, C::Matrix{<:Real})
	s = 0.0
	Ξ,K = size(fₜ₋₁)
	@inbounds for iaₜ = 1:Ξ
		for icₜ = 1:K
			for iaₜ₋₁ = 1:Ξ
				for icₜ₋₁ = 1:K
					s += fₜ₋₁[iaₜ₋₁,icₜ₋₁]*bₜ[iaₜ,icₜ]*Y[iaₜ,icₜ]*A[iaₜ,iaₜ₋₁]*C[icₜ, icₜ₋₁]
				end
			end
		end
	end
	return s/D
end

"""
	sum_product_over_states(D, b,Y,A,C)

Multiply terms across different states of the latent variables at a single time step and sum

ARGUMENT
-`Y`: similar to η, element Y[i,j] corresponds to i-th state of the accumulator at time t and the j-th state of the coupling at time t
-`A`: element A[i] corresponds to the i-th state of the accumulator at time t
-`C`: element C[j] corresponds to the j-th state of the coupling at time t-1

RETURN
-`s`: sum of the product across all states of the two latent variables at a single time step
"""
function sum_product_over_states(D::Real, b::Matrix{<:Real}, Y::Matrix{<:Real}, A::Vector{<:Real}, C::Vector{<:Real})
	s = 0.0
	Ξ,K = size(b)
	@inbounds for iaₜ = 1:Ξ
		for icₜ = 1:K
			s += b[iaₜ,icₜ]*Y[iaₜ,icₜ]*A[iaₜ]*C[icₜ]
		end
	end
	return s/D
end

"""
	expectation_derivative_logp𝑑_wrt_ψ(choice, γ, ψ)

Expectation of the derivative of the choice log-likelihood with respect to the lapse rate

ARGUMENT
-`choice`: a Boolean specifying whether the choice was to the right
-`γ`: a matrix or vector of representing the posterior probability of the latent variables at the end of the trial (i.e., last time step). Element `γ_end[i,j]` = p(aᵢ=1, cⱼ=1 ∣ 𝐘, d). Rows correspond to states of the accumulator state variable 𝐚, and columns to states of the coupling variable 𝐜.
-`ψ`: a floating-point number specifying the lapse rate

RETURN
-the expectation under the posterior probability of the late variables
"""
function expectation_derivative_logp𝑑_wrt_ψ(choice::Bool, γ::Array{<:Real}, ψ::Real)
	if choice
		dlogp𝑑_dψ_ξ⁻ = 1/ψ
		dlogp𝑑_dψ_ξ⁺ = 1/(ψ-2)
	else
		dlogp𝑑_dψ_ξ⁻ = 1/(ψ-2)
		dlogp𝑑_dψ_ξ⁺ = 1/ψ
	end
	Ξ = size(γ,1)
	zeroindex = cld(Ξ,2)
	edll = 0.0 # expectation of the derivative of the log-likelihood
	@inbounds for j = 1:size(γ,2)
		for i = 1:zeroindex-1
			edll += γ[i,j]*dlogp𝑑_dψ_ξ⁻
		end
		for i = zeroindex+1:Ξ
			edll += γ[i,j]*dlogp𝑑_dψ_ξ⁺
		end
	end
	return edll
end

"""
	expectation_second_derivative_logp𝑑_wrt_ψ(choice, γ, ψ)

Expectation of the second derivative of the choice log-likelihood with respect to the lapse rate

ARGUMENT
-`choice`: a Boolean specifying whether the choice was to the right
-`γ`: a matrix or vector of representing the posterior probability of the latent variables at the end of the trial (i.e., last time step). Element `γ_end[i,j]` = p(aᵢ=1, cⱼ=1 ∣ 𝐘, d). Rows correspond to states of the accumulator state variable 𝐚, and columns to states of the coupling variable 𝐜.
-`ψ`: a floating-point number specifying the lapse rate

RETURN
-the expectation under the posterior probability of the late variables
"""
function expectation_second_derivative_logp𝑑_wrt_ψ(choice::Bool, γ::Array{<:Real}, ψ::Real)
	if choice
		d²logp𝑑_dψdψ_ξ⁻ = -ψ^-2
		d²logp𝑑_dψdψ_ξ⁺ = -(ψ-2)^-2
	else
		d²logp𝑑_dψdψ_ξ⁻ = -(ψ-2)^-2
		d²logp𝑑_dψdψ_ξ⁺ = -ψ^-2
	end
	Ξ = size(γ,1)
	zeroindex = cld(Ξ,2)
	ed²ll = 0.0 # expectation of the second derivative of the log-likelihood
	@inbounds for j = 1:size(γ,2)
		for i = 1:zeroindex-1
			ed²ll += γ[i,j]*d²logp𝑑_dψdψ_ξ⁻
		end
		for i = zeroindex+1:Ξ
			ed²ll += γ[i,j]*d²logp𝑑_dψdψ_ξ⁺
		end
	end
	return ed²ll
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
	@unpack Δt, K, minpa, Ξ = options
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
	indexθ = indexparameters(model; includeunfit=true)
	indexθ_py = collect(collect(concatenateparameters(glmθ; includeunfit=true) for glmθ in glmθ) for glmθ in indexθ.glmθ)
	indexθ_pY = collect(vcat((glmθ for glmθ in glmθ)...) for glmθ in indexθ_py)
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
	P = Probabilityvector(Δt, minpa, θnative, Ξ)
	update_for_∇∇transition_probabilities!(P)
	∇∇Aᵃsilent = map(i->zeros(Ξ,Ξ), CartesianIndices((nθ_paₜaₜ₋₁,nθ_paₜaₜ₋₁)))
	∇Aᵃsilent = map(i->zeros(Ξ,Ξ), 1:nθ_paₜaₜ₋₁)
	Aᵃsilent = ones(typeof(θnative.B[1]), Ξ, Ξ).*minpa
	one_minus_Ξminpa = 1.0-Ξ*minpa
	Aᵃsilent[1,1] += one_minus_Ξminpa
	Aᵃsilent[Ξ, Ξ] += one_minus_Ξminpa
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

"""
	Memoryforhessian(model)
"""
function Memoryforhessian(model::Model, S::Sameacrosstrials)
	@unpack options, θnative, θreal, trialsets = model
	@unpack Δt, K, minpa, Ξ = options
	maxclicks = maximum_number_of_clicks(model)
	maxtimesteps = maximum_number_of_time_steps(model)
	maxneurons = maximum(map(trialset-> length(trialset.mpGLMs), trialsets))
	max_nθ_pY=maximum(S.nθ_pY)
	max_nθ_py=maximum(map(n->maximum(n), S.nθ_py))
	∇D=collect(zeros(S.nθ_alltrialsets) for t=1:maxtimesteps)
	f = collect(zeros(Ξ,K) for t=1:maxtimesteps)
	∇f = collect(collect(zeros(Ξ,K) for q=1:S.nθ_alltrialsets) for t=1:maxtimesteps)
	∇b = collect(zeros(Ξ,K) for q=1:S.nθ_alltrialsets)
	λ = collect(collect(zeros(Ξ,K) for t=1:maxtimesteps) for n = 1:maxneurons)
	𝐋 = linearpredictor(model.trialsets)
	∇logpy = collect(collect(collect(zeros(Ξ,K) for q=1:max_nθ_py) for n=1:maxneurons) for t=1:maxtimesteps)
	∇∇logpy=map(1:maxtimesteps) do t
				map(1:maxneurons) do n
					map(CartesianIndices((max_nθ_py,max_nθ_py))) do q
						zeros(Ξ,K)
					end
				end
			end
	one_minus_Ξminpa = 1.0-Ξ*minpa
	Aᵃinput=map(1:maxclicks) do t
				A = ones(Ξ,Ξ).*minpa
				A[1,1] += one_minus_Ξminpa
				A[Ξ,Ξ] += one_minus_Ξminpa
				return A
			end
	∇Aᵃinput = collect(collect(zeros(Ξ,Ξ) for q=1:S.nθ_paₜaₜ₋₁) for t=1:maxclicks)
	∇∇Aᵃinput = map(1:maxclicks) do t
					map(CartesianIndices((S.nθ_paₜaₜ₋₁,S.nθ_paₜaₜ₋₁))) do ij
						zeros(Ξ,Ξ)
					end
				end
	∇pa₁ = collect(zeros(Ξ) for q=1:S.nθ_pa₁)
	∇∇pa₁ = map(CartesianIndices((S.nθ_pa₁,S.nθ_pa₁))) do q
				zeros(Ξ)
			end
	pY = collect(zeros(Ξ,K) for t=1:maxtimesteps)
	∇pY = collect(collect(zeros(Ξ,K) for q=1:max_nθ_pY) for t=1:maxtimesteps)
	𝛚 = map(model.trialsets) do trialset
			map(trialset.mpGLMs) do mpGLM
				transformaccumulator(mpGLM)
			end
		end
	d𝛚_db = map(model.trialsets) do trialset
			map(trialset.mpGLMs) do mpGLM
				dtransformaccumulator(mpGLM)
			end
		end
	d²𝛚_db² = map(model.trialsets) do trialset
			map(trialset.mpGLMs) do mpGLM
				d²transformaccumulator(mpGLM)
			end
		end
	Memoryforhessian(Aᵃinput=Aᵃinput,
					∇Aᵃinput=∇Aᵃinput,
					∇∇Aᵃinput=∇∇Aᵃinput,
					f=f,
					∇f=∇f,
					∇b=∇b,
					D = zeros(maxtimesteps),
					∇D=∇D,
					glmderivatives=GLMDerivatives(model.trialsets[1].mpGLMs[1]),
					indexθglms = collect(collect(indexparameters(mpGLM.θ; includeunfit=true) for mpGLM in trialset.mpGLMs) for trialset in model.trialsets),
					∇ℓ=zeros(S.nθ_alltrialsets),
					∇∇ℓ=zeros(S.nθ_alltrialsets,S.nθ_alltrialsets),
					∂pY𝑑_∂ψ=zeros(Ξ,K),
					λ=λ,
					𝐋=𝐋,
					∇logpy=∇logpy,
					∇∇logpy=∇∇logpy,
					𝛂 = collect(collect(inverselink(mpGLM.θ.a[1]) for mpGLM in trialset.mpGLMs) for trialset in model.trialsets),
					𝛚 = 𝛚,
					d𝛚_db = d𝛚_db,
					d²𝛚_db² = d²𝛚_db²,
					P = Probabilityvector(Δt, minpa, θnative, Ξ),
					∇pa₁=∇pa₁,
					∇∇pa₁=∇∇pa₁,
					pY=pY,
					∇pY=∇pY)
end

"""
	∇∇scalechoiceLL!(ℓ, ∇ℓ, ∇∇ℓ, model)

Adjust the log-likelihood, its gradient, and hessian by the scaling factor of the log-likelihood of choices

MODIFIED ARGUMENT
-`ℓ`: log-likelihood of the data before scaling up the log-likelihood of the choices
-`∇ℓ`: gradient of the log-likelihood of the data before scaling up the log-likelihood of the choices. This is before sorting to remove parameters that are not being fitted.
-`∇∇ℓ`: hessian of the log-likelihood of the data before scaling up the log-likelihood of the choices. This is before sorting to remove parameters that are not being fitted.

UNMODIFIED ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters
"""
function ∇∇scalechoiceLL!(ℓ::Vector{<:AbstractFloat}, ∇ℓ::Vector{<:AbstractFloat}, ∇∇ℓ::Matrix{<:AbstractFloat}, model::Model)
	s = scale_factor_choiceLL(model)
	ℓˢ, ∇ℓˢ, ∇∇ℓˢ = ∇∇choiceLL(model)
	ℓ[1] += (s-1)*ℓˢ
	nθchoice = length(∇ℓˢ)
	indices = zeros(Int, nθchoice)
	i = 0
	j = 0
	for field in fieldnames(Latentθ)
		i += 1
		if field == :Aᶜ₁₁ || field == :Aᶜ₂₂ || field == :πᶜ₁
		else
			j += 1
			indices[j] = i
		end
	end
	for j = 1:nθchoice
		i = indices[j]
		∇ℓ[i] += (s-1)*∇ℓˢ[j]
		for n = 1:nθchoice
			m = indices[n]
			∇∇ℓ[i,m] += (s-1)*∇∇ℓˢ[j,n]
		end
	end
	return nothing
end
