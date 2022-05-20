"""
	check_twopasshessian(model)

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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_19_test/T176_2018_05_03/data.mat")
julia> absdiffℓ, absdiff∇, absdiff∇∇ = FHMDDM.check_twopasshessian(model)
```
"""
function check_twopasshessian(model::Model)
	concatenatedθ, indexθ = concatenateparameters(model)
	ℓhand, ∇hand, ∇∇hand = twopasshessian!(model,concatenatedθ,indexθ)
	f(x) = loglikelihood(x, indexθ, model)
	ℓauto = f(concatenatedθ)
	∇auto = ForwardDiff.gradient(f, concatenatedθ)
	∇∇auto = ForwardDiff.hessian(f, concatenatedθ)
	return abs(ℓauto-ℓhand), abs.(∇auto .- ∇hand), abs.(∇∇auto .- ∇∇hand)
end

"""
	twopasshessian!(model, concatenatedθ, indexθ)

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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"; randomize=true)
julia> concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
julia> ℓ, ∇ℓ, ∇∇ℓ = FHMDDM.twopasshessian!(model, concatenatedθ, indexθ)
```
"""
function twopasshessian!(model::Model, concatenatedθ::Vector{<:Real}, indexθ::Indexθ)
	sortparameters!(model, concatenatedθ, indexθ)
	ℓ, ∇ℓ, ∇∇ℓ = twopasshessian(model)
	native2real!(∇ℓ, ∇∇ℓ, model)
	∇ℓ = sortparameters(indexθ.latentθ, ∇ℓ)
	∇∇ℓ = sortparameters(indexθ.latentθ, ∇∇ℓ)
	return ℓ, ∇ℓ, ∇∇ℓ
end

"""
	twopasshessian(model)

Compute the hessian as the Jacobian of the expectation conjugate gradient

ARGUMENT
-`model`: a structure containing the parameters, data, and hyperparameters of a factorial hidden Markov drift-diffusion model

RETURN
-`ℓ`: log-likelihood
-`∇ℓ`: gradient of the log-likelihood with respect to fitted parameters in real space
-`∇∇ℓ`: Hessian matrix of the log-likelihood with respect to fitted parameters in real space

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat")
julia> ℓ, ∇ℓ, ∇∇ℓ = FHMDDM.twopasshessian(model)
```
"""
function twopasshessian(model::Model)
	@unpack trialsets = model
	sameacrosstrials = Sameacrosstrials(model)
	memoryforhessian = Memoryforhessian(model, sameacrosstrials)
	@inbounds for trialsetindex in eachindex(trialsets)
		𝐋 = linearpredictor(trialsets[trialsetindex].mpGLMs)
		offset = 0
		for trialindex in eachindex(trialsets[trialsetindex].trials)
			twopasshessian!(memoryforhessian, 𝐋, model, sameacrosstrials, offset, trialindex, trialsetindex)
			offset+=model.trialsets[trialsetindex].trials[trialindex].ntimesteps
		end
	end
	@unpack ℓ, ∇ℓ, ∇∇ℓ = memoryforhessian
	@inbounds for i = 1:size(∇∇ℓ,1)
		for j = i+1:size(∇∇ℓ,2)
			∇∇ℓ[j,i] = ∇∇ℓ[i,j]
		end
	end
	return ℓ[1], ∇ℓ, ∇∇ℓ
end

"""
	twopasshessian!

Compute the hessian for one trial as the Jacobian of the expectation conjugate gradient

MODIFIED ARGUMENT
-`memoryforhessian`: a structure containing quantities used in each trial

UNMODIFIED ARGUMENT
-`𝐋`: a nested array whose element 𝐋[n][j,k][t] corresponds to n-th neuron, j-th accumualtor state, k-th coupling state, and the t-th time bin in the trialset
"""
function twopasshessian!(memoryforhessian::Memoryforhessian,
						 𝐋::Vector{<:Matrix{<:Vector{<:Real}}},
						 model::Model,
						 sameacrosstrials::Sameacrosstrials,
						 offset::Integer,
						 trialindex::Integer,
						 trialsetindex::Integer)
	trial = model.trialsets[trialsetindex].trials[trialindex]
	@unpack mpGLMs = model.trialsets[trialsetindex]
	@unpack clicks = trial
	@unpack θnative = model
	@unpack ℓ, ∇ℓ, ∇∇ℓ, f, ∇f, D, ∇D, ∇b = memoryforhessian
	@unpack P, ∇pa₁, ∇∇pa₁, Aᵃinput, ∇Aᵃinput, ∇∇Aᵃinput = memoryforhessian
	@unpack λ, ∇logpy, ∇∇logpy, pY, ∇pY, ∂pY𝑑_∂ψ = memoryforhessian
	@unpack d𝛏_dB, Δt, K, Ξ = sameacrosstrials
	@unpack Aᵃsilent, ∇Aᵃsilent, ∇∇Aᵃsilent, Aᶜ, Aᶜᵀ, ∇Aᶜ, ∇Aᶜᵀ, πᶜ, πᶜᵀ, ∇πᶜ, ∇πᶜᵀ = sameacrosstrials
	@unpack indexθ_pa₁, indexθ_paₜaₜ₋₁, indexθ_paₜaₜ₋₁only, indexθ_pc₁, indexθ_pcₜcₜ₋₁, indexθ_ψ,  nθ_pa₁, nθ_paₜaₜ₋₁, nθ_pc₁, nθ_pcₜcₜ₋₁, nθ_ψ, index_pa₁_in_θ, index_paₜaₜ₋₁_in_θ, index_pc₁_in_θ, index_pcₜcₜ₋₁_in_θ, index_ψ_in_θ = sameacrosstrials
	indexθ_py = sameacrosstrials.indexθ_py[trialsetindex]
	nθ_py = sameacrosstrials.nθ_py[trialsetindex]
	indexθ_pY = sameacrosstrials.indexθ_pY[trialsetindex]
	nθ_pY = sameacrosstrials.nθ_pY[trialsetindex]
	index_pY_in_θ = sameacrosstrials.index_pY_in_θ[trialsetindex]
	indexθ_trialset = sameacrosstrials.indexθ_trialset[trialsetindex]
	nθ_trialset = sameacrosstrials.nθ_trialset[trialsetindex]
	adaptedclicks = ∇∇adapt(clicks, θnative.k[1], θnative.ϕ[1])
	update_emissions!(λ, ∇logpy, ∇∇logpy, pY, ∇pY, Δt, 𝐋, mpGLMs, trial.ntimesteps, offset)
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
					∇f[t][q] = ∂pY𝑑_∂ψ .* Aᵃ⨉f⨉Aᶜᵀ
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
    linearpredictor(mpGLMs)

Linear combination of the weights in the j-th accumulator state and k-th coupling state

ARGUMENT
-`mpGLMs`: a vector of mixture of Poisson generalized linear models

RETURN
-`𝐋`: a nested array whose element 𝐋[n][j,k][t] corresponds to n-th neuron, j-th accumualtor state, k-th coupling state, and the t-th time bin in the trialset
"""
function linearpredictor(mpGLMs::Vector{<:MixturePoissonGLM})
	map(mpGLMs) do mpGLM
		Ξ = length(mpGLM.d𝛏_dB)
		K = length(mpGLM.θ.𝐯)
		map(CartesianIndices((Ξ,K))) do index
			j = index[1]
			k = index[2]
			linearpredictor(mpGLM, j, k)
		end
	end
end

"""
	update_emissions!(λ, ∇logpy, ∇∇logpy, pY, ∇pY, Δt, 𝐋, mpGLMs, offset)

Update the conditional likelihood of spiking and its gradient and the gradient and Hessian of the conditional log-likelihoods

MODIFIED ARGUMENT
-`λ`:: conditional rate of each neuron at each time step. Element `λ[n][t][i,j]` corresponds to the n-th neuron, t-th time step in a trial, i-th accumulator state, and j-th coupling state
-`∇logpy`: partial derivatives of the conditional log-likelihood of each neuron at each time step. Element `∇logpy[t][n][q][i,j]` corresponds to the t-th time step in a trial, n-th neuron, q-th parameter, i-th accumulator state, and j-th coupling state.
 -`∇∇logpy`: partial derivatives of the conditional log-likelihood of each neuron at each time step. Element `∇∇logpy[t][n][q,r][i,j]` corresponds to the t-th time step in a trial, n-th neuron, q-th and r-th parameter, i-th accumulator state, and j-th coupling state.
-`pY`: conditional likelihood of the spiking of all neurons. Element `pY[t][i,j]` corresponds to the t-th time step in a trial, i-th accumulator state, and j-th coupling state
-`∇pY`: partial derivative of the conditional likelihood of the spiking of all neurons. Element `∇pY[t][q][i,j]` corresponds to the t-th time step in a trial, q-th paramaeter,  i-th accumulator state, and j-th coupling state

UNMODIFIED ARGUMENT
-`Δt`: duration of each time step, in second
-`𝐋`: linear predictors. Element `𝐋[n][i,j][τ]` corresponds to the n-th neuron, the i-th accumulator state and j-th coupling state for the τ-timestep in the trialset
-`mpGLMs`: Mixture of Poisson GLM of each neuron
-`ntimesteps`: number of time steps in the trial
-`offset`: the time index in the trialset corresponding to the time index 0 in the trial
"""
function update_emissions!(λ::Vector{<:Vector{<:Matrix{<:Real}}},
						∇logpy::Vector{<:Vector{<:Vector{<:Matrix{<:Real}}}},
						∇∇logpy::Vector{<:Vector{<:Matrix{<:Matrix{<:Real}}}},
						pY::Vector{<:Matrix{<:Real}},
						∇pY::Vector{<:Vector{<:Matrix{<:Real}}},
						Δt::Real,
						𝐋::Vector{<:Matrix{<:Vector{<:Real}}},
						mpGLMs::Vector{<:MixturePoissonGLM},
						ntimesteps::Integer,
						offset::Integer)
	dL_d𝐯 = zeros(length(mpGLMs[1].θ.𝐯[1]))
	@inbounds for n in eachindex(mpGLMs)
		conditionalrate!(λ[n], 𝐋[n], ntimesteps, offset)
		for t = 1:ntimesteps
			τ = t + offset
			∇∇conditional_log_likelihood!(∇logpy[t][n], ∇∇logpy[t][n], dL_d𝐯, Δt, 𝐋[n], λ[n][t], mpGLMs[n], τ)
		end
	end
	nneurons = length(mpGLMs)
	Ξ = length(mpGLMs[1].d𝛏_dB)
	K = length(mpGLMs[1].θ.𝐯)
	@inbounds for t = 1:ntimesteps
		τ = t + offset
		for ij in eachindex(pY[t])
			pY[t][ij] = 1.0
			for n=1:nneurons
				pY[t][ij] *= poissonlikelihood(λ[n][t][ij]*Δt, mpGLMs[n].𝐲[τ])
			end
		end
		r = 0
		for n=1:nneurons
			for q in eachindex(∇logpy[t][n])
				r+=1
				for i=1:Ξ
					for j=1:K
						∇pY[t][r][i,j] = ∇logpy[t][n][q][i,j]*pY[t][i,j]
					end
				end
			end
		end
	end
	return nothing
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
	conditionalrate!(λ, 𝐋, offset)

MODIFIED ARGUMENT
-`λ`: matrix whose element `λ[t][i,j]` is the Poisson rate at the t-th timestep of a trial given the i-th accumulator state and j-th coupling state

UNMODIFIED ARGUMENT
-`𝐋`: matrix whose element `𝐋[i,j][τ]` is the linear predictor given the i-th accumulator state and j-th coupling state for the τ-timestep in the trialset
-`offset`: the time index in the trialset corresponding to the time index 0 in the trial
"""
function conditionalrate!(λ::Vector{<:Matrix{<:Real}},
						  𝐋::Matrix{<:Vector{<:Real}},
						  ntimesteps::Integer,
						  offset::Integer)
	for t = 1:ntimesteps
		τ = t + offset
		for jk in eachindex(λ[t])
			λ[t][jk] = softplus(𝐋[jk][τ])
		end
	end
	return nothing
end

"""
	∇∇conditional_log_likelihood!(∇logpy, ∇∇logpy, dL_d𝐯, Δt, 𝐋, λ, mpGLM, τ)

Gradient and Hessian of the conditional log-likelihood of one neuron at single timestep

MODIFIED ARGUMENT
-`∇logpy`: Gradient of the conditional log-likelihood of a neuron's response at a single time. Element ∇logpy[i][j,k] correponds to the partial derivative of log p{y(n,t) ∣ a(t)=ξ(j), c(t)=k} with respect to the i-th parameter of the neuron's GLM
-`∇∇logpy`: Hessian of the conditional log-likelihood. Element ∇∇logpy[i,j][k,l] correponds to the partial derivative of log p{y(n,t) ∣ a(t)=ξ(k), c(t)=l} with respect to the i-th and j-th parameters of the neuron's GLM
-`dL_d𝐯`: memory for computing the derivative of the linear predictor with respect to the linear filters of the accumulator. The element `dL_d𝐯[q]` corresponds to the q-th linear filter in one of the coupling states.

UNMODIFIED ARGUMENT
-`Δt`: width of time step
-`𝐋`: linear predictors. Element `𝐋[i,j][τ]` corresponds to the i-th accumulator state and j-th coupling state for the τ-timestep in the trialset
-`λ`: Conditional Poisson whose element λ[i,j] corresponds to a(t)=ξ(i), c(t)=j
-`offset`: the time index in the trialset corresponding to the time index 0 in the trial
-`τ` time step in the trialset
"""
function ∇∇conditional_log_likelihood!(∇logpy::Vector{<:Matrix{<:Real}},
										∇∇logpy::Matrix{<:Matrix{<:Real}},
										dL_d𝐯::Vector{<:Real},
										Δt::Real,
										𝐋::Matrix{<:Vector{<:Real}},
										λ::Matrix{<:Real},
										mpGLM::MixturePoissonGLM,
										τ::Integer)
	@unpack d𝛏_dB, 𝐗, 𝐕, 𝐲 = mpGLM
	@unpack 𝐮, 𝐯 = mpGLM.θ
	n𝐮 = length(𝐮)
	K = length(𝐯)
	n𝐯 = length(𝐯[1])
	Ξ = length(d𝛏_dB)
	for i = 1:Ξ
		for q=1:n𝐯
			dL_d𝐯[q] = 𝐕[τ,q]*d𝛏_dB[i]
		end
		for j = 1:K
			dlogp_dL, d²logp_dL = differentiate_twice_loglikelihood_wrt_linearpredictor(Δt, 𝐋[i,j][τ], λ[i,j], 𝐲[τ])
			for q=1:n𝐮
				∇logpy[q][i,j] = dlogp_dL*𝐗[τ,q]
			end
			for q=1:n𝐯
				s = n𝐮+(j-1)*n𝐯+q
				∇logpy[s][i,j] = dlogp_dL*dL_d𝐯[q]
			end
			for q=1:n𝐮
				for r=q:n𝐮
					∇∇logpy[q,r][i,j] = d²logp_dL*𝐗[τ,q]*𝐗[τ,r]
				end
				for r=1:n𝐯
					s = n𝐮+(j-1)*n𝐯+r
					∇∇logpy[q,s][i,j] = d²logp_dL*𝐗[τ,q]*dL_d𝐯[r]
				end
			end
			for q=1:n𝐯
				for r=q:n𝐯
					s1 = n𝐮+(j-1)*n𝐯+q
					s2 = n𝐮+(j-1)*n𝐯+r
					∇∇logpy[s1,s2][i,j] = d²logp_dL * dL_d𝐯[q] * dL_d𝐯[r]
				end
			end
		end
	end
	return nothing
end

"""
    differentiate_twice_loglikelihood_wrt_linearpredictor

Differentiate the log-likelihood of a Poisson GLM with respect to the linear predictor

The Poisson GLM is assumed to have a a softplus nonlinearity

ARGUMENT
-`Δt`: duration of time step
-`L`: linear predictor at one time step
-`λ`: Poisson rate
-`y`: observation at that time step

RETURN
-the first derivative with respect to the linear predictor
-the second derivative with respect to the linear predictor

EXAMPLE
```julia-repl
julia> using FHMDDM, ForwardDiff, LogExpFunctions
julia> Δt = 0.01
julia> y = 2
julia> f(x) = let λΔt = softplus(x[1])*Δt; y*log(λΔt)-λΔt+log(factorial(y)); end
julia> x = rand(1)
julia> d1auto = ForwardDiff.gradient(f, x)
julia> d2auto = ForwardDiff.hessian(f, x)
julia> d1hand, d2hand = FHMDDM.differentiate_twice_loglikelihood_wrt_linearpredictor(Δt, x[1], softplus(x[1]), y)
julia> abs(d1hand - d1auto[1])
julia> abs(d2hand - d2auto[1])
```
"""
function differentiate_twice_loglikelihood_wrt_linearpredictor(Δt::AbstractFloat, L::Real, λ::Real, y::Integer)
	dλ_dL = logistic(L)
	d²λ_dLdL = dλ_dL*(1-dλ_dL)
    if y > 0
        if L > -100.0
            dℓ_dL = dλ_dL*(y/λ - Δt)
        else
            dℓ_dL = y - dλ_dL*Δt  # the limit of `dλ_dL/λ` as x goes to -∞ is 1
        end
		if L > -50.0
			d²ℓ_dLdL = y*(λ*d²λ_dLdL - dλ_dL^2)/λ^2 - d²λ_dLdL*Δt # the limit of first second term is 0 as L goes to -∞
		else
			d²ℓ_dLdL = -d²λ_dLdL*Δt
		end
    else
        dℓ_dL = -dλ_dL*Δt
		d²ℓ_dLdL = -d²λ_dLdL*Δt
    end
	return dℓ_dL, d²ℓ_dLdL
end

"""
    differentiate_loglikelihood_wrt_linearpredictor

Differentiate the log-likelihood of a Poisson GLM with respect to the linear predictor

The Poisson GLM is assumed to have a a softplus nonlinearity

ARGUMENT
-`Δt`: duration of time step
-`L`: linear predictor at one time step
-`λ`: Poisson rate
-`y`: observation at that time step

RETURN
-the first derivative with respect to the linear predictor

EXAMPLE
```julia-repl
julia> using FHMDDM, ForwardDiff, LogExpFunctions
julia> Δt = 0.01
julia> y = 2
julia> f(x) = let λΔt = softplus(x[1])*Δt; y*log(λΔt)-λΔt+log(factorial(y)); end
julia> x = rand(1)
julia> d1auto = ForwardDiff.gradient(f, x)
julia> d1hand = FHMDDM.differentiate_loglikelihood_wrt_linearpredictor(Δt, x[1], softplus(x[1]), y)
julia> abs(d1hand - d1auto[1])
```
"""
function differentiate_loglikelihood_wrt_linearpredictor(Δt::AbstractFloat, L::Real, λ::Real, y::Integer)
	dλ_dL = logistic(L)
    if y > 0
        if L > -100.0
            dℓ_dL = dλ_dL*(y/λ - Δt)
        else
            dℓ_dL = y - dλ_dL*Δt  # the limit of `dλ_dL/λ` as x goes to -∞ is 1
        end
    else
        dℓ_dL = -dλ_dL*Δt
    end
	return dℓ_dL
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
	indexθ_py = map(trialsets) do trialset
					map(trialset.mpGLMs) do mpGLM
						q = length(mpGLM.θ.𝐮) + sum(length.(mpGLM.θ.𝐯))
						zeros(Int,q)
					end
				end
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

"""
	Memoryforhessian(model)

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat"; randomize=true)
julia> S = FHMDDM.Sameacrosstrials(model)
julia> M = FHMDDM.Memoryforhessian(model, S)
```
"""
function Memoryforhessian(model::Model, S::Sameacrosstrials)
	@unpack options, θnative, θreal, trialsets = model
	@unpack Δt, K, Ξ = options
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
	∇logpy = collect(collect(collect(zeros(Ξ,K) for q=1:max_nθ_py) for n=1:maxneurons) for t=1:maxtimesteps)
	∇∇logpy=map(1:maxtimesteps) do t
				map(1:maxneurons) do n
					map(CartesianIndices((max_nθ_py,max_nθ_py))) do q
						zeros(Ξ,K)
					end
				end
			end
	Aᵃinput=map(1:maxclicks) do t
				A = zeros(Ξ,Ξ)
				A[1,1] = A[Ξ,Ξ] = 1.0
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
	Memoryforhessian(Aᵃinput=Aᵃinput,
					∇Aᵃinput=∇Aᵃinput,
					∇∇Aᵃinput=∇∇Aᵃinput,
					f=f,
					∇f=∇f,
					∇b=∇b,
					D = zeros(maxtimesteps),
					∇D=∇D,
					∇ℓ=zeros(S.nθ_alltrialsets),
					∇∇ℓ=zeros(S.nθ_alltrialsets,S.nθ_alltrialsets),
					∂pY𝑑_∂ψ=zeros(Ξ,K),
					λ=λ,
					∇logpy=∇logpy,
					∇∇logpy=∇∇logpy,
					P = Probabilityvector(Δt, θnative, Ξ),
					∇pa₁=∇pa₁,
					∇∇pa₁=∇∇pa₁,
					pY=pY,
					∇pY=∇pY)
end
