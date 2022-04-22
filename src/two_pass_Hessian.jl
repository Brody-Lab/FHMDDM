
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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true)
julia> absdiffℓ, absdiff∇, absdiff∇∇ = FHMDDM.check_twopasshessian(model)
```
"""
function check_twopasshessian(model::Model)
	concatenatedθ, indexθ = concatenateparameters(model)
	ℓhand, ∇hand, ∇∇hand = FHMDDM.twopasshessian!(model,concatenatedθ,indexθ)
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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true)
julia> concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
julia> ℓ, ∇ℓ, ∇∇ℓ = FHMDDM.twopasshessian!(model, concatenatedθ, indexθ)
```
"""
function twopasshessian!(model::Model, concatenatedθ::Vector{<:Real}, indexθ::Indexθ)
	sortparameters!(model, concatenatedθ, indexθ)
	ℓ, ∇ℓ, ∇∇ℓ = twopasshessian(model)
	native2real!(∇ℓ, ∇∇ℓ, indexθ.latentθ, model)
	∇ℓ = sortparameters(indexθ.latentθ, ∇ℓ)
	∇∇ℓ = sortparameters(indexθ.latentθ, ∇∇ℓ)
	return ℓ, ∇ℓ, ∇∇ℓ
end

"""
	twopasshessian!(model)

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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true)
julia> ℓ, ∇ℓ, ∇∇ℓ = FHMDDM.twopasshessian(model)
```
"""
function twopasshessian(model::Model)
	@unpack trialsets = model
	sameacrosstrials = FHMDDM.Sameacrosstrials(model)
	memoryforhessian = FHMDDM.Memoryforhessian(model, sameacrosstrials)
	for s in eachindex(trialsets)
		glmθs = collect(trialsets[s].mpGLMs[n].θ for n = 1:length(trialsets[s].mpGLMs))
		for m in eachindex(trialsets[s].trials)
			twopasshessian!(memoryforhessian, glmθs, s, sameacrosstrials, model.θnative, trialsets[s].trials[m])
		end
	end
	@unpack ℓ, ∇ℓ, ∇∇ℓ = memoryforhessian
	for i = 1:size(∇∇ℓ,1)
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
-`ℓ`: log-likelihood
-`∇ℓ`: gradient of the log-likelihood with respect to fitted parameters in real space
-`∇∇ℓ`: Hessian matrix of the log-likelihood with respect to fitted parameters in real space
-`P`: structure containing pre-allocated memory fro computing the transition matrix

UNMODIFIED ARGUMENT
-`glmθs`: a vector whose each element is a structure containing the parameters of of the generalized linear model of a neuron
-`s`: index of the trialset
-`memoryforhessian`: a structure containing quantities used in each trial
-`θnative`: a structure containing parameters specifying the latent variables in their native space
-`trial`: a structure containing information on the sensory stimuli, spike trains, input to each neuron's GLM, and behavioral choice
"""
function twopasshessian!(memoryforhessian::Memoryforhessian,
						 glmθs::Vector{<:GLMθ},
						 s::Integer,
						 sameacrosstrials::Sameacrosstrials,
 						 θnative::Latentθ,
						 trial::Trial)
	@unpack ℓ, ∇ℓ, ∇∇ℓ, f, ∇f, D, ∇D, ∇b, ∇η = memoryforhessian
	@unpack P, ∇pa₁, ∇∇pa₁, Aᵃinput, ∇Aᵃinput, ∇∇Aᵃinput = memoryforhessian
	@unpack L, λ, ∇logpy, ∇∇logpy, pY, ∇pY, ∂pY𝑑_∂ψ = memoryforhessian
	@unpack d𝛏_dB, Δt, K, Ξ = sameacrosstrials
	@unpack Aᵃsilent, ∇Aᵃsilent, ∇∇Aᵃsilent, Aᶜ, Aᶜᵀ, ∇Aᶜ, ∇Aᶜᵀ, πᶜ, πᶜᵀ, ∇πᶜ, ∇πᶜᵀ = sameacrosstrials
	@unpack indexθ_pa₁, indexθ_paₜaₜ₋₁, indexθ_pc₁, indexθ_pcₜcₜ₋₁, indexθ_ψ,  nθ_pa₁, nθ_paₜaₜ₋₁, nθ_pc₁, nθ_pcₜcₜ₋₁, nθ_ψ, index_pa₁_in_θ, index_paₜaₜ₋₁_in_θ, index_pc₁_in_θ, index_pcₜcₜ₋₁_in_θ, index_ψ_in_θ = sameacrosstrials
	indexθ_py = sameacrosstrials.indexθ_py[s]
	nθ_py = sameacrosstrials.nθ_py[s]
	indexθ_pY = sameacrosstrials.indexθ_pY[s]
	nθ_pY = sameacrosstrials.nθ_pY[s]
	index_pY_in_θ = sameacrosstrials.index_pY_in_θ[s]
	indexθ_trialset = sameacrosstrials.indexθ_trialset[s]
	nθ_trialset = sameacrosstrials.nθ_trialset[s]
	nneurons = length(trial.spiketrainmodels)
	@unpack clicks, spiketrainmodels = trial
	adaptedclicks = FHMDDM.∇∇adapt(clicks, θnative.k[1], θnative.ϕ[1])
	# conditional likelihood of population spiking and its gradient; gradient and Hessian of the conditional log-likelihood of individual neurons' spiking
	for n in eachindex(L)
		FHMDDM.conditional_linear_predictor!(L[n], d𝛏_dB, spiketrainmodels[n], glmθs[n])
		for t = 1:trial.ntimesteps
			FHMDDM.conditionalrate!(λ[n][t], L[n][t])
			FHMDDM.∇∇conditional_log_likelihood!(∇logpy[t][n], ∇∇logpy[t][n], L[n][t], λ[n][t], Δt, d𝛏_dB, t, trial.spiketrainmodels[n])
		end
	end
	for t = 1:trial.ntimesteps
		for jk in eachindex(pY[t])
			pY[t][jk] = FHMDDM.Poissonlikelihood(λ[1][t][jk]*Δt, spiketrainmodels[1].𝐲[t])
			for n=2:nneurons
				pY[t][jk] *= FHMDDM.Poissonlikelihood(λ[n][t][jk]*Δt, spiketrainmodels[n].𝐲[t])
			end
		end
		q = 0
		for n=1:nneurons
			for i in eachindex(∇logpy[t][n])
				q+=1
				for j=1:Ξ
					for k=1:K
						∇pY[t][q][j,k] = ∇logpy[t][n][i][j,k]*pY[t][j,k]
					end
				end
			end
		end
	end
	differentiate_pY𝑑_wrt_ψ!(∂pY𝑑_∂ψ, pY[trial.ntimesteps], trial.choice)
	conditionallikelihood!(pY[trial.ntimesteps], trial.choice, θnative.ψ[1])
	for i in eachindex(∇pY[trial.ntimesteps])
		conditionallikelihood!(∇pY[trial.ntimesteps][i], trial.choice, θnative.ψ[1])
	end
	# first forward step
	for q in eachindex(∇f[1])
		∇f[1][q] .= 0
	end
	FHMDDM.∇∇priorprobability!(∇∇pa₁, ∇pa₁, P, trial.previousanswer)
	pa₁ = copy(P.𝛑) # save for later
	pY₁⨀pc₁ = pY[1] .* πᶜᵀ
	pa₁⨀pc₁ = pa₁ .* πᶜᵀ
	for i = 1:nθ_pY
		for j=1:Ξ
			for k = 1:K
				∇f[1][indexθ_pY[i]][j,k] = ∇pY[1][i][j,k]*pa₁⨀pc₁[j,k]
			end
		end
	end
	for i = 1:nθ_pa₁
		for j=1:Ξ
			for k = 1:K
				∇f[1][indexθ_pa₁[i]][j,k] = pY₁⨀pc₁[j,k]*∇pa₁[i][j]
			end
		end
	end
	for i = 1:nθ_pc₁
		for j=1:Ξ
			for k = 1:K
				∇f[1][indexθ_pc₁[1]][j,k] = pY[1][j,k]*pa₁[j]*∇πᶜ[i][k]
			end
		end
	end
	for j=1:Ξ
		for k = 1:K
			f[1][j,k] = pY[1][j,k] * pa₁⨀pc₁[j,k]
		end
	end
	D[1] = sum(f[1])
	forward!(∇D[1], f[1], ∇f[1], ℓ, ∇ℓ, D[1])
	for t=2:trial.ntimesteps
		if t ∈ clicks.inputtimesteps
			update_for_∇∇transition_probabilities!(P, adaptedclicks, clicks, t)
			∇∇transitionmatrix!(∇∇Aᵃinput[t], ∇Aᵃinput[t], Aᵃinput[t], P)
			Aᵃ = Aᵃinput[t]
			∇Aᵃ = ∇Aᵃinput[t]
			∇∇Aᵃ = ∇∇Aᵃinput[t]
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
		forward!(∇D[t], f[t], ∇f[t], ℓ, ∇ℓ, D[t])
	end
	bₜ = ones(Ξ,K)
	for t = trial.ntimesteps:-1:1
		γ = f[t] # resuse memory
		∇γ = ∇f[trial.ntimesteps] # resuse memory
		if t == trial.ntimesteps
			for q in indexθ_trialset
				∇b[q] .= 0
			end
			# the p(𝑑 ∣ aₜ, cₜ) term
			q = indexθ_ψ[1]
			∇ℓ[q] += FHMDDM.expectation_derivative_logp𝑑_wrt_ψ(trial.choice, γ, θnative.ψ[1])
			∇∇ℓ[q,q] += FHMDDM.expectation_second_derivative_logp𝑑_wrt_ψ(trial.choice, γ, θnative.ψ[1])
			for r in indexθ_trialset
				if r < q
					∇∇ℓ[r,q] += FHMDDM.expectation_derivative_logp𝑑_wrt_ψ(trial.choice, ∇γ[r], θnative.ψ[1])
				else
					∇∇ℓ[q,r] += FHMDDM.expectation_derivative_logp𝑑_wrt_ψ(trial.choice, ∇γ[r], θnative.ψ[1])
				end
			end
		else
			if t+1 ∈ clicks.inputtimesteps
				Aᵃₜ₊₁ = Aᵃinput[t+1]
				∇Aᵃₜ₊₁ = ∇Aᵃinput[t+1]
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
				elseif (i_ψ > 0) && (t == trial.ntimesteps-1)
					∇b[q] = (Aᵃₜ₊₁ᵀ * (bₜ₊₁.*∂pY𝑑_∂ψ) * Aᶜ .- bₜ.*∇D[t+1][q])./D[t+1]
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
				Aᵃ = Aᵃinput[t]
				∇Aᵃ = ∇Aᵃinput[t]
				∇∇Aᵃ = ∇∇Aᵃinput[t]
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
				for r in indexθ_trialset
 					if (r >= q) && r != indexθ_ψ[1]
						∇∇ℓ[q,r] += sum_product_over_states(D[t], ∇f[t-1][r], bₜ, pY[t], ∇Aᵃ[i], Aᶜ)
						∇∇ℓ[q,r] += sum_product_over_states(D[t], f[t-1], ∇b[r], pY[t], ∇Aᵃ[i], Aᶜ)
						∇∇ℓ[q,r] -= η*∇D[t][r]/D[t]
						j = index_pY_in_θ[r]
						if j > 0
							∇∇ℓ[q,r] += sum_product_over_states(D[t], f[t-1], bₜ, ∇pY[t][j], ∇Aᵃ[i], Aᶜ)
						end
						j = index_paₜaₜ₋₁_in_θ[r]
						if j > 0
							∇∇ℓ[q,r] += sum_product_over_states(D[t], f[t-1], bₜ, pY[t], ∇∇Aᵃ[i,j], Aᶜ)
						end
						j = index_pcₜcₜ₋₁_in_θ[r]
						if j > 0
							∇∇ℓ[q,r] += sum_product_over_states(D[t], f[t-1], bₜ, pY[t], ∇Aᵃ[i], ∇Aᶜ[j])
						end
					end
				end
			end
			# the p(cₜ ∣ cₜ₋₁) term
			for i = 1:nθ_pcₜcₜ₋₁
				q = indexθ_pcₜcₜ₋₁[i]
				η = sum_product_over_states(D[t], f[t-1], bₜ, pY[t], Aᵃ, ∇Aᶜ[i])
				∇ℓ[q] += η
				for r in indexθ_trialset
 					if (r >= q) && r != indexθ_ψ[1]
						∇∇ℓ[q,r] += sum_product_over_states(D[t], ∇f[t-1][r], bₜ, pY[t], Aᵃ, ∇Aᶜ[i])
						∇∇ℓ[q,r] += sum_product_over_states(D[t], f[t-1], ∇b[r], pY[t], Aᵃ, ∇Aᶜ[i])
						∇∇ℓ[q,r] -= η*∇D[t][r]/D[t]
						j = index_pY_in_θ[r]
						if j > 0
							∇∇ℓ[q,r] += sum_product_over_states(D[t], f[t-1], bₜ, ∇pY[t][j], Aᵃ, ∇Aᶜ[i])
						end
						j = index_paₜaₜ₋₁_in_θ[r]
						if j > 0
							∇∇ℓ[q,r] += sum_product_over_states(D[t], f[t-1], bₜ, pY[t], ∇Aᵃ[j], ∇Aᶜ[i])
						end
					end
				end
			end
		end
		# the p(yₙₜ ∣ aₜ, cₜ) term
		for n = 1:length(indexθ_py)
			for i = 1:nθ_py[n]
				q = indexθ_py[n][i]
				∇ℓ[q] += sum_product_over_states(γ, ∇logpy[t][n][i])
				for r in indexθ_trialset
					if r >=q
						∇∇ℓ[q,r] += sum_product_over_states(∇γ[r], ∇logpy[t][n][i])
					end
				end
				for j = i:nθ_py[n]
					r = indexθ_py[n][j]
					∇∇ℓ[q,r] += sum_product_over_states(γ, ∇∇logpy[t][n][i,j])
				end
			end
		end
	end
	#last backward step
	t = 1
	# the p(a₁) term
	for i = 1:nθ_pa₁
		q = indexθ_pa₁[i]
		η = sum_product_over_states(D[t], bₜ, pY[t], ∇pa₁[i], πᶜ)
		∇ℓ[q] += η
		for r in indexθ_trialset
			if (r >= q) && r != indexθ_ψ[1]
				∇∇ℓ[q,r] += sum_product_over_states(D[t], ∇b[r], pY[t], ∇pa₁[i], πᶜ)
				∇∇ℓ[q,r] -= η*∇D[t][r]/D[t]
				j = index_pY_in_θ[r]
				if j > 0
					∇∇ℓ[q,r] += sum_product_over_states(D[t], bₜ, ∇pY[t][j], ∇pa₁[i], πᶜ)
				end
				j = index_pa₁_in_θ[r]
				if j > 0
					∇∇ℓ[q,r] += sum_product_over_states(D[t], bₜ, pY[t], ∇∇pa₁[i,j], πᶜ)
				end
				j = index_pc₁_in_θ[r]
				if j > 0
					∇∇ℓ[q,r] += sum_product_over_states(D[t], bₜ, pY[t], ∇pa₁[i], ∇πᶜ[j])
				end
			end
		end
	end
	# the p(c₁) term
	for i = 1:nθ_pc₁
		q = indexθ_pc₁[i]
		η = sum_product_over_states(D[t], bₜ, pY[t], pa₁, ∇πᶜ[i])
		∇ℓ[q] += η
		for r in indexθ_trialset
			if (r >= q) && r != indexθ_ψ[1]
				∇∇ℓ[q,r] += sum_product_over_states(D[t], ∇b[r], pY[t], pa₁, ∇πᶜ[i])
				∇∇ℓ[q,r] -= η*∇D[t][r]/D[t]
				j = index_pY_in_θ[r]
				if j > 0
					∇∇ℓ[q,r] += sum_product_over_states(D[t], bₜ, ∇pY[t][j], pa₁, ∇πᶜ[i])
				end
				j = index_pa₁_in_θ[r]
				if j > 0
					∇∇ℓ[q,r] += sum_product_over_states(D[t], bₜ, pY[t], ∇pa₁[j], ∇πᶜ[i])
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
-`∇ℓ`: gradient of the log-likelihood for the data up to the current time-step
-`ℓ`: log-likelihood for the data before the up to the current time-step

UNMODIFIED ARGUMENT
-`D`: past-conditioned emissions likelihood: `p{𝐘(t) ∣ 𝐘(1:t-1))`

"""
function forward!(∇D::Vector{<:Real},
				f::Matrix{<:Real},
				∇f::Vector{<:Matrix{<:Real}},
				ℓ::Vector{<:Real},
				∇ℓ::Vector{<:Real},
				D::Real)
	f ./= D
	for i in eachindex(∇D)
		∇D[i] = sum(∇f[i])
	end
	for i in eachindex(∇f)
		for jk in eachindex(∇f[i])
			∇f[i][jk] = (∇f[i][jk] - f[jk]*∇D[i])/D
		end
	end
	ℓ[1] += log(D)
	# for i in eachindex(∇ℓ)
	# 	∇ℓ[i] += ∇D[i]/D
	# end
end

"""
	conditionalrate!(λ, L)

MODIFIED ARGUMENT
-`λ`: matrix whose element `λ[i,j]` is the Poisson rate given the i-th accumulator state and j-th coupling state

UNMODIFIED ARGUMENT
-`L`: matrix whose element `L[i,j]` is the linear predictor given the i-th accumulator state and j-th coupling state
"""
function conditionalrate!(λ::Matrix{<:Real}, L::Matrix{<:Real})
	Ξ = size(λ,1)
	for i = 1:Ξ
		λ[i,1] = softplus(L[i,1])
	end
	λ[:,2] .= λ[cld(Ξ,2),1]
	return nothing
end

"""
	Poissonlikelihood(λΔt, y)

Likelihood of observation `y` given Poisson rate `λΔt`
"""
function Poissonlikelihood(λΔt::Real, y::Integer)
	if y==0
		exp(-λΔt)
	elseif y==1
		λΔt/exp(λΔt)
	elseif y == 2
		λΔt^2 / exp(λΔt) / 2
	else
		λΔt^y / exp(λΔt) / factorial(y)
	end
end


"""
	conditional_linear_predictor!(L, d𝛏_dB, glmio, glmθ)

MODIFIED ARGUMENT
-`L`: A vector of matrices whose element `L[t][i,j]` corresponds to the linear predictor at time t given the i-th accumulator state and j-th coupling state

UNMODFIED ARGUMENT
-`d𝛏_dB`: normalized values of the accumulator
-`glmio`: input and observations of a neuron's glm
-`glmθ`: parameters of a neuron's glm
"""
function conditional_linear_predictor!(L::Vector{<:Matrix{<:Real}},
									d𝛏_dB::Vector{<:Real},
									glmio::SpikeTrainModel,
									glmθ::GLMθ)
	@unpack 𝐔, 𝚽 = glmio
	@unpack 𝐮, 𝐯 = glmθ
	𝐔𝐮 = 𝐔*𝐮
	𝚽𝐯 = 𝚽*𝐯
	Ξ = length(d𝛏_dB)
	zeroindex = cld(Ξ,2)
	for t = 1:length(𝐔𝐮)
		for i=1:Ξ-1
			L[t][i,1] = 𝐔𝐮[t] + 𝚽𝐯[t]*d𝛏_dB[i]
		end
		L[t][zeroindex,1] = 𝐔𝐮[t]
		for i=zeroindex+1:Ξ
			L[t][i,1] = 𝐔𝐮[t] + 𝚽𝐯[t]*d𝛏_dB[i]
		end
		L[t][:,2] .= L[t][zeroindex,1]
	end
	return nothing
end

"""
	∇∇conditional_log_likelihood!(∇logpy, ∇∇logpy, L, λ, Δt, d𝛏_dB, t, glmio)

Gradient and Hessian of the conditional log-likelihood

MODIFIED ARGUMENT
-`∇logpy`: Gradient of the conditional log-likelihood of a neuron's response at a single time. Element ∇logpy[i][j,k] correponds to the partial derivative of log p{y(n,t) ∣ a(t)=ξ(j), c(t)=k} with respect to the i-th parameter of the neuron's GLM
-`∇∇logpy`: Hessian of the conditional log-likelihood. Element ∇logpy[i.j][k,l] correponds to the partial derivative of log p{y(n,t) ∣ a(t)=ξ(k), c(t)=l} with respect to the i-th and j-th parameters of the neuron's GLM

UNMODIFIED ARGUMENT
-`L`: Conditional linear predictor whose element L[i,j] corresponds to a(t)=ξ(i), c(t)=j
-`λ`: Conditional Poisson whose element λ[i,j] corresponds to a(t)=ξ(i), c(t)=j
-`Δt`: width of time step
-`d𝛏_dB`: normalized value into which the accumulator is discretzed
-`t` time step
-`glmio`: input and observations of a neuron's Poisson mixture GLM
"""
function ∇∇conditional_log_likelihood!(∇logpy::Vector{<:Matrix{<:Real}},
									∇∇logpy::Matrix{<:Matrix{<:Real}},
									L::Matrix{<:Real},
									λ::Matrix{<:Real},
									Δt::Real,
									d𝛏_dB::Vector{<:Real},
									t::Integer,
									glmio::SpikeTrainModel)
	@unpack 𝐔, 𝚽, 𝐲 = glmio
	n𝐮 = size(𝐔,2)
	n𝐯 = size(𝚽,2)
	dL_d𝐯 = zeros(n𝐯)
	Ξ = size(L,1)
	zeroindex = cld(Ξ,2)
	for i = 1:Ξ
		dlogp_dL, d²logp_dL = differentiate_twice_loglikelihood_wrt_linearpredictor(Δt, L[i,1], λ[i,1], 𝐲[t])
		for j=1:n𝐮
			∇logpy[j][i,1] = dlogp_dL*𝐔[t,j]
		end
		for j=1:n𝐯
			dL_d𝐯[j] = 𝚽[t,j]*d𝛏_dB[i]
			∇logpy[j+n𝐮][i,1] = dlogp_dL*dL_d𝐯[j]
		end
		for j=1:n𝐮
			for k=j:n𝐮
				∇∇logpy[j,k][i,1] = d²logp_dL*𝐔[t,j]*𝐔[t,k]
			end
			for k=1:n𝐯
				∇∇logpy[j,k+n𝐮][i,1] = d²logp_dL*𝐔[t,j]*dL_d𝐯[k]
			end
		end
		for j=1:n𝐯
			for k=j:n𝐯
				∇∇logpy[j+n𝐮,k+n𝐮][i,1] = d²logp_dL * dL_d𝐯[j] * dL_d𝐯[k]
			end
		end
	end
	n𝐮𝐯 = n𝐮+n𝐯
	for j = 1:n𝐮𝐯
		∇logpy[j][:,2] .= ∇logpy[j][zeroindex,1]
		for k = j:n𝐮𝐯
			∇∇logpy[j,k][:,2] .= ∇∇logpy[j,k][zeroindex,1]
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
	for j = 1:K
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
	for iaₜ = 1:Ξ
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
	for iaₜ = 1:Ξ
		for icₜ = 1:K
			s += b[iaₜ,icₜ]*Y[iaₜ,icₜ]*A[iaₜ]*C[icₜ]
		end
	end
	return s/D
end

"""
	sum_product_over_states(γ,Y)

Multiply terms across different states of the latent variables at consecutive time step and sum

ARGUMENT
-`γ`: element γ[i,j] corresponds to i-th state of the accumulator at time t and the j-th state of the coupling at time t
-`Y`: similar to η, element Y[i,j] corresponds to i-th state of the accumulator at time t and the j-th state of the coupling at time t

RETURN
-`s`: sum of the product across all states of the two latent variables at two consecutive time steps
"""
function sum_product_over_states(γ::Matrix{<:Real}, Y::Matrix{<:Real})
	s = 0.0
	Ξ,K = size(γ)
	for j = 1:Ξ
		for k = 1:K
			s+= γ[j,k]*Y[j,k]
		end
	end
	return s
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
	for j = 1:size(γ,2)
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
	for j = 1:size(γ,2)
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
	Memoryforhessian(model)

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_15/data.mat"; randomize=true)
julia> S = FHMDDM.Sameacrosstrials(model)
julia> M = FHMDDM.Memoryforhessian(model, S)
```
"""
function Memoryforhessian(model::Model, S::Sameacrosstrials)
	@unpack options, θnative, θreal, trialsets = model
	@unpack Δt, K, Ξ = options
	maxtimesteps=maximum(map(trialset->maximum(map(trial->trial.ntimesteps, trialset.trials)), trialsets))
	max_nθ_pY=maximum(S.nθ_pY)
	max_nθ_py=maximum(map(n->maximum(n), S.nθ_py))
	maxneurons = maximum(map(trialset-> length(trialset.mpGLMs), trialsets))
	∇D=collect(zeros(S.nθ_alltrialsets) for t=1:maxtimesteps)
	f = collect(zeros(Ξ,K) for t=1:maxtimesteps)
	∇f = collect(collect(zeros(Ξ,K) for q=1:S.nθ_alltrialsets) for t=1:maxtimesteps)
	∇b = collect(zeros(Ξ,K) for q=1:S.nθ_alltrialsets)
	∇η = collect(zeros(Ξ,K) for q=1:S.nθ_alltrialsets)
	L = collect(collect(zeros(Ξ,K) for t=1:maxtimesteps) for n = 1:maxneurons)
	λ = collect(collect(zeros(Ξ,K) for t=1:maxtimesteps) for n = 1:maxneurons)
	∇logpy = collect(collect(collect(zeros(Ξ,K) for q=1:max_nθ_py) for n=1:maxneurons) for t=1:maxtimesteps)
	∇∇logpy=map(1:maxtimesteps) do t
				map(1:maxneurons) do n
					map(CartesianIndices((max_nθ_py,max_nθ_py))) do q
						zeros(Ξ,K)
					end
				end
			end
	Aᵃinput=map(1:maxtimesteps) do t
				A = zeros(Ξ,Ξ)
				A[1,1] = A[Ξ,Ξ] = 1.0
				return A
			end
	∇Aᵃinput = collect(collect(zeros(Ξ,Ξ) for q=1:S.nθ_paₜaₜ₋₁) for t=1:maxtimesteps)
	∇∇Aᵃinput = map(1:maxtimesteps) do t
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
					∇η=∇η,
					D = zeros(maxtimesteps),
					∇D=∇D,
					∇ℓ=zeros(S.nθ_alltrialsets),
					∇∇ℓ=zeros(S.nθ_alltrialsets,S.nθ_alltrialsets),
					∂pY𝑑_∂ψ=zeros(Ξ,K),
					L=L,
					λ=λ,
					∇logpy=∇logpy,
					∇∇logpy=∇∇logpy,
					P = Probabilityvector(Δt, θnative, Ξ),
					∇pa₁=∇pa₁,
					∇∇pa₁=∇∇pa₁,
					pY=pY,
					∇pY=∇pY)
end
