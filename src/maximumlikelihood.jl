"""
    maximizelikelihood!(model)

Learn the parameters of the factorial hidden Markov drift-diffusion model by maximizing the likelihood of the data

MODIFIED ARGUMENT
- a structure containing information for a factorial hidden Markov drift-diffusion model

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
function maximizelikelihood!(model::Model;
			                 extended_trace::Bool=true,
			                 f_tol::AbstractFloat=1e-9,
			                 g_tol::AbstractFloat=1e-8,
			                 iterations::Integer=1000,
							 outer_iterations::Integer=10,
			                 show_every::Integer=1,
			                 show_trace::Bool=true,
			                 x_tol::AbstractFloat=1e-5)
	shared = Shared(model)
	@unpack K, Ξ = model.options
	γ =	map(model.trialsets) do trialset
			map(CartesianIndices((Ξ,K))) do index
				zeros(trialset.ntimesteps)
			end
		end
    f(concatenatedθ) = -loglikelihood!(model, shared, concatenatedθ)
    g!(∇, concatenatedθ) = ∇negativeloglikelihood!(∇, γ, model, shared, concatenatedθ)
	# lowerbounds, upperbounds = concatenatebounds(shared.indexθ, model.options)
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
	θ₀ = deepcopy(shared.concatenatedθ)
	# optimizationresults = Optim.optimize(f, g!, lowerbounds, upperbounds, θ₀, algorithm, Optim_options)
	optimizationresults = Optim.optimize(f, g!, θ₀, algorithm, Optim_options)
    println(optimizationresults)
    maximumlikelihoodθ = Optim.minimizer(optimizationresults)
	sortparameters!(model, maximumlikelihoodθ, shared.indexθ)
    return nothing
end

"""
    loglikelihood!(model, shared, concatenatedθ)

Compute the log-likelihood

ARGUMENT
-`model`: an instance of FHM-DDM
-`shared`: a container of variables used by both the log-likelihood and gradient computation

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values

RETURN
-log-likelihood
"""
function loglikelihood!(model::Model,
						shared::Shared,
					    concatenatedθ::Vector{<:Real})
	if concatenatedθ != shared.concatenatedθ
		update!(model, shared, concatenatedθ)
	end
	@unpack options, θnative, trialsets = model
	trialinvariant = Trialinvariant(options, θnative; purpose="loglikelihood")
	ℓ = map(trialsets, shared.p𝐘𝑑) do trialset, p𝐘𝑑
			pmap(trialset.trials, p𝐘𝑑) do trial, p𝐘𝑑
				loglikelihood(p𝐘𝑑, θnative, trial, trialinvariant)
			end
		end
	return sum(sum(ℓ))
end

"""
	loglikelihood(p𝐘𝑑, trialinvariant, θnative, trial)

Compute the log-likelihood of the data from one trial

ARGUMENT
-`p𝐘𝑑`: a matrix whose element `p𝐘𝑑[t][i,j]` represents the conditional likelihood `p(𝐘ₜ, d ∣ 𝐚ₜ=i, 𝐜ₜ=j)`
-`θnative`: model parameters in their native space
-`trial`: stimulus and behavioral information of one trial
-`trialinvariant`: a structure containing quantities that are used in each trial

RETURN
-`ℓ`: log-likelihood of the data from one trial
"""
function loglikelihood(p𝐘𝑑::Vector{<:Matrix{<:Real}},
					   θnative::Latentθ,
					   trial::Trial,
					   trialinvariant::Trialinvariant)
	@unpack clicks = trial
	@unpack Aᵃsilent, Aᶜᵀ, Δt, πᶜᵀ, 𝛏, Ξ = trialinvariant
	C = adapt(clicks, θnative.k[1], θnative.ϕ[1])
	μ = θnative.μ₀[1] + trial.previousanswer*θnative.wₕ[1]
	σ = √θnative.σ²ᵢ[1]
	πᵃ = probabilityvector(μ, σ, 𝛏)
	f = p𝐘𝑑[1] .* πᵃ .* πᶜᵀ
	D = sum(f)
	f /= D
	ℓ = log(D)
	T = eltype(p𝐘𝑑[1])
	Aᵃ = zeros(T, Ξ, Ξ)
	@inbounds for t = 2:trial.ntimesteps
		if isempty(clicks.inputindex[t])
			f = Aᵃsilent * f * Aᶜᵀ
		else
			cL = sum(C[clicks.left[t]])
			cR = sum(C[clicks.right[t]])
			stochasticmatrix!(Aᵃ, cL, cR, trialinvariant, θnative)
			f = Aᵃ * f * Aᶜᵀ
		end
		f .*= p𝐘𝑑[t]
		D = sum(f)
		f /= D
		ℓ += log(D)
	end
	return ℓ
end

"""
    ∇negativeloglikelihood!(∇, γ, model, shared, concatenatedθ)

Gradient of the negative log-likelihood of the factorial hidden Markov drift-diffusion model

MODIFIED INPUT
-`∇`: a vector of partial derivatives
-`γ`: posterior probability of the latent variables
-`model`: a structure containing information of the factorial hidden Markov drift-diffusion model
-`shared`: structure containing variables shared between computations of the model's log-likelihood and its gradient

UNMODIFIED ARGUMENT
-`concatenatedθ`: parameter values concatenated into a vetor
"""
function ∇negativeloglikelihood!(∇::Vector{<:AbstractFloat},
								 γ::Vector{<:Matrix{<:Vector{<:AbstractFloat}}},
								 model::Model,
								 shared::Shared,
								 concatenatedθ::Vector{<:AbstractFloat})
	if concatenatedθ != shared.concatenatedθ
		update!(model, shared, concatenatedθ)
	end
	@unpack indexθ, p𝐘𝑑 = shared
	@unpack options, θnative, θreal, trialsets = model
	@unpack K = options
	trialinvariant = Trialinvariant(options, θnative; purpose="gradient")
	output=	map(trialsets, p𝐘𝑑) do trialset, p𝐘𝑑
				pmap(trialset.trials, p𝐘𝑑) do trial, p𝐘𝑑
					∇loglikelihood(p𝐘𝑑, trialinvariant, θnative, trial)
				end
			end
	latent∇ = output[1][1][1] # reuse this memory
	for field in fieldnames(Latentθ)
		latent∂ = getfield(latent∇, field)
		for i in eachindex(output)
			start = i==1 ? 2 : 1
			for m in start:length(output[i])
				latent∂[1] += getfield(output[i][m][1], field)[1] #output[i][m][1] are the partial derivatives
			end
		end
	end
	latent∇.B[1] *= θnative.B[1]*logistic(-θreal.B[1])
	latent∇.k[1] *= θnative.k[1]
	latent∇.ϕ[1] *= θnative.ϕ[1]*(1.0 - θnative.ϕ[1])
	latent∇.σ²ₐ[1] *= θnative.σ²ₐ[1]
	latent∇.σ²ᵢ[1] *= θnative.σ²ᵢ[1]
	latent∇.σ²ₛ[1] *= θnative.σ²ₛ[1]
	for field in fieldnames(Latentθ)
		index = getfield(indexθ.latentθ,field)[1]
		if index != 0
			∇[index] = -getfield(latent∇,field)[1] # note the negative sign
		end
	end
	@inbounds for i in eachindex(output)
        t = 0
        for m in eachindex(output[i])
            for tₘ in eachindex(output[i][m][2]) # output[i][m][2] is `fb` of trial `m` in trialset `i`
                t += 1
                for jk in eachindex(output[i][m][2][tₘ])
                    γ[i][jk][t] = output[i][m][2][tₘ][jk]
                end
            end
        end
    end
	Pᵤ = length(trialsets[1].mpGLMs[1].𝐮)
	Pₗ = length(trialsets[1].mpGLMs[1].𝐥)
	for i in eachindex(trialsets)
		∇𝐰 = pmap(mpGLM->∇negativeexpectation(γ[i], mpGLM), trialsets[i].mpGLMs)
		for n in eachindex(trialsets[i].mpGLMs)
			∇[indexθ.𝐮[i][n]] .= ∇𝐰[n][1:Pᵤ]
			∇[indexθ.𝐥[i][n]] .= ∇𝐰[n][Pᵤ+1:Pᵤ+Pₗ]
			∇[indexθ.𝐫[i][n]] .= ∇𝐰[n][Pᵤ+Pₗ+1:end]
		end
	end
	return nothing
end

"""
	∇loglikelihood(p𝐘𝑑, trialinvariant, θnative, trial)

Compute quantities needed for the gradient of the log-likelihood of the data observed in one trial

ARGUMENT
-`p𝐘𝑑`: A vector of matrices of floating-point numbers whose element `p𝐘𝑑[t][i,j]` represents the likelihood of the emissions (spike trains and choice) at time step `t` conditioned on the accumulator variable being in state `i` and the coupling variable in state `j`
-`trialinvariant`: structure containing quantities used across trials
-`θnative`: parameters for the latent variables in their native space
-`trial`: information on the stimuli and behavioral choice of one trial

RETURN
-`latent∇`: gradient of the log-likelihood of the data observed in one trial with respect to the parameters specifying the latent variables
-`fb`: joint posterior probabilities of the accumulator and coupling variables
"""
function ∇loglikelihood(p𝐘𝑑::Vector{<:Matrix{<:AbstractFloat}},
						trialinvariant::Trialinvariant,
						θnative::Latentθ,
						trial::Trial)
	@unpack clicks = trial
	@unpack inputtimesteps, inputindex = clicks
	@unpack Aᵃsilent, dAᵃsilentdμ, dAᵃsilentdσ², dAᵃsilentdB, Aᶜ, Aᶜᵀ, Δt, K, 𝛚, πᶜᵀ, Ξ, 𝛏 = trialinvariant
	dℓdk, dℓdλ, dℓdϕ, dℓdσ²ₐ, dℓdσ²ₛ, dℓdB = 0., 0., 0., 0., 0., 0.
	∑χᶜ = zeros(K,K)
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
	D, f = forward(Aᵃ, inputindex, πᵃ, p𝐘𝑑, trialinvariant)
	fb = f # reuse memory
	b = ones(Ξ,K)
	Aᶜreshaped = reshape(Aᶜ, 1, 1, K, K)
	λΔt = θnative.λ[1]*Δt
	expλΔt = exp(λΔt)
	dμdΔc = (expλΔt - 1.0)/λΔt
	η = (expλΔt - dμdΔc)/θnative.λ[1]
	𝛏ᵀΔtexpλΔt = transpose(𝛏)*Δt*expλΔt
	@inbounds for t = trial.ntimesteps:-1:1
		if t < trial.ntimesteps # backward step
			Aᵃₜ₊₁ = isempty(inputindex[t+1]) ? Aᵃsilent : Aᵃ[inputindex[t+1][1]]
			b .*= p𝐘𝑑[t+1]
			b = transpose(Aᵃₜ₊₁) * b * Aᶜ / D[t+1]
			fb[t] .*= b
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
			χ_oslash_Aᵃ = reshape(p𝐘𝑑[t].*b, Ξ, 1, K, 1) .* reshape(f[t-1], 1, Ξ, 1, K) .* Aᶜreshaped ./ D[t]
	        ∑χᶜ += dropdims(sum(χ_oslash_Aᵃ.*Aᵃₜ, dims=(1,2)); dims=(1,2))
			χᵃ_Aᵃ = dropdims(sum(χ_oslash_Aᵃ, dims=(3,4)); dims=(3,4))
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
	dℓdAᶜ₁₁ = ∑χᶜ[1,1]*Aᶜ[2,1] - ∑χᶜ[2,1]*Aᶜ[1,1]
	dℓdAᶜ₂₂ = ∑χᶜ[2,2]*Aᶜ[1,2] - ∑χᶜ[1,2]*Aᶜ[2,2]
	∑γᶜ₁ = sum(fb[1], dims=1)
	dℓdxπᶜ₁ = ∑γᶜ₁[1] - θnative.πᶜ₁[1]
	γᵃ₁_oslash_πᵃ = sum(p𝐘𝑑[1] .* πᶜᵀ ./ D[1] .* b, dims=2)
	∑_γᵃ₁_dlogπᵃdμ = γᵃ₁_oslash_πᵃ ⋅ dπᵃdμ # similar to above, γᵃ₁⊙ d/dμ{log(πᵃ)} = γᵃ₁⊘ πᵃ⊙ d/dμ{πᵃ}
	dℓdμ₀ = ∑_γᵃ₁_dlogπᵃdμ
	dℓdwₕ = ∑_γᵃ₁_dlogπᵃdμ * trial.previousanswer
	dℓdσ²ᵢ = γᵃ₁_oslash_πᵃ ⋅ dπᵃdσ²
	dℓdB += γᵃ₁_oslash_πᵃ ⋅ dπᵃdB
	dℓdxψ = differentiateℓ_wrt_xψ(trial.choice, f[end], θnative.ψ[1])
	latent∇ = Latentθ(	Aᶜ₁₁ = [dℓdAᶜ₁₁],
						Aᶜ₂₂ = [dℓdAᶜ₂₂],
						k	 = [dℓdk],
						λ	 = [dℓdλ],
						μ₀	 = [dℓdμ₀],
						ϕ	 = [dℓdϕ],
						πᶜ₁	 = [dℓdxπᶜ₁],
						ψ	 = [dℓdxψ],
						σ²ₐ	 = [dℓdσ²ₐ],
						σ²ᵢ	 = [dℓdσ²ᵢ],
						σ²ₛ	 = [dℓdσ²ₛ],
						wₕ	 = [dℓdwₕ],
						B	 = [dℓdB])
	return latent∇, fb
end

"""
	forward(Aᵃ, inputindex, πᵃ, p𝐘d, trialinvariant)

Forward pass of the forward-backward algorithm

ARGUMENT
-`Aᵃ`: transition probabilities of the accumulator variable. Aᵃ[t][j,k] ≡ p(aₜ=ξⱼ ∣ aₜ₋₁=ξₖ)
`inputindex`: index of the time steps with auditory input. For time step `t`, if the element `inputindex[t]` is nonempty, then `Aᵃ[inputindex[t][1]]` is the transition matrix for that time step. If `inputindex[t]` is empty, then the corresponding transition matrix is `Aᵃsilent`.
-`πᵃ`: a vector of floating-point numbers specifying the prior probability of each accumulator state
-`p𝐘𝑑`: likelihood of the emissions in each time bin in this trial. p𝐘𝑑[t][j,k] = ∏ₙ p(𝐲ₙ(t) ∣ 𝑎ₜ=ξⱼ, 𝑧ₜ=k) and p𝐘𝑑[end][j,k] = p(𝑑∣ 𝑎ₜ=ξⱼ, 𝑧ₜ=k) ∏ₙ p(𝐲ₙ(t) ∣ 𝑎ₜ=ξⱼ, 𝑧ₜ=k)
-`trialinvariant`: a structure containing quantities that are used in each trial

RETURN
-`D`: scaling factors with element `D[t]` ≡ p(𝐘ₜ ∣ 𝐘₁, ... 𝐘ₜ₋₁)
-`f`: Forward recursion terms. `f[t][j,k]` ≡ p(aₜ=ξⱼ, zₜ=k ∣ 𝐘₁, ... 𝐘ₜ) where 𝐘 refers to all the spike trains

"""
function forward(Aᵃ::Vector{<:Matrix{<:AbstractFloat}},
 				 inputindex::Vector{<:Vector{<:Integer}},
				 πᵃ::Vector{<:AbstractFloat},
				 p𝐘𝑑::Vector{<:Matrix{<:AbstractFloat}},
				 trialinvariant::Trialinvariant)
	@unpack Aᵃsilent, Aᶜᵀ, K, πᶜᵀ, Ξ, 𝛏 = trialinvariant
	ntimesteps = length(inputindex)
	f = map(x->zeros(Ξ,K), 1:ntimesteps)
	D = zeros(ntimesteps)
	f[1] = p𝐘𝑑[1] .* πᵃ .* πᶜᵀ
	D[1] = sum(f[1])
	f[1] /= D[1]
	@inbounds for t = 2:ntimesteps
		if isempty(inputindex[t])
			Aᵃₜ = Aᵃsilent
		else
			i = inputindex[t][1]
			Aᵃₜ = Aᵃ[i]
		end
		f[t] = Aᵃₜ * f[t-1] * Aᶜᵀ
		f[t] .*= p𝐘𝑑[t]
		D[t] = sum(f[t])
		f[t] /= D[t]
	end
	return D,f
end

"""
	differentiateℓwrtψ(choice, γ_end, ψ)

Partial derivative of the log-likelihood of the data from one trial with respect to the lapse rate ψ

ARGUMENT
-`choice`: a Boolean specifying whether the choice was to the right
-`γ_end`: a matrix of floating-point numbers representing the posterior likelihood of the latent variables at the end of the trial (i.e., last time step). Element `γ_end[i,j]` = p(aᵢ=1, cⱼ=1 ∣ 𝐘, d). Rows correspond to states of the accumulator state variable 𝐚, and columns to states of the coupling variable 𝐜.
-`ψ`: a floating-point number specifying the lapse rate

RETURN
-a floating-point number quantifying the partial derivative of the log-likelihood of one trial's data with respect to the lapse rate ψ
"""
function differentiateℓwrtψ(choice::Bool, γ_end::Array{<:AbstractFloat}, ψ::AbstractFloat)
	γᵃ_end = sum(γ_end, dims=2)
	zeroindex = cld(length(γᵃ_end), 2)
	# γᵃ_end0_div2 = γᵃ_end[zeroindex]/2
	if choice
		choiceconsistent   = sum(γᵃ_end[zeroindex+1:end])
		choiceinconsistent = sum(γᵃ_end[1:zeroindex-1])
	else
		choiceconsistent   = sum(γᵃ_end[1:zeroindex-1])
		choiceinconsistent = sum(γᵃ_end[zeroindex+1:end])
	end
	return choiceconsistent/(ψ-2) + choiceinconsistent/ψ
end

"""
	differentiateℓ_wrt_xψ(choice, γ_end, ψ)

Partial derivative of the log-likelihood of the data from one trial with respect to the lapse rate ψ in real space

ARGUMENT
-`choice`: a Boolean specifying whether the choice was to the right
-`γ_end`: a matrix of floating-point numbers representing the posterior likelihood of the latent variables at the end of the trial (i.e., last time step). Element `γ_end[i,j]` = p(aᵢ=1, cⱼ=1 ∣ 𝐘, d). Rows correspond to states of the accumulator state variable 𝐚, and columns to states of the coupling variable 𝐜.
-`ψ`: a floating-point number specifying the lapse rate

RETURN
-a floating-point number quantifying the partial derivative of the log-likelihood of one trial's data with respect to the lapse rate ψ
"""
function differentiateℓ_wrt_xψ(choice::Bool, γ_end::Array{<:AbstractFloat}, ψ::AbstractFloat)
	γᵃ_end = sum(γ_end, dims=2)
	zeroindex = cld(length(γᵃ_end), 2)
	# γᵃ_end0_div2 = γᵃ_end[zeroindex]/2
	if choice
		choiceconsistent   = sum(γᵃ_end[zeroindex+1:end])
		choiceinconsistent = sum(γᵃ_end[1:zeroindex-1])
	else
		choiceconsistent   = sum(γᵃ_end[1:zeroindex-1])
		choiceinconsistent = sum(γᵃ_end[zeroindex+1:end])
	end
	return (1-ψ)*(choiceconsistent*ψ/(ψ-2) + choiceinconsistent)
end

"""
	Trialinvariant(options, θnative)

Compute quantities that are used in each trial for computing gradient of the log-likelihood

ARGUMENT
-`options`: model settings
-`θnative`: model parameters in their native space
"""
function Trialinvariant(options::Options, θnative::Latentθ; purpose="gradient")
	@unpack Δt, K, Ξ = options
	λ = θnative.λ[1]
	B = θnative.B[1]
	Aᶜ₁₁ = θnative.Aᶜ₁₁[1]
	Aᶜ₂₂ = θnative.Aᶜ₂₂[1]
	πᶜ₁ = θnative.πᶜ₁[1]
	Aᶜ = [Aᶜ₁₁ 1-Aᶜ₂₂; 1-Aᶜ₁₁ Aᶜ₂₂]
	Aᶜᵀ = [Aᶜ₁₁ 1-Aᶜ₁₁; 1-Aᶜ₂₂ Aᶜ₂₂]
	πᶜᵀ = [πᶜ₁ 1-πᶜ₁]
	𝛏 = B*(2collect(1:Ξ) .- Ξ .- 1)/(Ξ-2)
	𝛍 = conditionedmean(0.0, Δt, θnative.λ[1], 𝛏)
	σ = √(θnative.σ²ₐ[1]*Δt)
	Aᵃsilent = zeros(eltype(𝛍),Ξ,Ξ)
	if purpose=="gradient"
		𝛚 = (2collect(1:Ξ) .- Ξ .- 1)/2
		Ω = 𝛚 .- transpose(𝛚).*exp.(λ.*Δt)
		dAᵃsilentdμ = zeros(Ξ,Ξ)
		dAᵃsilentdσ² = zeros(Ξ,Ξ)
		dAᵃsilentdB = zeros(Ξ,Ξ)
		stochasticmatrix!(Aᵃsilent, dAᵃsilentdμ, dAᵃsilentdσ², dAᵃsilentdB, 𝛍, σ, Ω, 𝛏)
		Trialinvariant( Aᵃsilent=Aᵃsilent,
					Aᶜ=Aᶜ,
					Aᶜᵀ=Aᶜᵀ,
					dAᵃsilentdμ=dAᵃsilentdμ,
					dAᵃsilentdσ²=dAᵃsilentdσ²,
					dAᵃsilentdB=dAᵃsilentdB,
					Δt=options.Δt,
					𝛚=𝛚,
					Ω=Ω,
					πᶜᵀ=πᶜᵀ,
					Ξ=Ξ,
 				    K=K,
					𝛏=𝛏)
	elseif purpose=="loglikelihood"
		stochasticmatrix!(Aᵃsilent, 𝛍, σ, 𝛏)
		Trialinvariant(Aᵃsilent=Aᵃsilent,
				   Aᶜᵀ=Aᶜᵀ,
				   Δt=options.Δt,
				   πᶜᵀ=πᶜᵀ,
				   𝛏=𝛏,
				   K=K,
				   Ξ=Ξ)
	end
end

"""
    likelihood!(p𝐘𝑑, trialset, ψ)

Update the conditional likelihood of the emissions (spikes and/or behavioral choice)

MODIFIED ARGUMENT
-`p𝐘𝑑`: Conditional probability of the emissions (spikes and/or choice) at each time bin. For time bins of each trial other than the last, it is the product of the conditional likelihood of all spike trains. For the last time bin, it corresponds to the product of the conditional likelihood of the spike trains and the choice. Element p𝐘𝑑[i][m][t][j,k] corresponds to ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ, zₜ=k) across N neural units at the t-th time bin in the m-th trial of the i-th trialset. The last element p𝐘𝑑[i][m][end][j,k] of each trial corresponds to p(𝑑 | aₜ = ξⱼ, zₜ=k) ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ, zₜ=k)

UNMODIFIED ARGUMENT
-`trialsets`: data used to constrain the model
-`ψ`: lapse rate

RETURN
-`nothing`
"""
function likelihood!(p𝐘𝑑::Vector{<:Vector{<:Vector{<:Matrix{<:AbstractFloat}}}},
                     trialsets::Vector{<:Trialset},
                     ψ::Real)
	Ξ = size(p𝐘𝑑[1][1][end],1)
	K = size(p𝐘𝑑[1][1][end],2)
	zeroindex = cld(Ξ,2)
    @inbounds for i in eachindex(p𝐘𝑑)
		N = length(trialsets[i].mpGLMs)
		𝐩decoupled = likelihood(trialsets[i].mpGLMs[1], zeroindex, 2)
		for n = 2:N
			likelihood!(𝐩decoupled, trialsets[i].mpGLMs[n], zeroindex, 2)
		end
	    for j = 1:Ξ
	        for k = 1:K
	            if k == 2 || j==zeroindex
					𝐩 = 𝐩decoupled
				else
					𝐩 = likelihood(trialsets[i].mpGLMs[1], j, k)
		            for n = 2:N
					    likelihood!(𝐩, trialsets[i].mpGLMs[n], j, k)
		            end
				end
	            t = 0
	            for m in eachindex(p𝐘𝑑[i])
	                for tₘ in eachindex(p𝐘𝑑[i][m])
	                    t += 1
	                    p𝐘𝑑[i][m][tₘ][j,k] = 𝐩[t]
	                end
	            end
	        end
	    end
		for m in eachindex(p𝐘𝑑[i])
			likelihood!(p𝐘𝑑[i][m][end], trialsets[i].trials[m].choice, ψ; zeroindex=zeroindex)
		end
    end
    return nothing
end

"""
    likelihood!(p𝐘ₜ𝑑, choice, ψ)

Multiply against the conditional probability of a right choice given the state of the accumulator

MODIFIED ARGUMENT
-`p𝐘ₜ𝑑`: A matrix whose element p𝐘ₜ𝑑[j,k] ≡ p(𝐘ₜ, 𝑑 ∣ aₜ = ξⱼ, zₜ = k) for time bin t that is the at the end of the trial

UNMODIFIED ARGUMENT
-`choice`: the observed choice, either right (`choice`=true) or left.
-`ψ`: the prior probability of a lapse state

OPTIONAL ARGUMENT
- `zeroindex`: the index of the bin for which the accumulator variable equals zero
"""
function likelihood!(p𝐘ₜ𝑑,
		             choice::Bool,
		             ψ::Real;
		             zeroindex=cld(size(p𝐘ₜ𝑑,1),2))
    p𝐘ₜ𝑑[zeroindex,:] .*= 0.5
    if choice
        p𝐘ₜ𝑑[1:zeroindex-1,:] .*= ψ/2
        p𝐘ₜ𝑑[zeroindex+1:end,:] .*= 1-ψ/2
    else
        p𝐘ₜ𝑑[1:zeroindex-1,:]   .*= 1-ψ/2
        p𝐘ₜ𝑑[zeroindex+1:end,:] .*= ψ/2
    end
    return nothing
end

"""
	sortparameters!(model, concatenatedθ, indexθ)

Sort a vector of concatenated parameter values and convert the values from real space to native space

MODIFIED ARGUMENT
-`model`: a factorial hidden Markov drift-diffusion model

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values
"""
function sortparameters!(model::Model,
				 		 concatenatedθ::Vector{<:AbstractFloat},
				 		 indexθ::Indexθ)
	@unpack options, θnative, θreal, trialsets = model
	for field in fieldnames(Latentθ) # `Latentθ` is the type of `indexθ.latentθ`
		index = getfield(indexθ.latentθ, field)[1]
		if index != 0 # an index of 0 indicates that the parameter is not being fit
			getfield(θreal, field)[1] = concatenatedθ[index]
		end
	end
	for field in (:𝐮, :𝐥, :𝐫)
		index = getfield(indexθ, field)
		for i in eachindex(index)
			for n in eachindex(index[i])
				if !isempty(index[i][n])
					getfield(trialsets[i].mpGLMs[n], field) .= concatenatedθ[index[i][n]]
				end
			end
		end
	end
	real2native!(θnative, options, θreal)
	return nothing
end

"""
    concatenateparameters(model)

Concatenate values of parameters being fitted into a vector of floating point numbers

ARGUMENT
-`model`: the factorial hidden Markov drift-diffusion model

RETURN
-`concatenatedθ`: a vector of the concatenated values of the parameters being fitted
-`indexθ`: a structure indicating the index of each model parameter in the vector of concatenated values
"""
function concatenateparameters(model::Model)
    @unpack options, θreal, trialsets = model
	concatenatedθ = zeros(0)
    counter = 0
	latentθ = Latentθ(collect(zeros(Int64,1) for i in fieldnames(Latentθ))...)
	tofit = true
	for field in fieldnames(Latentθ)
		if field == :Aᶜ₁₁ || field == :Aᶜ₂₂ || field == :πᶜ₁
			tofit = options.K == 2
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
    𝐮 = map(trialset->map(mpGLM->zeros(Int, length(mpGLM.𝐮)), trialset.mpGLMs), trialsets)
    𝐥 = map(trialset->map(mpGLM->zeros(Int, length(mpGLM.𝐥)), trialset.mpGLMs), trialsets)
    𝐫 = map(trialset->map(mpGLM->zeros(Int, length(mpGLM.𝐫)), trialset.mpGLMs), trialsets)
    for i in eachindex(trialsets)
        for n in eachindex(trialsets[i].mpGLMs)
            concatenatedθ = vcat(concatenatedθ, trialsets[i].mpGLMs[n].𝐮)
            p = length(trialsets[i].mpGLMs[n].𝐮)
            𝐮[i][n] = collect(counter+1:counter+p)
            counter += p
            concatenatedθ = vcat(concatenatedθ, trialsets[i].mpGLMs[n].𝐥)
            p = length(trialsets[i].mpGLMs[n].𝐥)
            𝐥[i][n] = collect(counter+1:counter+p)
            counter += p
            concatenatedθ = vcat(concatenatedθ, trialsets[i].mpGLMs[n].𝐫)
            p = length(trialsets[i].mpGLMs[n].𝐫)
            𝐫[i][n] = collect(counter+1:counter+p)
            counter += p
        end
    end
    indexθ = Indexθ(latentθ=latentθ,
					𝐮=𝐮,
					𝐥=𝐥,
					𝐫=𝐫)
    return concatenatedθ, indexθ
end

"""
    concatenatebounds(options, θindices; boundtype)

Concatenate lower and upper bounds of the parameters being fitted

INPUT
-`indexθ`: a structure indicating the index of each model parameter in the vector of concatenated values
-`options`: settings of the model

RETURN
-two vectors representing the lower and upper bounds of the parameters being fitted, respectively
"""
function concatenatebounds(indexθ::Indexθ, options::Options)
	lowerbounds, upperbounds = zeros(0), zeros(0)
	for field in fieldnames(typeof(indexθ.latentθ))
		if getfield(indexθ.latentθ, field)[1] != 0
			field_in_options = Symbol("bounds_"*String(field))
			bounds = getfield(options, field_in_options)
			lowerbounds = vcat(lowerbounds, bounds[1])
			upperbounds = vcat(upperbounds, bounds[2])
		end
	end
	nweights = 0
	for i in eachindex(indexθ.𝐮)
		for n in eachindex(indexθ.𝐮[i])
			nweights += length(indexθ.𝐮[i][n]) +
				  		length(indexθ.𝐥[i][n]) +
						length(indexθ.𝐫[i][n])
		end
	end
	lowerbounds = vcat(lowerbounds, -Inf*ones(nweights))
	upperbounds = vcat(upperbounds,  Inf*ones(nweights))
	return lowerbounds, upperbounds
end

"""
	Shared(model)

Create variables that are shared by the computations of the log-likelihood and its gradient

ARGUMENT
-`model`: structure with information about the factorial hidden Markov drift-diffusion model

OUTPUT
-an instance of the custom type `Shared`, which contains the shared quantities
"""
function Shared(model::Model)
	@unpack K, Ξ = model.options
	p𝐘𝑑=map(model.trialsets) do trialset
			map(trialset.trials) do trial
				map(1:trial.ntimesteps) do t
					ones(Ξ,K)
				end
			end
		end
	likelihood!(p𝐘𝑑, model.trialsets, model.θnative.ψ[1])
	concatenatedθ, indexθ = concatenateparameters(model)
	Shared(	concatenatedθ=concatenatedθ,
			indexθ=indexθ,
			p𝐘𝑑=p𝐘𝑑)
end

"""
	update!(model, shared, concatenatedθ)

Update the model and the shared quantities according to new parameter values

ARGUMENT
-`model`: structure with information concerning a factorial hidden Markov drift-diffusion model
-`shared`: structure containing variables shared between computations of the model's log-likelihood and its gradient
-`concatenatedθ`: newest values of the model's parameters
"""
function update!(model::Model,
				 shared::Shared,
				 concatenatedθ::Vector{<:Real})
	shared.concatenatedθ .= concatenatedθ
	sortparameters!(model, shared.concatenatedθ, shared.indexθ)
	if !isempty(shared.p𝐘𝑑[1][1][1])
	    likelihood!(shared.p𝐘𝑑, model.trialsets, model.θnative.ψ[1])
	end
	return nothing
end
