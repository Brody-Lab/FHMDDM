"""
	likelihood(model)

Compute the likelihood of the emissions at each timestep

ARGUMENT
-`model`: custom type containing the settings, data, and parameters of a factorial hidden Markov drift-diffusion model

RETURN
-`p𝐘𝑑`: Conditional probability of the emissions (spikes and/or choice) at each time bin. For time bins of each trial other than the last, it is the product of the conditional likelihood of all spike trains. For the last time bin, it corresponds to the product of the conditional likelihood of the spike trains and the choice. Element p𝐘𝑑[i][m][t][j,k] corresponds to ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ, zₜ=k) across N neural units at the t-th time bin in the m-th trial of the i-th trialset. The last element p𝐘𝑑[i][m][end][j,k] of each trial corresponds to p(𝑑 | aₜ = ξⱼ, zₜ=k) ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ, zₜ=k)
"""
function likelihood(model::Model)
	@unpack K, Ξ = model.options
	T = eltype(model.trialsets[1].mpGLMs[1].θ.𝐮)
	p𝐘𝑑=map(model.trialsets) do trialset
			map(trialset.trials) do trial
				map(1:trial.ntimesteps) do t
					ones(T,Ξ,K)
				end
			end
		end
	likelihood!(p𝐘𝑑, model.trialsets, model.θnative.ψ[1])
	return p𝐘𝑑
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
function likelihood!(p𝐘𝑑::Vector{<:Vector{<:Vector{<:Matrix{<:Real}}}},
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
function likelihood!(p𝐘ₜ𝑑::Matrix{<:Real},
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
function forward(Aᵃ::Vector{<:Matrix{type}},
 				 inputindex::Vector{<:Vector{<:Integer}},
				 πᵃ::Vector{<:Real},
				 p𝐘𝑑::Vector{<:Matrix{<:Real}},
				 trialinvariant::Trialinvariant) where {type<:Real}
	@unpack Aᵃsilent, Aᶜᵀ, K, πᶜᵀ, Ξ, 𝛏 = trialinvariant
	ntimesteps = length(inputindex)
	f = map(x->zeros(type,Ξ,K), 1:ntimesteps)
	D = zeros(type,ntimesteps)
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
	posteriors(model)

Compute the joint posteriors of the latent variables at each time step.

ARGUMENT
-`model`: custom type containing the settings, data, and parameters of a factorial hidden Markov drift-diffusion model

RETURN
-`γ`: joint posterior probabilities of the accumulator and coupling variables at each time step conditioned on the emissions at all time steps in the trialset. Element `γ[i][j,k][t]` represent the posterior probability at the t-th timestep in the i-th trialset: p(aₜⱼ=1, cₜₖ=1 ∣ 𝐘, 𝑑)
"""
function posteriors(model::Model)
	@unpack options, θnative, trialsets = model
	@unpack K, Ξ = options
	trialinvariant = Trialinvariant(model; purpose="gradient")
	p𝐘𝑑 = likelihood(model)
	fb = map(trialsets, p𝐘𝑑) do trialset, p𝐘𝑑
			pmap(trialset.trials, p𝐘𝑑) do trial, p𝐘𝑑
				posteriors(p𝐘𝑑, θnative, trial, trialinvariant)
			end
		end
	type = typeof(fb[1][1][1][1,1])
	γ =	map(trialsets) do trialset
			map(CartesianIndices((Ξ,K))) do index
				zeros(type, trialset.ntimesteps)
			end
		end
	@inbounds for i in eachindex(fb)
        t = 0
        for m in eachindex(fb[i])
            for tₘ in eachindex(fb[i][m])
                t += 1
                for jk in eachindex(fb[i][m][tₘ])
                	γ[i][jk][t] = fb[i][m][tₘ][jk]
                end
            end
        end
    end
	return γ, fb
end

"""
	posteriors(p𝐘𝑑, trialinvariant, θnative, trial)

Compute the joint posteriors of the latent variables at each time step.

ARGUMENT
-`p𝐘𝑑`: A vector of matrices of floating-point numbers whose element `p𝐘𝑑[t][i,j]` represents the likelihood of the emissions (spike trains and choice) at time step `t` conditioned on the accumulator variable being in state `i` and the coupling variable in state `j`
-`trialinvariant`: structure containing quantities used across trials
-`θnative`: parameters for the latent variables in their native space
-`trial`: information on the stimuli and behavioral choice of one trial

RETURN
-`fb`: joint posterior probabilities of the accumulator and coupling variables at each time step conditioned on the emissions at all time steps in the trial. Element `fb[t][j,k]` represent the posterior probability at the t-th timestep: p(aₜⱼ=1, cₜₖ=1 ∣ 𝐘, 𝑑)
"""
function posteriors(p𝐘𝑑::Vector{<:Matrix{type}},
					θnative::Latentθ,
					trial::Trial,
					trialinvariant::Trialinvariant) where {type<:Real}
	@unpack clicks = trial
	@unpack inputtimesteps, inputindex = clicks
	@unpack Aᵃsilent, Aᶜ, K, Ξ, 𝛏 = trialinvariant
	μ = θnative.μ₀[1] + trial.previousanswer*θnative.wₕ[1]
	σ = √θnative.σ²ᵢ[1]
	πᵃ = probabilityvector(μ, σ, 𝛏)
	n_steps_with_input = length(clicks.inputtimesteps)
	Aᵃ = map(x->zeros(type,Ξ,Ξ), clicks.inputtimesteps)
	C = adapt(clicks, θnative.k[1], θnative.ϕ[1])
	for i in 1:n_steps_with_input
		t = clicks.inputtimesteps[i]
		cL = sum(C[clicks.left[t]])
		cR = sum(C[clicks.right[t]])
		stochasticmatrix!(Aᵃ[i], cL, cR, trialinvariant, θnative)
	end
	D, fb = forward(Aᵃ, inputindex, πᵃ, p𝐘𝑑, trialinvariant)
	b = ones(type,Ξ,K)
	@inbounds for t = trial.ntimesteps-1:-1:1
		Aᵃₜ₊₁ = isempty(inputindex[t+1]) ? Aᵃsilent : Aᵃ[inputindex[t+1][1]]
		b .*= p𝐘𝑑[t+1]
		b = (transpose(Aᵃₜ₊₁) * b * Aᶜ) ./ D[t+1]
		fb[t] .*= b
	end
	return fb
end

"""
	choiceposteriors(model)

Compute the joint posteriors of the latent variables conditioned on only the choice at each time step.

ARGUMENT
-`model`: custom type containing the settings, data, and parameters of a factorial hidden Markov drift-diffusion model

RETURN
-`γ`: joint posterior probabilities of the accumulator and coupling variables at each time step conditioned on the choices (𝑑). Element `γ[i][j,k][t]` represent the posterior probability at the t-th timestep in the i-th trialset: p(aₜⱼ=1, cₜₖ=1 ∣ 𝑑)
"""
function choiceposteriors(model::Model)
	@unpack options, θnative, trialsets = model
	@unpack K, Ξ = options
	trialinvariant = Trialinvariant(model; purpose="gradient")
	p𝐘𝑑 = choicelikelihood(model)
	fb = map(trialsets, p𝐘𝑑) do trialset, p𝐘𝑑
			pmap(trialset.trials, p𝐘𝑑) do trial, p𝐘𝑑
				posteriors(p𝐘𝑑, θnative, trial, trialinvariant)
			end
		end
	γ =	map(trialsets) do trialset
			map(CartesianIndices((Ξ,K))) do index
				zeros(trialset.ntimesteps)
			end
		end
	@inbounds for i in eachindex(fb)
        t = 0
        for m in eachindex(fb[i])
            for tₘ in eachindex(fb[i][m])
                t += 1
                for jk in eachindex(fb[i][m][tₘ])
                	γ[i][jk][t] = fb[i][m][tₘ][jk]
                end
            end
        end
    end
	return γ
end

"""
	choicelikelihood(model)

Compute the likelihood of the choice at each timestep

ARGUMENT
-`model`: custom type containing the settings, data, and parameters of a factorial hidden Markov drift-diffusion model

RETURN
-`p𝐘𝑑`: Conditional probability of the emissions (spikes and/or choice) at each time bin. For time bins of each trial other than the last, it is the product of the conditional likelihood of all spike trains. For the last time bin, it corresponds to the product of the conditional likelihood of the spike trains and the choice. Element p𝐘𝑑[i][m][t][j,k] corresponds to ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ, zₜ=k) across N neural units at the t-th time bin in the m-th trial of the i-th trialset. The last element p𝐘𝑑[i][m][end][j,k] of each trial corresponds to p(𝑑 | aₜ = ξⱼ, zₜ=k) ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ, zₜ=k)
"""
function choicelikelihood(model::Model)
	@unpack options, trialsets = model
	@unpack K, Ξ = options
	p𝐘𝑑=map(model.trialsets) do trialset
			map(trialset.trials) do trial
				map(1:trial.ntimesteps) do t
					ones(Ξ,K)
				end
			end
		end
	zeroindex = cld(Ξ, 2)
	@inbounds for i in eachindex(p𝐘𝑑)
		for m in eachindex(p𝐘𝑑[i])
			likelihood!(p𝐘𝑑[i][m][end], trialsets[i].trials[m].choice, model.θnative.ψ[1]; zeroindex=zeroindex)
		end
    end
	return p𝐘𝑑
end

"""
	posteriorcoupled(model)

Posterior probability of being coupled

ARGUMENT
-`model`: instance of the factorial hidden markov drift diffusion model

OUTPUT
-`fbz`: a nested array whose element `fbz[i][m][t]` represents the posterior porobability that the neural population is coupled to the accumulator in the timestep t of trial m of trialset i.
"""
function posteriorcoupled(model::Model)
	γ, fb = posteriors(model)
	map(fb) do fb # trialset
		map(fb) do fb # trial
			map(fb) do fb #timestep
				sum(fb[:,1])
			end
		end
	end
end
