"""
Gradient of the joint posterior probability of the latent variables

The gradient is computed for the m-th trial of the i-th trialset, the t-th timestep in that trialet, and for accumulator state iᵃ and coupling variable state iᶜ:
    ∇p(a = ξ(iᵃ), c = iᶜ ∣ 𝐘ᵢₘ, dᵢₘ)

The recommended dataset should have a few number of trials

ARGUMENT
-`i`: trialset
-`m`: trial
-`t`: time step
-`iᵃ`: index of the accumulator
-`iᶜ`: index of the coupling variable
-`model`: structure containing the data, parameters, and hyperparameters of a factorial hidden-Markov drift-diffusion model

RETURN
-a vector corresponding to the gradient of the posterior probability
"""
function ∇posterior(i::Integer, m::Integer, t::Integer, iᵃ::Integer, iᶜ::Integer, model::Model)
    @unpack options, θnative, θreal, trialsets = model
	@unpack Ξ, K = options
	trialinvariant = Trialinvariant(model; purpose="gradient")
	p𝐘𝑑=map(model.trialsets) do trialset
			map(trialset.trials) do trial
				map(1:trial.ntimesteps) do t
					ones(T,Ξ,K)
				end
			end
		end
    likelihood!(p𝐘𝑑, trialsets, θnative.ψ[1]) # `p𝐘𝑑` is the conditional likelihood p(𝐘ₜ, d ∣ aₜ, zₜ)
	∇posterior(p𝐘𝑑[i][m], mpGLMs, trialinvariant, θnative, trialsets[i].trials[m])
end

"""
Compute the gradient of the posterior probabilities for each time step in one trial

RETURN
-a nested array whose element [t][i][j][q] corresponds the t-th time step, j-th accumulator state, j-th coupling state, and q-th parameter
"""
function ∇posterior(glmθs::Vector{<:GLMθ}, θnative::Latentθ, trial::Trial, trialinvariant::Trialinvariant)
	@unpack clicks = trial
	@unpack inputtimesteps, inputindex = clicks
	@unpack Aᵃsilent, dAᵃsilentdμ, dAᵃsilentdσ², dAᵃsilentdB, Aᶜ, Aᶜᵀ, Δt, K, 𝛚, πᶜᵀ, Ξ, 𝛏 = trialinvariant
	dℓdk, dℓdλ, dℓdϕ, dℓdσ²ₐ, dℓdσ²ₛ, dℓdB = 0., 0., 0., 0., 0., 0.
	∑χᶜ = zeros(T, K,K)
	μ = θnative.μ₀[1] + trial.previousanswer*θnative.wₕ[1]
	σ = √θnative.σ²ᵢ[1]
	πᵃ, dπᵃdμ, dπᵃdσ², dπᵃdB = zeros(T, Ξ), zeros(T, Ξ), zeros(T, Ξ), zeros(T, Ξ)
	probabilityvector!(πᵃ, dπᵃdμ, dπᵃdσ², dπᵃdB, μ, 𝛚, σ, 𝛏)
	n_steps_with_input = length(clicks.inputtimesteps)
	Aᵃ = map(x->zeros(T, Ξ,Ξ), clicks.inputtimesteps)
	dAᵃdμ = map(x->zeros(T, Ξ,Ξ), clicks.inputtimesteps)
	dAᵃdσ² = map(x->zeros(T, Ξ,Ξ), clicks.inputtimesteps)
	dAᵃdB = map(x->zeros(T, Ξ,Ξ), clicks.inputtimesteps)
	Δc = zeros(T, n_steps_with_input)
	∑c = zeros(T, n_steps_with_input)
	C, dCdk, dCdϕ = ∇adapt(clicks, θnative.k[1], θnative.ϕ[1])
	for i in 1:n_steps_with_input
		t = clicks.inputtimesteps[i]
		cL = sum(C[clicks.left[t]])
		cR = sum(C[clicks.right[t]])
		stochasticmatrix!(Aᵃ[i], dAᵃdμ[i], dAᵃdσ²[i], dAᵃdB[i], cL, cR, trialinvariant, θnative)
		Δc[i] = cR-cL
		∑c[i] = cL+cR
	end

	p𝐘𝑑, ∂p𝐘𝑑_∂w, ∂p𝐘𝑑_∂ψ = ∇conditionalikelihood(choice, glmθs, θnative.ψ[1], trial.spiketrainmodels, trialinvariant) #to write

	D, f = forward(Aᵃ, inputindex, πᵃ, p𝐘𝑑, trialinvariant)
end

"""
	∇conditionallikelihood(glmθs, spiketrainmodels, trialinvariant)

Gradient of the conditional likelihood of the spiking of simultaneously recorded neurons

ARGUMENT
-`glmθs`: a vector whose each element contains the parameters of the Poisson mixture generalized linear model of a neuron's spike train. Each element corresponds to a neuron.
-`spiketrainmodels`: a vector whose element contains one trial's data of the Poisson mixture generalized linear model of a neuron's spike train. Each element corresponds to a neuron.
-`trialinvariant`: a structure containing quantities that are used in each trial

RETURN
-`∇p𝐘`: partial derivatives; element ∇p𝐘[q][t][j,k] corresponds to the partial derivative of the product of the likelihood of all neurons' spike count at time step t conditioned on the accumulator in the j-th state and the coupling in the k-th state, with respect to the q-th parameter. The parameters of the GLMs of all neurons are concatenated.
-`p𝐘`: conditional likelihood; element p𝐘[t][j,k] corresponds to the product of the likelihood of all neurons' spike count at time step t conditioned on the accumulator in the j-th state and the coupling in the k-th state

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat")
julia> trialinvariant = Trialinvariant(model)
julia> glmθs = map(glm->glm.θ, model.trialsets[1].mpGLMs)
julia> ∇p𝐘, p𝐘 = FHMDDM.∇conditionallikelihood(glmθs, model.trialsets[1].trials[1].spiketrainmodels, trialinvariant)
```
"""
function ∇conditionallikelihood(glmθs::Vector{<:GLMθ},
							    spiketrainmodels::Vector{<:SpikeTrainModel},
							    trialinvariant::Trialinvariant)
	@unpack Δt, K, Ξ = trialinvariant
	𝛏 = (2collect(1:Ξ) .- Ξ .- 1)./(Ξ-1) # normalized
	ntimesteps = length(spiketrainmodels[1].𝐲)
	nneurons = length(spiketrainmodels)
	n𝐮 = length(glmθs[1].𝐮)
	n𝐯 = length(glmθs[1].𝐯)
	nparameters_per_neuron = n𝐮+n𝐯+1
	nparameters = nneurons*nparameters_per_neuron
	p𝐘 = map(t->ones(Ξ,K), 1:ntimesteps)
	∇p𝐘 = map(q->map(t->zeros(Ξ,K), 1:ntimesteps), 1:nparameters)
	for n = 1:nneurons
		f𝛏 = map(ξ->transformaccumulator(glmθs[n].b[1], ξ), 𝛏)
		∂f𝛏_∂b = map(ξ->dtransformaccumulator(glmθs[n].b[1], ξ), 𝛏)
		index1 = (n-1)*nparameters_per_neuron+1
		indices𝐮 = index1 : index1+n𝐮-1
		indices𝐯 = index1+n𝐮 : index1+n𝐮+n𝐯-1
		indexb = index1+n𝐮+n𝐯
		𝚽𝐯 = spiketrainmodels[n].𝚽 * glmθs[n].𝐯
		for t=1:ntimesteps
			for i = 1:n𝐮
				q = indices𝐮[i]
				∇p𝐘[q][t] .= spiketrainmodels[n].𝐔[t,i]
			end
			for i = 1:n𝐯
				q = indices𝐯[i]
				∇p𝐘[q][t][:,1] .= spiketrainmodels[n].𝚽[t,i].*f𝛏
			end
			∇p𝐘[indexb][t][:,1] .= 𝚽𝐯[t].*∂f𝛏_∂b
		end
		indicesbefore = 1:index1-1
		indices = index1:index1+nparameters_per_neuron-1
		indicesafter = indices[end]+1:nparameters
		for j = 1:Ξ
			for k = 1:K
				𝐗𝐰 = spiketrainmodels[n].𝐔 * glmθs[n].𝐮
				(k == 1) && (𝐗𝐰 .+= 𝚽𝐯.*f𝛏[j])
				for t=1:ntimesteps
					∂p𝑦ₙₜ_∂Xw, p𝑦ₙₜ = dPoissonlikelihood(Δt, 𝐗𝐰[t], spiketrainmodels[n].𝐲[t])
					p𝐘[t][j,k] *= p𝑦ₙₜ
					for q in indicesbefore
						∇p𝐘[q][t][j,k] *= p𝑦ₙₜ
					end
					for q in indices
						∇p𝐘[q][t][j,k] *= ∂p𝑦ₙₜ_∂Xw
					end
					for q in indicesafter
						∇p𝐘[q][t][j,k] *= p𝑦ₙₜ
					end
				end
			end
		end
	end
	return ∇p𝐘, p𝐘
end

"""
	compare∇p𝐘(model)

Compare the automatically computed and hand coded gradient of the conditional likelihood of the population spiking

ARGUMENT
-`model`: a structure containing the data, parametes, and hyperparameters of a factorial hidden-Markov drift-diffusion model

RETURN
-a vector whose each element shows the maximum absolute difference between the two partial derivatives with respect to each parameter.

EXAMPLE
```julia-repl
using FHMDDM
model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat")
maxabsdiff = FHMDDM.compare∇p𝐘(model)
```
"""
function compare∇p𝐘(model::Model)
	@unpack trialsets = model
	@unpack Δt, K, Ξ = model.options
	trialinvariant = Trialinvariant(model)
	maxbsdiff = 0.0
	glmθs = map(glm->glm.θ, model.trialsets[1].mpGLMs)
	concatenatedθ = zeros(0)
	for n in eachindex(glmθs)
		concatenatedθ = vcat(concatenatedθ, glmθs[n].𝐮, glmθs[n].𝐯, glmθs[n].b)
	end
	nparameters = length(concatenatedθ)
	automaticgradient = similar(concatenatedθ)
	maxabsdiff = zeros(nparameters)
	for m in eachindex(model.trialsets[1].trials)
		trial = model.trialsets[1].trials[m]
		∇p𝐘, p𝐘 = ∇conditionallikelihood(glmθs, trial.spiketrainmodels, trialinvariant)
		for t = 1:trial.ntimesteps
			for j = 1:Ξ
				for k = 1:K
					f(x) = conditionallikelihood(Δt,j,k,trial.spiketrainmodels,t,Ξ,x)
					automaticgradient = ForwardDiff.gradient!(automaticgradient, f, concatenatedθ)
					for q=1:nparameters
						maxabsdiff[q] = max(maxabsdiff[q], abs(automaticgradient[q] - ∇p𝐘[q][t][j,k]))
					end
				end
			end
		end
	end
	maxabsdiff
end
