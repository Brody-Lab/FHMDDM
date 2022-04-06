"""
	∇∇loglikelihood

Hessian of the observations in one trial
"""
function ∇∇loglikelihood(glmθs::Vector{<:GLMθ},
						 θnative::Latentθ,
						 trial::Trial,
						 trialinvariant::Trialinvariant)
	@unpack clicks = trial
	@unpack inputtimesteps, inputindex = clicks
	@unpack Aᵃsilent, dAᵃsilentdμ, dAᵃsilentdσ², dAᵃsilentdB, Aᶜ, Aᶜᵀ, Δt, K, 𝛚, πᶜᵀ, Ξ, 𝛏 = trialinvariant # need second derivative of the silent transition matrix and silent prior probability (without including `previousreward`)
	C, dCdk, dCdϕ, dCdkdk, dCdkdϕ, dCdϕdϕ = FHMDDM.∇∇adapt(clicks, θnative.k[1], θnative.ϕ[1])
	for t =1:trial.ntimesteps
		∇∇conditionallikelihood!(∇∇pY, ∇pY, pY, glmθs, t, trial.spiketrainmodels, trialinvariant)
		# then, compute using ∇∇p𝐘, ∇p𝐘, p𝐘
	end
end

"""
	∇∇conditionallikelihood!

Hessian of the conditional likelihood of the population spiking at one time step

MODIFIED ARGUMENTS
-`∇∇pY`: a nested array whose element ∇∇pY[q,r][i,j] corresponds to the second partial derivative of the conditional likelihood of population spiking, given that the accumulator is in the i-th state and the coupling in the j-th state, with respect the q-th and r-th parameters
-`∇pY`: a nested array whose element ∇pY[q][i,j] corresponds to the partial derivative of the conditional likelihood of population spiking, given that the accumulator is in the i-th state and the coupling in the j-th state, with respect the q-th parameter
-`pY`: an array whose element pY[i,j] corresponds to the conditional likelihood of population spiking, given that the accumulator is in the i-th state and the coupling in the j-th state

UNMODFIED ARGUMENTS
-`glmθs`: a vector whose each element is a structure containing the parameters of of the generalized linear model of a neuron
-`t`: time step
-`spiketrainmodels`: a vector whose each element is a structure containing the input and observations of the generalized linear model of a neuron
-`trialinvariant`: a structure containing quantities used in each trial

EXAMPLE
```julia-repl
using FHMDDM
model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat")
glmθs = map(x->x.θ, model.trialsets[1].mpGLMs)
t = 10
spiketrainmodels = model.trialsets[1].trials[1].spiketrainmodels
trialinvariant = Trialinvariant(model; purpose = "gradient")
nparameters = length(glmθs)*(length(glmθs[1].𝐮) + length(glmθs[1].𝐯))
Ξ = model.options.Ξ
K = model.options.K
pY = zeros(Ξ,K)
∇pY = collect(zeros(Ξ,K) for n=1:nparameters)
∇∇pY = map(index->zeros(Ξ,K), CartesianIndices((nparameters,nparameters)))
FHMDDM.∇∇conditionallikelihood!(∇∇pY, ∇pY, pY, glmθs, t, spiketrainmodels, trialinvariant)
```
"""
function ∇∇conditionallikelihood!(∇∇pY::Matrix{<:Matrix{<:Real}},
								  ∇pY::Vector{<:Matrix{<:Real}},
								  pY::Matrix{<:Real},
								  glmθs::Vector{<:GLMθ},
								  t::Integer,
								  spiketrainmodels::Vector{<:SpikeTrainModel},
								  trialinvariant::Trialinvariant)
	@unpack Δt, K, Ξ, 𝛏 = trialinvariant
	nneurons = length(spiketrainmodels)
	n𝐮 = length(glmθs[1].𝐮)
	n𝐯 = length(glmθs[1].𝐯)
	nparameters_per_neuron = n𝐮+n𝐯
	nparameters = nneurons*nparameters_per_neuron
	zeroindex = cld(Ξ,2)
	pY .= 1.0
	for n = 1:nneurons
		𝐔ₜ𝐮 = spiketrainmodels[n].𝐔[t,:] ⋅ glmθs[n].𝐮
		𝚽ₜ𝐯 = spiketrainmodels[n].𝚽[t,:] ⋅ glmθs[n].𝐯
		index1 = (n-1)*nparameters_per_neuron+1
		indices𝐮 = index1 : index1+n𝐮-1
		indices𝐯 = index1+n𝐮 : index1+n𝐮+n𝐯-1
		indices_thisneuron = index1:index1+n𝐮+n𝐯-1
		indices_previousneurons = 1:index1-1
		indices_subsequentneurons = index1+n𝐮+n𝐯:nparameters
		for i = 1:Ξ
			L = 𝐔ₜ𝐮 + 𝚽ₜ𝐯*𝛏[i]
			∂²py_∂L∂L, ∂py_∂L, py = ddPoissonlikelihood(Δt, L, spiketrainmodels[n].𝐲[t])
			pY[i,1] *= py
			for j=1:n𝐮
				q = indices𝐮[j]
				∇pY[q][i,1] = ∂py_∂L*spiketrainmodels[n].𝐔[t,j]/py #∂p(yₙ)/∂u * [1/p(yₙ)]
			end
			for j=1:n𝐯
				q = indices𝐯[j]
				∇pY[q][i,1] = ∂py_∂L*spiketrainmodels[n].𝚽[t,j]*𝛏[i]/py #∂p(yₙ)/∂v * [1/p(yₙ)]
			end
			for j = 1:n𝐮
				q = indices𝐮[j]
				for k = j:n𝐮
					r = indices𝐮[k]
					∇∇pY[q,r][i,1] = ∇∇pY[r,q][i,1] = ∂²py_∂L∂L * spiketrainmodels[n].𝐔[t,j] * spiketrainmodels[n].𝐔[t,k] / py
				end
			end
			for j = 1:n𝐮
				q = indices𝐮[j]
				for k = 1:n𝐯
					r = indices𝐯[k]
					∇∇pY[q,r][i,1] = ∇∇pY[r,q][i,1] = ∂²py_∂L∂L * spiketrainmodels[n].𝐔[t,j] * spiketrainmodels[n].𝚽[t,k]*𝛏[i] / py
				end
			end
			for j = 1:n𝐯
				q = indices𝐯[j]
				for k = j:n𝐯
					r = indices𝐯[k]
					∇∇pY[q,r][i,1] = ∇∇pY[r,q][i,1] = ∂²py_∂L∂L * spiketrainmodels[n].𝚽[t,j] * spiketrainmodels[n].𝚽[t,k]*𝛏[i]^2 / py
				end
			end
			for q in indices_thisneuron
				for r in indices_previousneurons
					∇∇pY[q,r][i,1] *= ∇pY[q][i,1]
					∇∇pY[r,q][i,1] = ∇∇pY[q,r][i,1]
				end
				for r in indices_subsequentneurons
					∇∇pY[q,r][i,1] = ∇∇pY[r,q][i,1] = ∇pY[q][i,1]
				end
			end
		end
	end
	for i = 1:Ξ
		for q = 1:nparameters
			∇pY[q][i,1] *= pY[i,1]
			for r = q:nparameters
				∇∇pY[q,r][i,1] *= pY[i,1]
				∇∇pY[r,q][i,1] = ∇∇pY[q,r][i,1]
			end
		end
	end
	if K > 1
		pY[:,2] .= pY[zeroindex,1]
		indices𝐮 = vcat(collect((n-1)*nparameters_per_neuron+1:(n-1)*nparameters_per_neuron+n𝐮 for n = 1:nneurons)...)
		for q in indices𝐮
			∇pY[q][:,2] .= ∇pY[q][zeroindex,1]
			for r in indices𝐮
				∇∇pY[q,r][:,2] .= ∇∇pY[q,r][zeroindex,1]
			end
		end
	end
end

"""
	compare_conditional_likelihood(model)

Compare the automatic computed and hand-coded derivatives of the conditional likelihood of population spiking

The second and first partial derivatives are compared at each time step in each trial and for each combination of latent states.

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of a factorial hidden-Markov drift-diffusion model

RETURN
-a matrix whose each element shows the maximum absolute difference between the two second-order partial derivatives with respect to each parameter.
-a vector whose each element shows the maximum absolute difference between the two first-order partial derivatives with respect to each parameter.

EXAMPLE
```julia-repl
using FHMDDM
model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat")
ΔH, Δg, Δp = FHMDDM.compare_conditional_likelihood(model)
```
"""
function compare_conditional_likelihood(model::Model)
	@unpack trialsets = model
	@unpack Δt, K, Ξ = model.options
	trialinvariant = Trialinvariant(model)
	glmθs = map(glm->glm.θ, model.trialsets[1].mpGLMs)
	concatenatedθ = zeros(0)
	for n in eachindex(glmθs)
		concatenatedθ = vcat(concatenatedθ, glmθs[n].𝐮, glmθs[n].𝐯)
	end
	Δp = 0.0
	nparameters = length(concatenatedθ)
	gauto, Δg = zeros(nparameters), zeros(nparameters)
	Hauto, ΔH = zeros(nparameters, nparameters), zeros(nparameters, nparameters)
	phand = zeros(Ξ,K)
	ghand = collect(zeros(Ξ,K) for n=1:nparameters)
	Hhand = map(index->zeros(Ξ,K), CartesianIndices((nparameters,nparameters)))
	for m in eachindex(model.trialsets[1].trials)
		trial = model.trialsets[1].trials[m]
		for t = 1:trial.ntimesteps
			∇∇conditionallikelihood!(Hhand, ghand, phand, glmθs, t, trial.spiketrainmodels, trialinvariant)
			for j = 1:Ξ
				for k = 1:K
					f(x) = conditionallikelihood(j,k,trial.spiketrainmodels,t,trialinvariant,x)
					ForwardDiff.hessian!(Hauto, f, concatenatedθ)
					ForwardDiff.gradient!(gauto, f, concatenatedθ)
					Δp = max(Δp, abs(f(concatenatedθ) - phand[j,k]))
					for q=1:nparameters
						Δg[q] = max(Δg[q], abs(gauto[q] - ghand[q][j,k]))
						for r = q:nparameters
							ΔH[q,r] = ΔH[r,q] = max(ΔH[q,r], abs(Hauto[q,r] - Hhand[q,r][j,k]))
						end
					end
				end
			end
		end
	end
	ΔH, Δg, Δp
end

"""
	conditionallikelihood(j,k,spiketrainmodels,t,trialinvariant,x)

Conditional likelihood of the population spiking response, for automatic differentiation

ARGUMENT
-`j`: index of the state of the accumulator
-`k`: index of the state of the coupling
-`spiketrainmodels`: a vector whose element contains one trial's data of the Poisson mixture generalized linear model of a neuron's spike train. Each element corresponds to a neuron.
-`t`: index of the time step
-`trialinvariant`: a structure containing quantities used in each trial
-`x`: parameters of each neuron's generalized linear model, concatenated

RETURN
-likelihood of the population spiking at time step t conditioned on the accumulator being in the j-th state and the coupling in the i-th state

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat")
julia> nparameters = length(model.trialsets[1].mpGLMs) * (length(model.trialsets[1].mpGLMs[1].θ.𝐮) + length(model.trialsets[1].mpGLMs[1].θ.𝐯) + 1)
julia> x = rand(MersenneTwister(1234), nparameters)
julia>
julia> FHMDDM.conditionallikelihood(27, 1, model.trialsets[1].trials[1].spiketrainmodels, Trialinvariant(model), x)
	0.013017384655839466
```
"""
function conditionallikelihood(j::Integer,
							   k::Integer,
							   spiketrainmodels::Vector{<:SpikeTrainModel},
							   t::Integer,
                               trialinvariant::Trialinvariant,
							   x::Vector{<:Real})
    @unpack Δt, 𝛏, Ξ = trialinvariant
	n𝐮 = size(spiketrainmodels[1].𝐔,2)
	n𝐯 = size(spiketrainmodels[1].𝚽,2)
	q = 0
	p = 1.0
	for n in eachindex(spiketrainmodels)
		𝐮 = x[q+1:q+n𝐮]
		q+=n𝐮
		𝐯 = x[q+1:q+n𝐯]
		q+=n𝐯
		L = spiketrainmodels[n].𝐔[t,:] ⋅ 𝐮
		if k == 1
			L += 𝛏[j]*(spiketrainmodels[n].𝚽[t,:] ⋅ 𝐯)
		end
		λ = softplus(L)
        p *= Poissonlikelihood(λ*Δt, spiketrainmodels[n].𝐲[t])
	end
	return p
end
