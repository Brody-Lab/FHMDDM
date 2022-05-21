"""
      SpikeTrainModel(ntimesteps_each_trial, 𝝫, 𝐔ₑ, 𝐔ₕ, 𝐘)

Make a spike train model for each neuron in each trial

INPUT
-`ntimesteps_each_trial`: a vector of integers indicating the number time steps in each trial
-`𝝫`: A matrix whose rows correspond to time steps concatenated across trials and whose columns correspond to the temporal basis functions of accumulated evidence
-`𝐔ₑ`: A matrix whose rows correspond to time steps concatenated across trials and whose columns correspond to the temporal basis functions of events
-`𝐔ₕ`: A vector whose elements correspond to neurons and are matrices. Rows of each matrix correspond to time steps concatenated across trials, and its columns correspond to the value of each temporal basis functions of autoregressive input
-`𝐘`: A vector whose elements correspond to neurons and are vectors. Elements of each inner vector indicate the spike count at each time step. Time steps are concatenated across trials.

RETURN
-a nested set of vectors whose element [m][n] corresponds to the spike train model of the n-th neuron in the m-th trial
"""
function SpikeTrainModel(ntimesteps_each_trial::Vector{<:Integer},
                          𝝫::Matrix{<:AbstractFloat},
                          𝐔ₑ::Matrix{<:AbstractFloat},
                          𝐔ₕ::Vector{<:Matrix{<:AbstractFloat}},
                          𝐘::Vector{<:Vector{<:Integer}})
    firstindices = cumsum(vcat(1, ntimesteps_each_trial[1:end-1]))
    lastindices = cumsum(ntimesteps_each_trial)
    timeindices_each_trial = map((i,j)->i:j, firstindices, lastindices)
    map(timeindices_each_trial) do indices
        Uₑ = 𝐔ₑ[indices,:]
        𝚽 = 𝝫[indices,:]
        map(𝐔ₕ, 𝐘) do Uₕ, 𝐲
            if isempty(Uₕ)
                𝐔 = Uₑ
            else
                𝐔 = hcat(Uₕ[indices,:], Uₑ)
            end
            SpikeTrainModel(𝚽=𝚽, 𝐔=𝐔, 𝐲=𝐲[indices])
        end
    end
end

"""
	Poissonlikelihood(Δt, L, y)

Probability of a Poisson observation

ARGUMENT
-`Δt`: size of time step. Assumed to be a positive real number, otherwise the results are nonsensical. No error checking because this is performance-critical code.
-`L`: linear predictor. This is a real scalar.
-`y`: observation.

OUTPUT
-the likelihood
"""
function Poissonlikelihood(Δt::Real, L::Real, y::Integer)
    λ = softplus(L)
    λΔt = λ*Δt
	if y==0
		exp(-λΔt)
	elseif y==1
		λΔt / exp(λΔt)
	else
		λΔt^y / exp(λΔt) / factorial(y)
	end
end

"""
    dPoissonlikelihood(Δt, L, y)

Compute the derivative of the likelihood of a Poisson observation with respect to the input.

The likelihood itself is also computed, and the Poisson model rectifies its input with a softplus nonlinearity and multiplies this result by the step size Δt to specify the rate: λ = Δt*softplus(x)

ARGUMENT
-`Δt`: size of time step. Assumed to be a positive real number, otherwise the results are nonsensical. No error checking because this is performance-critical code.
-`L`: linear predictor. This is a real scalar.
-`y`: observation.

RETURN
-`∂p_∂x`: derivative of the likelihood
-`p`: the likelihood

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> ∂p_∂L, p = FHMDDM.dPoissonlikelihood(0.01, 20.0, 2)
    (0.0014737153526074402, 0.016374615064597187)
julia> ∂p_∂L, p = FHMDDM.dPoissonlikelihood(-0.01, 20.0, 2) #even though Δt<0, no error is thrown!
    (0.0026870860627713614, 0.02442805516874189)
```
"""
function dPoissonlikelihood(Δt::Real, L::Real, y::Integer)
    λ = softplus(L)
    λΔt = λ*Δt
    expλΔt = exp(λΔt)
    if y==0
        p = 1/expλΔt
        ∂p_∂λ = -Δt*p
    elseif y==1
        p = λΔt/expλΔt
        ∂p_∂λ = Δt*(1/expλΔt - p)
    elseif y==2
        p = λΔt^2 / expλΔt / 2
        ∂p_∂λ = Δt*(λΔt/expλΔt - p)
    elseif y ==3
        p = λΔt^3 / expλΔt / 6
        ∂p_∂λ = Δt*(λΔt^2/expλΔt/2 - p)
    else
        p = λΔt^y / expλΔt / factorial(y)
        ∂p_∂λ = Δt*(λΔt^(y-1)/expλΔt/factorial(y-1) - p)
    end
    ∂λ_∂L = logistic(L)
    ∂p_∂L = ∂p_∂λ*∂λ_∂L
    return ∂p_∂L, p
end

"""
    ddPoissonlikelihood(Δt, L, y)

Second-order derivative of the likelihood of an observation under a Poisson generalized linear model with respect to the linear predictor.

The likelihood itself is also computed, and the Poisson model rectifies its input with a softplus nonlinearity and multiplies this result by the step size Δt to specify the rate: λ = Δt*softplus(x)

ARGUMENT
-`Δt`: size of time step. Assumed to be a positive real number, otherwise the results are nonsensical. No error checking because this is performance-critical code.
-`L`: linear predictor. This is a real number.
-`y`: observation.

RETURN
-`∂²p_∂L∂L`: second derivative of the likelihood
-`∂p_∂L`: first derivative of the likelihood
-`p`: the likelihood

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> ∂²p_∂L∂L, ∂p_∂L, p = FHMDDM.ddPoissonlikelihood(0.01, 5.0, 2)
    (8.765456536921317e-5, 0.00046119249651132817, 0.0011921527840907144)
```
"""
function ddPoissonlikelihood(Δt::Real, L::Real, y::Integer)
    λ = softplus(L)
    λΔt = λ*Δt
    expλΔt = exp(λΔt)
    if y==0
        p = 1/expλΔt
        ∂p_∂λ = -Δt*p
        ∂²p_∂λ∂λ = Δt^2*p
    elseif y==1
        p = λΔt/expλΔt
        ∂p_∂λ = Δt*(1/expλΔt - p)
        ∂²p_∂λ∂λ = Δt^2*(p - 2/expλΔt)
    elseif y==2
        p = λΔt^2 / expλΔt / 2
        ∂p_∂λ = Δt*(λΔt/expλΔt - p)
        ∂²p_∂λ∂λ = Δt^2*(p + (1-2λΔt)/expλΔt)
    elseif y ==3
        p = λΔt^3 / expλΔt / 6
        ∂p_∂λ = Δt*(λΔt^2/expλΔt/2 - p)
        ∂²p_∂λ∂λ = Δt^2*(p + (1-λΔt)*λΔt/expλΔt)
    else
        p = λΔt^y / expλΔt / factorial(y)
        ∂p_∂λ = Δt*(λΔt^(y-1)/expλΔt/factorial(y-1) - p)
        ∂²p_∂λ∂λ = Δt^2*p + λ^(y-2)*Δt^y/expλΔt/factorial(y-1)*(y-1-2λΔt)
    end
    ∂λ_∂L = logistic(L)
    ∂²λ_∂L∂L = 1.0 - ∂λ_∂L
    ∂p_∂L = ∂p_∂λ*∂λ_∂L
    ∂²p_∂L∂L = ∂²p_∂λ∂λ*∂λ_∂L^2 + ∂p_∂L*∂²λ_∂L∂L
    return ∂²p_∂L∂L, ∂p_∂L, p
end

"""
    comparederivatives(Δt, L, y)

Compare automatically and analytically computed first and second derivatives of the likelihood of an observation under a Poisson generalized linear model with respect to the linear predictor

ARGUMENT
-`Δt`: size of time step. Assumed to be a positive real number, otherwise the results are nonsensical. No error checking because this is performance-critical code.
-`L`: linear predictor. This is a real number.
-`y`: observation.

RETURN
-absolute difference in the second derivative
-absolute difference in the first derivative

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> FHMDDM.comparederivatives(0.01, 15, 3)

"""
function comparederivatives(Δt::Real, L::Real, y::Integer)
    f(x) = ddPoissonlikelihood(Δt, x[1], y)[3]
    automatic_1st = ForwardDiff.gradient(f, [L])[1]
    automatic_2nd = ForwardDiff.hessian(f, [L])[1]
    handcoded_2nd, handcoded_1st, p = ddPoissonlikelihood(Δt, L, y)
    return abs(handcoded_2nd-automatic_2nd), abs(handcoded_1st-automatic_1st)
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
-`sameacrosstrials`: a structure containing quantities used in each trial

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat")
julia> glmθs = map(x->x.θ, model.trialsets[1].mpGLMs)
julia> t = 10
julia> spiketrainmodels = model.trialsets[1].trials[1].spiketrainmodels
julia> sameacrosstrials = Sameacrosstrials(model)
julia> nparameters = length(glmθs)*(length(glmθs[1].𝐮) + length(glmθs[1].𝐯))
julia> Ξ = model.options.Ξ
julia> K = model.options.K
julia> pY = zeros(Ξ,K)
julia> ∇pY = collect(zeros(Ξ,K) for n=1:nparameters)
julia> ∇∇pY = map(index->zeros(Ξ,K), CartesianIndices((nparameters,nparameters)))
julia> FHMDDM.∇∇conditionallikelihood!(∇∇pY, ∇pY, pY, glmθs, t, spiketrainmodels, sameacrosstrials)
```
"""
function ∇∇conditionallikelihood!(∇∇pY::Matrix{<:Matrix{<:Real}},
								  ∇pY::Vector{<:Matrix{<:Real}},
								  pY::Matrix{<:Real},
								  glmθs::Vector{<:GLMθ},
								  t::Integer,
								  spiketrainmodels::Vector{<:SpikeTrainModel},
								  sameacrosstrials::Sameacrosstrials)
	@unpack Δt, K, Ξ, d𝛏_dB = sameacrosstrials
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
			L = 𝐔ₜ𝐮 + 𝚽ₜ𝐯*d𝛏_dB[i]
			∂²py_∂L∂L, ∂py_∂L, py = ddPoissonlikelihood(Δt, L, spiketrainmodels[n].𝐲[t])
			pY[i,1] *= py
			for j=1:n𝐮
				q = indices𝐮[j]
				∇pY[q][i,1] = ∂py_∂L*spiketrainmodels[n].𝐔[t,j]/py #∂p(yₙ)/∂u * [1/p(yₙ)]
			end
			for j=1:n𝐯
				q = indices𝐯[j]
				∇pY[q][i,1] = ∂py_∂L*spiketrainmodels[n].𝚽[t,j]*d𝛏_dB[i]/py #∂p(yₙ)/∂v * [1/p(yₙ)]
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
					∇∇pY[q,r][i,1] = ∇∇pY[r,q][i,1] = ∂²py_∂L∂L * spiketrainmodels[n].𝐔[t,j] * spiketrainmodels[n].𝚽[t,k]*d𝛏_dB[i] / py
				end
			end
			for j = 1:n𝐯
				q = indices𝐯[j]
				for k = j:n𝐯
					r = indices𝐯[k]
					∇∇pY[q,r][i,1] = ∇∇pY[r,q][i,1] = ∂²py_∂L∂L * spiketrainmodels[n].𝚽[t,j] * spiketrainmodels[n].𝚽[t,k]*d𝛏_dB[i]^2 / py
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
	return nothing
end

"""
	∇conditionallikelihood!

Gradient of the conditional likelihood of the population spiking at one time step

MODIFIED ARGUMENTS
-`∇pY`: a nested array whose element ∇pY[q][i,j] corresponds to the partial derivative of the conditional likelihood of population spiking, given that the accumulator is in the i-th state and the coupling in the j-th state, with respect the q-th parameter
-`pY`: an array whose element pY[i,j] corresponds to the conditional likelihood of population spiking, given that the accumulator is in the i-th state and the coupling in the j-th state

UNMODFIED ARGUMENTS
-`glmθs`: a vector whose each element is a structure containing the parameters of of the generalized linear model of a neuron
-`t`: time step
-`spiketrainmodels`: a vector whose each element is a structure containing the input and observations of the generalized linear model of a neuron
-`sameacrosstrials`: a structure containing quantities used in each trial

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat")
julia> glmθs = map(x->x.θ, model.trialsets[1].mpGLMs)
julia> t = 10
julia> spiketrainmodels = model.trialsets[1].trials[1].spiketrainmodels
julia> sameacrosstrials = Sameacrosstrials(model)
julia> nparameters = length(glmθs)*(length(glmθs[1].𝐮) + length(glmθs[1].𝐯))
julia> Ξ = model.options.Ξ
julia> K = model.options.K
julia> pY = zeros(Ξ,K)
julia> ∇pY = collect(zeros(Ξ,K) for n=1:nparameters)
julia> FHMDDM.∇conditionallikelihood!(∇pY, pY, glmθs, t, spiketrainmodels, sameacrosstrials)
```
"""
function ∇conditionallikelihood!(∇pY::Vector{<:Matrix{<:Real}},
								pY::Matrix{<:Real},
								glmθs::Vector{<:GLMθ},
								t::Integer,
								spiketrainmodels::Vector{<:SpikeTrainModel},
								sameacrosstrials::Sameacrosstrials)
	@unpack Δt, K, Ξ, d𝛏_dB = sameacrosstrials
	nneurons = length(spiketrainmodels)
	n𝐮 = length(glmθs[1].𝐮)
	n𝐯 = length(glmθs[1].𝐯)
	nparameters_per_neuron = n𝐮+n𝐯
	nparameters = nneurons*nparameters_per_neuron
	zeroindex = cld(Ξ,2)
	pY .= 1.0
	for n = 1:nneurons
		𝐔ₜ𝐮 = spiketrainmodels[n].𝐔[t,1]*glmθs[n].𝐮[1]
		for i=2:n𝐮
			𝐔ₜ𝐮 += spiketrainmodels[n].𝐔[t,i]*glmθs[n].𝐮[i]
		end
		𝚽ₜ𝐯 = spiketrainmodels[n].𝚽[t,1]*glmθs[n].𝐯[1]
		for i=2:n𝐯
			𝚽ₜ𝐯 += spiketrainmodels[n].𝚽[t,i]*glmθs[n].𝐯[i]
		end
		index1 = (n-1)*nparameters_per_neuron+1
		indices𝐮 = index1 : index1+n𝐮-1
		indices𝐯 = index1+n𝐮 : index1+n𝐮+n𝐯-1
		for i = 1:Ξ
			L = 𝐔ₜ𝐮 + 𝚽ₜ𝐯*d𝛏_dB[i]
			∂py_∂L, py = dPoissonlikelihood(Δt, L, spiketrainmodels[n].𝐲[t])
			pY[i,1] *= py
			for j=1:n𝐮
				q = indices𝐮[j]
				∇pY[q][i,1] = ∂py_∂L*spiketrainmodels[n].𝐔[t,j]/py #∂p(yₙ)/∂u * [1/p(yₙ)]
			end
			for j=1:n𝐯
				q = indices𝐯[j]
				∇pY[q][i,1] = ∂py_∂L*spiketrainmodels[n].𝚽[t,j]*d𝛏_dB[i]/py #∂p(yₙ)/∂v * [1/p(yₙ)]
			end
		end
	end
	for i = 1:Ξ
		for q = 1:nparameters
			∇pY[q][i,1] *= pY[i,1]
		end
	end
	if K > 1
		pY[:,2] .= pY[zeroindex,1]
		q = 0
		for n = 1:nneurons
			for i = 1:n𝐮
				q +=1
				∇pY[q][:,2] .= ∇pY[q][zeroindex,1]
			end
			q+=n𝐯
		end
	end
	return nothing
end

"""
    conditionallikelihood!

Hessian of the conditional likelihood of the population spiking at one time step

MODIFIED ARGUMENTS
-`pY`: an array whose element pY[i,j] corresponds to the conditional likelihood of population spiking, given that the accumulator is in the i-th state and the coupling in the j-th state

UNMODFIED ARGUMENTS
-`glmθs`: a vector whose each element is a structure containing the parameters of of the generalized linear model of a neuron
-`t`: time step
-`spiketrainmodels`: a vector whose each element is a structure containing the input and observations of the generalized linear model of a neuron
-`sameacrosstrials`: a structure containing quantities used in each trial
"""
function conditionallikelihood!(pY::Matrix{<:Real},
                                Δt::Real,
                                d𝛏_dB::Vector{<:Real},
                                glmθs::Vector{<:GLMθ},
                                K::Integer,
                                t::Integer,
                                spiketrainmodels::Vector{<:SpikeTrainModel})
	Ξ = length(d𝛏_dB)
	pY .= 1.0
	n𝐮 = length(glmθs[1].𝐮)
	n𝐯 = length(glmθs[1].𝐯)
	zeroindex = cld(Ξ,2)
	for n = 1:length(glmθs)
		𝐔ₜ𝐮 = spiketrainmodels[n].𝐔[t,1]*glmθs[n].𝐮[1]
		for i=2:n𝐮
			𝐔ₜ𝐮 += spiketrainmodels[n].𝐔[t,i]*glmθs[n].𝐮[i]
		end
		𝚽ₜ𝐯 = spiketrainmodels[n].𝚽[t,1]*glmθs[n].𝐯[1]
		for i=2:n𝐯
			𝚽ₜ𝐯 += spiketrainmodels[n].𝚽[t,i]*glmθs[n].𝐯[i]
		end
		for i = 1:Ξ
			L = 𝐔ₜ𝐮 + 𝚽ₜ𝐯*d𝛏_dB[i]
			pY[i,1] *= Poissonlikelihood(Δt, L, spiketrainmodels[n].𝐲[t])
		end
	end
	if K > 1
		pY[:,2] .= pY[zeroindex,1]
    end
	return nothing
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
	sameacrosstrials = Sameacrosstrials(model)
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
			∇∇conditionallikelihood!(Hhand, ghand, phand, glmθs, t, trial.spiketrainmodels, sameacrosstrials)
			for j = 1:Ξ
				for k = 1:K
					f(x) = conditionallikelihood(j,k,trial.spiketrainmodels,t,sameacrosstrials,x)
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
	conditionallikelihood(j,k,spiketrainmodels,t,sameacrosstrials,x)

Conditional likelihood of the population spiking response, for automatic differentiation

ARGUMENT
-`j`: index of the state of the accumulator
-`k`: index of the state of the coupling
-`spiketrainmodels`: a vector whose element contains one trial's data of the Poisson mixture generalized linear model of a neuron's spike train. Each element corresponds to a neuron.
-`t`: index of the time step
-`sameacrosstrials`: a structure containing quantities used in each trial
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
julia> FHMDDM.conditionallikelihood(27, 1, model.trialsets[1].trials[1].spiketrainmodels, Sameacrosstrials(model), x)
	0.013017384655839466
```
"""
function conditionallikelihood(j::Integer,
							   k::Integer,
							   spiketrainmodels::Vector{<:SpikeTrainModel},
							   t::Integer,
                               sameacrosstrials::Sameacrosstrials,
							   x::Vector{<:Real})
    @unpack Δt, 𝛏, Ξ = sameacrosstrials
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
