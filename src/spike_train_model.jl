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
    dPoissonlikelihood(Δt, x, y)

Compute the derivative of the likelihood of a Poisson observation with respect to the input.

The likelihood itself is also computed, and the Poisson model rectifies its input with a softplus nonlinearity and multiplies this result by the step size Δt to specify the rate: λ = Δt*softplus(x)

ARGUMENT
-`Δt`: size of time step. Assumed to be a positive real number, otherwise the results are nonsensical. No error checking because this is performance-critical code.
-`x`: input. This is a real number.
-`y`: observation.

RETURN
-`∂p_∂x`: derivative of the likelihood
-`p`: the likelihood

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> ∂p_∂x, p = FHMDDM.dPoissonlikelihood(0.01, -1, 2)
    (0.0008385398922087353, 4.89129765681903e-6)
julia> ∂p_∂x, p = FHMDDM.dPoissonlikelihood(-0.01, -1, 2) #even though Δt<0, no error is thrown!
    (-0.0008464575130532024, 4.922038980212279e-6)
```
"""
function dPoissonlikelihood(Δt::Real, x::Real, y::Integer)
    λ = softplus(x)
    λΔt = λ*Δt
    expλΔt = exp(λΔt)
    η = logistic(x)
    if y==0
        p = 1/expλΔt
        ∂p_∂x = -η*p
    elseif y==1
        p = λΔt/expλΔt
        ∂p_∂x = η*(1/expλΔt - p)
    elseif y==2
        p = λΔt^2 / expλΔt / 2
        ∂p_∂x = η*(λΔt/expλΔt - p)
    else
        p = λΔt^y / expλΔt / factorial(y)
        ∂p_∂x = η*(λΔt^(y-1)/expλΔt/factorial(y-1) - p)
    end
    return ∂p_∂x, p
end

"""
	conditionallikelihood(Δt,j,k,spiketrainmodels,t,Ξ,x)

Conditional likelihood of the spiking of a population, for automatic differentiation

ARGUMENT
-`Δt`: time step size
-`j`: index of the state of the accumulator
-`k`: index of the state of the coupling
-`spiketrainmodels`: a vector whose element contains one trial's data of the Poisson mixture generalized linear model of a neuron's spike train. Each element corresponds to a neuron.
-`t`: index of the time step
-`x`: parameters of each neuron's generalized linear model, concatenated

RETURN
-likelihood of the population spiking at time step t conditioned on the accumulator being in the j-th state and the coupling in the i-th state

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat")
julia> nparameters = length(model.trialsets[1].mpGLMs) * (length(model.trialsets[1].mpGLMs[1].θ.𝐮) + length(model.trialsets[1].mpGLMs[1].θ.𝐯) + 1)
julia> x = rand(MersenneTwister(1234), nparameters)
julia> FHMDDM.conditionallikelihood(model.options.Δt, 27, 1, model.trialsets[1].trials[1].spiketrainmodels, 10, model.options.Ξ, x)
	0.013017384655839466
```
"""
function conditionallikelihood(Δt::Real,
							   j::Integer,
							   k::Integer,
							   spiketrainmodels::Vector{<:SpikeTrainModel},
							   t::Integer,
							   Ξ::Integer,
							   x::Vector{<:Real})
	ξ = (2j-Ξ-1)/(Ξ-1) # normalized
	n𝐮 = size(spiketrainmodels[1].𝐔,2)
	n𝐯 = size(spiketrainmodels[1].𝚽,2)
	q = 0
	p = 1.0
	for n in eachindex(spiketrainmodels)
		𝐮 = x[q+1:q+n𝐮]
		q+=n𝐮
		𝐯 = x[q+1:q+n𝐯]
		q+=n𝐯
		b = x[q+1]
		q+=1
		Xw = spiketrainmodels[n].𝐔[t,:] ⋅ 𝐮
		if k == 1
			Xw += transformaccumulator(b,ξ)*(spiketrainmodels[n].𝚽[t,:] ⋅ 𝐯)
		end
		λ = softplus(Xw)
	    λΔt = λ*Δt
	    expλΔt = exp(λΔt)
		y = spiketrainmodels[n].𝐲[t]
		p *= λΔt^y / expλΔt / factorial(y)
	end
	return p
end
