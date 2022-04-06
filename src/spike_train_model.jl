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
	Poissonlikelihood(λΔt, y)

Probability of a Poisson observation

ARGUMENT
-`λΔt`: the expected value
-`y`: the observation

OUTPUT
-the likelihood
"""
function Poissonlikelihood(λΔt::Real, y::Integer)
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
