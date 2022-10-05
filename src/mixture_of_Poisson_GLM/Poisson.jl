"""
	poissonlikelihood(Δt, L, y)

Probability of a Poisson observation

ARGUMENT
-`Δt`: time step size
-`L`: linear predictor
-`y`: observation

OUTPUT
-the likelihood
"""
function poissonlikelihood(Δt::Real, L::Real, y::Integer)
	λΔt = softplus(L)*Δt
	poissonlikelihood(λΔt, y)
end

"""
	poissonlikelihood(λΔt, y)

Likelihood of observation `y` given intensity `λΔt`
"""
function poissonlikelihood(λΔt::Real, y::Integer)
	if y==0
		exp(-λΔt)
	elseif y==1
		λΔt*exp(-λΔt)
	else
		λΔt^y * exp(-λΔt) / factorial(y)
	end
end

"""
	poissonloglikelihood(λΔt, y)

Log-likelihood of an observation under a Poisson GLM

ARGUMENT
-`λΔt`: Poisson intensity per second
-`y`: observation

RETURN
-log-likelihood
"""
function poissonloglikelihood(λΔt::Real, y::Integer)
	if y == 0
		-λΔt
	elseif y == 1
		log(λΔt) - λΔt
	else
		y*log(λΔt) - λΔt
	end
end

"""
    poissonloglikelihood

Log-likelihood of an observation under a Poisson GLM with a softplus nonlinearity

ARGUMENT
-`Δt`: duration of time step
-`L`: linear predictor
-`y`: observation

RETURN
-log-likelihood
"""
poissonloglikelihood(Δt::AbstractFloat, L::Real, y::Integer) = poissonloglikelihood(softplus(L)*Δt, y)


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
end

"""
    differentiate_loglikelihood_wrt_linearpredictor(Δt, L, y)

First derivative of the log-likelihood of a Poisson GLM with respect to the linear predictor
"""
differentiate_loglikelihood_wrt_linearpredictor(Δt::AbstractFloat, L::Real, y::Integer) = differentiate_loglikelihood_wrt_linearpredictor(Δt, L, softplus(L), y)

"""
    differentiate_twice_loglikelihood_wrt_linearpredictor

Second derivative of the log-likelihood of a Poisson GLM with respect to the linear predictor

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
julia> y = 3
julia> f(x) = FHMDDM.poissonloglikelihood(Δt, x, y)
julia> g(x) = ForwardDiff.derivative(f, x)
julia> h(x) = ForwardDiff.derivative(g, x)
julia> x₀ = 1-2rand()
julia> d1auto = g(x₀)
julia> d2auto = h(x₀)
julia> d2hand, d1hand = FHMDDM.differentiate_twice_loglikelihood_wrt_linearpredictor(Δt, x₀, softplus(x₀), y)
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
	return d²ℓ_dLdL, dℓ_dL
end

"""
	differentiate_twice_loglikelihood_wrt_linearpredictor(Δt, L, y)

Second derivative of the log-likelihood of a Poisson GLM with respect to the linear predictor

ARGUMENT
-`Δt`: duration of time step
-`L`: linear predictor at one time step
-`y`: observation at that time step

RETURN
-the second derivative with respect to the linear predictor
-the first derivative with respect to the linear predictor
-the log-likelihood
"""
function differentiate_twice_loglikelihood_wrt_linearpredictor(Δt::AbstractFloat, L::Real, y::Integer)
	λ = softplus(L)
	λΔt = λ*Δt
	ℓ = poissonloglikelihood(λΔt, y)
	d²ℓ_dL², dℓ_dL = differentiate_twice_loglikelihood_wrt_linearpredictor(Δt, L, λ, y)
	return d²ℓ_dL², dℓ_dL, ℓ
end

"""
	scaledpoissonlikelihood(Δt, L, s, y)

Probability of a Poisson observation raised to a scale factor

ARGUMENT
-`Δt`: time step size
-`L`: linear predictor
-`s`: scale factor
-`y`: observation

OUTPUT
-the likelihood
"""
function scaledpoissonlikelihood(Δt::Real, L::Real, s::Real, y::Integer)
	λΔt = softplus(L)*Δt
	if y==0
		exp(-λΔt*s)
	elseif y==1
		λΔt^s*exp(-λΔt*s)
	else
		(λΔt^y/factorial(y))^s * exp(-λΔt*s)
	end
end
