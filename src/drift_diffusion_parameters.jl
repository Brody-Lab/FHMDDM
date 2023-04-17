"""
	randomize_latent_parameters(options)

Initialize the value of each model parameters in native space by sampling from a Uniform random variable""

RETURN
-values of model parameter in native space
"""
function randomize_latent_parameters(options::Options)
	θnative = Latentθ()
	randomize_latent_parameters!(θnative, options)
	return θnative
end

"""
	randomize_latent_parameters!(model)

Set the value of each latent-variable parameter as a sample from a Uniform distribution.

Only parameters being fit are randomized

MODIFIED ARGUMENT
-`model`: a structure containing the parameters, hyperparameters, and data of the factorial hidden-Markov drift-diffusion model. Its fields `θnative` and `θreal` are modified.
"""
function randomize_latent_parameters!(model::Model)
	@unpack options, θnative, θreal = model
	randomize_latent_parameters!(θnative, options)
	native2real!(θreal, options, θnative)
end

"""
	randomize_latent_parameters!(θnative, options)

Set the value of each latent-variable parameter as a sample from a Uniform distribution.

Only parameters being fit are randomized

MODIFIED ARGUMENT
-`θnative`: latent variables' parameters in native space

UNMODIFIED ARGUMENT
-`options`: settings of the model
"""
function randomize_latent_parameters!(θnative::Latentθ, options::Options)
	for field in fieldnames(typeof(θnative))
		fit = is_parameter_fit(options, field)
		l, q, u = lower_zero_upper(options, field)
		getfield(θnative, field)[1] = fit ? l + (u-l)*rand() : q
	end
	return nothing
end

"""
	lower_zero_upper(parametername)

RETURN the native values of a latent-variable parameter corresponding to its real values of negative infinity, zero, and infinity

ARGUMENT
-`options`: a struct containing the fixed hyperparameters
-`parametername`: a `Symbol` indicating the name of the latent variable

RETURN
1. the native value corresponding to the real value of negative infinity
2. the native value corresponding to the real value of zero
3. the native value corresponding to the real value of infinity

EXAMPLE
```julia
julia> l,q,u = FHMDDM.lower_zero_upper(options, :λ)
```
"""
function lower_zero_upper(options::Options, parametername::Symbol)
	l = getfield(options, Symbol(string(parametername)*"_l"))
	q = getfield(options, Symbol(string(parametername)*"_q"))
	u = getfield(options, Symbol(string(parametername)*"_u"))
	return l,q,u
end

"""
    real2native!(θnative, options, θreal)

Map values of model parameters from real space to native space

MODIFIED ARGUMENT
-`θnative: values of model parameters in native space

UNMODIFIED ARGUMENT
-`options`: model settings
-`θreal': values of model parameters in real space
"""
function real2native!(θnative::Latentθ, options::Options, θreal::Latentθ)
	for field in fieldnames(Latentθ)
		l, q, u = lower_zero_upper(options, field)
		n = getfield(θnative, field)
		r = getfield(θreal, field)[1]
		n[1] = real2native(r,q,l,u)
	end
	return nothing
end

"""
    real2native(options, θreal)

Map values of model parameters from real space to native space

ARGUMENT
-`options`: model settings
-`θreal: values of model parameters in real space

RETURN
-values of model parameters in their native space
"""
function real2native(options::Options, θreal::Latentθ{<:Vector{type}}) where type<:Real
	θnative = Latentθ((zeros(type,1) for field in fieldnames(Latentθ))...)
	real2native!(θnative, options, θreal)
	return θnative
end

"""
    native2real!(θreal, options, θnative)

Map values of model parameters from native space to real space

MODIFIED ARGUMENT
-`θreal: values of model parameters in real space

UNMODIFIED ARGUMENT
-`options`: model settings
-`θnative: values of model parameters in their native space
"""
function native2real!(θreal::Latentθ, options::Options, θnative::Latentθ)
	for field in fieldnames(Latentθ)
		l, q, u = lower_zero_upper(options, field)
		n = getfield(θnative, field)[1]
		r = getfield(θreal, field)
		r[1] = native2real(n,q,l,u)
	end
	return nothing
end

"""
    native2real(options, θnative)

Map values of model parameters from native space to real space

ARGUMENT
-`options`: model settings
-`θnative: values of model parameters in their native space`

RETURN
-values of model parameters in real space
"""
function native2real(options::Options, θnative::Latentθ)
	θreal = Latentθ()
	native2real!(θreal,options,θnative)
	return θreal
end

"""
	native2real!(∇ℓ, ∇∇ℓ, latentθindex, model)

Convert the gradient and hessian from being with respect to the parameters in native space to parameters in real space

ARGUMENT
-`∇ℓ`: gradient of the log-likelihood with respect to all parameters in native space
-`∇∇ℓ`: Hessian matrix of the log-likelihood with respect to all parameters in native space
-`latentθindex`: index of each latent parameter in the gradient and Hessian
-`model`: a structure containing the data, parameters, and hyperparameters of an FHMDDM

MODIFIED ARGUMENT
-`∇ℓ`: gradient of the log-likelihood with respect to all parameters in real space
-`∇∇ℓ`: Hessian matrix of the log-likelihood with respect to all parameters in real space
"""
function native2real!(∇ℓ::Vector{<:Real}, ∇∇ℓ::Matrix{<:Real}, model::Model)
	firstderivatives = differentiate_native_wrt_real(model)
	secondderivatives = differentiate_twice_native_wrt_real(model)
	parameternames = fieldnames(Latentθ)
	for i = 1:length(parameternames)
		d1 = getfield(firstderivatives, parameternames[i])[1]
		d2 = getfield(secondderivatives, parameternames[i])[1]
		∇∇ℓ[i,:] .*= d1
		∇∇ℓ[:,i] .*= d1
		∇∇ℓ[i,i] += d2*∇ℓ[i]
		∇ℓ[i] *= d1
	end
	return nothing
end

"""
	native2real!(∇ℓ, latentθindex, model)

Convert the gradient from being with respect to the parameters in native space to parameters in real space

ARGUMENT
-`∇ℓ`: gradient of the log-likelihood with respect to all parameters in native space
-`latentθindex`: index of each latent parameter in the gradient and Hessian
-`model`: a structure containing the data, parameters, and hyperparameters of an FHMDDM

MODIFIED ARGUMENT
-`∇ℓ`: gradient of the log-likelihood with respect to all parameters in real space
"""
function native2real!(∇ℓ::Vector{<:Real}, indexθ::Latentθ, model::Model)
	firstderivatives = differentiate_native_wrt_real(model)
	for parametername in fieldnames(Latentθ)
		i = getfield(indexθ, parametername)[1]
		if i > 0
			∇ℓ[i] *= getfield(firstderivatives, parametername)[1]
		end
	end
	return nothing
end

"""
	differentiate_native_wrt_real(model)

Derivative of each latent-variable-related parameter in its native space with respect to its value in real space

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of an FHMDDM

RETURN
-`derivatives`: an instance of `Latentθ` containing the derivative of each parameter in its native space with respect to its value in real space
"""
function differentiate_native_wrt_real(model::Model)
	@unpack options, θreal = model
	derivatives = Latentθ()
	for field in fieldnames(Latentθ)
		l, q, u = lower_zero_upper(options, field)
		r = getfield(θreal, field)[1]
		d = getfield(derivatives, field)
 		d[1] = differentiate_native_wrt_real(r,q,l,u)
	end
	return derivatives
end

"""
	differentiate_twice_native_wrt_real(model)

Second derivative of each latent-variable-related parameter in its native space with respect to its value in real space

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of an FHMDDM

RETURN
-`derivatives`: an instance of `Latentθ` containing the derivative of each parameter in its native space with respect to its value in real space
"""
function differentiate_twice_native_wrt_real(model::Model)
	@unpack options, θreal = model
	derivatives = Latentθ()
	for field in fieldnames(Latentθ)
		l, q, u = lower_zero_upper(options, field)
		r = getfield(θreal, field)[1]
		d = getfield(derivatives, field)
 		d[1] = differentiate_twice_native_wrt_real(r,q,l,u)
	end
	return derivatives
end

"""
	julianame(name)

Return the Symbol of a parameter in this module corresponding its valid variable name in MATLAB
"""
function julianame(name::String)
	if name == "B"
		:B
	elseif name == "k"
		:k
	elseif name == "lambda"
		:λ
	elseif name == "mu0"
		:μ₀
	elseif name == "phi"
		:ϕ
	elseif name == "psi"
		:ψ
	elseif name == "sigma2_a"
		:σ²ₐ
	elseif name == "sigma2_i"
		:σ²ᵢ
	elseif name == "sigma2_s"
		:σ²ₛ
	elseif name == "w_h"
		:wₕ
	end
end

"""
	matlabname(name)

A string containing the Valid variable name in MATLAB
"""
function matlabname(name::Symbol)
	if name == :B
		"B"
	elseif name == :k
		"k"
	elseif name == :λ
		"lambda"
	elseif name == :μ₀
		"mu0"
	elseif name == :ϕ
		"phi"
	elseif name == :ψ
		"psi"
	elseif name == :σ²ₐ
		"sigma2_a"
	elseif name == :σ²ᵢ
		"sigma2_i"
	elseif name == :σ²ₛ
		"sigma2_s"
	elseif name == :wₕ
		"w_h"
	end
end

"""
    dictionary(θ::Latentθ)

Convert an instance of `Latentθ` to a dictionary
"""
dictionary(θ::Latentθ) = Dict((matlabname(name)=>getfield(θ,name)[1] for name in fieldnames(Latentθ))...)

"""
    Latentθ(θ)

Create an instance of `Latentθ` from a Dict
"""
Latentθ(θ::Dict) = Latentθ(([θ[matlabname(name)]] for name in fieldnames(Latentθ))...)
