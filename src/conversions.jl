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
		lqu = getfield(options, Symbol("lqu_"*string(field)))
		l = lqu[1]
		q = lqu[2]
		u = lqu[3]
		n = getfield(θnative, field)
		r = getfield(θreal, field)[1]
		n[1] = real2native(r,q,l,u)
	end
	return nothing
end

"""
	real2native(r,q,l,u)

Convert a parameter from real space to latent space

ARGUMENT
-`r`: value in real space
-`q`: value in native space equal to a zero-valued in real space
-`l`: lower bound in native space
-`u`: upper bound in native space

RETURN
-scalar representing the value in native space
"""
function real2native(r::Real, q::Real, l::Real, u::Real)
	if q == l
		l + (u-l)*logistic(r)
	else
		l + (u-l)*logistic(r+logit((q-l)/(u-l)))
	end
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
		lqu = getfield(options, Symbol("lqu_"*string(field)))
		l = lqu[1]
		q = lqu[2]
		u = lqu[3]
		n = getfield(θnative, field)[1]
		r = getfield(θreal, field)
		r[1] = native2real(n,q,l,u)
	end
	return nothing
end

"""
	real2native(r,q,l,u)

Convert a parameter from real space to latent space

ARGUMENT
-`n`: value in native space
-`q`: value in native space equal to a zero-valued in real space
-`l`: lower bound in native space
-`u`: upper bound in native space

RETURN
-scalar representing the value in real space
"""
function native2real(n::Real, q::Real, l::Real, u::Real)
	if q == l
		logit((n-l)/(u-l))
	else
		logit((n-l)/(u-l)) - logit((q-l)/(u-l))
	end
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
		lqu = getfield(options, Symbol("lqu_"*string(field)))
		l = lqu[1]
		q = lqu[2]
		u = lqu[3]
		r = getfield(θreal, field)[1]
		d = getfield(derivatives, field)
 		d[1] = differentiate_native_wrt_real(r,q,l,u)
	end
	return derivatives
end

"""
	differentiate_native_wrt_real(r,q,l,u)

Derivative of the native value of a parameter with respect to its value in real space

ARGUMENT
-`r`: value in real space
-`q`: value in native space equal to a zero-valued in real space
-`l`: lower bound in native space
-`u`: upper bound in native space

RETURN
-a scalar representing the derivative
"""
function differentiate_native_wrt_real(r::Real, q::Real, l::Real, u::Real)
	if q == l
		x = logistic(r)
	else
		x = logistic(r + logit((q-l)/(u-l)))
	end
	(u-l)*x*(1.0-x)
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
		lqu = getfield(options, Symbol("lqu_"*string(field)))
		l = lqu[1]
		q = lqu[2]
		u = lqu[3]
		r = getfield(θreal, field)[1]
		d = getfield(derivatives, field)
 		d[1] = differentiate_twice_native_wrt_real(r,q,l,u)
	end
	return derivatives
end

"""
	differentiate_twice_native_wrt_real(r,q,l,u)

Second derivative of the native value of a parameter with respect to its value in real space

ARGUMENT
-`r`: value in real space
-`q`: value in native space equal to a zero-valued in real space
-`l`: lower bound in native space
-`u`: upper bound in native space

RETURN
-a scalar representing the second derivative
"""
function differentiate_twice_native_wrt_real(r::Real, q::Real, l::Real, u::Real)
	if q == l
		x = logistic(r)
	else
		x = logistic(r + logit((q-l)/(u-l)))
	end
	(u-l)*(x*(1.0-x)^2 - x^2*(1-x))
end

"""
	differentiate_native_wrt_real(r,l,u)

Derivative of the native value of hyperparameters with respect to values in real space

ARGUMENT
-`r`: value in real space
-`l`: lower bound in native space
-`u`: upper bound in native space

RETURN
-derivatives
"""
function differentiate_native_wrt_real(r::Real, l::Real, u::Real)
	x = logistic(r)
	(u-l)*x*(1.0-x)
end

"""
    dictionary(options)

Convert an instance of `Options` to a dictionary
"""
function dictionary(options::Options)
	Dict(	"a_latency_s"=>options.a_latency_s,
			"b_scalefactor"=>options.b_scalefactor/options.nunits,
			"choiceobjective"=>options.choiceobjective,
			"datapath"=>options.datapath,
			"dt"=>options.Δt,
			"fit_Ac11"=>options.fit_Aᶜ₁₁,
			"fit_Ac22"=>options.fit_Aᶜ₂₂,
			"fit_B"=>options.fit_B,
			"fit_b"=>options.fit_b,
			"fit_beta"=>options.fit_𝛃,
			"fit_k"=>options.fit_k,
			"fit_lambda"=>options.fit_λ,
			"fit_mu0"=>options.fit_μ₀,
			"fit_phi"=>options.fit_ϕ,
			"fit_pic1"=>options.fit_πᶜ₁,
			"fit_psi"=>options.fit_ψ,
			"fit_sigma2_a"=>options.fit_σ²ₐ,
			"fit_sigma2_i"=>options.fit_σ²ᵢ,
			"fit_sigma2_s"=>options.fit_σ²ₛ,
			"fit_w_h"=>options.fit_wₕ,
			"g_tol"=>options.g_tol,
			"K"=>options.K,
			"L2_b_fit"=>options.L2_b_fit,
			"L2_b_max"=>options.L2_b_max,
			"L2_b_min"=>options.L2_b_min,
			"L2_choices_max"=>options.L2_choices_max,
			"L2_choices_min"=>options.L2_choices_min,
			"L2_gain_fit"=>options.L2_gain_fit,
			"L2_gain_max"=>options.L2_gain_max,
			"L2_gain_min"=>options.L2_gain_min,
			"L2_hist_fit"=>options.L2_hist_fit,
			"L2_hist_max"=>options.L2_hist_max,
			"L2_hist_min"=>options.L2_hist_min,
			"L2_latent_fit"=>options.L2_latent_fit,
			"L2_latent_max"=>options.L2_latent_max,
			"L2_latent_min"=>options.L2_latent_min,
			"L2_move_fit"=>options.L2_move_fit,
			"L2_move_max"=>options.L2_move_max,
			"L2_move_min"=>options.L2_move_min,
			"L2_phot_fit"=>options.L2_phot_fit,
			"L2_phot_max"=>options.L2_phot_max,
			"L2_phot_min"=>options.L2_phot_min,
			"L2_time_fit"=>options.L2_time_fit,
			"L2_time_max"=>options.L2_time_max,
			"L2_time_min"=>options.L2_time_min,
			"L2_v_fit"=>options.L2_v_fit,
			"L2_v_max"=>options.L2_v_max,
			"L2_v_min"=>options.L2_v_min,
			"lqu_Ac11"=>options.lqu_Aᶜ₁₁,
			"lqu_Ac22"=>options.lqu_Aᶜ₂₂,
			"lqu_B"=>	options.lqu_B,
			"lqu_k"=>	options.lqu_k,
			"lqu_lambda"=>	options.lqu_λ,
			"lqu_mu0"=>	options.lqu_μ₀,
			"lqu_phi"=>	options.lqu_ϕ,
			"lqu_pic1"=>	options.lqu_πᶜ₁,
			"lqu_psi"=>	options.lqu_ψ,
			"lqu_sigma2_a"=>	options.lqu_σ²ₐ,
			"lqu_sigma2_i"=>	options.lqu_σ²ᵢ,
			"lqu_sigma2_s"=>	options.lqu_σ²ₛ,
			"lqu_w_h"=>	options.lqu_wₕ,
    		"minpa"=>	options.minpa,
    		"objective"=> options.objective,
			"resultspath"=>options.resultspath,
			"scalechoiceLL"=>options.scalechoiceLL,
    		"sf_y"=>options.sf_y,
			"tbf_accu_begins0"=>options.tbf_accu_begins0,
			"tbf_accu_ends0"=>options.tbf_accu_ends0,
			"tbf_accu_hz"=>options.tbf_accu_hz,
			"tbf_accu_scalefactor"=>options.tbf_accu_scalefactor/options.nunits,
			"tbf_accu_stretch"=>options.tbf_accu_stretch,
			"tbf_gain_scalefactor"=>options.tbf_gain_scalefactor,
			"tbf_hist_begins0"=>options.tbf_hist_begins0,
			"tbf_hist_dur_s"=>options.tbf_hist_dur_s,
			"tbf_hist_ends0"=>options.tbf_hist_ends0,
			"tbf_hist_hz"=>options.tbf_hist_hz,
			"tbf_hist_linear"=>options.tbf_hist_linear,
			"tbf_hist_scalefactor"=>options.tbf_hist_scalefactor/options.nunits,
			"tbf_hist_stretch"=>options.tbf_hist_stretch,
			"tbf_move_begins0"=>options.tbf_move_begins0,
			"tbf_move_dur_s"=>options.tbf_move_dur_s,
			"tbf_move_ends0"=>options.tbf_move_ends0,
			"tbf_move_hz"=>options.tbf_move_hz,
			"tbf_move_linear"=>options.tbf_move_linear,
			"tbf_move_scalefactor"=>options.tbf_move_scalefactor/options.nunits,
			"tbf_move_stretch"=>options.tbf_move_stretch,
			"tbf_period"=>options.tbf_period,
			"tbf_phot_begins0"=>options.tbf_phot_begins0,
			"tbf_phot_ends0"=>options.tbf_phot_ends0,
			"tbf_phot_hz"=>options.tbf_phot_hz,
			"tbf_phot_linear"=>options.tbf_phot_linear,
			"tbf_phot_scalefactor"=>options.tbf_phot_scalefactor/options.nunits,
			"tbf_phot_stretch"=>options.tbf_phot_stretch,
			"tbf_time_begins0"=>options.tbf_time_begins0,
			"tbf_time_dur_s"=>options.tbf_time_dur_s,
			"tbf_time_ends0"=>options.tbf_time_ends0,
			"tbf_time_hz"=>options.tbf_time_hz,
			"tbf_time_linear"=>options.tbf_time_linear,
			"tbf_time_scalefactor"=>options.tbf_time_scalefactor/options.nunits,
			"tbf_time_stretch"=>options.tbf_time_stretch,
			"Xi"=>options.Ξ)
end

"""
    dictionary(glmθ)

Convert into a dictionary the parameters of a mixture of Poisson generalized linear model
"""
function dictionary(glmθ::GLMθ)
    Dict("b"=>glmθ.b,
		"b_scalefactor"=>glmθ.b_scalefactor,
		"u"=>glmθ.𝐮,
		"v"=>glmθ.𝐯,
		"beta"=>glmθ.𝛃,
		("u_"*string(field)=>glmθ.𝐮[getfield(glmθ.indices𝐮, field)] for field in fieldnames(Indices𝐮))...)
end

"""
	julianame(name)

Return the Symbol of a parameter in this module corresponding its valid variable name in MATLAB
"""
function julianame(name::String)
	if name == "Ac11"
		:Aᶜ₁₁
	elseif name == "Ac22"
		:Aᶜ₂₂
 	elseif name == "B"
		:B
	elseif name == "k"
		:k
	elseif name == "lambda"
		:λ
	elseif name == "mu0"
		:μ₀
	elseif name == "phi"
		:ϕ
	elseif name == "pic1"
		:πᶜ₁
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
	if name == :Aᶜ₁₁
		"Ac11"
	elseif name == :Aᶜ₂₂
		"Ac22"
 	elseif name == :B
		"B"
	elseif name == :k
		"k"
	elseif name == :λ
		"lambda"
	elseif name == :μ₀
		"mu0"
	elseif name == :ϕ
		"phi"
	elseif name == :πᶜ₁
		"pic1"
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

"""
    Options(options::Dict)

Create an instance of `Options` from a Dict
"""
function Options(nunits::Integer, options::Dict)
	Options(a_latency_s = options["a_latency_s"],
			b_scalefactor = options["b_scalefactor"]*nunits,
			choiceobjective=options["choiceobjective"],
			datapath = options["datapath"],
			Δt = options["dt"],
			fit_Aᶜ₁₁= options["fit_Ac11"],
			fit_Aᶜ₂₂= options["fit_Ac22"],
			fit_B = options["fit_B"],
			fit_b = options["fit_b"],
			fit_𝛃 = options["fit_beta"],
			fit_k = options["fit_k"],
			fit_λ = options["fit_lambda"],
			fit_μ₀ = options["fit_mu0"],
			fit_ϕ = options["fit_phi"],
			fit_πᶜ₁	= options["fit_pic1"],
			fit_ψ = options["fit_psi"],
			fit_σ²ₐ = options["fit_sigma2_a"],
			fit_σ²ᵢ = options["fit_sigma2_i"],
			fit_σ²ₛ = options["fit_sigma2_s"],
			fit_wₕ = options["fit_w_h"],
			g_tol = options["g_tol"],
			K = convert(Int, options["K"]),
			L2_b_fit = options["L2_b_fit"],
			L2_b_max = options["L2_b_max"],
			L2_b_min = options["L2_b_min"],
			L2_choices_max = options["L2_choices_max"],
			L2_choices_min = options["L2_choices_min"],
			L2_gain_fit = options["L2_gain_fit"],
			L2_gain_max = options["L2_gain_max"],
			L2_gain_min = options["L2_gain_min"],
			L2_hist_fit = options["L2_hist_fit"],
			L2_hist_max = options["L2_hist_max"],
			L2_hist_min = options["L2_hist_min"],
			L2_latent_fit = options["L2_latent_fit"],
			L2_latent_max = options["L2_latent_max"],
			L2_latent_min = options["L2_latent_min"],
			L2_move_fit = options["L2_move_fit"],
			L2_move_max = options["L2_move_max"],
			L2_move_min = options["L2_move_min"],
			L2_phot_fit = options["L2_phot_fit"],
			L2_phot_max = options["L2_phot_max"],
			L2_phot_min = options["L2_phot_min"],
			L2_time_fit = options["L2_time_fit"],
			L2_time_max = options["L2_time_max"],
			L2_time_min = options["L2_time_min"],
			L2_v_fit = options["L2_v_fit"],
			L2_v_max = options["L2_v_max"],
			L2_v_min = options["L2_v_min"],
			lqu_Aᶜ₁₁= vec(options["lqu_Ac11"]),
			lqu_Aᶜ₂₂= vec(options["lqu_Ac22"]),
			lqu_B 	= vec(options["lqu_B"]),
			lqu_k	= vec(options["lqu_k"]),
			lqu_λ	= vec(options["lqu_lambda"]),
			lqu_μ₀	= vec(options["lqu_mu0"]),
			lqu_ϕ	= vec(options["lqu_phi"]),
			lqu_πᶜ₁	= vec(options["lqu_pic1"]),
			lqu_ψ	= vec(options["lqu_psi"]),
			lqu_σ²ₐ	= vec(options["lqu_sigma2_a"]),
			lqu_σ²ᵢ	= vec(options["lqu_sigma2_i"]),
			lqu_σ²ₛ	= vec(options["lqu_sigma2_s"]),
			lqu_wₕ	= vec(options["lqu_w_h"]),
			minpa = options["minpa"],
			nunits = nunits,
			objective = options["objective"],
			resultspath = options["resultspath"],
			scalechoiceLL = options["scalechoiceLL"],
			sf_y = options["sf_y"],
			tbf_accu_begins0 = options["tbf_accu_begins0"],
			tbf_accu_ends0 = options["tbf_accu_ends0"],
			tbf_accu_hz = options["tbf_accu_hz"],
			tbf_accu_scalefactor = options["tbf_accu_scalefactor"]*nunits,
			tbf_accu_stretch = options["tbf_accu_stretch"],
			tbf_gain_scalefactor = options["tbf_gain_scalefactor"]*nunits,
			tbf_hist_begins0 = options["tbf_hist_begins0"],
			tbf_hist_dur_s = options["tbf_hist_dur_s"],
			tbf_hist_ends0 = options["tbf_hist_ends0"],
			tbf_hist_hz = options["tbf_hist_hz"],
			tbf_hist_linear = options["tbf_hist_linear"],
			tbf_hist_scalefactor = options["tbf_hist_scalefactor"]*nunits,
			tbf_hist_stretch = options["tbf_hist_stretch"],
			tbf_move_begins0 = options["tbf_move_begins0"],
			tbf_move_dur_s = options["tbf_move_dur_s"],
			tbf_move_ends0 = options["tbf_move_ends0"],
			tbf_move_hz = options["tbf_move_hz"],
			tbf_move_linear = options["tbf_move_linear"],
			tbf_move_scalefactor = options["tbf_move_scalefactor"]*nunits,
			tbf_move_stretch = options["tbf_move_stretch"],
			tbf_period = options["tbf_period"],
			tbf_phot_begins0 = options["tbf_phot_begins0"],
			tbf_phot_ends0 = options["tbf_phot_ends0"],
			tbf_phot_hz = options["tbf_phot_hz"],
			tbf_phot_linear = options["tbf_phot_linear"],
			tbf_phot_scalefactor = options["tbf_phot_scalefactor"]*nunits,
			tbf_phot_stretch = options["tbf_phot_stretch"],
			tbf_time_begins0 = options["tbf_time_begins0"],
			tbf_time_dur_s = options["tbf_time_dur_s"],
			tbf_time_ends0 = options["tbf_time_ends0"],
			tbf_time_hz = options["tbf_time_hz"],
			tbf_time_linear = options["tbf_time_linear"],
			tbf_time_scalefactor = options["tbf_time_scalefactor"]*nunits,
			tbf_time_stretch = options["tbf_time_stretch"],
			Ξ = convert(Int, options["Xi"]))
end

"""
	sortbytrial(γ, model)

Sort concatenated posterior probability or spike response by trials

ARGUMENT
-`γ`: a nested array whose element γ[s][j,k][τ] corresponds to the τ-th time step in the s-th trialset and the j-th accumulator state and k-th coupling state
-`model`: structure containing data, parameters, and hyperparameters

RETURN
-`fb`: a nested array whose element fb[s][m][t][j,k] corresponds to the t-th time step in the m-th trial of the s-th trialset and the j-th accumulator state and k-th coupling state
"""
function sortbytrial(γ::Vector{<:Matrix{<:Vector{T}}}, model::Model) where {T<:Real}
	@unpack K, Ξ = model.options
	fb = map(model.trialsets) do trialset
			map(trialset.trials) do trial
				collect(zeros(T, Ξ, K) for i=1:trial.ntimesteps)
			end
		end
	for s in eachindex(fb)
		τ = 0
		for m in eachindex(fb[s])
			for t in eachindex(fb[s][m])
				τ += 1
				for j=1:Ξ
					for k=1:K
						fb[s][m][t][j,k] = γ[s][j,k][τ]
					end
				end
			end
		end
	end
	return fb
end
