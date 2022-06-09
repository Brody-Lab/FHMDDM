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
	real2native(r,q,l,u)

Convert a hyperparameter from real space to native space.

ARGUMENT
-`r`: vector of values in real space
-`l`: lower bound in native space
-`u`: upper bound in native space

RETURN
-scalar representing the value in native space
"""
function real2native(r::Real, l::Real, u::Real)
	l + (u-l)*logistic(r)
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
function real2native(options::Options, θreal::Latentθ)
	θnative = Latentθ()
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
	real2native(r,q,l,u)

Convert hyperparameters from native space to real space

ARGUMENT
-`n`: value in native space
-`l`: lower bound in native space
-`u`: upper bound in native space

RETURN
-vector representing the values in real space

"""
function native2real(n::Real, l::Real, u::Real)
	logit((n-l)/(u-l))
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
	Dict(	"a_basis_per_s"=>options.a_basis_per_s,
			"a_latency_s"=>options.a_latency_s,
			"basistype"=>options.basistype,
			"datapath"=>options.datapath,
			"dt"=>options.Δt,
			"K"=>options.K,
			"fit_B"=>options.fit_B,
			"fit_k"=>options.fit_k,
			"fit_lambda"=>options.fit_λ,
			"fit_mu0"=>options.fit_μ₀,
			"fit_phi"=>options.fit_ϕ,
			"fit_psi"=>options.fit_ψ,
			"fit_sigma2_a"=>options.fit_σ²ₐ,
			"fit_sigma2_i"=>options.fit_σ²ᵢ,
			"fit_sigma2_s"=>options.fit_σ²ₛ,
			"fit_w_h"=>options.fit_wₕ,
			"alpha0"=>options.α₀,
			"alpha0_choices"=>options.α₀_choices,
			"lqu_Ac11"=>options.lqu_Aᶜ₁₁,
			"lqu_Ac22"=>options.lqu_Aᶜ₂₂,
			"lqu_B"=>	options.lqu_B,
			"lqu_k"=>	options.lqu_k,
			"lqu_λ"=>	options.lqu_λ,
			"lqu_μ₀"=>	options.lqu_μ₀,
			"lqu_ϕ"=>	options.lqu_ϕ,
			"lqu_πᶜ₁"=>	options.lqu_πᶜ₁,
			"lqu_ψ"=>	options.lqu_ψ,
			"lqu_σ²ₐ"=>	options.lqu_σ²ₐ,
			"lqu_σ²ᵢ"=>	options.lqu_σ²ᵢ,
			"lqu_σ²ₛ"=>	options.lqu_σ²ₛ,
			"lqu_wₕ"=>	options.lqu_wₕ,
    		"minpa"=>	options.minpa,
    		"objective"=> options.objective,
			"resultspath"=>options.resultspath,
			"Xi"=>options.Ξ)
end

"""
    dictionary(trialset)

Convert an instance of `trialsetdata` into a `Dict`
"""
function dictionary(trialset::Trialset)
    Dict("mpGLMs" => map(mpGLM->Dict(mpGLM), trialset.mpGLMs),
         "trials" => map(trial->Dict(trial), trialset.trials))
end

"""
    dictionary(trial)

Convert an instance of `trialdata` into a `Dict`
"""
function dictionary(trial::Trial)
    Dict("choice" => trial.choice,
         "clicks" => Dict(trial.clicks),
		 "ntimesteps"=> trial.ntimesteps,
		 "previousanswer" => trial.previousanswer,
         "a"=>trial.a,
         "z"=>trial.z)
end

"""
    dictionary(clicks)

Convert an instance of `clicks` into a `Dict`
"""
function dictionary(clicks::Clicks)
    Dict("time" => 		clicks.time,
		 "source" => 	clicks.source,
         "left" => 		clicks.left,
         "right" =>		clicks.right)
end

"""
    dictionary(mpGLM::MixturePoissonGLM)

Convert into a dictionary a mixture of Poisson generalized linear model
"""
function dictionary(mpGLM::MixturePoissonGLM)
    Dict("dt"=>mpGLM.Δt,
	     "dxi_dB"=>mpGLM.d𝛏_dB,
		 "max_spikehistory_lag"=>mpGLM.max_spikehistory_lag,
         "Phi"=>mpGLM.Φ,
         "theta"=>dictionary(mpGLM.θ),
		 "V"=>mpGLM.𝐕,
		 "X"=>mpGLM.𝐗,
         "y"=>mpGLM.𝐲)
end

"""
    dictionary(θ::GLMθ)

Convert into a dictionary the parameters of a mixture of Poisson generalized linear model
"""
function dictionary(θ::GLMθ)
    Dict("u"=>θ.𝐮,
         "v"=>θ.𝐯)
end

"""
    dictionary(θ::Latentθ)

Convert an instance of `Latentθ` to a dictionary
"""
function dictionary(θ::Latentθ)
    Dict("Ac11"=>θ.Aᶜ₁₁[1],
		"Ac22"=>θ.Aᶜ₂₂[1],
		"B"=>θ.B[1],
		"k"=>θ.k[1],
		"lambda"=>θ.λ[1],
		"mu0"=>θ.μ₀[1],
		"phi"=>θ.ϕ[1],
		"pic1"=>θ.πᶜ₁[1],
		"psi"=>θ.ψ[1],
		"sigma2_a"=>θ.σ²ₐ[1],
		"sigma2_i"=>θ.σ²ᵢ[1],
		"sigma2_s"=>θ.σ²ₛ[1],
		"w_h"=>θ.wₕ[1])
end

"""
	dictionary(cvindices)

Convert an instance of 'CVIndices' to a dictionary
"""
function dictionary(cvindices::CVIndices)
	Dict("testingtrials" => cvindices.testingtrials,
		 "trainingtrials" => cvindices.trainingtrials,
		 "testingtimesteps" => cvindices.testingtimesteps,
		 "trainingtimesteps" => cvindices.trainingtimesteps)
end

"""
	dictionary(cvresults)

Convert an instance of `CVResults` to a dictionary
"""
function dictionary(cvresults::CVResults)
	Dict("cvindices" => map(dictionary, cvresults.cvindices),
		"theta0_native" => map(dictionary, cvresults.θ₀native),
		"theta_native" => map(dictionary, cvresults.θnative),
		"thetaglm" => map(glmθ->map(glmθ->map(glmθ->dictionary(glmθ), glmθ), glmθ), cvresults.glmθ),
		"lambdaDeltat" => cvresults.λΔt,
		"pchoice" => cvresults.pchoice,
		"rll_choice"=>cvresults.rll_choice,
		"rll_spikes"=>cvresults.rll_spikes)
end

"""
    Options(options::Dict)

Create an instance of `Options` from a Dict
"""
function Options(options::Dict)
	Options(a_basis_per_s = convert(Int64, options["a_basis_per_s"]),
			a_latency_s = options["a_latency_s"],
			basistype = options["basistype"],
			datapath = options["datapath"],
			Δt = options["dt"],
			K = convert(Int64, options["K"]),
			fit_B = options["fit_B"],
			fit_k = options["fit_k"],
			fit_λ = options["fit_lambda"],
			fit_μ₀ = options["fit_mu0"],
			fit_ϕ = options["fit_phi"],
			fit_ψ = options["fit_psi"],
			fit_σ²ₐ = options["fit_sigma2_a"],
			fit_σ²ᵢ = options["fit_sigma2_i"],
			fit_σ²ₛ = options["fit_sigma2_s"],
			fit_wₕ = options["fit_w_h"],
			α₀=options["alpha0"],
			α₀_choices=options["alpha0_choices"],
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
			objective = options["objective"],
			resultspath = options["resultspath"],
			Ξ = convert(Int64, options["Xi"]))
end

"""
    MixturePoissonGLM(dict)

Convert a dictionary into an instance of `MixturePoissonGLM`
"""
function MixturePoissonGLM(mpGLM::Dict)
    MixturePoissonGLM(Δt=mpGLM["dt"],
					d𝛏_dB=vec(mpGLM["dxi_dB"]),
					max_spikehistory_lag=mpGLM["max_spikehistory_lag"],
					Φ=mpGLM["Phi"],
                    θ=GLMθ(mpGLM["theta"]),
					𝐕=mpGLM["𝐕"],
					𝐗=mpGLM["𝐗"],
                    𝐲=vec(mpGLM["y"]))
end

"""
    GLMθ(dict)

Convert a dictionary into an instance of `GLMθ`
"""
function GLMθ(θ::Dict)
    GLMθ(𝐮=vec(mpGLM["u"]),
         𝐯=vec(map(𝐯ₖ->vec(𝐯ₖ), mpGLM["v"])))
end

"""
    Latentθ(θ)

Create an instance of `Latentθ` from a Dict
"""
function Latentθ(θ::Dict)
	Latentθ(Aᶜ₁₁=[θ["Ac11"]],
			Aᶜ₂₂=[θ["Ac22"]],
			B=[θ["B"]],
			k=[θ["k"]],
			λ=[θ["lambda"]],
			μ₀=[θ["mu0"]],
			ϕ=[θ["phi"]],
			πᶜ₁=[θ["pic1"]],
			ψ=[θ["psi"]],
			σ²ₐ=[θ["sigma2_a"]],
			σ²ᵢ=[θ["sigma2_i"]],
			σ²ₛ=[θ["sigma2_s"]],
			wₕ=[θ["w_h"]])
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
