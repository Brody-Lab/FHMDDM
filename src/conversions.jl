"""
    native2real(options, θnative)

Map values of model parameters from native space to real space

ARGUMENT
-`options`: model settings
-`θnative: values of model parameters in their native space`

RETURN
-values of model parameters in real space
"""
function native2real(options::Options,
                     θnative::Latentθ)
	if options.bound_ψ == 0.0 || options.q_ψ == 0.0
 		ψreal = logit(θnative.ψ[1])
	else
		ψreal = logit((θnative.ψ[1]-options.bound_ψ) / (1.0-2.0*options.bound_ψ)) - logit(options.q_ψ)
	end
	Latentθ(Aᶜ₁₁ = [logit((θnative.Aᶜ₁₁[1]-options.bound_z)/(1.0-2.0*options.bound_z)) - logit(options.q_Aᶜ₁₁)],
			Aᶜ₂₂ = [logit((θnative.Aᶜ₂₂[1]-options.bound_z)/(1.0-2.0*options.bound_z)) - logit(options.q_Aᶜ₂₂)],
			B = [logit((θnative.B[1]-options.bound_B)/2/options.q_B)],
			k = [log(θnative.k[1]/options.q_k)],
			λ = [θnative.λ[1]],
			μ₀ = [θnative.μ₀[1]],
			ϕ = [logit(θnative.ϕ[1]) - logit(options.q_ϕ)],
			πᶜ₁ = [logit((θnative.πᶜ₁[1]-options.bound_z)/(1.0-2.0*options.bound_z)) - logit(options.q_πᶜ₁)],
			ψ 	= [ψreal],
			σ²ₐ = [log((θnative.σ²ₐ[1]-options.bound_σ²)/options.q_σ²ₐ)],
			σ²ᵢ = [log((θnative.σ²ᵢ[1]-options.bound_σ²)/options.q_σ²ᵢ)],
			σ²ₛ = [log((θnative.σ²ₛ[1]-options.bound_σ²) /options.q_σ²ₛ)],
			wₕ = [θnative.wₕ[1]])
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
function native2real!(θreal::Latentθ,
					  options::Options,
					  θnative::Latentθ)
	θreal.Aᶜ₁₁[1] = logit((θnative.Aᶜ₁₁[1]-options.bound_z)/(1.0-2.0*options.bound_z)) - logit(options.q_Aᶜ₁₁)
	θreal.Aᶜ₂₂[1] = logit((θnative.Aᶜ₂₂[1]-options.bound_z)/(1.0-2.0*options.bound_z)) - logit(options.q_Aᶜ₂₂)
	θreal.B[1] = logit((θnative.B[1]-options.bound_B)/2/options.q_B)
	θreal.k[1] = log(θnative.k[1]/options.q_k)
	θreal.λ[1] = θnative.λ[1]
	θreal.μ₀[1] = θnative.μ₀[1]
	θreal.ϕ[1] = logit(θnative.ϕ[1]) - logit(options.q_ϕ)
	θreal.πᶜ₁[1] = logit((θnative.πᶜ₁[1]-options.bound_z)/(1.0-2.0*options.bound_z)) - logit(options.q_πᶜ₁)
	if options.bound_ψ == 0.0 || options.q_ψ == 0.0
 		ψreal = logit(θnative.ψ[1])
	else
		ψreal = logit((θnative.ψ[1]-options.bound_ψ) / (1.0-2.0*options.bound_ψ)) - logit(options.q_ψ)
	end
	θreal.ψ[1] = ψreal
	θreal.σ²ₐ[1] = log((θnative.σ²ₐ[1]-options.bound_σ²)/options.q_σ²ₐ)
	θreal.σ²ᵢ[1] = log((θnative.σ²ᵢ[1]-options.bound_σ²)/options.q_σ²ᵢ)
	θreal.σ²ₛ[1] = log((θnative.σ²ₛ[1]-options.bound_σ²)/options.q_σ²ₛ)
	θreal.wₕ[1] = θnative.wₕ[1]
	return nothing
end

"""
	native2real!(g, options, θnative, θreal)

Convert each partial derivative from native space to real space

This involves multiplying each partial derivative in native space by the derivative of the parameter in native space with respect to the parameter in real space

ARGUMENT
-`g`: gradient
-`options`: model settings
-`θnative`: values of the parameters in native space
-`θreal`: values of the parameters in real space
"""
function native2real!(g::Latentθ,
					  options::Options,
					  θnative::Latentθ,
					  θreal::Latentθ)
	tmpAᶜ₁₁ = logistic(θreal.Aᶜ₁₁[1] + logit(options.q_Aᶜ₁₁))
	tmpAᶜ₂₂ = logistic(θreal.Aᶜ₂₂[1] + logit(options.q_Aᶜ₂₂))
	tmpπᶜ₁ 	= logistic(θreal.πᶜ₁[1] + logit(options.q_πᶜ₁))
	if options.bound_ψ == 0.0 || options.q_ψ == 0.0
		dψnative_dψreal = θnative.ψ[1]*(1-θnative.ψ[1])
	else
		tmpψ 	= logistic(θreal.ψ[1] + logit(options.q_ψ))
		f_bound_ψ = 1.0-2.0*options.bound_ψ
		dψnative_dψreal = f_bound_ψ*tmpψ*(1.0 - tmpψ)
	end
	f_bound_z = 1.0-2.0*options.bound_z
	g.Aᶜ₁₁[1] *= f_bound_z*tmpAᶜ₁₁*(1.0 - tmpAᶜ₁₁)
	g.Aᶜ₂₂[1] *= f_bound_z*tmpAᶜ₂₂*(1.0 - tmpAᶜ₂₂)
	fB = logistic(θreal.B[1])
	g.B[1] *= 2options.q_B*fB*(1-fB)
	g.k[1] *= θnative.k[1]
	g.ϕ[1] *= θnative.ϕ[1]*(1.0 - θnative.ϕ[1])
	g.πᶜ₁[1] *= f_bound_z*tmpπᶜ₁*(1.0 - tmpπᶜ₁)
	g.ψ[1]   *= dψnative_dψreal
	g.σ²ₐ[1] *= options.q_σ²ₐ*exp(θreal.σ²ₐ[1])
	g.σ²ᵢ[1] *= options.q_σ²ᵢ*exp(θreal.σ²ᵢ[1])
	g.σ²ₛ[1] *= options.q_σ²ₛ*exp(θreal.σ²ₛ[1])
	return nothing
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
function native2real!(∇ℓ::Vector{<:Real}, ∇∇ℓ::Matrix{<:Real}, latentθindex::Latentθ, model::Model)
	firstderivatives = differentiate_native_wrt_real(model)
	secondderivatives = differentiate_twice_native_wrt_real(model)
	for parametername in fieldnames(Latentθ)
		d1 = getfield(firstderivatives, parametername)[1]
		d2 = getfield(secondderivatives, parametername)[1]
		if d1 != 1.0
			i = getfield(latentθindex, parametername)[1]
			∇∇ℓ[i,:] .*= d1
			∇∇ℓ[:,i] .*= d1
			∇∇ℓ[i,i] += d2*∇ℓ[i]
			∇ℓ[i] *= d1
		end
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
	@unpack options, θreal, θnative = model
	tmpAᶜ₁₁ = logistic(θreal.Aᶜ₁₁[1] + logit(options.q_Aᶜ₁₁))
	tmpAᶜ₂₂ = logistic(θreal.Aᶜ₂₂[1] + logit(options.q_Aᶜ₂₂))
	tmpπᶜ₁ 	= logistic(θreal.πᶜ₁[1] + logit(options.q_πᶜ₁))
	tmpψ 	= logistic(θreal.ψ[1] + logit(options.q_ψ))
	f_bound_z = 1.0-2.0*options.bound_z
	f_bound_ψ = 1.0-2.0*options.bound_ψ
	d = Latentθ()
	d.Aᶜ₁₁[1] = f_bound_z*tmpAᶜ₁₁*(1.0 - tmpAᶜ₁₁)
	d.Aᶜ₂₂[1] = f_bound_z*tmpAᶜ₂₂*(1.0 - tmpAᶜ₂₂)
	fB = logistic(θreal.B[1])
	d.B[1] = 2options.q_B*fB*(1-fB)
	d.k[1] = θnative.k[1]
	d.λ[1] = 1.0
	d.μ₀[1] = 1.0
	d.ϕ[1] = θnative.ϕ[1]*(1.0 - θnative.ϕ[1])
	d.πᶜ₁[1] = f_bound_z*tmpπᶜ₁*(1.0 - tmpπᶜ₁)
	d.ψ[1] = f_bound_ψ*tmpψ*(1.0 - tmpψ)
	d.σ²ₐ[1] = options.q_σ²ₐ*exp(θreal.σ²ₐ[1])
	d.σ²ᵢ[1] = options.q_σ²ᵢ*exp(θreal.σ²ᵢ[1])
	d.σ²ₛ[1] = options.q_σ²ₛ*exp(θreal.σ²ₛ[1])
	d.wₕ[1] = 1.0
	return d
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
	@unpack options, θreal, θnative = model
	tmpAᶜ₁₁ = logistic(θreal.Aᶜ₁₁[1] + logit(options.q_Aᶜ₁₁))
	tmpAᶜ₂₂ = logistic(θreal.Aᶜ₂₂[1] + logit(options.q_Aᶜ₂₂))
	tmpπᶜ₁ 	= logistic(θreal.πᶜ₁[1] + logit(options.q_πᶜ₁))
	tmpψ 	= logistic(θreal.ψ[1] + logit(options.q_ψ))
	f_bound_z = 1.0-2.0*options.bound_z
	f_bound_ψ = 1.0-2.0*options.bound_ψ
	d = Latentθ()
	d.Aᶜ₁₁[1] = f_bound_z*(tmpAᶜ₁₁*(1-tmpAᶜ₁₁)^2 - tmpAᶜ₁₁^2*(1-tmpAᶜ₁₁))
	d.Aᶜ₂₂[1] = f_bound_z*(tmpAᶜ₂₂*(1-tmpAᶜ₂₂)^2 - tmpAᶜ₂₂^2*(1-tmpAᶜ₂₂))
	fB = logistic(θreal.B[1])
	d.B[1] = 2options.q_B*(fB*(1-fB)^2 - fB^2*(1-fB))
	d.k[1] = θnative.k[1]
	d.λ[1] = 0.0
	d.μ₀[1] = 0.0
	d.ϕ[1] = θnative.ϕ[1]*(1.0 - θnative.ϕ[1])^2 - θnative.ϕ[1]^2*(1.0 - θnative.ϕ[1])
	d.πᶜ₁[1] = f_bound_z*(tmpπᶜ₁*(1-tmpπᶜ₁)^2 - tmpπᶜ₁^2*(1-tmpπᶜ₁))
	d.ψ[1] = f_bound_ψ*(tmpψ*(1-tmpψ)^2 - tmpψ^2*(1-tmpψ))
	d.σ²ₐ[1] = options.q_σ²ₐ*exp(θreal.σ²ₐ[1])
	d.σ²ᵢ[1] = options.q_σ²ᵢ*exp(θreal.σ²ᵢ[1])
	d.σ²ₛ[1] = options.q_σ²ₛ*exp(θreal.σ²ₛ[1])
	d.wₕ[1] = 0.0
	return d
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
function real2native(options::Options,
                     θreal::Latentθ)
	if options.q_ψ == 0.0
		ψnative = options.bound_ψ + (1.0-2.0*options.bound_ψ)*logistic(θreal.ψ[1])
	else
		ψnative = options.bound_ψ + (1.0-2.0*options.bound_ψ)*logistic(θreal.ψ[1] + logit(options.q_ψ))
	end
	Latentθ(Aᶜ₁₁ = [options.bound_z + (1.0-2.0*options.bound_z)*logistic(θreal.Aᶜ₁₁[1] + logit(options.q_Aᶜ₁₁))],
			Aᶜ₂₂ = [options.bound_z + (1.0-2.0*options.bound_z)*logistic(θreal.Aᶜ₂₂[1] + logit(options.q_Aᶜ₂₂))],
			B = [options.bound_B + 2options.q_B*logistic(θreal.B[1])],
			k = [options.q_k*exp(θreal.k[1])],
			λ = [1.0*θreal.λ[1]], # the multiplication by 1 is for ReverseDiff
			μ₀ = [1.0*θreal.μ₀[1]],
			ϕ = [logistic(θreal.ϕ[1] + logit(options.q_ϕ))],
			πᶜ₁ = [options.bound_z + (1.0-2.0*options.bound_z)*logistic(θreal.πᶜ₁[1] + logit(options.q_πᶜ₁))],
			ψ   = [ψnative],
			σ²ₐ = [options.bound_σ² + options.q_σ²ₐ*exp(θreal.σ²ₐ[1])],
			σ²ᵢ = [options.bound_σ² + options.q_σ²ᵢ*exp(θreal.σ²ᵢ[1])],
			σ²ₛ = [options.bound_σ² + options.q_σ²ₛ*exp(θreal.σ²ₛ[1])],
			wₕ = [1.0*θreal.wₕ[1]])
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
function real2native!(θnative::Latentθ,
					  options::Options,
					  θreal::Latentθ)
	θnative.Aᶜ₁₁[1] = options.bound_z + (1.0-2.0*options.bound_z)*logistic(θreal.Aᶜ₁₁[1] + logit(options.q_Aᶜ₁₁))
	θnative.Aᶜ₂₂[1] = options.bound_z + (1.0-2.0*options.bound_z)*logistic(θreal.Aᶜ₂₂[1] + logit(options.q_Aᶜ₂₂))
	θnative.B[1] = options.bound_B + 2options.q_B*logistic(θreal.B[1])
	θnative.k[1] = options.q_k*exp(θreal.k[1])
	θnative.λ[1] = θreal.λ[1]
	θnative.μ₀[1] = θreal.μ₀[1]
	θnative.ϕ[1] = logistic(θreal.ϕ[1] + logit(options.q_ϕ))
	θnative.πᶜ₁[1] = options.bound_z + (1.0-2.0*options.bound_z)*logistic(θreal.πᶜ₁[1] + logit(options.q_πᶜ₁))
	if options.q_ψ == 0.0
		θnative.ψ[1] = options.bound_ψ + (1.0-2.0*options.bound_ψ)*logistic(θreal.ψ[1])
	else
		θnative.ψ[1] = options.bound_ψ + (1.0-2.0*options.bound_ψ)*logistic(θreal.ψ[1] + logit(options.q_ψ))
	end
	θnative.σ²ₐ[1] = options.bound_σ² + options.q_σ²ₐ*exp(θreal.σ²ₐ[1])
	θnative.σ²ᵢ[1] = options.bound_σ² + options.q_σ²ᵢ*exp(θreal.σ²ᵢ[1])
	θnative.σ²ₛ[1] = options.bound_σ² + options.q_σ²ₛ*exp(θreal.σ²ₛ[1])
	θnative.wₕ[1] = θreal.wₕ[1]
	return nothing
end

"""
    dictionary(options)

Convert an instance of `Options` to a dictionary
"""
function dictionary(options::Options)
	Dict(	"a_basis_per_s"=>options.a_basis_per_s,
			"a_latency_s"=>options.a_latency_s,
			"basistype"=>options.basistype,
			"bound_B"=>options.bound_B,
			"bound_psi"=>options.bound_ψ,
			"bound_sigma2"=>options.bound_σ²,
			"bound_z"=>options.bound_z,
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
			"initial_glm_L2_coefficient"=>options.initial_glm_L2_coefficient,
			"initial_ddm_L2_coefficient"=>options.initial_ddm_L2_coefficient,
			"q_Ac11"=>options.q_Aᶜ₁₁,
			"q_Ac22"=>options.q_Aᶜ₂₂,
			"q_B"=>options.q_B,
			"q_k"=>options.q_k,
			"q_phi"=>options.q_ϕ,
			"q_pic1"=>options.q_πᶜ₁,
			"q_psi"=>options.q_ψ,
			"q_sigma2_a"=>options.q_σ²ₐ,
			"q_sigma2_i"=>options.q_σ²ᵢ,
			"q_sigma2_s"=>options.q_σ²ₛ,
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
		 "losses"=>cvresults.losses,
		 "gradientnorms"=>cvresults.gradientnorms,
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
			bound_B = options["bound_B"],
			bound_ψ = options["bound_psi"],
			bound_σ² = options["bound_sigma2"],
			bound_z = options["bound_z"],
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
			initial_glm_L2_coefficient=options["initial_glm_L2_coefficient"],
			initial_ddm_L2_coefficient=options["initial_ddm_L2_coefficient"],
			q_Aᶜ₁₁ = options["q_Ac11"],
			q_Aᶜ₂₂ = options["q_Ac22"],
			q_B = options["q_B"],
			q_k = options["q_k"],
			q_ϕ = options["q_phi"],
			q_πᶜ₁ = options["q_pic1"],
			q_ψ = options["q_psi"],
			q_σ²ₐ = options["q_sigma2_a"],
			q_σ²ᵢ = options["q_sigma2_i"],
			q_σ²ₛ = options["q_sigma2_s"],
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
