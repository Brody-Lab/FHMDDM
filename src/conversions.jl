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
	Latentθ(Aᶜ₁₁ = [logit(θnative.Aᶜ₁₁[1]) - logit(options.q_Aᶜ₁₁)],
			Aᶜ₂₂ = [logit(θnative.Aᶜ₂₂[1]) - logit(options.q_Aᶜ₂₂)],
			B = [logit(θnative.B[1]/2/options.q_B)],
			k = [log(θnative.k[1]/options.q_k)],
			λ = [θnative.λ[1]],
			μ₀ = [θnative.μ₀[1]],
			ϕ = [logit(θnative.ϕ[1]) - logit(options.q_ϕ)],
			πᶜ₁ = [logit(θnative.πᶜ₁[1]) - logit(options.q_πᶜ₁)],
			ψ = [logit((θnative.ψ[1]-options.bound_ψ)/(1.0-options.bound_ψ)) - logit(options.q_ψ)],
			σ²ₐ = [log(θnative.σ²ₐ[1]/options.q_σ²ₐ)],
			σ²ᵢ = [log((θnative.σ²ᵢ[1]-options.bound_σ²ᵢ)/options.q_σ²ᵢ)],
			σ²ₛ = [log(θnative.σ²ₛ[1]/options.q_σ²ₛ)],
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
	θreal.Aᶜ₁₁[1] = logit(θnative.Aᶜ₁₁[1]) - logit(options.q_Aᶜ₁₁)
	θreal.Aᶜ₂₂[1] = logit(θnative.Aᶜ₂₂[1]) - logit(options.q_Aᶜ₂₂)
	θreal.B[1] = logit(θnative.B[1]/2/options.q_B)
	θreal.k[1] = log(θnative.k[1]/options.q_k)
	θreal.ϕ[1] = logit(θnative.ϕ[1]) - logit(options.q_ϕ)
	θreal.πᶜ₁[1] = logit(θnative.πᶜ₁[1]) - logit(options.q_πᶜ₁)
	θreal.ψ[1] = logit((θnative.ψ[1]-options.bound_ψ)/(1.0-options.bound_ψ)) - logit(options.q_ψ)
	θreal.σ²ₐ[1] = log(θnative.σ²ₐ[1]/options.q_σ²ₐ)
	θreal.σ²ᵢ[1] = log((θnative.σ²ᵢ[1]-options.bound_σ²ᵢ)/options.q_σ²ᵢ)
	θreal.σ²ₛ[1] = log(θnative.σ²ₛ[1]/options.q_σ²ₛ)
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
function real2native(options::Options,
                     θreal::Latentθ)
	Latentθ(Aᶜ₁₁ = [logistic(θreal.Aᶜ₁₁[1] + logit(options.q_Aᶜ₁₁))],
			Aᶜ₂₂ = [logistic(θreal.Aᶜ₂₂[1] + logit(options.q_Aᶜ₂₂))],
			B = [2options.q_B*logistic(θreal.B[1])],
			k = [options.q_k*exp(θreal.k[1])],
			λ = [1.0*θreal.λ[1]], # the multiplication by 1 is for ReverseDiff
			μ₀ = [1.0*θreal.μ₀[1]],
			ϕ = [logistic(θreal.ϕ[1] + logit(options.q_ϕ))],
			πᶜ₁ = [logistic(θreal.πᶜ₁[1] + logit(options.q_πᶜ₁))],
			ψ = [options.bound_ψ + (1.0-options.bound_ψ)*logistic(θreal.ψ[1] + logit(options.q_ψ))],
			σ²ₐ = [options.q_σ²ₐ*exp(θreal.σ²ₐ[1])],
			σ²ᵢ = [options.bound_σ²ᵢ + options.q_σ²ᵢ*exp(θreal.σ²ᵢ[1])],
			σ²ₛ = [options.q_σ²ₛ*exp(θreal.σ²ₛ[1])],
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
	θnative.Aᶜ₁₁[1] = logistic(θreal.Aᶜ₁₁[1] + logit(options.q_Aᶜ₁₁))
	θnative.Aᶜ₂₂[1] = logistic(θreal.Aᶜ₂₂[1] + logit(options.q_Aᶜ₂₂))
	θnative.B[1] = 2options.q_B*logistic(θreal.B[1])
	θnative.k[1] = options.q_k*exp(θreal.k[1])
	θnative.λ[1] = θreal.λ[1]
	θnative.μ₀[1] = θreal.μ₀[1]
	θnative.ϕ[1] = logistic(θreal.ϕ[1] + logit(options.q_ϕ))
	θnative.πᶜ₁[1] = logistic(θreal.πᶜ₁[1] + logit(options.q_πᶜ₁))
	θnative.ψ[1] = options.bound_ψ + (1.0-options.bound_ψ)*logistic(θreal.ψ[1] + logit(options.q_ψ))
	θnative.σ²ₐ[1] = options.q_σ²ₐ*exp(θreal.σ²ₐ[1])
	θnative.σ²ᵢ[1] = options.bound_σ²ᵢ + options.q_σ²ᵢ*exp(θreal.σ²ᵢ[1])
	θnative.σ²ₛ[1] = options.q_σ²ₛ*exp(θreal.σ²ₛ[1])
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
			"spikehistorylags"=>options.spikehistorylags,
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
    dictionary(θ::MixturePoissonGLM)

Convert into a dictionary a mixture of Poisson generalized linear model
"""
function dictionary(mpGLM::MixturePoissonGLM)
    Dict("dt"=>mpGLM.Δt,
         "K"=>mpGLM.K,
         "U"=>mpGLM.𝐔,
         "u"=>mpGLM.𝐮,
         "l"=>mpGLM.𝐥,
         "r"=>mpGLM.𝐫,
         "bfPhi"=>mpGLM.𝚽,
         "Phi"=>mpGLM.Φ,
         "xi"=>mpGLM.𝛏,
         "y"=>mpGLM.𝐲)
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
    Options(options::Dict)

Create an instance of `Options` from a Dict
"""
function Options(options::Dict)
	spikehistorylags = vec(convert.(Int64, options["spikehistorylags"]))
    if typeof(spikehistorylags)<:Integer
        spikehistorylags = [spikehistorylags]
    end
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
                  spikehistorylags = spikehistorylags,
                  Ξ = convert(Int64, options["Xi"]))
end

"""
    MixturePoissonGLM(dict)

Convert a dictionary into an instance of `MixturePoissonGLM`
"""
function MixturePoissonGLM(mpGLM::Dict)
    MixturePoissonGLM(Δt=mpGLM["dt"],
                      K=convert(Int, mpGLM["K"]),
                      𝐔=mpGLM["U"],
                      𝐮=vec(mpGLM["u"]),
                      𝐥=vec(mpGLM["l"]),
                      𝐫=vec(mpGLM["r"]),
                      Φ=mpGLM["Phi"],
                      𝚽=mpGLM["bfPhi"],
                      𝛏=vec(mpGLM["xi"]),
                      𝐲=vec(mpGLM["y"]))
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
