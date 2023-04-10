"""
    dictionary(options)

Convert an instance of `Options` to a dictionary
"""
function dictionary(options::Options)
	s = options.nunits^options.choiceLL_scaling_exponent;
	Dict(	"a_latency_s"=>options.a_latency_s,
			"b_scalefactor"=>options.b_scalefactor/s,
			"choiceobjective"=>options.choiceobjective,
			"choiceLL_scaling_exponent"=>options.choiceLL_scaling_exponent,
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
			"fit_overdispersion"=>options.fit_overdispersion,
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
    		"sf_y"=>options.sf_y,
			"tbf_accu_begins0"=>options.tbf_accu_begins0,
			"tbf_accu_ends0"=>options.tbf_accu_ends0,
			"tbf_accu_hz"=>options.tbf_accu_hz,
			"tbf_accu_scalefactor"=>options.tbf_accu_scalefactor/s,
			"tbf_accu_stretch"=>options.tbf_accu_stretch,
			"tbf_gain_scalefactor"=>options.tbf_gain_scalefactor/s,
			"tbf_gain_maxfunctions"=>options.tbf_gain_maxfunctions,
			"tbf_hist_begins0"=>options.tbf_hist_begins0,
			"tbf_hist_dur_s"=>options.tbf_hist_dur_s,
			"tbf_hist_ends0"=>options.tbf_hist_ends0,
			"tbf_hist_hz"=>options.tbf_hist_hz,
			"tbf_hist_linear"=>options.tbf_hist_linear,
			"tbf_hist_scalefactor"=>options.tbf_hist_scalefactor/s,
			"tbf_hist_stretch"=>options.tbf_hist_stretch,
			"tbf_move_begins0"=>options.tbf_move_begins0,
			"tbf_move_dur_s"=>options.tbf_move_dur_s,
			"tbf_move_ends0"=>options.tbf_move_ends0,
			"tbf_move_hz"=>options.tbf_move_hz,
			"tbf_move_linear"=>options.tbf_move_linear,
			"tbf_move_scalefactor"=>options.tbf_move_scalefactor/s,
			"tbf_move_stretch"=>options.tbf_move_stretch,
			"tbf_period"=>options.tbf_period,
			"tbf_phot_begins0"=>options.tbf_phot_begins0,
			"tbf_phot_ends0"=>options.tbf_phot_ends0,
			"tbf_phot_hz"=>options.tbf_phot_hz,
			"tbf_phot_linear"=>options.tbf_phot_linear,
			"tbf_phot_scalefactor"=>options.tbf_phot_scalefactor/s,
			"tbf_phot_stretch"=>options.tbf_phot_stretch,
			"tbf_time_begins0"=>options.tbf_time_begins0,
			"tbf_time_dur_s"=>options.tbf_time_dur_s,
			"tbf_time_ends0"=>options.tbf_time_ends0,
			"tbf_time_hz"=>options.tbf_time_hz,
			"tbf_time_linear"=>options.tbf_time_linear,
			"tbf_time_scalefactor"=>options.tbf_time_scalefactor/s,
			"tbf_time_stretch"=>options.tbf_time_stretch,
			"Xi"=>options.Ξ)
end

"""
    dictionary(glmθ)

Convert into a dictionary the parameters of a mixture of Poisson generalized linear model
"""
function dictionary(glmθ::GLMθ)
    Dict("a"=>glmθ.a,
		"b"=>glmθ.b,
		"b_scalefactor"=>glmθ.b_scalefactor,
		"u"=>glmθ.𝐮,
		"v"=>glmθ.𝐯,
		"beta"=>glmθ.𝛃,
		("u_"*string(field)=>glmθ.𝐮[getfield(glmθ.indices𝐮, field)] for field in fieldnames(Indices𝐮))...)
end

"""
    Options(options::Dict)

Create an instance of `Options` from a Dict
"""
function Options(nunits::Integer, options::Dict)
	s = nunits^options["choiceLL_scaling_exponent"]
	Options(a_latency_s = options["a_latency_s"],
			b_scalefactor = options["b_scalefactor"]*s,
			choiceobjective=options["choiceobjective"],
			choiceLL_scaling_exponent=options["choiceLL_scaling_exponent"],
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
			fit_overdispersion = options["fit_overdispersion"],
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
			sf_y = options["sf_y"],
			tbf_accu_begins0 = options["tbf_accu_begins0"],
			tbf_accu_ends0 = options["tbf_accu_ends0"],
			tbf_accu_hz = options["tbf_accu_hz"],
			tbf_accu_scalefactor = options["tbf_accu_scalefactor"]*s,
			tbf_accu_stretch = options["tbf_accu_stretch"],
			tbf_gain_scalefactor = options["tbf_gain_scalefactor"]*s,
			tbf_gain_maxfunctions = convert(Int,options["tbf_gain_maxfunctions"]),
			tbf_hist_begins0 = options["tbf_hist_begins0"],
			tbf_hist_dur_s = options["tbf_hist_dur_s"],
			tbf_hist_ends0 = options["tbf_hist_ends0"],
			tbf_hist_hz = options["tbf_hist_hz"],
			tbf_hist_linear = options["tbf_hist_linear"],
			tbf_hist_scalefactor = options["tbf_hist_scalefactor"]*s,
			tbf_hist_stretch = options["tbf_hist_stretch"],
			tbf_move_begins0 = options["tbf_move_begins0"],
			tbf_move_dur_s = options["tbf_move_dur_s"],
			tbf_move_ends0 = options["tbf_move_ends0"],
			tbf_move_hz = options["tbf_move_hz"],
			tbf_move_linear = options["tbf_move_linear"],
			tbf_move_scalefactor = options["tbf_move_scalefactor"]*s,
			tbf_move_stretch = options["tbf_move_stretch"],
			tbf_period = options["tbf_period"],
			tbf_phot_begins0 = options["tbf_phot_begins0"],
			tbf_phot_ends0 = options["tbf_phot_ends0"],
			tbf_phot_hz = options["tbf_phot_hz"],
			tbf_phot_linear = options["tbf_phot_linear"],
			tbf_phot_scalefactor = options["tbf_phot_scalefactor"]*s,
			tbf_phot_stretch = options["tbf_phot_stretch"],
			tbf_time_begins0 = options["tbf_time_begins0"],
			tbf_time_dur_s = options["tbf_time_dur_s"],
			tbf_time_ends0 = options["tbf_time_ends0"],
			tbf_time_hz = options["tbf_time_hz"],
			tbf_time_linear = options["tbf_time_linear"],
			tbf_time_scalefactor = options["tbf_time_scalefactor"]*s,
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

"""
	dictionary(x)
"""
dictionary(x) = Dict((String(fieldname)=>getfield(x,fieldname) for fieldname in fieldnames(typeof(x)))...)
