"""
Gradient of the joint posterior probability of the latent variables

The gradient is computed for the m-th trial of the i-th trialset, the t-th timestep in that trialet, and for accumulator state iᵃ and coupling variable state iᶜ:
    ∇p(a = ξ(iᵃ), c = iᶜ ∣ 𝐘ᵢₘ, dᵢₘ)

The recommended dataset should have a few number of trials

ARGUMENT
-`i`: trialset
-`m`: trial
-`t`: time step
-`iᵃ`: index of the accumulator
-`iᶜ`: index of the coupling variable
-`model`: structure containing the data, parameters, and hyperparameters of a factorial hidden-Markov drift-diffusion model

RETURN
-a vector corresponding to the gradient of the posterior probability
"""
function ∇posterior(i::Integer, m::Integer, t::Integer, iᵃ::Integer, iᶜ::Integer, model::Model)
    @unpack options, θnative, θreal, trialsets = model
	@unpack Ξ, K = options
	trialinvariant = Trialinvariant(model; purpose="gradient")
	p𝐘𝑑=map(model.trialsets) do trialset
			map(trialset.trials) do trial
				map(1:trial.ntimesteps) do t
					ones(T,Ξ,K)
				end
			end
		end
    likelihood!(p𝐘𝑑, trialsets, θnative.ψ[1]) # `p𝐘𝑑` is the conditional likelihood p(𝐘ₜ, d ∣ aₜ, zₜ)
	∇posterior(p𝐘𝑑[i][m], mpGLMs, trialinvariant, θnative, trialsets[i].trials[m])
end
