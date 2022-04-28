"""
    maximizelikelihood!(model, optimizer)

Optimize the parameters of the factorial hidden Markov drift-diffusion model using a first-order optimizer in Optim

MODIFIED ARGUMENT
- a structure containing information for a factorial hidden Markov drift-diffusion model

UNMODIFIED ARGUMENT
-`optimizer`: an optimizer implemented by Optim.jl. The limited memory quasi-Newton algorithm `LBFGS()` does pretty well, and when using L-BFGS the `HagerZhang()` line search seems to do better than `BackTracking()`

OPTIONAL ARGUMENT
-`extended_trace`: save additional information
-`f_tol`: threshold for determining convergence in the objective value
-`g_tol`: threshold for determining convergence in the gradient
-`iterations`: number of inner iterations that will be run before the optimizer gives up
-`show_every`: trace output is printed every `show_every`th iteration.
-`show_trace`: should a trace of the optimizer's state be shown?
-`x_tol`: threshold for determining convergence in the input vector

RETURN
`losses`: value of the loss function (negative of the model's log-likelihood) across iterations. If `store_trace` were set to false, then these are NaN's
`gradientnorms`: 2-norm of the gradient of  of the loss function across iterations. If `store_trace` were set to false, then these are NaN's

EXAMPLE
```julia-repl
julia> using FHMDDM, Optim
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_02_18_test/data.mat"
julia> model = Model(datapath)
julia> losses, gradientnorms = maximizelikelihood!(model, LBFGS(linesearch = LineSearches.BackTracking()))
```
"""
function maximizelikelihood!(model::Model,
							 optimizer::Optim.FirstOrderOptimizer;
			                 extended_trace::Bool=false,
			                 f_tol::AbstractFloat=0.0,
			                 g_tol::AbstractFloat=1e-8,
			                 iterations::Integer=1000,
			                 show_every::Integer=10,
			                 show_trace::Bool=true,
							 store_trace::Bool=true,
			                 x_tol::AbstractFloat=0.0)
	memory = Memoryforgradient(model)
	@unpack K, Ξ = model.options
	γ =	map(model.trialsets) do trialset
			map(CartesianIndices((Ξ,K))) do index
				zeros(trialset.ntimesteps)
			end
		end
    f(concatenatedθ) = -loglikelihood!(model, memory, concatenatedθ)
    g!(∇, concatenatedθ) = ∇negativeloglikelihood!(∇, γ, model, memory, concatenatedθ)
    Optim_options = Optim.Options(extended_trace=extended_trace,
								  f_tol=f_tol,
                                  g_tol=g_tol,
                                  iterations=iterations,
                                  show_every=show_every,
                                  show_trace=show_trace,
								  store_trace=store_trace,
                                  x_tol=x_tol)
	θ₀ = copy(memory.concatenatedθ)
	optimizationresults = Optim.optimize(f, g!, θ₀, optimizer, Optim_options)
    maximumlikelihoodθ = Optim.minimizer(optimizationresults)
	sortparameters!(model, maximumlikelihoodθ, memory.indexθ)
	println(optimizationresults)
	losses, gradientnorms = fill(NaN, iterations+1), fill(NaN, iterations+1)
	if store_trace
		traces = Optim.trace(optimizationresults)
		for i in eachindex(traces)
			gradientnorms[i] = traces[i].g_norm
			losses[i] = traces[i].value
		end
	end
    return losses, gradientnorms
end

"""
    maximizelikelihood!(model, optimizer)

Optimize the parameters of the factorial hidden Markov drift-diffusion model using an algorithm implemented by Flux.jl

MODIFIED ARGUMENT
- a structure containing information for a factorial hidden Markov drift-diffusion model

UNMODIFIED ARGUMENT
-`optimizer`: optimization algorithm implemented by Flux.jl

OPTIONAL ARGUMENT
-`iterations`: number of inner iterations that will be run before the optimizer gives up

RETURN
`losses`: a vector of floating-point numbers indicating the value of the loss function (negative of the model's log-likelihood) across iterations. If `store_trace` were set to false, then these are NaN's
`gradientnorms`: a vector of floating-point numbers indicating the 2-norm of the gradient of  of the loss function across iterations. If `store_trace` were set to false, then these are NaN's

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> import Flux
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_02_18_test/data.mat"
julia> model = Model(datapath)
julia> losses, gradientnorms = maximizelikelihood!(model, Flux.ADAM())
```
"""
function maximizelikelihood!(model::Model,
							optimizer::Flux.Optimise.AbstractOptimiser;
							iterations::Integer = 3000)
	memory = Memoryforgradient(model)
	@unpack K, Ξ = model.options
	γ =	map(model.trialsets) do trialset
		map(CartesianIndices((Ξ,K))) do index
			zeros(trialset.ntimesteps)
		end
	end
	θ = copy(memory.concatenatedθ)
	∇ = similar(θ)
	local x, min_err, min_θ
	min_err = typemax(eltype(θ)) #dummy variables
	min_θ = copy(θ)
	losses, gradientnorms = fill(NaN, iterations+1), fill(NaN, iterations+1)
	optimizationtime = 0.0
	for i = 1:iterations
		iterationtime = @timed begin
			x = -loglikelihood!(model, memory, θ)
			∇negativeloglikelihood!(∇, γ, model, memory, θ)
			losses[i] = x
			gradientnorms[i] = norm(∇)
			if x < min_err  # found a better solution
				min_err = x
				min_θ = copy(θ)
			end
			Flux.update!(optimizer, θ, ∇)
		end
		optimizationtime += iterationtime[2]
		println("iteration=", i, ", loss= ", losses[i], ", gradient norm= ", gradientnorms[i], ", time(s)= ", optimizationtime)
	end
	sortparameters!(model, min_θ, memory.indexθ)
    return losses, gradientnorms
end

"""
    loglikelihood!(model, memory, concatenatedθ)

Compute the log-likelihood

ARGUMENT
-`model`: an instance of FHM-DDM
-`memory`: a container of variables used by both the log-likelihood and gradient computation

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values

RETURN
-log-likelihood

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_27_test/data.mat"; randomize=true);
julia> concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
julia> memory, P = FHMDDM.Memoryforgradient(model)
julia> ℓ = loglikelihood!(model, memory, memory.concatenatedθ)
julia> ℓ = loglikelihood!(model, memory, rand(length(memory.concatenatedθ)))
```
"""
function loglikelihood!(model::Model,
						memory::Memoryforgradient,
					    concatenatedθ::Vector{<:Real})
	if (concatenatedθ != memory.concatenatedθ) || isnan(memory.ℓ[1])
		P = update!(memory, model, concatenatedθ)
		memory.ℓ[1] = 0.0
		for s in eachindex(model.trialsets)
			for m in eachindex(model.trialsets[s].trials)
				memory.ℓ[1] += loglikelihood(memory.p𝐘𝑑[s][m], memory, P, model.θnative, model.trialsets[s].trials[m])
			end
		end
	end
	memory.ℓ[1]
end

"""
	loglikelihood(p𝐘𝑑, θnative, trial, trialinvariant)

Compute the log-likelihood of the data from one trial

ARGUMENT
-`p𝐘𝑑`: a matrix whose element `p𝐘𝑑[t][i,j]` represents the conditional likelihood `p(𝐘ₜ, d ∣ 𝐚ₜ=i, 𝐜ₜ=j)`
-`θnative`: model parameters in their native space
-`trial`: stimulus and behavioral information of one trial
-`trialinvariant`: a structure containing quantities that are used in each trial

RETURN
-`ℓ`: log-likelihood of the data from one trial
"""
function loglikelihood(p𝐘𝑑::Vector{<:Matrix{<:Real}},
   					   memory::Memoryforgradient,
					   P::Probabilityvector,
					   θnative::Latentθ,
					   trial::Trial)
	@unpack clicks = trial
	@unpack Aᵃinput, Aᵃsilent, Aᶜᵀ, πᶜᵀ = memory
	adaptedclicks = adapt(clicks, θnative.k[1], θnative.ϕ[1])
	priorprobability!(P, trial.previousanswer)
	pa₁ = P.𝛑
	f = p𝐘𝑑[1] .* pa₁ .* πᶜᵀ
	D = sum(f)
	f ./= D
	ℓ = log(D)
	@inbounds for t = 2:trial.ntimesteps
		if isempty(clicks.inputindex[t])
			Aᵃ = Aᵃsilent
		else
			Aᵃ = Aᵃinput[clicks.inputindex[t][1]]
			update_for_transition_probabilities!(P, adaptedclicks, clicks, t)
			transitionmatrix!(Aᵃ, P)
		end
		f = p𝐘𝑑[t].*(Aᵃ * f * Aᶜᵀ)
		D = sum(f)
		f ./= D
		ℓ += log(D)
	end
	return ℓ
end

"""
    loglikelihood(concatenatedθ, indexθ, model)

Compute the log-likelihood in a way that is compatible with ForwardDiff

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values
-`model`: an instance of FHM-DDM

RETURN
-log-likelihood

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_25_test/data.mat"; randomize=true);
julia> concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
julia> ℓ = loglikelihood(concatenatedθ, indexθ, model)
```
"""
function loglikelihood(	concatenatedθ::Vector{<:Real},
					    indexθ::Indexθ,
						model::Model)
	model = Model(concatenatedθ, indexθ, model)
	@unpack options, θnative, θreal, trialsets = model
	@unpack Ξ, K = options
	trialinvariant = Trialinvariant(model; purpose="loglikelihood")
	T = eltype(concatenatedθ)
	p𝐘𝑑=map(model.trialsets) do trialset
			map(trialset.trials) do trial
				map(1:trial.ntimesteps) do t
					ones(T,Ξ,K)
				end
			end
		end
    likelihood!(p𝐘𝑑, trialsets, θnative.ψ[1]) # `p𝐘𝑑` is the conditional likelihood p(𝐘ₜ, d ∣ aₜ, zₜ)
	ℓ = map(trialsets, p𝐘𝑑) do trialset, p𝐘𝑑
			map(trialset.trials, p𝐘𝑑) do trial, p𝐘𝑑
				loglikelihood(p𝐘𝑑, θnative, trial, trialinvariant)
			end
		end
	return sum(sum(ℓ))
end

"""
    ∇negativeloglikelihood!(∇, γ, model, memory, concatenatedθ)

Gradient of the negative log-likelihood of the factorial hidden Markov drift-diffusion model

MODIFIED INPUT
-`∇`: a vector of partial derivatives
-`γ`: posterior probability of the latent variables
-`model`: a structure containing information of the factorial hidden Markov drift-diffusion model
-`memory`: structure containing variables memory between computations of the model's log-likelihood and its gradient

UNMODIFIED ARGUMENT
-`concatenatedθ`: parameter values concatenated into a vetor
"""
function ∇negativeloglikelihood!(∇::Vector{<:AbstractFloat},
								 γ::Vector{<:Matrix{<:Vector{<:AbstractFloat}}},
								 model::Model,
								 memory::Memoryforgradient,
								 concatenatedθ::Vector{<:AbstractFloat})
	if concatenatedθ != memory.concatenatedθ
		update!(model, memory, concatenatedθ)
	end
	@unpack indexθ, p𝐘𝑑 = memory
	@unpack options, θnative, θreal, trialsets = model
	@unpack K = options
	trialinvariant = Trialinvariant(model; purpose="gradient")
	output=	map(trialsets, p𝐘𝑑) do trialset, p𝐘𝑑
				pmap(trialset.trials, p𝐘𝑑) do trial, p𝐘𝑑
					∇loglikelihood(p𝐘𝑑, trialinvariant, θnative, trial)
				end
			end
	latent∇ = output[1][1][1] # reuse this memory
	for field in fieldnames(Latentθ)
		latent∂ = getfield(latent∇, field)
		for i in eachindex(output)
			start = i==1 ? 2 : 1
			for m in start:length(output[i])
				latent∂[1] += getfield(output[i][m][1], field)[1] #output[i][m][1] are the partial derivatives
			end
		end
	end
	native2real!(latent∇, options, θnative, θreal)
	for field in fieldnames(Latentθ)
		index = getfield(indexθ.latentθ,field)[1]
		if index != 0
			∇[index] = -getfield(latent∇,field)[1] # note the negative sign
		end
	end
	@inbounds for i in eachindex(output)
        t = 0
        for m in eachindex(output[i])
            for tₘ in eachindex(output[i][m][2]) # output[i][m][2] is `fb` of trial `m` in trialset `i`
                t += 1
                for jk in eachindex(output[i][m][2][tₘ])
                    γ[i][jk][t] = output[i][m][2][tₘ][jk]
                end
            end
        end
    end
	∇glm = zeros(countparameters(trialsets[1].mpGLMs[1].θ))
	for i in eachindex(trialsets)
		for n in eachindex(trialsets[i].mpGLMs)
			∇negativeexpectation!(∇glm, γ[i], trialsets[i].mpGLMs[n])
			sortparameters!(∇, indexθ.glmθ[i][n], ∇glm)
		end
	end
	return nothing
end

"""
    ∇negativeloglikelihood(γ, model, memory, concatenatedθ)

Gradient of the negative log-likelihood implemented to be compatible with ForwardDiff

MODIFIED INPUT
-`γ`: posterior probability of the latent variables
-`model`: a structure containing information of the factorial hidden Markov drift-diffusion model
-`memory`: structure containing variables memory between computations of the model's log-likelihood and its gradient

UNMODIFIED ARGUMENT
-`concatenatedθ`: parameter values concatenated into a vetor
"""
function ∇negativeloglikelihood(concatenatedθ::Vector{T},
								indexθ::Indexθ,
								model::Model) where {T<:Real}
	model = sortparameters(concatenatedθ, indexθ, model)
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
	output=	map(trialsets, p𝐘𝑑) do trialset, p𝐘𝑑
				pmap(trialset.trials, p𝐘𝑑) do trial, p𝐘𝑑
					∇loglikelihood(p𝐘𝑑, trialinvariant, θnative, trial)
				end
			end
	latent∇ = output[1][1][1] # reuse this memory
	for field in fieldnames(Latentθ)
		latent∂ = getfield(latent∇, field)
		for i in eachindex(output)
			start = i==1 ? 2 : 1
			for m in start:length(output[i])
				latent∂[1] += getfield(output[i][m][1], field)[1] #output[i][m][1] are the partial derivatives
			end
		end
	end
	native2real!(latent∇, options, θnative, θreal)
	∇ = similar(concatenatedθ)
	for field in fieldnames(Latentθ)
		index = getfield(indexθ.latentθ,field)[1]
		if index != 0
			∇[index] = -getfield(latent∇,field)[1] # note the negative sign
		end
	end
	@unpack K, Ξ = model.options
	γ =	map(model.trialsets) do trialset
			map(CartesianIndices((Ξ,K))) do index
				zeros(T, trialset.ntimesteps)
			end
		end
	@inbounds for i in eachindex(output)
        t = 0
        for m in eachindex(output[i])
            for tₘ in eachindex(output[i][m][2]) # output[i][m][2] is `fb` of trial `m` in trialset `i`
                t += 1
                for jk in eachindex(output[i][m][2][tₘ])
                    γ[i][jk][t] = output[i][m][2][tₘ][jk]
                end
            end
        end
    end
	for i in eachindex(trialsets)
		for n in eachindex(trialsets[i].mpGLMs)
			∇glm = ∇negativeexpectation(γ[i], trialsets[i].mpGLMs[n])
			sortparameters!(∇, indexθ.glmθ[i][n], ∇glm)
		end
	end
	return ∇
end

"""
	∇loglikelihood(p𝐘𝑑, trialinvariant, θnative, trial)

Compute quantities needed for the gradient of the log-likelihood of the data observed in one trial

ARGUMENT
-`p𝐘𝑑`: A vector of matrices of floating-point numbers whose element `p𝐘𝑑[t][i,j]` represents the likelihood of the emissions (spike trains and choice) at time step `t` conditioned on the accumulator variable being in state `i` and the coupling variable in state `j`
-`trialinvariant`: structure containing quantities used across trials
-`θnative`: parameters for the latent variables in their native space
-`trial`: information on the stimuli and behavioral choice of one trial

RETURN
-`latent∇`: gradient of the log-likelihood of the data observed in one trial with respect to the parameters specifying the latent variables
-`fb`: joint posterior probabilities of the accumulator and coupling variables
"""
function ∇loglikelihood(p𝐘𝑑::Vector{<:Matrix{T}},
						trialinvariant::Trialinvariant,
						θnative::Latentθ,
						trial::Trial) where {T<:Real}
	@unpack clicks = trial
	@unpack inputtimesteps, inputindex = clicks
	@unpack Aᵃsilent, dAᵃsilentdμ, dAᵃsilentdσ², dAᵃsilentdB, Aᶜ, Aᶜᵀ, Δt, K, 𝛚, πᶜᵀ, Ξ, 𝛏 = trialinvariant
	dℓdk, dℓdλ, dℓdϕ, dℓdσ²ₐ, dℓdσ²ₛ, dℓdB = 0., 0., 0., 0., 0., 0.
	∑χᶜ = zeros(T, K,K)
	μ = θnative.μ₀[1] + trial.previousanswer*θnative.wₕ[1]
	σ = √θnative.σ²ᵢ[1]
	πᵃ, dπᵃdμ, dπᵃdσ², dπᵃdB = zeros(T, Ξ), zeros(T, Ξ), zeros(T, Ξ), zeros(T, Ξ)
	probabilityvector!(πᵃ, dπᵃdμ, dπᵃdσ², dπᵃdB, μ, 𝛚, σ, 𝛏)
	n_steps_with_input = length(clicks.inputtimesteps)
	Aᵃ = map(x->zeros(T, Ξ,Ξ), clicks.inputtimesteps)
	dAᵃdμ = map(x->zeros(T, Ξ,Ξ), clicks.inputtimesteps)
	dAᵃdσ² = map(x->zeros(T, Ξ,Ξ), clicks.inputtimesteps)
	dAᵃdB = map(x->zeros(T, Ξ,Ξ), clicks.inputtimesteps)
	Δc = zeros(T, n_steps_with_input)
	∑c = zeros(T, n_steps_with_input)
	adaptedclicks = ∇adapt(clicks, θnative.k[1], θnative.ϕ[1])
	@unpack C, dC_dk, dC_dϕ = adaptedclicks
	for i in 1:n_steps_with_input
		t = clicks.inputtimesteps[i]
		cL = sum(C[clicks.left[t]])
		cR = sum(C[clicks.right[t]])
		transitionmatrix!(Aᵃ[i], dAᵃdμ[i], dAᵃdσ²[i], dAᵃdB[i], cL, cR, trialinvariant, θnative)
		Δc[i] = cR-cL
		∑c[i] = cL+cR
	end
	D, f = forward(Aᵃ, inputindex, πᵃ, p𝐘𝑑, trialinvariant)
	fb = f # reuse memory
	b = ones(T, Ξ,K)
	Aᶜreshaped = reshape(Aᶜ, 1, 1, K, K)
	dμ_dΔc = differentiate_μ_wrt_Δc(Δt, θnative.λ[1])
	d²μ_dΔcdλ = differentiate_μ_wrt_Δc(Δt, θnative.λ[1])
	𝛏ᵀΔtexpλΔt = transpose(𝛏).*Δt.*exp(θnative.λ[1]*Δt)
	@inbounds for t = trial.ntimesteps:-1:1
		if t < trial.ntimesteps # backward step
			Aᵃₜ₊₁ = isempty(inputindex[t+1]) ? Aᵃsilent : Aᵃ[inputindex[t+1][1]]
			b = transpose(Aᵃₜ₊₁) * (b .* p𝐘𝑑[t+1] ./ D[t+1]) * Aᶜ
			fb[t] .*= b
		end
		if t > 1 # joint posterior over consecutive time bins, computations involving the transition matrix
			if isempty(inputindex[t])
				Aᵃₜ = Aᵃsilent
				dAᵃₜdμ = dAᵃsilentdμ
				dAᵃₜdσ² = dAᵃsilentdσ²
				dAᵃₜdB = dAᵃsilentdB
			else
				i = inputindex[t][1]
				Aᵃₜ = Aᵃ[i]
				dAᵃₜdμ = dAᵃdμ[i]
				dAᵃₜdσ² = dAᵃdσ²[i]
				dAᵃₜdB = dAᵃdB[i]
			end
			χ_oslash_Aᵃ = reshape(p𝐘𝑑[t].*b, Ξ, 1, K, 1) .* reshape(f[t-1], 1, Ξ, 1, K) .* Aᶜreshaped ./ D[t]
	        if K == 2
		        ∑χᶜ += dropdims(sum(χ_oslash_Aᵃ.*Aᵃₜ, dims=(1,2)); dims=(1,2))
			end
			χᵃ_Aᵃ = dropdims(sum(χ_oslash_Aᵃ, dims=(3,4)); dims=(3,4))
			χᵃ_dlogAᵃdμ = χᵃ_Aᵃ .* dAᵃₜdμ # χᵃ⊙ d/dμ{log(Aᵃ)} = χᵃ⊘ Aᵃ⊙ d/dμ{Aᵃ}
			∑_χᵃ_dlogAᵃdμ = sum(χᵃ_dlogAᵃdμ)
			∑_χᵃ_dlogAᵃdσ² = sum(χᵃ_Aᵃ .* dAᵃₜdσ²) # similarly, χᵃ⊙ d/dσ²{log(Aᵃ)} = χᵃ⊘ Aᵃ⊙ d/dσ²{Aᵃ}
			dℓdσ²ₐ += ∑_χᵃ_dlogAᵃdσ² # the Δt is multiplied after summing across time steps
			dℓdB += sum(χᵃ_Aᵃ .* dAᵃₜdB)
			if isempty(inputindex[t])
				dμdλ = 𝛏ᵀΔtexpλΔt
			else
				dμdλ = 𝛏ᵀΔtexpλΔt .+ Δc[i].*d²μ_dΔcdλ
				dℓdσ²ₛ += ∑_χᵃ_dlogAᵃdσ²*∑c[i]
				dcLdϕ = sum(dC_dϕ[clicks.left[t]])
				dcRdϕ = sum(dC_dϕ[clicks.right[t]])
				dcLdk = sum(dC_dk[clicks.left[t]])
				dcRdk = sum(dC_dk[clicks.right[t]])
				dσ²dϕ = θnative.σ²ₛ[1]*(dcLdϕ + dcRdϕ)
				dσ²dk = θnative.σ²ₛ[1]*(dcLdk + dcRdk)
				dℓdϕ += ∑_χᵃ_dlogAᵃdμ*dμ_dΔc*(dcRdϕ - dcLdϕ) + ∑_χᵃ_dlogAᵃdσ²*dσ²dϕ
				dℓdk += ∑_χᵃ_dlogAᵃdμ*dμ_dΔc*(dcRdk - dcLdk) + ∑_χᵃ_dlogAᵃdσ²*dσ²dk
			end
			dℓdλ += sum(χᵃ_dlogAᵃdμ.*dμdλ)
		end
	end
	dℓdσ²ₐ *= Δt
	if K == 2
		dℓdAᶜ₁₁ = ∑χᶜ[1,1]/Aᶜ[1,1] - ∑χᶜ[2,1]/Aᶜ[2,1]
		dℓdAᶜ₂₂ = ∑χᶜ[2,2]/Aᶜ[2,2] - ∑χᶜ[1,2]/Aᶜ[1,2]
		∑γᶜ₁ = sum(fb[1], dims=1)
		dℓdxπᶜ₁ = (∑γᶜ₁[1] - θnative.πᶜ₁[1])/θnative.πᶜ₁[1]/(1.0 - θnative.πᶜ₁[1])
	else
		dℓdAᶜ₁₁ = 0.0
		dℓdAᶜ₂₂ = 0.0
		dℓdxπᶜ₁ = 0.0
	end
	γᵃ₁_oslash_πᵃ = sum(p𝐘𝑑[1] .* πᶜᵀ ./ D[1] .* b, dims=2)
	∑_γᵃ₁_dlogπᵃdμ = γᵃ₁_oslash_πᵃ ⋅ dπᵃdμ # similar to above, γᵃ₁⊙ d/dμ{log(πᵃ)} = γᵃ₁⊘ πᵃ⊙ d/dμ{πᵃ}
	dℓdμ₀ = ∑_γᵃ₁_dlogπᵃdμ
	dℓdwₕ = ∑_γᵃ₁_dlogπᵃdμ * trial.previousanswer
	dℓdσ²ᵢ = γᵃ₁_oslash_πᵃ ⋅ dπᵃdσ²
	dℓdB += γᵃ₁_oslash_πᵃ ⋅ dπᵃdB
	dℓdψ = differentiateℓ_wrt_ψ(trial.choice, f[end], θnative.ψ[1])
	latent∇ = Latentθ(	Aᶜ₁₁ = [dℓdAᶜ₁₁],
						Aᶜ₂₂ = [dℓdAᶜ₂₂],
						k	 = [dℓdk],
						λ	 = [dℓdλ],
						μ₀	 = [dℓdμ₀],
						ϕ	 = [dℓdϕ],
						πᶜ₁	 = [dℓdxπᶜ₁],
						ψ	 = [dℓdψ],
						σ²ₐ	 = [dℓdσ²ₐ],
						σ²ᵢ	 = [dℓdσ²ᵢ],
						σ²ₛ	 = [dℓdσ²ₛ],
						wₕ	 = [dℓdwₕ],
						B	 = [dℓdB])
	return latent∇, fb
end

"""
	differentiateℓ_wrt_ψ(choice, γ_end, ψ)

Partial derivative of the log-likelihood of the data from one trial with respect to the lapse rate ψ

ARGUMENT
-`choice`: a Boolean specifying whether the choice was to the right
-`γ_end`: a matrix of floating-point numbers representing the posterior likelihood of the latent variables at the end of the trial (i.e., last time step). Element `γ_end[i,j]` = p(aᵢ=1, cⱼ=1 ∣ 𝐘, d). Rows correspond to states of the accumulator state variable 𝐚, and columns to states of the coupling variable 𝐜.
-`ψ`: a floating-point number specifying the lapse rate

RETURN
-a floating-point number quantifying the partial derivative of the log-likelihood of one trial's data with respect to the lapse rate ψ
"""
function differentiateℓ_wrt_ψ(choice::Bool, γ_end::Array{<:Real}, ψ::Real)
	γᵃ_end = sum(γ_end, dims=2)
	zeroindex = cld(length(γᵃ_end), 2)
	if choice
		choiceconsistent   = sum(γᵃ_end[zeroindex+1:end])
		choiceinconsistent = sum(γᵃ_end[1:zeroindex-1])
	else
		choiceconsistent   = sum(γᵃ_end[1:zeroindex-1])
		choiceinconsistent = sum(γᵃ_end[zeroindex+1:end])
	end
	return choiceconsistent/(ψ-2) + choiceinconsistent/ψ
end

"""
	Trialinvariant(options, θnative)

Compute quantities that are used in each trial for computing gradient of the log-likelihood

ARGUMENT
-`model`: custom type containing the settings, data, and parameters of a factorial hidden Markov drift-diffusion model
"""
function Trialinvariant(model::Model; purpose="gradient")
	@unpack options, θnative, θreal = model
	@unpack Δt, K, Ξ = options
	λ = θnative.λ[1]
	B = θnative.B[1]
	Aᶜ₁₁ = θnative.Aᶜ₁₁[1]
	Aᶜ₂₂ = θnative.Aᶜ₂₂[1]
	πᶜ₁ = θnative.πᶜ₁[1]
	if K == 2
		Aᶜ = [Aᶜ₁₁ 1-Aᶜ₂₂; 1-Aᶜ₁₁ Aᶜ₂₂]
		Aᶜᵀ = [Aᶜ₁₁ 1-Aᶜ₁₁; 1-Aᶜ₂₂ Aᶜ₂₂]
		πᶜᵀ = [πᶜ₁ 1-πᶜ₁]
	else
		Aᶜ = ones(1,1)
		Aᶜᵀ = ones(1,1)
		πᶜᵀ = ones(1,1)
	end
	𝛏 = B*(2collect(1:Ξ) .- Ξ .- 1)/(Ξ-2)
	𝛍 = conditionedmean(0.0, Δt, θnative.λ[1], 𝛏)
	σ = √(θnative.σ²ₐ[1]*Δt)
	T = eltype(𝛍)
	Aᵃsilent = zeros(T,Ξ,Ξ)
	if purpose=="gradient"
		𝛚 = (2collect(1:Ξ) .- Ξ .- 1)/2
		Ω = 𝛚 .- transpose(𝛚).*exp.(λ.*Δt)
		dAᵃsilentdμ = zeros(T,Ξ,Ξ)
		dAᵃsilentdσ² = zeros(T,Ξ,Ξ)
		dAᵃsilentdB = zeros(T,Ξ,Ξ)
		transitionmatrix!(Aᵃsilent, dAᵃsilentdμ, dAᵃsilentdσ², dAᵃsilentdB, 𝛍, σ, Ω, 𝛏)
		Trialinvariant( Aᵃsilent=Aᵃsilent,
					Aᶜ=Aᶜ,
					Aᶜᵀ=Aᶜᵀ,
					dAᵃsilentdμ=dAᵃsilentdμ,
					dAᵃsilentdσ²=dAᵃsilentdσ²,
					dAᵃsilentdB=dAᵃsilentdB,
					Δt=options.Δt,
					𝛚=𝛚,
					Ω=Ω,
					πᶜᵀ=πᶜᵀ,
					Ξ=Ξ,
 				    K=K,
					𝛏=𝛏)
	elseif purpose=="loglikelihood"
		transitionmatrix!(Aᵃsilent, 𝛍, σ, 𝛏)
		Trialinvariant(Aᵃsilent=Aᵃsilent,
				   Aᶜᵀ=Aᶜᵀ,
				   Δt=options.Δt,
				   πᶜᵀ=πᶜᵀ,
				   𝛏=𝛏,
				   K=K,
				   Ξ=Ξ)
	end
end

"""
	Memoryforgradient(model)

Create variables that are memory by the computations of the log-likelihood and its gradient

ARGUMENT
-`model`: structure with information about the factorial hidden Markov drift-diffusion model

OUTPUT
-an instance of the custom type `Memoryforgradient`, which contains the memory quantities
-`P`: an instance of `Probabilityvector`

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_27_test/data.mat"; randomize=true);
julia> memory, P = FHMDDM.Memoryforgradient(model)
```
"""
function Memoryforgradient(model::Model)
	@unpack options, θnative = model
	@unpack Δt, K, Ξ = options
	Aᶜ₁₁ = θnative.Aᶜ₁₁[1]
	Aᶜ₂₂ = θnative.Aᶜ₂₂[1]
	πᶜ₁ = θnative.πᶜ₁[1]
	if K == 2
		Aᶜ = [Aᶜ₁₁ 1-Aᶜ₂₂; 1-Aᶜ₁₁ Aᶜ₂₂]
		∇Aᶜ = [[1.0 0.0; -1.0 0.0], [0.0 -1.0; 0.0 1.0]]
		πᶜ = [πᶜ₁, 1-πᶜ₁]
		∇πᶜ = [[1.0, -1.0]]
	else
		Aᶜ = ones(1,1)
		∇Aᶜ = [zeros(1,1), zeros(1,1)]
		πᶜ = ones(1)
		∇πᶜ = [zeros(1)]
	end
	indexθ_pa₁ = [3,6,11,13]
	indexθ_paₜaₜ₋₁ = [3,4,5,7,10,12]
	indexθ_pc₁ = [8]
	indexθ_pcₜcₜ₋₁ = [1,2]
	indexθ_ψ = [9]
	nθ_pa₁ = length(indexθ_pa₁)
	nθ_paₜaₜ₋₁ = length(indexθ_paₜaₜ₋₁)
	maxclicks = 0
	maxtimesteps = 0
	@inbounds for trialset in model.trialsets
		for trial in trialset.trials
			maxclicks = max(maxclicks, length(trial.clicks.time))
			maxtimesteps = max(maxtimesteps, trial.ntimesteps)
		end
	end
	Aᵃinput=map(1:maxclicks) do t
				A = zeros(Ξ,Ξ)
				A[1,1] = A[Ξ,Ξ] = 1.0
				return A
			end
	∇Aᵃinput = collect(collect(zeros(Ξ,Ξ) for q=1:nθ_paₜaₜ₋₁) for t=1:maxclicks)
	Aᵃsilent = zeros(Ξ, Ξ)
	∇Aᵃsilent = map(i->zeros(Ξ,Ξ), 1:nθ_paₜaₜ₋₁)
	Aᵃsilent[1,1] = Aᵃsilent[Ξ, Ξ] = 1.0
	P = Probabilityvector(Δt, θnative, Ξ)
	update_for_∇transition_probabilities!(P)
	∇transitionmatrix!(∇Aᵃsilent, Aᵃsilent, P)
	p𝐘𝑑 = likelihood(model)
	concatenatedθ, indexθ = concatenateparameters(model)
	memory = Memoryforgradient(Aᵃinput=Aᵃinput,
								∇Aᵃinput=∇Aᵃinput,
								Aᵃsilent=Aᵃsilent,
								∇Aᵃsilent=∇Aᵃsilent,
								Aᶜ=Aᶜ,
								∇Aᶜ=∇Aᶜ,
								concatenatedθ=concatenatedθ,
								D = zeros(maxtimesteps),
								Δt=options.Δt,
								f=collect(zeros(Ξ,K) for t=1:maxtimesteps),
								indexθ=indexθ,
								indexθ_pa₁=indexθ_pa₁,
								indexθ_paₜaₜ₋₁=indexθ_paₜaₜ₋₁,
								indexθ_pc₁=indexθ_pc₁,
								indexθ_pcₜcₜ₋₁=indexθ_pcₜcₜ₋₁,
								indexθ_ψ=indexθ_ψ,
								K=K,
								∇pa₁ = collect(zeros(Ξ) for q=1:nθ_pa₁),
								πᶜ=πᶜ,
								∇πᶜ=∇πᶜ,
								p𝐘𝑑=p𝐘𝑑,
								Ξ=Ξ)
	return memory, P
end

"""
	update!(model, memory, concatenatedθ)

Update the model and the memory quantities according to new parameter values

MODIFIED ARGUMENT
-`memory`: structure containing variables memory between computations of the model's log-likelihood and its gradient
-`model`: structure with information concerning a factorial hidden Markov drift-diffusion model

ARGUMENT
-`concatenatedθ`: newest values of the model's parameters

RETURN
-`P`: an instance of `Probabilityvector`

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_27_test/data.mat"; randomize=true);
julia> memory, P = FHMDDM.Memoryforgradient(model)
julia> P = update!(model, memory, rand(length(memory.concatenatedθ)))
```
"""
function update!(memory::Memoryforgradient,
				 model::Model,
				 concatenatedθ::Vector{<:Real})
	memory.concatenatedθ .= concatenatedθ
	sortparameters!(model, memory.concatenatedθ, memory.indexθ)
	if !isempty(memory.p𝐘𝑑[1][1][1])
	    likelihood!(memory.p𝐘𝑑, model.trialsets, model.θnative.ψ[1])
	end
	@unpack options, θnative = model
	@unpack Δt, K, Ξ = options
	P = Probabilityvector(Δt, θnative, Ξ)
	update_for_∇transition_probabilities!(P)
	∇transitionmatrix!(memory.∇Aᵃsilent, memory.Aᵃsilent, P)
	if K == 2
		Aᶜ₁₁ = θnative.Aᶜ₁₁[1]
		Aᶜ₂₂ = θnative.Aᶜ₂₂[1]
		πᶜ₁ = θnative.πᶜ₁[1]
		memory.Aᶜ .= [Aᶜ₁₁ 1-Aᶜ₂₂; 1-Aᶜ₁₁ Aᶜ₂₂]
		memory.πᶜ .= [πᶜ₁, 1-πᶜ₁]
	end
	return P
end
