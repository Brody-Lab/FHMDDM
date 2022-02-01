"""
	Set of functions compatible with ReverseDiff.jl and ForwardDiff.jl
"""

"""
    loglikelihood!(concatenatedθ, indexθ, model)

Computation the log-likelihood meant for automatic differentiation

ARGUMENT
-`model`: an instance of FHM-DDM
-`p𝐘𝑑`: a nested matrix representing the conditional likelihood of the emissions.

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values

RETURN
-log-likelihood
"""
function loglikelihood(	concatenatedθ,
					    indexθ::Indexθ,
						model::FHMDDM)
	model = sortparameters(concatenatedθ, indexθ, model)
	@unpack options, θnative, θreal, trialsets = model
	@unpack Ξ, K = options
	trialinvariant = Trialinvariant(options, θnative; purpose="loglikelihood")
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
			pmap(trialset.trials, p𝐘𝑑) do trial, p𝐘𝑑
				loglikelihood(p𝐘𝑑, trialinvariant, θnative, trial)
			end
		end
	return sum(sum(ℓ))
end


"""
    likelihood!(p𝐘𝑑, trialset, ψ)

Update the conditional likelihood of the emissions (spikes and/or behavioral choice)

MODIFIED ARGUMENT
-`p𝐘𝑑`: Condition probability of the emissions (spikes and/or choice) at each time bin. For time bins of each trial other than the last, it is the product of the conditional likelihood of all spike trains. For the last time bin, it corresponds to the product of the conditional likelihood of the spike trains and the choice. Element p𝐘𝑑[i][m][t][j,k] corresponds to ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ, zₜ=k) across N neural units at the t-th time bin in the m-th trial of the i-th trialset. The last element p𝐘𝑑[i][m][end][j,k] of each trial corresponds to p(𝑑 | aₜ = ξⱼ, zₜ=k) ∏ₙᴺ p(𝐲ₙ(t) | aₜ = ξⱼ, zₜ=k)

UNMODIFIED ARGUMENT
-`trialsets`: data used to constrain the model
-`ψ`: lapse rate

RETURN
-`nothing`
"""
function likelihood!(p𝐘𝑑,
                     trialsets::Vector{<:Trialset},
                     ψ::T) where {T<:Real}
	Ξ = size(p𝐘𝑑[1][1][end],1)
	K = size(p𝐘𝑑[1][1][end],2)
	zeroindex = cld(Ξ,2)
    @inbounds for i in eachindex(p𝐘𝑑)
		N = length(trialsets[i].mpGLMs)
		𝐩decoupled = ones(T, size(trialsets[i].mpGLMs[1].𝐲))
		for n = 1:N
			likelihood!(𝐩decoupled, trialsets[i].mpGLMs[n], zeroindex, 2)
		end
	    for j = 1:Ξ
	        for k = 1:K
	            if k == 2 || j==zeroindex
					𝐩 = 𝐩decoupled
				else
					𝐩 = ones(T, size(trialsets[i].mpGLMs[1].𝐲))
		            for n = 1:N
					    likelihood!(𝐩, trialsets[i].mpGLMs[n], j, k)
		            end
				end
	            t = 0
	            for m in eachindex(p𝐘𝑑[i])
	                for tₘ in eachindex(p𝐘𝑑[i][m])
	                    t += 1
	                    p𝐘𝑑[i][m][tₘ][j,k] = 𝐩[t]
	                end
	            end
	        end
	    end
		for m in eachindex(p𝐘𝑑[i])
			likelihood!(p𝐘𝑑[i][m][end], trialsets[i].trials[m].choice, ψ; zeroindex=zeroindex)
		end
    end
    return nothing
end

"""
	sortparameters(concatenatedθ, indexθ, model)

Sort a vector of concatenated parameter values

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values
-`model`: the model with old parameter values

RETURN
-`model`: the model with new parameter values
"""
function sortparameters(concatenatedθ,
				 		indexθ::Indexθ,
						model::FHMDDM)
	T = eltype(concatenatedθ)
	θreal = Latentθ((zeros(T,1) for field in fieldnames(Latentθ))...)
	for field in fieldnames(Latentθ) # `Latentθ` is the type of `indexθ.latentθ`
		index = getfield(indexθ.latentθ, field)[1]
		if index != 0 # an index of 0 indicates that the parameter is not being fit
			getfield(θreal, field)[1] = concatenatedθ[index]
		end
	end
	θnative = real2native(model.options, θreal)
	trialsets = map(model.trialsets, indexθ.𝐮, indexθ.𝐥, indexθ.𝐫) do trialset, index𝐮, index𝐥, index𝐫
					mpGLMs =map(trialset.mpGLMs, index𝐮, index𝐥, index𝐫) do mpGLM, index𝐮, index𝐥, index𝐫
								MixturePoissonGLM(	Δt=mpGLM.Δt,
													K=mpGLM.K,
													𝚽=mpGLM.𝚽,
													Φ=mpGLM.Φ,
													𝐔=mpGLM.𝐔,
													𝐗=mpGLM.𝐗,
													𝛏=mpGLM.𝛏,
													𝐲=mpGLM.𝐲,
													𝐮=concatenatedθ[index𝐮],
													𝐥=concatenatedθ[index𝐥],
													𝐫=concatenatedθ[index𝐫])
							end
					Trialset(mpGLMs=mpGLMs, trials=trialset.trials)
				end
	FHMDDM(	options = model.options,
			θnative = θnative,
			θ₀native=model.θ₀native,
			θreal = θreal,
			trialsets=trialsets)
end
