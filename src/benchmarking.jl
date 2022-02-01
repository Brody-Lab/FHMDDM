"""
	automatic gradient
"""
function forwardgradient!(∇::Vector{<:AbstractFloat},
							concatenatedθ::Vector{<:AbstractFloat},
							indexθ::Indexθ,
							model::FHMDDM)
	f(concatenatedθ) = -loglikelihood(concatenatedθ, indexθ, model)
	∇ .= ForwardDiff.gradient(f, concatenatedθ)
	return nothing
end

"""
	automatic gradient in reverse mode
"""
function reversegradient!(∇::Vector{<:AbstractFloat},
							concatenatedθ::Vector{<:AbstractFloat},
							indexθ::Indexθ,
							model::FHMDDM)
	f(concatenatedθ) = -loglikelihood(concatenatedθ, indexθ, model)
	ReverseDiff.gradient!(∇, f, concatenatedθ)
	return nothing
end

"""
	compute γ
"""
function posteriors(model::FHMDDM)
	@unpack options, θnative, θreal, trialsets = model
	@unpack Ξ, K = options
	p𝐘𝑑=map(model.trialsets) do trialset
	        map(trialset.trials) do trial
	            map(1:trial.ntimesteps) do t
	                ones(Ξ,K)
	            end
	        end
	    end;
	γ =	map(model.trialsets) do trialset
	        map(CartesianIndices((Ξ,K))) do index
	            zeros(trialset.ntimesteps)
	        end
	    end;
	trialinvariant = Trialinvariant(options, θnative; purpose="gradient")
	likelihood!(p𝐘𝑑, trialsets, θnative.ψ[1]) # `p𝐘𝑑` is the conditional likelihood p(𝐘ₜ, d ∣ aₜ, zₜ)
	output=	map(trialsets, p𝐘𝑑) do trialset, p𝐘𝑑
	            pmap(trialset.trials, p𝐘𝑑) do trial, p𝐘𝑑
	                ∇loglikelihood(p𝐘𝑑, trialinvariant, θnative, trial)
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
	return γ
end

"""
"""
function ∇negativeexpectation_parallel(γ, mpGLMs)
    pmap(mpGLMs) do mpGLM
        ∇negativeexpectation(γ, mpGLM)
    end
end
"""
"""
function ∇negativeexpectation_serial(γ, mpGLMs)
    map(mpGLMs) do mpGLM
        ∇negativeexpectation(γ, mpGLM)
    end
end

"""
	loglikelihood_one_trial

Compute the log-likelihood of a single trial

ARGUMENT
-`concatenatedθ`: a vector of the concatenated values of the parameters being fitted
-`indexθ`: a structure indicating the index of each model parameter in the vector of concatenated values
-`model`: structure containing all the information about the factorial hidden markov drift-diffusion model
-`p𝐘𝑑`: conditioned likelihood of the emissions in each time bin

OPTIONAL ARGUMENT
-`i`: trialset index
-`m`: trial index

RETURN
-scalar representing the log-likelihood of that trial
"""
function loglikelihood_one_trial(concatenatedθ::Vector{<:Real},
								indexθ::Indexθ,
								model::FHMDDM,
								p𝐘𝑑::Vector{<:Vector{<:Vector{<:Matrix{<:Real}}}};
								i::Integer,
								m::Integer)
	@unpack options, trialsets = model
	sortparameters!(model, concatenatedθ, indexθ)
	real2native!(θnative, options, θreal)
	trialinvariant = Trialinvariant(options, θnative; purpose="loglikelihood")
	loglikelihood(p𝐘𝑑[i][m], trialinvariant, θnative, trialsets[i].trials[m])
end


"""
	compute the log-likelihood in parallel across trials
"""
function loglikelihood_parallel(model::FHMDDM,
								p𝐘𝑑::Vector{<:Vector{<:Vector{<:Matrix{<:Real}}}},
								trialinvariant::Trialinvariant)
	@unpack θnative, trialsets = model
	ℓ = map(trialsets, p𝐘𝑑) do trialset, p𝐘𝑑
			pmap(trialset.trials, p𝐘𝑑) do trial, p𝐘𝑑
				loglikelihood!(p𝐘𝑑, trialinvariant, θnative, trial)
			end
		end
	return sum(sum(ℓ))
end

"""
	compute the log-likelihood serially over trials
"""
function loglikelihood_serial(model::FHMDDM,
								p𝐘𝑑::Vector{<:Vector{<:Vector{<:Matrix{<:Real}}}},
								trialinvariant::Trialinvariant)
	@unpack θnative, trialsets = model
	ℓ = 0.0
	@inbounds for i = eachindex(trialsets)
		for m = eachindex(trialsets[i].trials)
			ℓ += loglikelihood!(p𝐘𝑑[i][m], trialinvariant, θnative, trialsets[i].trials[m])
		end
	end
	ℓ
end

"""
    parallelize over neurons
"""
function likelihood1!(p𝐘𝑑::Vector{<:Vector{<:Vector{<:Matrix{<:Real}}}},
                      trialsets::Vector{<:Trialset})
    Ξ = length(trialsets[1].mpGLMs[1].𝛏)
    K = trialsets[1].mpGLMs[1].K
    indices = CartesianIndices((Ξ, K))
    @inbounds for i in eachindex(trialsets)
        𝐏 = pmap(mpGLM->likelihood1(indices, mpGLM), trialsets[i].mpGLMs)
        t = 0
        for m in eachindex(p𝐘𝑑[i])
            for tₘ in eachindex(p𝐘𝑑[i][m])
                p𝐘𝑑[i][m][tₘ] .= 1.0
                t += 1
                for n in eachindex(trialsets[i].mpGLMs)
                    for j =1:Ξ
                        for k = 1:K
                            p𝐘𝑑[i][m][tₘ][j,k] *= 𝐏[n][j,k][t]
                        end
                    end
                end
            end
        end
    end
end

"""
    parallelize over neurons,  helper function
"""
function likelihood1(indices::CartesianIndices, mpGLM::MixturePoissonGLM)
    map(indices) do index
        @unpack Δt, K, 𝛏, 𝐲, 𝐲! = mpGLM
        𝛌 = lambda(mpGLM, index[1], index[2])
        (𝛌.*Δt).^𝐲 ./ exp.(𝛌.*Δt) ./ 𝐲!
    end
end

"""
    parallelize over states
"""
function likelihood2!(p𝐘𝑑::Vector{<:Vector{<:Vector{<:Matrix{<:Real}}}},
                      trialsets::Vector{<:Trialset})
    Ξ = length(trialsets[1].mpGLMs[1].𝛏)
    K = trialsets[1].mpGLMs[1].K
    indices = CartesianIndices((Ξ, K))
    @inbounds for i in eachindex(trialsets)
        𝐏 = pmap(index->likelihood2(index, trialsets[i].mpGLMs), indices)
        t = 0
        for m in eachindex(p𝐘𝑑[i])
            for tₘ in eachindex(p𝐘𝑑[i][m])
                t += 1
                for n in eachindex(trialsets[i].mpGLMs)
                    for j =1:Ξ
                        for k = 1:K
                            p𝐘𝑑[i][m][tₘ][j,k] = 𝐏[j,k][t]
                        end
                    end
                end
            end
        end
    end
end

"""
    parallelize over states, helper function
"""
function likelihood2(index::CartesianIndex, mpGLMs::Vector{<:MixturePoissonGLM})
    j = index[1]
    k = index[2]
    @unpack Δt = mpGLMs[1]
    𝛌 = lambda(mpGLMs[1], j, k)
    𝐩 = (𝛌.*Δt).^mpGLMs[1].𝐲 ./ exp.(𝛌.*Δt) ./ mpGLMs[1].𝐲!
    for n = 2:length(mpGLMs)
        𝛌 = lambda(mpGLMs[n], j, k)
        𝐩 .*= (𝛌.*Δt).^mpGLMs[n].𝐲 ./ exp.(𝛌.*Δt) ./ mpGLMs[n].𝐲!
    end
    return 𝐩
end

"""
    no parallelization
"""
function likelihood3!(p𝐘𝑑::Vector{<:Vector{<:Vector{<:Matrix{<:Real}}}},
                      trialsets::Vector{<:Trialset})
    Ξ = length(trialsets[1].mpGLMs[1].𝛏)
    K = trialsets[1].mpGLMs[1].K
    Δt = trialsets[1].mpGLMs[1].Δt
    @inbounds for i in eachindex(trialsets)
        for j =1:Ξ
            for k = 1:K
                𝛌 = lambda(trialsets[i].mpGLMs[1], j, k)
                𝐩 = (𝛌.*Δt).^trialsets[i].mpGLMs[1].𝐲 ./ exp.(𝛌.*Δt) ./ trialsets[i].mpGLMs[1].𝐲!
                for n = 2:length(trialsets[i].mpGLMs)
                    𝛌 = lambda(trialsets[i].mpGLMs[1], j, k)
                    𝐩 .*= (𝛌.*Δt).^trialsets[i].mpGLMs[n].𝐲 ./ exp.(𝛌.*Δt) ./ trialsets[i].mpGLMs[n].𝐲!
                end
                t=0
                for m in eachindex(p𝐘𝑑[i])
                    for tₘ in eachindex(p𝐘𝑑[i][m])
                        t += 1
                        for n in eachindex(trialsets[i].mpGLMs)
                            p𝐘𝑑[i][m][tₘ][j,k] = 𝐩[t]
                        end
                    end
                end
            end
        end
    end
end
