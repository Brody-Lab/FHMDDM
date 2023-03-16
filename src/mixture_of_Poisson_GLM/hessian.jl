"""
    update_emissions!(memoryforhessian, mpGLMs, offset, sameacrosstrials, trial)

MODIFIED ARGUMENT
-`memoryforhessian`: a structure containing quantities used in each trial
-`mpGLMs`: vector whose each element is a structure containing the data and parameters of a mixture of Poisson GLM
-`sameacrosstrials`: structure containing intermediate quantities that are fixed across trials
-`trial`: structure containing the data of the trial being used for computation
"""
function update_emissions!(memoryforhessian::Memoryforhessian, mpGLMs::Vector{<:MixturePoissonGLM}, sameacrosstrials::Sameacrosstrials, trial::Trial)
    @unpack ntimesteps, τ₀, trialsetindex = trial
    𝐋 = memoryforhessian.𝐋[trialsetindex]
    𝛚 = memoryforhessian.𝛚[trialsetindex]
	d𝛚_db = memoryforhessian.d𝛚_db[trialsetindex]
	d²𝛚_db² = memoryforhessian.d²𝛚_db²[trialsetindex]
    @unpack λ, ∇logpy, ∇∇logpy, pY, ∇pY = memoryforhessian
    nθ_py = sameacrosstrials.nθ_py[trialsetindex]
    dL_d𝐯 = zeros(length(mpGLMs[1].θ.𝐯[1]))
	@inbounds for n in eachindex(mpGLMs)
		conditionalrate!(λ[n], 𝐋[n], ntimesteps, τ₀)
		for t = 1:ntimesteps
			∇∇conditional_log_likelihood!(∇logpy[t][n], ∇∇logpy[t][n], dL_d𝐯, 𝐋[n], λ[n][t], mpGLMs[n], 𝛚[n], d𝛚_db[n], d²𝛚_db²[n], t+τ₀)
		end
	end
	Ξ = length(𝛚[1])
	K = length(mpGLMs[1].θ.𝐯)
	@inbounds for t = 1:ntimesteps
		for ij in eachindex(pY[t])
			pY[t][ij] = 1.0
			for n=1:nneurons
				pY[t][ij] *= poissonlikelihood(λ[n][t][ij]*Δt, mpGLMs[n].𝐲[t+τ₀])
			end
		end
		r = 0
		for n=1:nneurons
			for q = 1:nθ_py[n]
				r+=1
				for i=1:Ξ
					for j=1:K
						∇pY[t][r][i,j] = ∇logpy[t][n][q][i,j]*pY[t][i,j]
					end
				end
			end
		end
	end
	return nothing
end

"""
    linearpredictor(mpGLMs)

Linear combination of the weights in the j-th accumulator state and k-th coupling state

ARGUMENT
-`trialsets`: a vector of trialsets

RETURN
-`𝐋`: a nested array whose element 𝐋[i][n][j,k][t] corresponds to i-th trialset, n-th neuron, j-th accumualtor state, k-th coupling state, and the t-th time bin in the trialset
"""
function linearpredictor(trialsets::Vector{<:Trialset})
    map(trialsets) do trialset
        map(trialset.mpGLMs) do mpGLM
            Ξ = mpGLM.Ξ
            K = length(mpGLM.θ.𝐯)
            map(CartesianIndices((Ξ,K))) do index
                j = index[1]
                k = index[2]
                linearpredictor(mpGLM, j, k)
            end
        end
    end
end

"""
	conditionalrate!(λ, 𝐋, offset)

MODIFIED ARGUMENT
-`λ`: matrix whose element `λ[t][i,j]` is the Poisson rate at the t-th timestep of a trial given the i-th accumulator state and j-th coupling state

UNMODIFIED ARGUMENT
-`𝐋`: matrix whose element `𝐋[i,j][τ]` is the linear predictor given the i-th accumulator state and j-th coupling state for the τ-timestep in the trialset
-`offset`: the time index in the trialset corresponding to the time index 0 in the trial
"""
function conditionalrate!(λ::Vector{<:Matrix{<:Real}},
						  𝐋::Matrix{<:Vector{<:Real}},
						  ntimesteps::Integer,
						  offset::Integer)
	for t = 1:ntimesteps
		τ = t + offset
		for jk in eachindex(λ[t])
			λ[t][jk] = softplus(𝐋[jk][τ])
		end
	end
	return nothing
end

"""
	∇∇conditional_log_likelihood!(∇logpy, ∇∇logpy, dL_d𝐯, 𝐋, λ, mpGLM, τ)

Gradient and Hessian of the conditional log-likelihood of one neuron at single timestep

MODIFIED ARGUMENT
-`∇logpy`: Gradient of the conditional log-likelihood of a neuron's response at a single time. Element ∇logpy[i][j,k] correponds to the partial derivative of log p{y(n,t) ∣ a(t)=ξ(j), c(t)=k} with respect to the i-th parameter of the neuron's GLM
-`∇∇logpy`: Hessian of the conditional log-likelihood. Element ∇∇logpy[i,j][k,l] correponds to the partial derivative of log p{y(n,t) ∣ a(t)=ξ(k), c(t)=l} with respect to the i-th and j-th parameters of the neuron's GLM
-`dL_d𝐯`: memory for computing the derivative of the linear predictor with respect to the linear filters of the accumulator. The element `dL_d𝐯[q]` corresponds to the q-th linear filter in one of the coupling states.

UNMODIFIED ARGUMENT
-`α`: overdispersion parameter
-`Δt`: width of time step
-`𝐋`: linear predictors. Element `𝐋[i,j][τ]` corresponds to the i-th accumulator state and j-th coupling state for the τ-timestep in the trialset
-`λ`: Conditional Poisson whose element λ[i,j] corresponds to a(t)=ξ(i), c(t)=j
-`mpGLM`: a composite containing the data and parameters of a Poisson mixture generalized linear model
-`𝛚`: transformed values of the accumulator
-`d𝛚_db`: first derivative of the transformed values of the accumulator with respect to the transformation parameter
-`d²𝛚_db²`: second derivative of the transformed values of the accumulator with respect to the transformation parameter
-`τ` time step in the trialset
"""
function ∇∇conditional_log_likelihood!(∇logpy::Vector{<:Matrix{<:Real}},
										∇∇logpy::Matrix{<:Matrix{<:Real}},
										dL_d𝐯::Vector{<:Real},
										α::Real,
										𝐋::Matrix{<:Vector{<:Real}},
										λ::Matrix{<:Real},
										mpGLM::MixturePoissonGLM,
										𝛚::Vector{<:Real},
										d𝛚_db::Vector{<:Real},
										d²𝛚_db²::Vector{<:Real},
										τ::Integer)
	@unpack Δt, 𝐗, Ξ, 𝐕, 𝐲 = mpGLM
	@unpack b, 𝐮, 𝐯, 𝛃, fit_𝛃 = mpGLM.θ
	K = length(𝐯)
	n𝐮 = length(𝐮)
	n𝐯 = length(𝐯[1])
	indexb = n𝐮 + 2*K*n𝐯 + 1
	Vₜᵀ𝐯 = zeros(K)
	for j = 1:K
		for q=1:n𝐯
			Vₜᵀ𝐯[j] += 𝐕[τ,q]*𝐯[j][q]
		end
	end
	for i = 1:Ξ
		for m=1:n𝐯
			dL_d𝐯[m] = 𝐕[τ,m]*𝛚[i]
		end
		for j = 1:K
			d²ℓ_dL², dℓ_dL = FHMDDM.differentiate_twice_loglikelihood_wrt_linearpredictor(Δt, 𝐋[i,j][τ], λ[i,j], 𝐲[τ])
			dL_db = Vₜᵀ𝐯[j]*d𝛚_db[i]
			d²L_db² = Vₜᵀ𝐯[j]*d²𝛚_db²[i]
			offset𝐯 = n𝐮 + (j-1)*n𝐯
			offset𝛃 = n𝐮 + (K+j-1)*n𝐯
			for m=1:n𝐮
				∇logpy[m][i,j] = dℓ_dL*𝐗[τ,m]
			end
			offset = (fit_𝛃 && ((i==1) || (i==Ξ))) ? offset𝛃 : offset𝐯
			for m=1:n𝐯
				∇logpy[m+offset][i,j] = dℓ_dL*dL_d𝐯[m]
			end
			∇logpy[indexb][i,j] = dℓ_dL*dL_db
			for m=1:n𝐮
				for n=m:n𝐮
					∇∇logpy[m,n][i,j] = d²ℓ_dL²*𝐗[τ,m]*𝐗[τ,n]
				end
				for n=1:n𝐯
					∇∇logpy[m,n+offset][i,j] = d²ℓ_dL²*𝐗[τ,m]*dL_d𝐯[n]
				end
				∇∇logpy[m,indexb][i,j] = d²ℓ_dL²*𝐗[τ,m]*dL_db
			end
			for m=1:n𝐯
				for n=m:n𝐯
					∇∇logpy[m+offset, n+offset][i,j] = d²ℓ_dL² * dL_d𝐯[m] * dL_d𝐯[n]
				end
				d²L_dvdb = 𝐕[τ,m]*d𝛚_db[i]
				∇∇logpy[m+offset,indexb][i,j] = d²ℓ_dL²*dL_d𝐯[m]*dL_db + dℓ_dL*d²L_dvdb
			end
			∇∇logpy[indexb,indexb][i,j] = d²ℓ_dL²*dL_db^2 + dℓ_dL*d²L_db²
		end
	end
	return nothing
end
