"""
	functions

-linearpredictor(mpGLMs)
-spikcountderivatives!(memoryforhessian, mpGLMs, sameacrosstrials, trial)
-conditionallikelihood!(pY, 𝛂, 𝐋, mpGLMs, ntimesteps, τ₀)
-∇∇conditional_log_likelihood!(∇logpy, ∇∇logpy, glmderivatives, indexθ, 𝐋, mpGLM, 𝛚, d𝛚_db, d²𝛚_db², τ)
"""

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
	spikcountderivatives!(memoryforhessian, mpGLMs, sameacrosstrials, trial)

Derivatives related to the spike count response at each time step in a trial

The quantities that are computed include the first- and second-order partial derivatives of the log-likelihood of each neuron's spike count response (`∇logpy` and `∇∇logpy`) and the gradient of the likelihood of the population spike count response (`∇pY`) . 

MODIFIED ARGUMENT
-`memoryforhessian`: a structure containing quantities used in each trial

UNMODIFIED ARGUMENT
-`mpGLMs`: vector whose each element is a structure containing the data and parameters of a mixture of Poisson GLM
-`sameacrosstrials`: structure containing intermediate quantities that are fixed across trials
-`trial`: structure containing the data of the trial being used for computation
"""
function spikcountderivatives!(memoryforhessian::Memoryforhessian, mpGLMs::Vector{<:MixturePoissonGLM}, sameacrosstrials::Sameacrosstrials, trial::Trial)
    @unpack ntimesteps, τ₀, trialsetindex = trial
    𝐋 = memoryforhessian.𝐋[trialsetindex]
	𝛂 = memoryforhessian.𝛂[trialsetindex]
	indexθglms = memoryforhessian.indexθglms[trialsetindex]
    𝛚 = memoryforhessian.𝛚[trialsetindex]
	d𝛚_db = memoryforhessian.d𝛚_db[trialsetindex]
	d²𝛚_db² = memoryforhessian.d²𝛚_db²[trialsetindex]
    @unpack glmderivatives, ∇logpy, ∇∇logpy, pY, ∇pY = memoryforhessian
	scaledlikelihood!(pY, 𝛂, 𝐋, mpGLMs, ntimesteps, τ₀)
	@inbounds for n in eachindex(mpGLMs)
		differentiate_twice_overdispersion!(glmderivatives,mpGLMs[n].θ.a[1])
		for t = 1:ntimesteps
			τ = t+τ₀
			∇∇loglikelihood!(∇logpy[t][n], ∇∇logpy[t][n], glmderivatives, indexθglms[n], 𝐋[n], mpGLMs[n], 𝛚[n], d𝛚_db[n], d²𝛚_db²[n], τ)
		end
	end
	Ξ = length(𝛚[1])
	K = length(mpGLMs[1].θ.𝐯)
	nθ_py = sameacrosstrials.nθ_py[trialsetindex]
	@inbounds for t = 1:ntimesteps
		r = 0
		for n=eachindex(mpGLMs)
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
	conditionallikelihood!(pY, 𝛂, 𝐋, mpGLMs, ntimesteps, τ₀)

Conditional likelihood of the population spike count response at each time step

The spike count response model is negative binomial

MODIFIED ARGUMENT
-`pY`:nested array used for in-place computation of the conditional likelihood of the population response. Element `pY[t][i,j]` corresponds to the t-th time step in a trial, i-th accumulator state, and j-th coupling state

UNMODIFIED ARGUMENT
-`λ`: conditional mean of the Poisson or negative-binomial spike count response model at each time step. Element `λ[n][t][i,j]` corresponds to the n-th neuron, t-th time step in a trial, i-th accumulator state, and j-th coupling state
-`𝛂`: overdispersion parameters of each neuron
-`mpGLMs`: a vector of mixture of Poisson GLM's. Each element corresponds to a neuron
-`ntimesteps`: number of time steps in the trial
-`τ₀`: total number of time steps across trials preceding this trial
"""
function scaledlikelihood!(pY::Vector{<:Matrix{<:AbstractFloat}}, 𝛂::Vector{<:AbstractFloat}, 𝐋::Vector{<:Matrix{<:Vector{<:AbstractFloat}}}, mpGLMs::Vector{<:MixturePoissonGLM}, ntimesteps::Integer, τ₀::Integer)
	Δt = mpGLMs[1].Δt
	fit_overdispersion = mpGLMs[1].θ.fit_overdispersion
	likelihoodscalefactor = mpGLMs[1].likelihoodscalefactor
	@inbounds for t = 1:ntimesteps
		τ = t+τ₀
		for ij in eachindex(pY[t])
			pY[t][ij] = 1.0
			for n = eachindex(mpGLMs)
				L = 𝐋[n][ij][τ]
				y = mpGLMs[n].𝐲[τ]
				μ = inverselink(L)
				p = fit_overdispersion ? negbinlikelihood(𝛂[n], Δt, μ, y) : poissonlikelihood(μ*Δt, y)
				pY[t][ij] *= p*likelihoodscalefactor
			end
		end
	end
	return nothing
end

"""
	∇∇loglikelihood!(∇logpy, ∇∇logpy, glmderivatives, indexθ, 𝐋, mpGLM, 𝛚, d𝛚_db, d²𝛚_db², τ)

Gradient and Hessian of the conditional log-likelihood of one particular neuron at single timestep

MODIFIED ARGUMENT
-`∇logpy`: Gradient of the conditional log-likelihood of a neuron's response at a single time. Element ∇logpy[i][j,k] correponds to the partial derivative of log p{y(n,t) ∣ a(t)=ξ(j), c(t)=k} with respect to the i-th parameter of the neuron's GLM
-`∇∇logpy`: Hessian of the conditional log-likelihood. Element ∇∇logpy[q,r][i,j] correponds to the partial derivative of log p{y(n,t) ∣ a(t)=ξ(i), c(t)=j} with respect to the q-th and r-th parameters of the neuron's GLM

UNMODIFIED ARGUMENT
-`D`: an object for computing derivatives
-`indexθ`: an object containing indices of parameters
-`𝐋`: linear predictors. Element `𝐋[i,j][τ]` corresponds to the i-th accumulator state and j-th coupling state for the τ-timestep in the trialset
-`mpGLM`: a composite containing the data and parameters of a Poisson mixture generalized linear model
-`𝛚`: transformed values of the accumulator
-`d𝛚_db`: first derivative of the transformed values of the accumulator with respect to the transformation parameter
-`d²𝛚_db²`: second derivative of the transformed values of the accumulator with respect to the transformation parameter
-`τ` time step in the trialset
"""
function ∇∇loglikelihood!(∇logpy::Vector{<:Matrix{<:Real}},
						∇∇logpy::Matrix{<:Matrix{<:Real}},
						glmderivatives::GLMDerivatives,
						indexθ::GLMθ,
						𝐋::Matrix{<:Vector{<:Real}},
						mpGLM::MixturePoissonGLM,
						𝛚::Vector{<:Real},
						d𝛚_db::Vector{<:Real},
						d²𝛚_db²::Vector{<:Real},
						τ::Integer)
	@unpack Δt, 𝐗, Ξ, 𝐕, 𝐲 = mpGLM
	@unpack b, 𝐮, 𝐯, 𝛃, fit_𝛃, fit_overdispersion = mpGLM.θ
	y = mpGLM.𝐲[τ]
	K = length(𝐯)
	n𝐮 = length(𝐮)
	n𝐯 = length(𝐯[1])
	indexa = indexθ.a[1]
	indexb = indexθ.b[1]
	Vₜᵀ𝐯 = zeros(K)
	dL_d𝐯 = zeros(n𝐯)
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
			indices𝐯𝛃 = (fit_𝛃 && ((i==1) || (i==Ξ))) ? indexθ.𝛃[j] : indexθ.𝐯[j]
			L = 𝐋[i,j][τ]
			differentiate_twice_loglikelihood!(glmderivatives, L, y)
			dℓ_da = glmderivatives.dℓ_da[1]
			dℓ_dL = glmderivatives.dℓ_dL[1]
			d²ℓ_da² = glmderivatives.d²ℓ_da²[1]
			d²ℓ_dadL = glmderivatives.d²ℓ_dadL[1]
			d²ℓ_dL² = glmderivatives.d²ℓ_dL²[1]
			dL_db = Vₜᵀ𝐯[j]*d𝛚_db[i]
			d²L_db² = Vₜᵀ𝐯[j]*d²𝛚_db²[i]
			for m=1:n𝐮
				∇logpy[m][i,j] = dℓ_dL*𝐗[τ,m]
			end
			for m=1:n𝐯
				∇logpy[indices𝐯𝛃[m]][i,j] = dℓ_dL*dL_d𝐯[m]
			end
			∇logpy[indexa][i,j] = dℓ_da
			∇logpy[indexb][i,j] = dℓ_dL*dL_db
			for m=1:n𝐮
				for n=m:n𝐮
					∇∇logpy[m,n][i,j] = d²ℓ_dL²*𝐗[τ,m]*𝐗[τ,n]
				end
				for n=1:n𝐯
					∇∇logpy[m,indices𝐯𝛃[n]][i,j] = d²ℓ_dL²*𝐗[τ,m]*dL_d𝐯[n]
				end
				∇∇logpy[m,indexa][i,j] = d²ℓ_dadL*𝐗[τ,m]
				∇∇logpy[m,indexb][i,j] = d²ℓ_dL²*𝐗[τ,m]*dL_db
			end
			for m=1:n𝐯
				for n=m:n𝐯
					∇∇logpy[indices𝐯𝛃[m], indices𝐯𝛃[n]][i,j] = d²ℓ_dL² * dL_d𝐯[m] * dL_d𝐯[n]
				end
				∇∇logpy[indices𝐯𝛃[m],indexa][i,j] = d²ℓ_dadL*dL_d𝐯[m]
				d²L_dvdb = 𝐕[τ,m]*d𝛚_db[i]
				∇∇logpy[indices𝐯𝛃[m],indexb][i,j] = d²ℓ_dL²*dL_d𝐯[m]*dL_db + dℓ_dL*d²L_dvdb
			end
			∇∇logpy[indexa,indexa][i,j] = d²ℓ_da²
			∇∇logpy[indexa,indexb][i,j] = d²ℓ_dadL*dL_db
			∇∇logpy[indexb,indexb][i,j] = d²ℓ_dL²*dL_db^2 + dℓ_dL*d²L_db²
		end
	end
	return nothing
end