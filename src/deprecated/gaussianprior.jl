
"""
    variancematrices(indexθglm, Φₐ, Φₜ)

Return the variance matrix of each group of parameters representing a time-varying quantity being flattened

ARGUMENT
-`indexθglm`: a nested array indexing each parameter in each mixture of Poisson GLM. The element `indexθglm[i][n]` corresponds to the n-th neuron in the i-th trialset
-`Φₐ`: values of the temporal basis functions parametrizing hte time-varying encoding of the accumulator. Element `Φₐ[t,i]` corresponds to the value of the i-th temporal basis function at the t-th time step in each trial
-`Φₜ`: values of the temporal basis functions parametrizing time in each trial. Element `Φₜ[t,i]` corresponds to the value of the i-th temporal basis function at the t-th time step in each trial.

OUTPUT
-`𝚪`: A nest array of matrices. Element `𝚪[i]` corresponds to the Nᵢ×Nᵢ variance matrix of the i-th group of parameters, with N parameters in the group
-`index𝚪`: Element `index𝚪[i][j]` corresponds to the i-th group of parameters and the j-th parameter in that group. The value of the element indicates the index of that parameter in a vector concatenating all the parameters in the model that are being fit.
"""
function variancematrices(indexθglm::Vector{<:GLMθ}, Φₐ::Matrix{<:AbstractFloat}, Φₜ::Matrix{<:AbstractFloat})
	Γaccumulator = Φₐ'*variancematrix(size(Φₐ,1))*Φₐ
	Γtime = Φₜ'*variancematrix(size(Φₜ,1))*Φₜ
	𝚪 = Matrix{typeof(1.0)}[]
	index𝚪 = Vector{typeof(1)}[]
	@unpack 𝐮indices_time = indexθglm[1]
	for indexᵢₙ in indexθglm
		𝚪 = vcat(𝚪, [Γtime])
		index𝚪 = vcat(index𝚪, [indexᵢₙ.𝐮[𝐮indices_time]])
		for indexᵢₙ𝐯ₖ in indexᵢₙ.𝐯
			𝚪 = vcat(𝚪, [Γaccumulator])
			index𝚪 = vcat(index𝚪, [indexᵢₙ𝐯ₖ])
		end
	end
    return 𝚪, index𝚪
end

"""
    variancematrix(n)

Return a matrix that computes the variance of `n` elements

OUTPUT
-`Γ`: a matrix:
    (n-1)/n 	-1/n		-1/n		...		-1/n
	-1/n		(n-1)/n		-1/n		...		-1/n
	-1/n		-1/n		(n-1)/n		...		-1/n
										...
	-1/n		-1/n		-1/n		...		(n-1)/n
"""
variancematrix(n::Integer) = I/n - ones(n,n)./n^2

"""
	index_shrinkage_coefficients(model)

Create a structure indexing the precisions

ARGUMENT
-`indexθ`: structure indicating the order of each parameter if all parameters were concatenated into a vector

RETURN
-a vector of integers
```
"""
function index_shrinkage_coefficients(indexθ::Indexθ)
	index𝛂 = Int[]
	for field in fieldnames(Latentθ)
		i = getfield(indexθ.latentθ, field)[1]
		if i == 0 || field == :Aᶜ₁₁ || field == :Aᶜ₂₂ || field == :πᶜ₁
		else
			index𝛂 = vcat(index𝛂, i)
		end
	end
	for glmθ in indexθ.glmθ
		for glmθ in glmθ
			if length(glmθ.𝐠) > 1
				index𝛂 = vcat(index𝛂, glmθ.𝐠[2]:glmθ.𝐯[end][end])
			else
				index𝛂 = vcat(index𝛂, glmθ.𝐮[1]:glmθ.𝐯[end][end])
			end
		end
	end
	index𝛂
end

"""
    squared_difference_matrices(model)

Return the square of the difference matrix of each gorup of parameters being smoothed

ARGUMENT
-`indexθglm`: a nested array indexing each parameter in each mixture of Poisson GLM. The element `indexθglm[i][n]` corresponds to the n-th neuron in the i-th trialset
-`nbases_each_event`: number of temporal basis functions in each trial event

OUTPUT
-`𝐒`: A nest array of matrices. Element `𝐒[i]` corresponds to the Nᵢ×Nᵢ squared difference matrix of the i-th group parameters
"""
function squared_difference_matrices(indexθglm::Vector{<:Vector{<:GLMθ}}, nbases_each_event::Vector{<:Integer})
    θ = indexθglm[1][1]
    𝐒 = Matrix{eltype(nbases_each_event)}[]
    for n in nbases_each_event
        if n > 1
            𝐒 = vcat(𝐒, [squared_difference_matrix(n)])
        end
    end
    K𝐯 = length(θ.𝐯)
    B𝐯 = length(θ.𝐯[1])
    if B𝐯 > 1
        for k = 1:K𝐯
            𝐒 = vcat(𝐒, [squared_difference_matrix(B𝐯)])
        end
    end
    if K𝐯 > 1
		for j = 1:B𝐯
	        𝐒 = vcat(𝐒, [squared_difference_matrix(K𝐯)])
		end
    end
    return [S for indexθglm in indexθglm for indexθglm in indexθglm for S in 𝐒] # copies 𝐒 for each neuron
end

"""
    squared_difference_matrix(n)

Return the squared difference matrix for a group of parameters

ARGUMENT
-`n`: number of parameters in the group

OUTPUT
-`S`: a matrix:
    1   -1  0   0   ...     0   0
    -1  2   -1  0   ...     0   0
    0   -1  2   -1  ...     0   0
    0   0   -1  2   ...     0   0
                    ...
    0   0   0   0   ...     2   -1
    0   0   0   0   ...     -1   1
"""
function squared_difference_matrix(n::Integer)
    S = zeros(typeof(n), n, n) + I
    for i = 2:n-1
        S[i,i] += 1
	end
	for i = 2:n
        S[i,i-1] = S[i-1,i] = -1
    end
    return S
end

"""
    index_smoothing_coefficients(indexθglm, max_spikehistory_lag, nbases_each_event)

Index parameters that have a L2 smoothing penalty

ARGUMENT
-`indexθglm`: a nested array indexing each parameter in each mixture of Poisson GLM. The element `indexθglm[i][n]` corresponds to the n-th neuron in the i-th trialset
-`max_spikehistory_lag`: number of parameters to specify the weight of spike history
-`nbases_each_event`: number of temporal basis functions in each trial event

OUTPUT
-`index𝐒`: Element `index𝐒[i][j]` corresponds to the i-th group of parameters being smoothed and the j-th parameter in that group. The value of the element indicates the order of the parameter in a vector concatenating all the parameters in the model that are being fit.
"""
function index_smoothing_coefficients(indexθglm::Vector{<:Vector{<:GLMθ}}, max_spikehistory_lag::Integer, nbases_each_event::Vector{<:Integer})
    index𝐒 = Vector{eltype(nbases_each_event)}[] # empty vector of vectors
    for indexθglm in indexθglm
        for indexθglm in indexθglm
            for i in eachindex(nbases_each_event)
                n = nbases_each_event[i]
                if n > 1
                    indices_in_𝐮 = indexθglm.𝐮[max_spikehistory_lag .+ sum(nbases_each_event[1:i-1]) .+ (1:n)]
                    index𝐒 = vcat(index𝐒, [indices_in_𝐮])
                end
            end
            K𝐯 = length(indexθglm.𝐯)
            B𝐯 = length(indexθglm.𝐯[1])
            if B𝐯 > 1
                for k = 1:K𝐯
                    index𝐒 = vcat(index𝐒, [indexθglm.𝐯[k]])
                end
            end
            if K𝐯 > 1
				for j = 1:B𝐯
	                index𝐒 = vcat(index𝐒, [[indexθglm.𝐯[1][j], indexθglm.𝐯[2][j]]])
				end
            end
        end
    end
    return index𝐒
end

"""
	shrinkage_coefficients_limits(αrangeDDM, αrangeGLM, indexθ, N𝛂)

Limits of the L2 shrinkage coefficients

ARGUMENT
-`αrangeDDM`: minimum and maximum of the precisions on the priors of the DDM parameters
-`αrangeGLM`: minimum and maximum of the precisions on the priors of the GLM parameters
-`indexθ`: index of each parameter
-`N𝛂`: number of precisions being leared

RETURN
-`𝛂min`: vector of the minimum of each precision being learned
-`𝛂max`: vector of the maximum of each precision being learned
"""
function shrinkage_coefficients_limits(αrangeDDM::Vector{<:AbstractFloat}, αrangeGLM::Vector{<:AbstractFloat}, indexθ::Indexθ, N𝛂::Integer)
	𝛂min, 𝛂max = zeros(N𝛂), zeros(N𝛂)
	k = 0
	for field in fieldnames(Latentθ)
		i = getfield(indexθ.latentθ, field)[1]
		if i == 0 || field == :Aᶜ₁₁ || field == :Aᶜ₂₂ || field == :πᶜ₁
		else
			k += 1
			𝛂min[k] = αrangeDDM[1]
			𝛂max[k] = αrangeDDM[2]
		end
	end
	for glmθ in indexθ.glmθ
		for glmθ in glmθ
			for i = 2:length(glmθ.𝐠)
				k +=1
				𝛂min[k] = αrangeGLM[1]
				𝛂max[k] = αrangeGLM[2]
			end
			for u in glmθ.𝐮
				k +=1
				𝛂min[k] = αrangeGLM[1]
				𝛂max[k] = αrangeGLM[2]
			end
			for 𝐯ₖ in glmθ.𝐯
				for v in 𝐯ₖ
					k +=1
					𝛂min[k] = αrangeGLM[1]
					𝛂max[k] = αrangeGLM[2]
				end
			end
		end
	end
	@assert k == N𝛂
	return 𝛂min, 𝛂max
end
