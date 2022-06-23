"""
    GaussianPrior(options, trialsets)

Construct a structure containing information on the Gaussian prior on the model's parameters

ARGUMENT
-`options`: settings of the model
-`trialsets`: data for the model

OUTPUT
-an instance of `GaussianPrior`

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_20a_test/T176_2018_05_03_b3K2K2/data.mat")
julia> model.gaussianprior
julia>
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_20a_test/no_smoothing/data.mat")
julia> model.gaussianprior
julia>
```
"""
function GaussianPrior(options::Options, trialsets::Vector{<:Trialset})
    indexθ = indexparameters(options, trialsets)
    N = indexθ.glmθ[end][end].𝐯[end][end]
    𝚲 = zeros(N,N)
	if isnan(options.s₀)
		𝐒 = Matrix{eltype(options.nbases_each_event)}[]
		index𝐒 = Vector{eltype(options.nbases_each_event)}[]
		𝐬 = typeof(options.s₀)[]
	else
	    𝐒 = squared_difference_matrices(indexθ.glmθ, options.nbases_each_event)
	    index𝐒 = index_smoothing_coefficients(indexθ.glmθ, trialsets[1].mpGLMs[1].max_spikehistory_lag, options.nbases_each_event)
	    𝐬 = ones(length(𝐒)).*options.s₀
	end
	index𝛂 = index_shrinkage_coefficients(indexθ)
	𝛂 = ones(length(index𝛂)).*options.α₀
    gaussianprior = GaussianPrior(𝛂=𝛂, index𝛂=index𝛂, index𝐒=index𝐒, 𝚲=𝚲, 𝐒=𝐒, 𝐬=𝐬)
    precisionmatrix!(gaussianprior)
    return gaussianprior
end

"""
	GaussianPrior(options, trialsets, 𝛂𝐬)

Construct a structure containing information on the Gaussian prior on the model's parameters

ARGUMENT
-`options`: settings of the model
-`trialsets`: data for the model
-`𝛂𝐬`: a vector concatenating the L2 shrinkrage and smoothing coefficients

OUTPUT
-an instance of `GaussianPrior`

"""
function GaussianPrior(options::Options, trialsets::Vector{<:Trialset}, 𝛂𝐬::Vector{type}) where {type<:Real}
    indexθ = indexparameters(options, trialsets)
    N = indexθ.glmθ[end][end].𝐯[end][end]
    𝚲 = zeros(type,N,N)
	index𝛂 = index_shrinkage_coefficients(indexθ)
	𝛂 = 𝛂𝐬[1:length(index𝛂)]
	if isnan(options.s₀)
		𝐒 = Matrix{eltype(options.nbases_each_event)}[]
		index𝐒 = Vector{eltype(options.nbases_each_event)}[]
		𝐬 = type[]
	else
		index𝐒 = index_smoothing_coefficients(indexθ.glmθ, trialsets[1].mpGLMs[1].max_spikehistory_lag, options.nbases_each_event)
		𝐒 = squared_difference_matrices(indexθ.glmθ, options.nbases_each_event)
		𝐬 = 𝛂𝐬[length(index𝛂)+1:length(index𝛂)+length(index𝐒)]
	end
    gaussianprior = GaussianPrior(𝛂=𝛂, index𝛂=index𝛂, index𝐒=index𝐒, 𝚲=𝚲, 𝐒=𝐒, 𝐬=𝐬)
    precisionmatrix!(gaussianprior)
    return gaussianprior
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
    K𝐠 = length(θ.𝐠)
    if K𝐠 > 1
        𝐒 = vcat(𝐒, [squared_difference_matrix(K𝐠)])
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
            K𝐠 = length(indexθglm.𝐠)
            if K𝐠 > 1
                index𝐒 = vcat(index𝐒, [vcat(indexθglm.𝐠...)])
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
		if i == 0 || field == :Aᶜ₁₁ || field == :Aᶜ₂₂
		else
			index𝛂 = vcat(index𝛂, i)
		end
	end
	for glmθ in indexθ.glmθ
		for glmθ in glmθ
			if length(glmθ.𝐠) == 2
				index𝛂 = vcat(index𝛂, glmθ.𝐠[2][1]:glmθ.𝐯[end][end])
			else
				index𝛂 = vcat(index𝛂, glmθ.𝐮[1]:glmθ.𝐯[end][end])
			end
		end
	end
	index𝛂
end

"""
    precisionmatrix!(gaussianprior)

Update the precision matrix

MODIFIED ARGUMENT
-`gaussianprior`: structure containing information on the Gaussian prior on the values of the model parameters in real space. The precision matrix `𝚲` is updated with respect to the shrinkage coefficients 𝛂 and smoothing coefficients 𝐬
"""
function precisionmatrix!(gaussianprior::GaussianPrior)
    @unpack 𝛂, index𝛂, index𝐒, 𝚲, 𝐒, 𝐬, 𝚽, index𝚽  = gaussianprior
    𝚲 .= 0
    for i in eachindex(index𝛂)
        j = index𝛂[i]
        𝚲[j,j] = 𝛂[i]
    end
    for i in eachindex(index𝐒)
        𝚲[index𝐒[i],index𝐒[i]] .+= 𝐬[i].*𝐒[i]
    end
	𝚽 .= 𝚲[index𝚽, index𝚽]
    return nothing
end

"""
	precisionmatrix!(gaussianprior, 𝛂𝐬)

Update the precision matrix with new L2 coefficients

MODIFIED ARGUMENT
-`gaussianprior`: structure containing information on the Gaussian prior on the values of the model parameters in real space. The precision matrix `𝚲` is updated with respect to the shrinkage coefficients 𝛂 and smoothing coefficients 𝐬

UNMODFIED ARGUMENT
-`𝛂𝐬`: vector concatenating the values of the L2 shrinkage coefficients and the L2 smoothing coefficcients
"""
function precisionmatrix!(gaussianprior::GaussianPrior, 𝛂𝐬::Vector{<:AbstractFloat})
	length𝛂 = length(gaussianprior.𝛂)
	for i = 1:length𝛂
		gaussianprior.𝛂[i] = 𝛂𝐬[i]
	end
	length𝐬 = length(gaussianprior.𝐬)
	for i = 1:length𝐬
		j = i + length𝛂
		gaussianprior.𝐬[i] = 𝛂𝐬[j]
	end
	precisionmatrix!(gaussianprior)
end

"""
	precisionmatrix!(gaussianprior, 𝛂, 𝐬)

Update the precision matrix with new L2 coefficients

MODIFIED ARGUMENT
-`gaussianprior`: structure containing information on the Gaussian prior on the values of the model parameters in real space. The precision matrix `𝚲` is updated with respect to the shrinkage coefficients 𝛂 and smoothing coefficients 𝐬

UNMODFIED ARGUMENT
-`𝛂`: L2 shrinkage coefficients
-`𝐬`: L2 smoothing coefficcients
"""
function precisionmatrix!(gaussianprior::GaussianPrior, 𝛂::Vector{<:AbstractFloat}, 𝐬::Vector{<:AbstractFloat})
	length𝛂 = length(gaussianprior.𝛂)
	for i = 1:length𝛂
		gaussianprior.𝛂[i] = 𝛂[i]
	end
	length𝐬 = length(gaussianprior.𝐬)
	for i = 1:length𝐬
		gaussianprior.𝐬[i] = 𝐬[i]
	end
	precisionmatrix!(gaussianprior)
end
