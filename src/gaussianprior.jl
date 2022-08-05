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
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_08_04e_test/T176_2018_05_03/data.mat"
julia> model = Model(datapath)
julia> model.gaussianprior
julia>
```
"""
function GaussianPrior(options::Options, trialsets::Vector{<:Trialset})
    indexθ = indexparameters(options, trialsets)
	𝐀 = Matrix{typeof(1.0)}[]
	index𝐀 = Vector{typeof(1)}[]
	𝛂min, 𝛂max = typeof(1.0)[], typeof(1.0)[]
	for i = 1:length(trialsets)
		mpGLM = trialsets[i].mpGLMs[1]
		𝐀_lv, index𝐀_lv = shrinkagematrices(indexθ.latentθ)
		𝐀_glm, index𝐀_glm = shrinkagematrices(indexθ.glmθ[i], options.glminputscaling, mpGLM.max_spikehistory_lag, mpGLM.Φₘ, mpGLM.Φₜ)
		𝚪_glm, index𝚪_glm = variancematrices(indexθ.glmθ[i], mpGLM.max_spikehistory_lag, mpGLM.Φₐ, mpGLM.Φₜ)
		𝐀 = vcat(𝐀, 𝐀_lv, 𝐀_glm, 𝚪_glm)
		index𝐀 = vcat(index𝐀, index𝐀_lv, index𝐀_glm, index𝚪_glm)
		𝛂min_t, 𝛂max_t = L2penalty_coeffcients_limits(options, length(index𝐀_lv), length(index𝐀_glm), length(index𝚪_glm))
		𝛂min = vcat(𝛂min, 𝛂min_t)
		𝛂max = vcat(𝛂max, 𝛂max_t)
	end
	𝛂 = sqrt.(𝛂min.*𝛂max)
    N = indexθ.glmθ[end][end].𝐯[end][end]
    gaussianprior = GaussianPrior(𝐀=𝐀, 𝛂=𝛂, 𝛂min=𝛂min, 𝛂max=𝛂max, index𝐀=index𝐀, 𝚲=zeros(N,N))
    precisionmatrix!(gaussianprior)
    return gaussianprior
end

"""
	GaussianPrior(options, trialsets, 𝛂)

Construct a structure containing information on the Gaussian prior on the model's parameters

ARGUMENT
-`options`: settings of the model
-`trialsets`: data for the model
-`𝛂`: a vector concatenating the L2 shrinkrage and smoothing coefficients

OUTPUT
-an instance of `GaussianPrior`
"""
function GaussianPrior(gaussianprior::GaussianPrior, 𝛂::Vector{type}) where {type<:Real}
	N = length(𝛂)
	GaussianPrior(𝐀=gaussianprior.𝐀, 𝛂=𝛂, 𝛂min=gaussianprior.𝛂min, 𝛂max=gaussianprior.𝛂max, index𝐀=gaussianprior.index𝐀, 𝚲=zeros(type,N,N))
    precisionmatrix!(gaussianprior)
    return gaussianprior
end

"""
	sum_of_square_matrices(indexθlatent)

Return the sum of squares matrix of the latent variable parameters

ARGUMENT
-`indexθlatent`: structure indicating the order of each latent variable parameter if all parameters were concatenated into a vector

RETURN
-`𝐀`: A nest array of matrices. Element `𝐀[i]` corresponds to the Nᵢ×Nᵢ sum-of-squares matrix of the i-th group of parameters, with N parameters in the group
-`index𝐀`: Element `index𝐀[i][j]` corresponds to the i-th group of parameters and the j-th parameter in that group. The value of the element indicates the index of that parameter in a vector concatenating all the parameters in the model that are being fit.
"""
function shrinkagematrices(indexθlatent::Latentθ)
	𝐀 = Matrix{typeof(1.0)}[]
	index𝐀 = Vector{typeof(1)}[]
	for field in fieldnames(Latentθ)
		i = getfield(indexθlatent, field)[1]
		if i == 0 || field == :Aᶜ₁₁ || field == :Aᶜ₂₂ || field == :πᶜ₁
		else
			𝐀 = vcat(𝐀, [ones(1,1)])
			index𝐀 = vcat(index𝐀, [[i]])
		end
	end
	return 𝐀, index𝐀
end

"""
	shrinkagematrices(indexθ, max_spikehistory_lag, Φₐ, Φₘ, Φₜ)

Matrices that compute can compute the time average of the squares of each kernel

ARGUMENT
-`indexθ`: structure indicating the order of each parameter if all parameters were concatenated into a vector
-`glminputscaling`: scaling factor of GLM inputs
-`max_spikehistory_lag`: number of parameters controlling the effect of spike history
-`Φₘ`: values of the temporal basis functions parametizing the kernel of the timing of movement
-`Φₜ`: values of the temporal basis functions parametrizing time in each trial. Element `Φₜ[t,i]` corresponds to the value of the i-th temporal basis function at the t-th time step in each trial.

RETURN
-`𝐀`: A nest array of matrices. Element `𝐀[i]` corresponds to the Nᵢ×Nᵢ sum-of-squares matrix of the i-th group of parameters, with N parameters in the group
-`index𝐀`: Element `index𝐀[i][j]` corresponds to the i-th group of parameters and the j-th parameter in that group. The value of the element indicates the index of that parameter in a vector concatenating all the parameters in the model that are being fit.
"""
function shrinkagematrices(indexθglm::Vector{<:GLMθ}, glminputscaling::AbstractFloat, max_spikehistory_lag::Integer, Φₘ::Matrix{<:AbstractFloat}, Φₜ::Matrix{<:AbstractFloat})
	length𝐮 = length(indexθglm[1].𝐮)
	nbasestime = size(Φₜ,2)
	nbasesmove = size(Φₘ,2)
	nbasesaccu = length(indexθglm[1].𝐯[1])
	index𝐮hist = indexθglm[1].𝐠[end] .+ (1:max_spikehistory_lag)
	index𝐮time = index𝐮hist[end] .+ (1:nbasestime)
	index𝐮move = index𝐮time[end] .+ (1:nbasesmove)
	s² = glminputscaling^2
	Again = fill(s²,1,1)
	Ahist = zeros(max_spikehistory_lag,max_spikehistory_lag) + s²*I # computations with `Diagonal` are slower
	Atime = zeros(nbasestime,nbasestime) + s²*I
	Amove = zeros(nbasesmove,nbasesmove) + s²*I
	Aaccu = zeros(nbasesaccu,nbasesaccu) + s²*I
	𝐀 = Matrix{typeof(1.0)}[]
	index𝐀 = Vector{typeof(1)}[]
	for indexᵢₙ in indexθglm
		for k = 2:length(indexᵢₙ.𝐠)
			𝐀 = vcat(𝐀, [Again])
			index𝐀 = vcat(index𝐀, [indexᵢₙ.𝐠[k:k]])
		end
		if max_spikehistory_lag > 0
			𝐀 = vcat(𝐀, [Ahist])
			index𝐀 = vcat(index𝐀, [indexᵢₙ.𝐮[index𝐮hist]])
		end
		𝐀 = vcat(𝐀, [Atime])
		index𝐀 = vcat(index𝐀, [indexᵢₙ.𝐮[index𝐮time]])
		𝐀 = vcat(𝐀, [Amove])
		index𝐀 = vcat(index𝐀, [indexᵢₙ.𝐮[index𝐮move]])
		for indexᵢₙ𝐯ₖ in indexᵢₙ.𝐯
			𝐀 = vcat(𝐀, [Aaccu])
			index𝐀 = vcat(index𝐀, [indexᵢₙ𝐯ₖ])
		end
	end
	return 𝐀, index𝐀
end

"""
    variancematrices(indexθglm, max_spikehistory_lag, Φₐ, Φₜ)

Return the variance matrix of each group of parameters representing a time-varying quantity being flattened

ARGUMENT
-`indexθglm`: a nested array indexing each parameter in each mixture of Poisson GLM. The element `indexθglm[i][n]` corresponds to the n-th neuron in the i-th trialset
-`max_spikehistory_lag`: number of parameters for the spike history effect. This is needed only for indexing. Spike history effects are not being flattened.
-`Φₐ`: values of the temporal basis functions parametrizing hte time-varying encoding of the accumulator. Element `Φₐ[t,i]` corresponds to the value of the i-th temporal basis function at the t-th time step in each trial
-`Φₜ`: values of the temporal basis functions parametrizing time in each trial. Element `Φₜ[t,i]` corresponds to the value of the i-th temporal basis function at the t-th time step in each trial.

OUTPUT
-`𝚪`: A nest array of matrices. Element `𝚪[i]` corresponds to the Nᵢ×Nᵢ variance matrix of the i-th group of parameters, with N parameters in the group
-`index𝚪`: Element `index𝚪[i][j]` corresponds to the i-th group of parameters and the j-th parameter in that group. The value of the element indicates the index of that parameter in a vector concatenating all the parameters in the model that are being fit.
"""
function variancematrices(indexθglm::Vector{<:GLMθ}, max_spikehistory_lag::Integer, Φₐ::Matrix{<:AbstractFloat}, Φₜ::Matrix{<:AbstractFloat})
	Γaccumulator = Φₐ'*variancematrix(size(Φₐ,1))*Φₐ
	Γtime = Φₜ'*variancematrix(size(Φₜ,1))*Φₜ
	𝚪 = Matrix{typeof(1.0)}[]
	index𝚪 = Vector{typeof(1)}[]
	index𝐮time = max_spikehistory_lag .+ (1:size(Φₜ,2))
	for indexᵢₙ in indexθglm
		𝚪 = vcat(𝚪, [Γtime])
		index𝚪 = vcat(index𝚪, [indexᵢₙ.𝐮[index𝐮time]])
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
	L2penalty_coeffcients_limits(options, N_shrinkage_DDM, N_shrinkage_GLM, N_flattening_GLM)

Minimum and maximum of the coefficients of the L2 penalties

ARGUMENT
-`options`: Settings of the model
-`N_shrinkage_DDM`: number of shrinkage coefficients related to DDM parameters
-`N_shrinkage_GLM`: number of shrinkage coefficients related to GLM parameters
-`N_flattening_GLM`: number of flattening coefficients related to GLM parameters

OUTPUT
-`𝛂min`: vector of the minimum of the coefficient of each L2 penalty being learned
-`𝛂max`: vector of the maximum of the coefficient of each L2 penalty being learned
"""
function L2penalty_coeffcients_limits(options::Options, N_shrinkage_LV::Integer, N_shrinkage_GLM::Integer, N_flattening_GLM::Integer)
	𝛂min = vcat(options.L2shrinkage_LV_min	.*ones(N_shrinkage_LV),
				options.L2shrinkage_GLM_min	.*ones(N_shrinkage_GLM),
				options.L2flattening_GLM_min.*ones(N_flattening_GLM))
	𝛂max = vcat(options.L2shrinkage_LV_max .*ones(N_shrinkage_LV),
				options.L2shrinkage_GLM_max	.*ones(N_shrinkage_GLM),
 				options.L2flattening_GLM_max.*ones(N_flattening_GLM))
	return 𝛂min, 𝛂max
end

"""
    precisionmatrix!(gaussianprior)

Update the precision matrix

MODIFIED ARGUMENT
-`gaussianprior`: structure containing information on the Gaussian prior on the values of the model parameters in real space. The precision matrix `𝚲` is updated with respect to the L2 penalty coefficients 𝛂. The square submatrix 𝚽 of 𝚲 is also updated.
"""
function precisionmatrix!(gaussianprior::GaussianPrior)
    @unpack 𝐀, 𝛂, index𝐀, 𝚲, 𝚽, index𝚽  = gaussianprior
    𝚲 .= 0
    for i in eachindex(index𝐀)
        𝚲[index𝐀[i],index𝐀[i]] .+= 𝛂[i].*𝐀[i]
    end
	for i = 1:length(index𝚽)
		for j = 1:length(index𝚽)
			𝚽[i,j] = 𝚲[index𝚽[i], index𝚽[j]]
		end
	end
    return nothing
end
