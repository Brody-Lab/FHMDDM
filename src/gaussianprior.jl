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
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_08_08a_test/T176_2018_05_03/data.mat"
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
		𝐀_b, index𝐀_b = shrinkagematrices(indexθ.glmθ[i], options.b_scalefactor)
		𝐀_glm, index𝐀_glm = shrinkagematrices(indexθ.glmθ[i], options)
		𝐀 = vcat(𝐀, 𝐀_lv, 𝐀_b, 𝐀_glm)
		index𝐀 = vcat(index𝐀, index𝐀_lv, index𝐀_b, index𝐀_glm)
		𝛂min_t, 𝛂max_t = L2penalty_coeffcients_limits(options, length(index𝐀_lv), length(index𝐀_b), length(index𝐀_glm))
		𝛂min = vcat(𝛂min, 𝛂min_t)
		𝛂max = vcat(𝛂max, 𝛂max_t)
	end
	𝛂 = sqrt.(𝛂min.*𝛂max)
	if indexθ.glmθ[end][end].fit_𝛃
		N = indexθ.glmθ[end][end].𝛃[end][end]
	else
	    N = indexθ.glmθ[end][end].𝐯[end][end]
	end
    gaussianprior = GaussianPrior(𝐀=𝐀, 𝛂=𝛂, 𝛂min=𝛂min, 𝛂max=𝛂max, index𝐀=index𝐀, 𝚲=zeros(N,N))
    precisionmatrix!(gaussianprior)
    return gaussianprior
end

"""
	native2real(gaussianprior)

Convert the L2 penalty coefficients from native to real space

ARGUMENT
-`gaussianprior`: a structure with information on the Gaussian prior. The field `𝛂` is modified.

RETURN
-`𝐱`: a vector of the L2 penalty coefficients in real space
"""
function native2real(gaussianprior::GaussianPrior)
	collect(native2real(α, αmin, αmax) for (α, αmin, αmax) in zip(gaussianprior.𝛂, gaussianprior.𝛂min, gaussianprior.𝛂max))
end

"""
	native2real(n,l,u)

Convert hyperparameters from native space to real space

ARGUMENT
-`n`: value in native space
-`l`: lower bound in native space
-`u`: upper bound in native space

RETURN
-vector representing the values in real space

"""
function native2real(n::Real, l::Real, u::Real)
	logit((n-l)/(u-l))
end

"""
	real2native!(gaussianprior, 𝐱)

Convert the L2 penalty coefficients from real to native space

MODIFIED ARGUMENT
-`gaussianprior`: a structure with information on the Gaussian prior. The field `𝛂` is modified.

UNMODIFIED ARGUMENT
-`𝐱`: a vector of the L2 penalty coefficients in real space
"""
function real2native!(gaussianprior::GaussianPrior, 𝐱::Vector{<:Real})
	@unpack 𝛂, 𝛂min, 𝛂max = gaussianprior
	for i = 1:length(𝛂)
		𝛂[i] = real2native(𝐱[i], 𝛂min[i], 𝛂max[i])
	end
	return nothing
end

"""
	real2native(gaussianprior, 𝐱)

Convert the L2 penalty coefficients from real to native space

ARGUMENT
-`gaussianprior`: a structure with information on the Gaussian prior. The field `𝛂` is modified.
-`𝐱`: a vector of the L2 penalty coefficients in real space

RETURN
-`gaussianpiror`
"""
function real2native(gaussianprior::GaussianPrior, 𝐱::Vector{type}) where {type<:Real}
	@unpack 𝛂min, 𝛂max = gaussianprior
	𝛂 = collect(real2native(x, αmin, αmax) for (x, αmin, αmax) in zip(𝐱, 𝛂min, 𝛂max))
	gaussianprior = GaussianPrior(𝐀=gaussianprior.𝐀, 𝛂=𝛂, 𝛂min=gaussianprior.𝛂min, 𝛂max=gaussianprior.𝛂max, index𝐀=gaussianprior.index𝐀, 𝚲=zeros(type,size(gaussianprior.𝚲)))
    precisionmatrix!(gaussianprior)
    return gaussianprior
end

"""
	real2native(r,l,u)

Convert a hyperparameter from real space to native space.

ARGUMENT
-`r`: vector of values in real space
-`l`: lower bound in native space
-`u`: upper bound in native space

RETURN
-scalar representing the value in native space
"""
function real2native(r::Real, l::Real, u::Real)
	l + (u-l)*logistic(r)
end

"""
	shrinkagematrices(indexθlatent)

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
	shrinkagematrices(indexθ, options)

Matrices that compute can compute the time average of the squares of each kernel

ARGUMENT
-`indexθ`: structure indicating the order of each parameter if all parameters were concatenated into a vector
-`options`: settings of the model

RETURN
-`𝐀`: A nest array of matrices. Element `𝐀[i]` corresponds to the Nᵢ×Nᵢ sum-of-squares matrix of the i-th group of parameters, with N parameters in the group
-`index𝐀`: Element `index𝐀[i][j]` corresponds to the i-th group of parameters and the j-th parameter in that group. The value of the element indicates the index of that parameter in a vector concatenating all the parameters in the model that are being fit.
"""
function shrinkagematrices(indexθglm::Vector{<:GLMθ}, options::Options)
	@unpack 𝐮indices_hist, 𝐮indices_time, 𝐮indices_move = indexθglm[1]
	nbaseshist = length(𝐮indices_hist)
	nbasestime = length(𝐮indices_time)
	nbasesmove = length(𝐮indices_move)
	nbasesaccu = length(indexθglm[1].𝐯[1])
	Again = ones(1,1)
	Ahist = zeros(nbaseshist,nbaseshist) + options.tbf_hist_scalefactor^2*I # computations with `Diagonal` are slower
	Atime = zeros(nbasestime,nbasestime) + options.tbf_time_scalefactor^2*I
	Amove = zeros(nbasesmove,nbasesmove) + options.tbf_move_scalefactor^2*I
	Aaccu = zeros(nbasesaccu,nbasesaccu) + options.tbf_accu_scalefactor^2*I
	𝐀 = Matrix{typeof(1.0)}[]
	index𝐀 = Vector{typeof(1)}[]
	for indexᵢₙ in indexθglm
		for k = 2:length(indexᵢₙ.𝐠)
			𝐀 = vcat(𝐀, [Again])
			index𝐀 = vcat(index𝐀, [indexᵢₙ.𝐠[k:k]])
		end
		if nbaseshist > 0
			𝐀 = vcat(𝐀, [Ahist])
			index𝐀 = vcat(index𝐀, [indexᵢₙ.𝐮[𝐮indices_hist]])
		end
		if nbasestime > 0
			𝐀 = vcat(𝐀, [Atime])
			index𝐀 = vcat(index𝐀, [indexᵢₙ.𝐮[𝐮indices_time]])
		end
		if nbasesmove > 0
			𝐀 = vcat(𝐀, [Amove])
			index𝐀 = vcat(index𝐀, [indexᵢₙ.𝐮[𝐮indices_move]])
		end
		if nbasesaccu > 0
			for indexᵢₙ𝐯ₖ in indexᵢₙ.𝐯
				𝐀 = vcat(𝐀, [Aaccu])
				index𝐀 = vcat(index𝐀, [indexᵢₙ𝐯ₖ])
			end
			if indexᵢₙ.fit_𝛃
				for indexᵢₙ𝛃ₖ in indexᵢₙ.𝛃
					𝐀 = vcat(𝐀, [Aaccu])
					index𝐀 = vcat(index𝐀, [indexᵢₙ𝛃ₖ])
				end
			end
		end
	end
	return 𝐀, index𝐀
end

"""
	shrinkagematrices(indexθ)

Matrices that compute the L2 penalty for the nonlinearity parameter in accumulator transformations

ARGUMENT
-`indexθ`: structure indicating the order of each parameter if all parameters were concatenated into a vector
-`b_scalefactor`: scale factor of the nonlinearity parameter

RETURN
-`𝐀`: A nest array of matrices. Element `𝐀[i]` corresponds to the Nᵢ×Nᵢ sum-of-squares matrix of the i-th group of parameters, with N parameters in the group
-`index𝐀`: Element `index𝐀[i][j]` corresponds to the i-th group of parameters and the j-th parameter in that group. The value of the element indicates the index of that parameter in a vector concatenating all the parameters in the model that are being fit.
"""
function shrinkagematrices(indexθglm::Vector{<:GLMθ}, b_scalefactor::Real)
	A = ones(1,1)*b_scalefactor^2
	𝐀 = Matrix{typeof(1.0)}[]
	index𝐀 = Vector{typeof(1)}[]
	for indexᵢₙ in indexθglm
		if indexᵢₙ.b[1] > 0
			𝐀 = vcat(𝐀, [A])
			index𝐀 = vcat(index𝐀, [indexᵢₙ.b])
		end
	end
	return 𝐀, index𝐀
end

"""
	L2penalty_coeffcients_limits(options, N_shrinkage_DDM, N_shrinkage_GLM)

Minimum and maximum of the coefficients of the L2 penalties

ARGUMENT
-`options`: Settings of the model
-`N_shrinkage_DDM`: number of shrinkage coefficients related to DDM parameters
-`N_shrinkage_GLM`: number of shrinkage coefficients related to GLM parameters

OUTPUT
-`𝛂min`: vector of the minimum of the coefficient of each L2 penalty being learned
-`𝛂max`: vector of the maximum of the coefficient of each L2 penalty being learned
"""
function L2penalty_coeffcients_limits(options::Options, N_shrinkage_LV::Integer, N_shrinkage_AT::Integer, N_shrinkage_GLM::Integer)
	𝛂min = vcat(options.L2shrinkage_LV_min	.*ones(N_shrinkage_LV),
				options.L2shrinkage_b_min	.*ones(N_shrinkage_AT),
				options.L2shrinkage_GLM_min	.*ones(N_shrinkage_GLM))
	𝛂max = vcat(options.L2shrinkage_LV_max .*ones(N_shrinkage_LV),
				options.L2shrinkage_b_max .*ones(N_shrinkage_AT),
				options.L2shrinkage_GLM_max	.*ones(N_shrinkage_GLM))
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
