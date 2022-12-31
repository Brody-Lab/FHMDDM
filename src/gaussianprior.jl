"""
    GaussianPrior(options, trialsets)

Construct a structure containing information on the Gaussian prior on the model's parameters

ARGUMENT
-`options`: settings of the model
-`trialsets`: data for the model

OUTPUT
-an instance of `GaussianPrior`
"""
function GaussianPrior(options::Options, trialsets::Vector{<:Trialset})
    indexθ = indexparameters(options, trialsets)
	𝐀, index𝐀 = shrinkagematrices(indexθ.latentθ, options)
	𝛂max = options.L2_latent_max .*ones(length(𝐀))
	𝛂min = options.L2_latent_min .*ones(length(𝐀))
	for glmθs_each_trialset in indexθ.glmθ
		𝐀_glm, index𝐀_glm, 𝛂max_glm, 𝛂min_glm = shrinkagematrices(glmθs_each_trialset, options)
		𝐀 = vcat(𝐀, 𝐀_glm)
		index𝐀 = vcat(index𝐀, index𝐀_glm)
		𝛂max = vcat(𝛂max, 𝛂max_glm)
		𝛂min = vcat(𝛂min, 𝛂min_glm)
	end
	nparameters = concatenate(indexθ)[end]
    gaussianprior = GaussianPrior(𝐀=𝐀, 𝛂=sqrt.(𝛂min.*𝛂max), 𝛂min=𝛂min, 𝛂max=𝛂max, index𝐀=index𝐀, 𝚲=zeros(nparameters,nparameters))
    precisionmatrix!(gaussianprior)
    return gaussianprior
end

"""
	shrinkagematrices(indexθlatent, options)

Return the sum of squares matrix of the latent variable parameters

ARGUMENT
-`indexθlatent`: structure indicating the order of each latent variable parameter if all parameters were concatenated into a vector
-`options`: settings of the model

RETURN
-`𝐀`: A nest array of matrices. Element `𝐀[i]` corresponds to the Nᵢ×Nᵢ sum-of-squares matrix of the i-th group of parameters, with N parameters in the group
-`index𝐀`: Element `index𝐀[i][j]` corresponds to the i-th group of parameters and the j-th parameter in that group. The value of the element indicates the index of that parameter in a vector concatenating all the parameters in the model that are being fit.
"""
function shrinkagematrices(indexθlatent::Latentθ, options::Options)
	𝐀 = Matrix{typeof(1.0)}[]
	index𝐀 = Vector{typeof(1)}[]
	if options.L2_latent_fit
		for field in fieldnames(Latentθ)
			i = getfield(indexθlatent, field)[1]
			if i == 0 || field == :Aᶜ₁₁ || field == :Aᶜ₂₂ || field == :πᶜ₁
			else
				𝐀 = vcat(𝐀, [ones(1,1)])
				index𝐀 = vcat(index𝐀, [[i]])
			end
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
-`𝛂max`: a vector containing the maximum precision of the prior on each parameter
-`𝛂min`: a vector containing the minimum precision of the prior on each parameter
"""
function shrinkagematrices(indexθglm::Vector{<:GLMθ}, options::Options)
	@unpack 𝐮indices_hist, 𝐮indices_time, 𝐮indices_move, 𝐮indices_phot = indexθglm[1]
	nbaseshist = length(𝐮indices_hist)
	nbasestime = length(𝐮indices_time)
	nbasesmove = length(𝐮indices_move)
	nbasesphot = length(𝐮indices_phot)
	nbasesaccu = length(indexθglm[1].𝐯[1])
	Ahist = zeros(nbaseshist,nbaseshist) + options.tbf_hist_scalefactor^2*I # computations with `Diagonal` are slower
	Atime = zeros(nbasestime,nbasestime) + options.tbf_time_scalefactor^2*I
	Amove = zeros(nbasesmove,nbasesmove) + options.tbf_move_scalefactor^2*I
	Aphot = zeros(nbasesphot,nbasesphot) + options.tbf_phot_scalefactor^2*I
	Aevtr = ones(1,1)*options.b_scalefactor^2
	Aaccu = zeros(nbasesaccu,nbasesaccu) + options.tbf_accu_scalefactor^2*I
	𝐀 = Matrix{typeof(1.0)}[]
	index𝐀 = Vector{typeof(1)}[]
	𝛂max = typeof(1.0)[]
	𝛂min = typeof(1.0)[]
	for indexᵢₙ in indexθglm
		if nbaseshist > 0 & options.L2_hist_fit
			𝐀 = vcat(𝐀, [Ahist])
			index𝐀 = vcat(index𝐀, [indexᵢₙ.𝐮[𝐮indices_hist]])
			𝛂max = vcat(𝛂max, options.L2_hist_max)
			𝛂min = vcat(𝛂min, options.L2_hist_min)
		end
		if nbasestime > 0 & options.L2_time_fit
			𝐀 = vcat(𝐀, [Atime])
			index𝐀 = vcat(index𝐀, [indexᵢₙ.𝐮[𝐮indices_time]])
			𝛂max = vcat(𝛂max, options.L2_time_max)
			𝛂min = vcat(𝛂min, options.L2_time_min)
		end
		if nbasesmove > 0 & options.L2_move_fit
			𝐀 = vcat(𝐀, [Amove])
			index𝐀 = vcat(index𝐀, [indexᵢₙ.𝐮[𝐮indices_move]])
			𝛂max = vcat(𝛂max, options.L2_move_max)
			𝛂min = vcat(𝛂min, options.L2_move_min)
		end
		if nbasesphot > 0 options.L2_phot_fit
			𝐀 = vcat(𝐀, [Aphot])
			index𝐀 = vcat(index𝐀, [indexᵢₙ.𝐮[𝐮indices_phot]])
			𝛂max = vcat(𝛂max, options.L2_phot_max)
			𝛂min = vcat(𝛂min, options.L2_phot_min)
		end
		if options.L2_v_fit
			for indexᵢₙ𝐯ₖ in indexᵢₙ.𝐯
				𝐀 = vcat(𝐀, [Aaccu])
				index𝐀 = vcat(index𝐀, [indexᵢₙ𝐯ₖ])
				𝛂max = vcat(𝛂max, options.L2_v_max)
				𝛂min = vcat(𝛂min, options.L2_v_min)
			end
		end
		if options.L2_Δ𝐯_fit
			if indexᵢₙ.fit_Δ𝐯
				for indexᵢₙΔ𝐯ₖ in indexᵢₙ.Δ𝐯
					𝐀 = vcat(𝐀, [Aaccu])
					index𝐀 = vcat(index𝐀, [indexᵢₙΔ𝐯ₖ])
					𝛂max = vcat(𝛂max, options.L2_Δ𝐯_max)
					𝛂min = vcat(𝛂min, options.L2_Δ𝐯_min)
				end
			end
		end
		if indexᵢₙ.fit_b & options.L2_b_fit
			𝐀 = vcat(𝐀, [Aevtr])
			index𝐀 = vcat(index𝐀, [indexᵢₙ.b])
			𝛂max = vcat(𝛂max, options.L2_b_max)
			𝛂min = vcat(𝛂min, options.L2_b_min)
		end
	end
	return 𝐀, index𝐀, 𝛂max, 𝛂min
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
