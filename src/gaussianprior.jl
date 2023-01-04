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
    gaussianprior = GaussianPrior(𝐀=𝐀, 𝛂=sqrt.(𝛂min.*𝛂max), 𝛂min=𝛂min, 𝛂max=𝛂max, index𝐀=index𝐀, 𝚲=zeros(nparameters,nparameters), penaltynames=namepenalties(indexθ, options))
    precisionmatrix!(gaussianprior)
    return gaussianprior
end

"""
	namepenalties(indexθ, options)

Name of each L2 regularization term to the optimization problem

ARGUMENT
-`indexθ`: a composite containing the indices of the parameters if they were concatenated into a vector
-`options`: a composite containining fixed hyperparameters

RETURN
-a vector of String
"""
function namepenalties(indexθ::Indexθ, options::Options)
	if options.L2_latent_fit
		return vcat(namepenalties(indexθ.latentθ), namepenalties(indexθ.glmθ, options))
	else
		return namepenalties(indexθ.glmθ, options)
	end
end

"""
	namepenalties(indices)

Name of each L2 regularization term on the latent-variable parameters

ARGUMENT
-`indexθ`: a composite containing the indices of the latent-variable parameters if they were concatenated into a vector

RETURN
-a vector of String
"""
function namepenalties(indices::Latentθ)
	parameternames = String[]
	for name in fieldnames(Latentθ)
		i = getfield(indices, name)[1]
		if (i == 0) || (name == :Aᶜ₁₁) || (name == :Aᶜ₂₂) || (name == :πᶜ₁)
		else
			parameternames = vcat(parameternames, matlabname(name))
		end
	end
	return parameternames
end

"""
	namepenalties(nested_glmθs)

Name of each L2 regularization term of each Poisson mixture GLM

ARGUMENT
-`nested_glmθs`: a nested array whose element `nested_glmθs[i][n]` is a composite containing the parameter values or indices of the Poisson mixture generalized linear model of the n-th neuron in the i-th trialset
-`options`: a composite containing fixed hyperparameters

RETURN
-a vector of String
"""
function namepenalties(nested_glmθs::Vector{<:Vector{<:GLMθ}}, options::Options)
	ntrialsets = length(nested_glmθs)
	if ntrialsets > 1
		reduce(vcat, (namepenalties(i,n,nested_glmθs[i][n],options) for i in eachindex(nested_glmθs) for n in eachindex(nested_glmθs[i][n])))
	else
		reduce(vcat, (namepenalties(n,nested_glmθs[1][n],options) for n in eachindex(nested_glmθs[1])))
	end
end

"""
	namepenalties(trialsetindex, neuronindex, glmθ, options)

Name of each L2 regularization term of each Poisson mixture GLM
"""
function namepenalties(trialsetindex::Integer, neuronindex::Integer, glmθ::GLMθ, options::Options)
	map(namepenalties(glmθ,options)) do name
		"trialset"*string(trialsetindex)*"_neuron"*string(neuronindex)*"_"*name
	end
end

"""
	namepenalties(neuronindex, glmθ, options)

Name of each L2 regularization term of each Poisson mixture GLM
"""
function namepenalties(neuronindex::Integer, glmθ::GLMθ, options::Options)
	map(namepenalties(glmθ,options)) do name
		"neuron"*string(neuronindex)*"_"*name
	end
end

"""
	namepenalties(indices𝐮, options)

Name of each L2 regularization term on the accumulator-independent filters
"""
function namepenalties(indices𝐮::Indices𝐮, options::Options)
	penaltynames = String[]
	for filtername in fieldnames(Indices𝐮)
		if (filtername == :postspike) & (length(getfield(indices𝐮,filtername)) > 0) & options.L2_hist_fit
			penaltynames = vcat(penaltynames, string(filtername))
		elseif (filtername == :poststereoclick) & (length(getfield(indices𝐮,filtername)) > 0) & options.L2_time_fit
			penaltynames = vcat(penaltynames, string(filtername))
		elseif (filtername == :premovement) & (length(getfield(indices𝐮,filtername)) > 0) & options.L2_move_fit
			penaltynames = vcat(penaltynames, string(filtername))
		elseif (filtername == :postphotostimulus) & (length(getfield(indices𝐮,filtername)) > 0) options.L2_phot_fit
			penaltynames = vcat(penaltynames, string(filtername))
		end
	end
	return penaltynames
end

"""
	namepenalties(glmθ, options)

Name of each L2 regularization term of a Poisson mixture GLM

Had the implementation use `reduce(vcat, ...)`, the resulting vector may contain 'nothing'.
"""
function namepenalties(glmθ::GLMθ, options::Options)
	penaltynames = String[]
	for name in glmθ.concatenationorder
		if name == :𝐮
			penaltynames = vcat(penaltynames, namepenalties(glmθ.indices𝐮, options))
		elseif (name == :𝐯) & options.L2_v_fit
			penaltynames = vcat(penaltynames, "accumulator_encoding")
		elseif (name == :b) & glmθ.fit_b & options.L2_b_fit
			penaltynames = vcat(penaltynames, "accumulator_transformation")
		end
	end
	return penaltynames
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
			if (i == 0) || (field == :Aᶜ₁₁) || (field == :Aᶜ₂₂) || (field == :πᶜ₁)
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
	nbaseshist = length(indexθglm[1].indices𝐮.postspike)
	nbasestime = length(indexθglm[1].indices𝐮.poststereoclick)
	nbasesmove = length(indexθglm[1].indices𝐮.premovement)
	nbasesphot = length(indexθglm[1].indices𝐮.postphotostimulus)
	Ahist = zeros(nbaseshist,nbaseshist) + options.tbf_hist_scalefactor^2*I # computations with `Diagonal` are slower
	Atime = zeros(nbasestime,nbasestime) + options.tbf_time_scalefactor^2*I
	Amove = zeros(nbasesmove,nbasesmove) + options.tbf_move_scalefactor^2*I
	Aphot = zeros(nbasesphot,nbasesphot) + options.tbf_phot_scalefactor^2*I
	Aevtr = ones(1,1)*options.b_scalefactor^2
	if indexθglm[1].fit_Δ𝐯
		A = [2.0 -1.0; -1.0 2.0].*options.tbf_accu_scalefactor^2
		Aaccu = cat((A for k in eachindex(indexθglm[1].𝐯) for q in eachindex(indexθglm[1].𝐯[k]) )...,dims=(1,2))
	else
		nbasesaccu = length(vcat(indexθglm[1].𝐯...))
		Aaccu = zeros(nbasesaccu,nbasesaccu) + options.tbf_accu_scalefactor^2*I
	end
	𝐀 = Matrix{typeof(1.0)}[]
	index𝐀 = Vector{typeof(1)}[]
	𝛂max = typeof(1.0)[]
	𝛂min = typeof(1.0)[]
	for indexᵢₙ in indexθglm
		for name in indexᵢₙ.concatenationorder
			if name == :𝐮
				for filtername in fieldnames(Indices𝐮)
					if (filtername == :postspike) & (nbaseshist > 0) & options.L2_hist_fit
						𝐀 = vcat(𝐀, [Ahist])
						parameterindices = indexᵢₙ.𝐮[getfield(indexᵢₙ.indices𝐮, filtername)]
						index𝐀 = vcat(index𝐀, [parameterindices])
						𝛂max = vcat(𝛂max, options.L2_hist_max)
						𝛂min = vcat(𝛂min, options.L2_hist_min)
					elseif (filtername == :poststereoclick) & (nbasestime > 0) & options.L2_time_fit
						𝐀 = vcat(𝐀, [Atime])
						parameterindices = indexᵢₙ.𝐮[getfield(indexᵢₙ.indices𝐮, filtername)]
						index𝐀 = vcat(index𝐀, [parameterindices])
						𝛂max = vcat(𝛂max, options.L2_time_max)
						𝛂min = vcat(𝛂min, options.L2_time_min)
					elseif (filtername == :premovement) & (nbasesmove > 0) & options.L2_move_fit
						𝐀 = vcat(𝐀, [Amove])
						parameterindices = indexᵢₙ.𝐮[getfield(indexᵢₙ.indices𝐮, filtername)]
						index𝐀 = vcat(index𝐀, [parameterindices])
						𝛂max = vcat(𝛂max, options.L2_move_max)
						𝛂min = vcat(𝛂min, options.L2_move_min)
					elseif (filtername == :postphotostimulus) & (nbasesphot > 0) options.L2_phot_fit
						𝐀 = vcat(𝐀, [Aphot])
						parameterindices = indexᵢₙ.𝐮[getfield(indexᵢₙ.indices𝐮, filtername)]
						index𝐀 = vcat(index𝐀, [parameterindices])
						𝛂max = vcat(𝛂max, options.L2_phot_max)
						𝛂min = vcat(𝛂min, options.L2_phot_min)
					end
				end
			elseif (name == :𝐯) & options.L2_v_fit
				𝐀 = vcat(𝐀, [Aaccu])
				if indexᵢₙ.fit_Δ𝐯
					parameterindices = vcat(([indexᵢₙ.𝐯[k][q], indexᵢₙ.Δ𝐯[k][q]] for k in eachindex(indexᵢₙ.𝐯) for q in eachindex(indexᵢₙ.𝐯[k]))...)
				else
					parameterindices = vcat(indexᵢₙ.𝐯...)
				end
				index𝐀 = vcat(index𝐀, [parameterindices])
				𝛂max = vcat(𝛂max, options.L2_v_max)
				𝛂min = vcat(𝛂min, options.L2_v_min)
			elseif (name == :b) & indexᵢₙ.fit_b & options.L2_b_fit
				𝐀 = vcat(𝐀, [Aevtr])
				index𝐀 = vcat(index𝐀, [indexᵢₙ.b])
				𝛂max = vcat(𝛂max, options.L2_b_max)
				𝛂min = vcat(𝛂min, options.L2_b_min)
			end
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
	gaussianprior = GaussianPrior(𝐀=gaussianprior.𝐀, 𝛂=𝛂, 𝛂min=gaussianprior.𝛂min, 𝛂max=gaussianprior.𝛂max, index𝐀=gaussianprior.index𝐀, 𝚲=zeros(type,size(gaussianprior.𝚲)), penaltynames=gaussianprior.penaltynames)
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
