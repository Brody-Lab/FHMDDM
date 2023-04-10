"""
	contents

GaussianPrior(options, trialsets)
namepenalties(indexθ, options)
namepenalties(indices)
shrinkagematrices(indexθlatent, options)
precision_matrix_components(nestedindices, options)
addprior(𝐀, 𝛂max, 𝛂min, index𝐀, options, parameterindices, parametername)
precisionmatrix!(gaussianprior)
native2real(gaussianprior)
native2real(n,l,u)
real2native!(gaussianprior, 𝐱)
real2native(gaussianprior, 𝐱)
real2native(r,l,u)
"""

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
	𝐀_glm, index𝐀_glm, 𝛂max_glm, 𝛂min_glm = precision_matrix_components(indexθ.glmθ, options)
	𝐀 = vcat(𝐀, 𝐀_glm)
	index𝐀 = vcat(index𝐀, index𝐀_glm)
	𝛂max = vcat(𝛂max, 𝛂max_glm)
	𝛂min = vcat(𝛂min, 𝛂min_glm)
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
namepenalties(indexθ::Indexθ, options::Options) = vcat(namepenalties(indexθ.latentθ), namepenalties(indexθ.glmθ, options))

"""
	namepenalties(indices)

Name of each L2 regularization term on the latent-variable parameters

ARGUMENT
-`indexθ`: a composite containing the indices of the latent-variable parameters if they were concatenated into a vector

RETURN
-a vector of String
"""
function namepenalties(indices::Latentθ)
	penaltynames = String[]
	for name in fieldnames(Latentθ)
		i = getfield(indices, name)[1]
		if (i == 0) || (name == :Aᶜ₁₁) || (name == :Aᶜ₂₂) || (name == :πᶜ₁)
		else
			penaltynames = vcat(penaltynames, matlabname(name))
		end
	end
	return penaltynames
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
function namepenalties(nestedindices::Vector{<:Vector{<:GLMθ}}, options::Options)
	penaltynames = String[]
	for name in nestedindices[1][1].concatenationorder
		if name == :𝐮
			for filtername in fieldnames(Indices𝐮)
				parameterindices = reduce(vcat, reduce(vcat, index.𝐮[getfield(index.indices𝐮,filtername)] for index in indices) for indices in nestedindices)
				nparameters = length(parameterindices)
				if nparameters > 0
					penaltynames = vcat(penaltynames, string(filtername))
				end
			end
		elseif name == :𝐯
			penaltynames = vcat(penaltynames, "accumulator_encoding")
		elseif (name == :b) & options.fit_b
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
	for field in fieldnames(Latentθ)
		i = getfield(indexθlatent, field)[1]
		if (i == 0) || (field == :Aᶜ₁₁) || (field == :Aᶜ₂₂) || (field == :πᶜ₁)
		else
			𝐀 = vcat(𝐀, [ones(1,1)])
			index𝐀 = vcat(index𝐀, [[i]])
		end
	end
	return 𝐀, index𝐀
end

"""
	precision_matrix_components(nestedindices, options)

Components of the precision matrix

ARGUMENT
-`nestedindices`: structure indicating the order of each parameter if all parameters were concatenated into a vector
-`options`: settings of the model

RETURN
-`𝐀`: A nest array of matrices. Element `𝐀[i]` corresponds to the Nᵢ×Nᵢ sum-of-squares matrix of the i-th group of parameters, with N parameters in the group.
-`index𝐀`: Element `index𝐀[i][j]` corresponds to the i-th group of parameters and the j-th parameter in that group. The value of the element indicates the index of that parameter in a vector concatenating all the parameters in the model that are being fit.
-`𝛂max`: a vector containing the maximum precision of the prior on each parameter
-`𝛂min`: a vector containing the minimum precision of the prior on each parameter
"""
function precision_matrix_components(nestedindices::Vector{<:Vector{<:GLMθ}}, options::Options)
	𝐀 = Matrix{typeof(1.0)}[]
	index𝐀 = Vector{typeof(1)}[]
	𝛂max = typeof(1.0)[]
	𝛂min = typeof(1.0)[]
	for name in nestedindices[1][1].concatenationorder
		if name == :𝐮
			for parametername in fieldnames(Indices𝐮)
				parameterindices = reduce(vcat, reduce(vcat, index.𝐮[getfield(index.indices𝐮, parametername)] for index in indices) for indices in nestedindices)
				𝐀, 𝛂max, 𝛂min, index𝐀 = addprior(𝐀, 𝛂max, 𝛂min, index𝐀, options, parameterindices, parametername)
			end
		elseif (name == :𝐯)
			if options.fit_𝛃
				parameterindices = reduce(vcat, reduce(vcat, reduce(vcat, reduce(vcat, vcat(v,Δv) for (v, Δv) in zip(𝐯ₖ, 𝛃ₖ)) for (𝐯ₖ, 𝛃ₖ) in zip(index.𝐯, index.𝛃)) for index in indices) for indices in nestedindices)
			else
				parameterindices = reduce(vcat, reduce(vcat, vcat(index.𝐯...) for index in indices) for indices in nestedindices)
			end
			𝐀, 𝛂max, 𝛂min, index𝐀 = addprior(𝐀, 𝛂max, 𝛂min, index𝐀, options, parameterindices, :accumulator)
		elseif (name == :b) & options.fit_b
			parameterindices = reduce(vcat, collect(index.b[1] for index in indices) for indices in nestedindices)
			𝐀, 𝛂max, 𝛂min, index𝐀 = addprior(𝐀, 𝛂max, 𝛂min, index𝐀, options, parameterindices, :b)
		end
	end
	return 𝐀, index𝐀, 𝛂max, 𝛂min
end

"""
	addprior(𝐀, 𝛂max, 𝛂min, index𝐀, options, parameterindices, parametername)

Extend the list of gaussian priors

ARGUMENTS
-`𝐀`: A nest array of matrices. Element `𝐀[i]` corresponds to the Nᵢ×Nᵢ sum-of-squares matrix of the i-th group of parameters, with N parameters in the group.
-`𝛂max`: a vector containing the maximum precision of the prior on each parameter
-`𝛂min`: a vector containing the minimum precision of the prior on each parameter
-`index𝐀`: Element `index𝐀[i][j]` corresponds to the i-th group of parameters and the j-th parameter in that group. The value of the element indicates the index of that parameter in a vector concatenating all the parameters in the model that are being fit.
-`options`: fixed hyperparameters
-`parameterindices`: indices of the parameter
-`parametername`: name of the class of parameters

RETURN
-`𝐀`
-`𝛂max`
-`𝛂min`
-`index𝐀`
"""
function addprior(𝐀::Vector{<:Matrix{<:AbstractFloat}},
				𝛂max::Vector{<:AbstractFloat},
				𝛂min::Vector{<:AbstractFloat},
				index𝐀::Vector{<:Vector{<:Integer}},
				options::Options,
				parameterindices::Vector{<:Integer},
				parametername::Symbol)
	nparameters = length(parameterindices)
	if nparameters > 0
		scalefactor = getfield(options, Symbol("tbf_"*String(parametername)*"_scalefactor"))*options.sf_tbf[1]
		𝐀 = vcat(𝐀, [zeros(nparameters,nparameters) + scalefactor^2*I])
		index𝐀 = vcat(index𝐀, [parameterindices])
		𝛂max = vcat(𝛂max, getfield(options, Symbol("L2_"*String(parametername)*"_max")))
		𝛂min = vcat(𝛂min, getfield(options, Symbol("L2_"*String(parametername)*"_max")))
	end
	return 𝐀, 𝛂max, 𝛂min, index𝐀
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
