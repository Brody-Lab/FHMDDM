"""
	concatenate(indexθ)

Concatenate the index of each parameter into a vector
"""
function concatenate(indexθ::Indexθ; includeunfit=false)
	if includeunfit
		latentθindices = concatenateparameters(indexθ.latentθ)
	else
		latentθindices = concatenateparameters(indexθ.latentθ, indexθ.latentθ)
	end
	glmθindices = vcat((vcat((concatenateparameters(glmθ; includeunfit=includeunfit) for glmθ in glmθ)...) for glmθ in indexθ.glmθ)...)
	vcat(latentθindices, glmθindices)
end

"""
    concatenate_choice_related_parameters(model)

Concatenate values of parameters being fitted into a vector of floating point numbers

ARGUMENT
-`model`: the factorial hidden Markov drift-diffusion model

RETURN
-`concatenatedθ`: a vector of the concatenated values of the parameters being fitted
-`indexθ`: a structure indicating the index of each model parameter in the vector of concatenated values
"""
function concatenate_choice_related_parameters(model::Model)
    @unpack options, θreal, trialsets = model
	concatenatedθ = zeros(0)
    counter = 0
	indexθ = Latentθ(collect(zeros(Int64,1) for i in fieldnames(Latentθ))...)
	for field in fieldnames(Latentθ)
		tofit = is_parameter_fit(options, field)
		if tofit
			counter += 1
			getfield(indexθ, field)[1] = counter
			concatenatedθ = vcat(concatenatedθ, getfield(θreal, field)[1])
		else
			getfield(indexθ, field)[1] = 0
		end
	end
	indexθglm = collect(collect(GLMθ(Int, mpGLM.θ) for mpGLM in trialset.mpGLMs) for trialset in model.trialsets)
    return concatenatedθ, Indexθ(latentθ=indexθ, glmθ = indexθglm)
end

"""
	concatenateparameters(latentθ, options)

Concatenate the latent-variable-parameters that are being fit into a vector
"""
concatenateparameters(latentθ::Latentθ, options::Options) = concatenateparameters(indexparameters(options), latentθ)

"""
	concatenateparameters(indices, values)

Concatenate the parameters of the latent variables into a vector

ARGUMENT
-`indices`: a composite indicating whether each latent-variable-parameter is fit, and if so, the position of the parameter in a vector
-values: a composite of the values of the latent-variable-parameters

RETURN
-a vector of floats
"""
function concatenateparameters(indices::Latentθ{<:Vector{<:Integer}}, values::Latentθ{<:Vector{<:type}}) where type<:Real
	vcat((getfield(indices, name)[1] > 0 ? getfield(values, name) : zeros(type,0) for name in fieldnames(Latentθ))...)
end

"""
	concatenateparameters(latentθ)

Concatenate latent-variable-parameters, even those not being fit, into a vector
"""
concatenateparameters(latentθ::Latentθ) = collect(getfield(latentθ, name)[1] for name in fieldnames(Latentθ))

"""
	concatenateparameters(model)

Concatenate values of parameters being fitted into a vector of floating point numbers
"""
concatenateparameters(model::Model; includeunfit::Bool=false) = concatenateparameters(model.options, model.θreal, model.trialsets; includeunfit=includeunfit)

"""
    concatenateparameters(options, θreal, trialsets)

Concatenate values of parameters being fitted into a vector of floating point numbers

ARGUMENT
-`options`: fixed hyperparameters
-`θreal`: parameters of the latent variables in real space
-`trialsets`: data and parameters of the Poisson mixture generalized linear models

RETURN
-`concatenatedθ`: a vector of the concatenated values of the parameters being fitted
"""
function concatenateparameters(options::Options, θreal::Latentθ, trialsets::Vector{<:Trialset}; includeunfit::Bool=false)
	latentθvalues = includeunfit ? concatenateparameters(θreal) : concatenateparameters(θreal, options)
	glmθvalues = vcat((concatenateparameters(trialset; includeunfit=includeunfit) for trialset in trialsets)...)
	vcat(latentθvalues, glmθvalues)
end

"""
	copy(latentθ)

Copy a composite containing the parameters of the latent variables
"""

FHMDDM.copy(latentθ::Latentθ) = Latentθ(([getfield(latentθ, f)...] for f in fieldnames(Latentθ))...)

"""
	indexparameters(model)

Position of each parameters if they were concatenated into a vector

ARGUMENT
-`model`: a composite containing the data, parameters, and hyperparameters

RETURN
-a composite of the type 'Indexθ'
"""
indexparameters(model::Model; includeunfit::Bool=false) = indexparameters(model.options, model.trialsets; includeunfit=includeunfit)

"""
	indexparameters(options, trialsets)

Position of each parameters if they were concatenated into a vector

ARGUMENT
-`options`: settings of the model
-`trialsets`: data for the model

RETURN
-a composite of the type 'Indexθ'
"""
function indexparameters(options::Options, trialsets::Vector{<:Trialset}; includeunfit::Bool=false)
	if includeunfit
		latentθindices = Latentθ(([i] for i=1:length(fieldnames(Latentθ)))...)
	else
		latentθindices = indexparameters(options)
	end
	nlatentθ = maximum(getfield(latentθindices,name)[1] for name in fieldnames(Latentθ))
	glmθindices = indexparameters(trialsets; includeunfit=includeunfit, offset=nlatentθ)
	Indexθ(latentθ=latentθindices, glmθ=glmθindices)
end

"""
	indexparameters(options)

Return the position of each latent-variable parameter if they were concatenated into a vector

ARGUMENT
-`options`: settings of the model

RETURN
-a composite of the type `Latentθ`
"""
function indexparameters(options::Options)
	indexθ = Latentθ(collect(zeros(typeof(0),1) for i in fieldnames(Latentθ))...)
    counter = 0
	for field in fieldnames(Latentθ)
		if is_parameter_fit(options, field)
			counter += 1
			getfield(indexθ, field)[1] = counter
		else
			getfield(indexθ, field)[1] = 0
		end
	end
	return indexθ
end

"""
	is_parameter_fit(options, parametername)

Check whether a parameter is fit

ARGUMENT
-`options`: settings of the model
-`parametername`: name of the parameter as a symbol

RETURN
-a Bool
"""
function is_parameter_fit(options::Options, parametername::Symbol)
	options_field = Symbol("fit_"*String(parametername))
	if hasfield(typeof(options), options_field)
		tofit = getfield(options, options_field)
	else
		error("Unrecognized field: "*String(parametername))
	end
	return tofit
end

"""
	Latentθ(concatenatedθ, index, old)

Create a structure containing the parameters for the latent variable with updated values

ARGUMENT
-`concatenatedθ`: a vector of new parameter values
-`index`: index of each parameter in the vector of values
-`old`: structure with old parameter values

OUTPUT
-`new`: a structure with updated parameter values
"""
function Latentθ(concatenatedθ::Vector{T}, index::Latentθ, old::Latentθ) where {T<:Real}
	new = Latentθ((similar(getfield(old, field), T) for field in fieldnames(Latentθ))...)
	for field in fieldnames(Latentθ)
		if getfield(index, field)[1] == 0
			getfield(new, field)[1] = getfield(old, field)[1]
		else
			getfield(new, field)[1] = concatenatedθ[getfield(index, field)[1]]
		end
	end
	return new
end

"""
	Model(concatenatedθ, indexθ, model)

Create a new model using the concatenated parameter values

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values
-`model`: the model with old parameter values

RETURN
-`model`: the model with new parameter values
```
"""
function Model(concatenatedθ::Vector{type}, indexθ::Indexθ, model::Model) where {type<:Real}
	θreal = Latentθ(concatenatedθ, indexθ.latentθ, model.θreal)
	θnative = real2native(model.options, θreal)
	trialsets = map(model.trialsets, indexθ.glmθ) do trialset, glmθindex
					mpGLMs =map(trialset.mpGLMs, glmθindex) do mpGLM, glmθindex
								offset = concatenateparameters(glmθindex)[1] - 1
								MixturePoissonGLM(concatenatedθ, mpGLM; offset=offset)
							end
					Trialset(mpGLMs=mpGLMs, trials=trialset.trials)
				end
	Model(	options = model.options,
			gaussianprior = model.gaussianprior,
			θnative = θnative,
			θ₀native=model.θ₀native,
			θreal = θreal,
			trialsets=trialsets)
end

"""
	Model(concatenatedθ, indexθ, model)

Create a new model using the concatenated values of the parameters controlling the latent variable

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values of the parameters controlling the latnet variables
-`model`: the model with old parameter values

RETURN
-`model`: the model with new parameter values
```
"""
function Model(concatenatedθ::Vector{type}, indexθ::Latentθ, model::Model) where {type<:Real}
	θreal = Latentθ(concatenatedθ, indexθ, model.θreal)
	θnative = Latentθ((zeros(type,1) for field in fieldnames(Latentθ))...)
	real2native!(θnative, model.options, θreal)
	Model(	options = model.options,
			gaussianprior = model.gaussianprior,
			θnative = θnative,
			θ₀native=model.θ₀native,
			θreal = θreal,
			trialsets=model.trialsets)
end

"""
	nameparameters(model)

Names of the parameters of the model

ARGUMENT
-`model`: a composite containing the data, parameters, and hyperparameters of the model

OPTIONAL ARGUMENT
-`includeunfit`: whether

RETURN
-a vector of String
"""
nameparameters(model::Model) = nameparameters(indexparameters(model))

"""
	nameparameters(indexθ)

Name of the parameters of the model

ARGUMENT
-`indexθ`: a composite containing the indices of the parameters when they are concatenated into a vector

RETURN
-a vector of String
"""
function nameparameters(indexθ::Indexθ)
	ntrialsets = length(indexθ.glmθ)
	if ntrialsets > 1
		glmθnames = vcat((vcat((nameparameters(i,n,indexθ.glmθ[i][n]) for n in eachindex(indexθ.glmθ[i]))...) for i in eachindex(indexθ.glmθ))...)
	else
		glmθnames = vcat((nameparameters(n,indexθ.glmθ[1][n]) for n in eachindex(indexθ.glmθ[1]))...)
	end
	vcat(nameparameters(indexθ.latentθ), glmθnames)
end

"""
	nameparameters(indices)

Name of each parameter associated with a latent variable

ARGUMENT
-`indices`: a composite containing the indices of the parameters when they are concatenated into a vector

RETURN
-a vector of String
"""
function nameparameters(indices::Latentθ{<:Vector{<:Integer}})
	parameternames = String[]
	for name in fieldnames(Latentθ)
		if getfield(indices, name)[1] > 0
			parameternames = vcat(parameternames, matlabname(name))
		end
	end
	return parameternames
end

"""
	sortparameters!(latentθ, dict)

Update the parameters of the latent variables contained in the composite `latentθ` with values in the `Dict` `dict`
"""
function sortparameters!(latentθ::Latentθ, dict::Dict)
	for fieldname in fieldnames(Latentθ)
		getfield(latentθ, fieldname)[1] = dict[matlabname(fieldname)]
	end
end

"""
	sortparameters!(model, concatenatedθ, indexθ)

Sort a vector of concatenated parameter values and convert the values from real space to native space

MODIFIED ARGUMENT
-`model`: a factorial hidden Markov drift-diffusion model

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values
"""
function sortparameters!(model::Model, concatenatedθ::Vector{<:AbstractFloat}, indexθ::Indexθ)
	sortparameters!(model, concatenatedθ, indexθ.latentθ)
	@unpack options, θnative, θreal, trialsets = model
	for i in eachindex(indexθ.glmθ)
		for n in eachindex(indexθ.glmθ[i])
			@unpack θ = trialsets[i].mpGLMs[n]
			index = indexθ.glmθ[i][n]
			sortparameters!(trialsets[i].mpGLMs[n].θ, concatenatedθ, index)
		end
	end
	return nothing
end

"""
	sortparameters!(model, concatenatedθ, indexθ)

Sort the parameters of the latent variables

MODIFIED ARGUMENT
-`model`: a factorial hidden Markov drift-diffusion model

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated values of the latent paraeters
-`indexθ`: struct indexing of each latentparameter in the vector of concatenated values
"""
function sortparameters!(model::Model, concatenatedθ::Vector{<:Real}, indexθ::Latentθ)
	@unpack θreal = model
	for field in fieldnames(Latentθ)
		index = getfield(indexθ, field)[1]
		if index != 0
			getfield(θreal, field)[1] = concatenatedθ[index]
		end
	end
	return nothing
end

"""
	sortparameters!(model, filepath)

Update the parameters and hyperparameters in the composite `model` with values from a MAT file located at `filepath`
"""
function sortparameters!(model::Model, filepath::String)
	matfile = matopen(filepath)
	sortparameters!(model.θnative, 	read(matfile, "thetanative"))
	sortparameters!(model.θreal, 	read(matfile, "thetareal"))
	sortparameters!(model.θ₀native, read(matfile, "theta0native"))
	thetaglm = read(matfile, "thetaglm")
	for (trialset, thetaglm) in zip(model.trialsets, thetaglm)
		for (mpGLM, thetaglm) in zip(trialset.mpGLMs, thetaglm)
			sortparameters!(mpGLM.θ, thetaglm)
		end
	end
	model.gaussianprior.𝛂 .= vec(read(matfile, "penaltycoefficients"))
    precisionmatrix!(model.gaussianprior)
	return nothing
end
