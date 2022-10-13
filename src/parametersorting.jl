"""
	concatenate(indexθ)

Concatenate the index of each parameter into a vector
"""
function concatenate(indexθ::Indexθ; includeunfit=includeunfit)
	latentθnames = fieldnames(FHMDDM.Latentθ)
	nθ = length(latentθnames)
	nglmθ = countparameters(indexθ.glmθ[1][1]; includeunfit=includeunfit)
	for glmθindex in indexθ.glmθ
		for glmθindex in glmθindex
			nθ += nglmθ
		end
	end
	indices = zeros(Int, nθ)
	counter = 0
	latentθnames = fieldnames(FHMDDM.Latentθ)
	for latentθname in latentθnames
		indices[counter+=1] = getfield(indexθ.latentθ, latentθname)[1]
	end
	for glmθindex in indexθ.glmθ
		for glmθindex in glmθindex
			indices[counter+1:counter+nglmθ] = concatenateparameters(glmθindex; includeunfit=includeunfit)
			counter+=nglmθ
		end
	end
	return indices
end

"""
    concatenate_glm_parameters(model, startingcounter)

Concatenate values of parameters of all glms into a vector of floating point numbers

ARGUMENT
-`model`: the factorial hidden Markov drift-diffusion model
-`offset`: number of latent parameters being fitted

RETURN
-`concatenatedθ`: a vector of the concatenated values of the parameters from all glms
-`indexθ`: a structure indicating the index of each model parameter in the vector of concatenated values
"""
function concatenate_glm_parameters(model::Model, offset::Integer)
	concatenate_glm_parameters(offset, model.trialsets)
end

"""
	concatenate_glm_parameters(offset, trialsets)

Concatenate values of parameters of all glms into a vector of floating point numbers

ARGUMENT
-`offset`: number of latent parameters being fitted
-`trialsets`: data in the model

RETURN
-`concatenatedθ`: a vector of the concatenated values of the parameters from all glms
-`indexθ`: a structure indicating the index of each model parameter in the vector of concatenated values
"""
function concatenate_glm_parameters(offset::Integer, trialsets::Vector{<:Trialset})
	indexθ = map(trialsets) do trialset
				map(trialset.mpGLMs) do mpGLM
					GLMθ(mpGLM.θ, Int)
				end
			end
	nglmparameters = 0
	for trialset in trialsets
		for mpGLM in trialset.mpGLMs
			nglmparameters += countparameters(mpGLM.θ)
		end
	end
	concatenatedθ = zeros(nglmparameters)
	counter = 0
	for i in eachindex(trialsets)
        for n in eachindex(trialsets[i].mpGLMs)
			@unpack θ = trialsets[i].mpGLMs[n]
			if θ.fit_b
				counter += 1
				concatenatedθ[counter] = θ.b[1]
				indexθ[i][n].b[1] = offset + counter
			end
			indexθ[i][n].𝐠[1] = 0
			for k = 2:length(θ.𝐠)
				counter += 1
				concatenatedθ[counter] = θ.𝐠[k]
				indexθ[i][n].𝐠[k] = offset + counter
			end
			for q in eachindex(θ.𝐮)
				counter += 1
				concatenatedθ[counter] = θ.𝐮[q]
				indexθ[i][n].𝐮[q] = offset + counter
			end
			for k in eachindex(θ.𝐯)
				for q in eachindex(θ.𝐯[k])
					counter += 1
					concatenatedθ[counter] = θ.𝐯[k][q]
					indexθ[i][n].𝐯[k][q] = offset + counter
				end
			end
			if θ.fit_𝛃
				for k in eachindex(θ.𝛃)
					for q in eachindex(θ.𝛃[k])
						counter += 1
						concatenatedθ[counter] = θ.𝛃[k][q]
						indexθ[i][n].𝛃[k][q] = offset + counter
					end
				end
			end
		end
	end
    return concatenatedθ, indexθ
end

"""
	    concatenate_latent_parameters(model)

Concatenate values of latent parameters being fitted into a vector of floating point numbers

ARGUMENT
-`model`: the factorial hidden Markov drift-diffusion model

RETURN
-`concatenatedθ`: a vector of the concatenated values of the latent parameters being fitted
-`indexθ`: a structure indicating the index of each latent parameter in the vector of concatenated values
"""
function concatenate_latent_parameters(model::Model)
    @unpack options, θreal = model
	indexθ = index_latent_parameters(options)
	concatenatedθ = zeros(0)
	for field in fieldnames(Latentθ)
		if getfield(indexθ, field)[1] > 0
			concatenatedθ = vcat(concatenatedθ, getfield(θreal, field)[1])
		end
	end
	return concatenatedθ, indexθ
end

"""
    concatenateparameters(model)

Concatenate values of parameters being fitted into a vector of floating point numbers

ARGUMENT
-`model`: the factorial hidden Markov drift-diffusion model

RETURN
-`concatenatedθ`: a vector of the concatenated values of the parameters being fitted
-`indexθ`: a structure indicating the index of each model parameter in the vector of concatenated values

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true);
julia> concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
```
"""
function concatenateparameters(model::Model)
	concatenated_latentθ, index_latentθ = concatenate_latent_parameters(model)
	concatenated_glmθ, index_glmθ = concatenate_glm_parameters(model, length(concatenated_latentθ))
	concatenatedθ = vcat(concatenated_latentθ, concatenated_glmθ)
    indexθ = Indexθ(latentθ=index_latentθ, glmθ = index_glmθ)
    return concatenatedθ, indexθ
end

"""
	concatenateparameters(θ)

Concatenate the parameters of one neuron's Poisson mixture GLM

ARGUMENT
-`θ`: parameters organized in an instance of `GLMθ`

RETURN
-`concatenatedθ`: a vector concatenating the values of the parameters
-`indexθ`: an instance of `GLMθ` indexing each parameter in the vector of concatenated values
"""
function concatenateparameters(θ::GLMθ; includeunfit::Bool=false, initialization::Bool=false)
	concatenatedθ = zeros(eltype(θ.𝐮), countparameters(θ;includeunfit=includeunfit, initialization=initialization))
	counter = 0
	if includeunfit || (θ.fit_b && !initialization)
		counter += 1
		concatenatedθ[counter] = θ.b[1]
	end
	for k = 2:length(θ.𝐠)
		counter += 1
		concatenatedθ[counter] = θ.𝐠[k]
	end
	for q in eachindex(θ.𝐮)
		counter += 1
		concatenatedθ[counter] = θ.𝐮[q]
	end
	for k in eachindex(θ.𝐯)
		for q in eachindex(θ.𝐯[k])
			counter += 1
			concatenatedθ[counter] = θ.𝐯[k][q]
		end
	end
	if includeunfit || (θ.fit_𝛃 && !initialization)
		for k in eachindex(θ.𝛃)
			for q in eachindex(θ.𝛃[k])
				counter += 1
				concatenatedθ[counter] = θ.𝛃[k][q]
			end
		end
	end
	return concatenatedθ
end

"""
	index_latent_parameters(options)

Create a structure indexing the parameters of the latent variables

ARGUMENT
-`options`: settings of the model

RETURN
-an instance of `Latenθ`
"""
function index_latent_parameters(options::Options)
	indexθ = Latentθ(collect(zeros(Int64,1) for i in fieldnames(Latentθ))...)
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
	indexparameters(options, trialsets)

Index of each parameter if all parameters were concatenated into a vector

ARGUMENT
-`options`: settings of the model
-`trialsets`: data for the model

RETURN
-a structure indicating the index of each model parameter in the vector of concatenated values
"""
function indexparameters(options::Options, trialsets::Vector{<:Trialset})
	indexθlatent = index_latent_parameters(options)
	indexθglm = concatenate_glm_parameters(count_latent_parameters_being_fitted(options), trialsets)[2]
    return Indexθ(latentθ=indexθlatent, glmθ=indexθglm)
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
	if parametername == :Aᶜ₁₁ || parametername == :Aᶜ₂₂ || parametername == :πᶜ₁
		tofit = tofit && (options.K == 2)
	end
	return tofit
end

"""
	count_latent_parameters_being_fitted(options)

Count the number of parameters of latent variables being fitted

ARGUMENT
-`options`: settings of the model

RETURN
-a positive integer
"""
function count_latent_parameters_being_fitted(options::Options)
	counter = 0
	for field in fieldnames(Latentθ)
		counter += is_parameter_fit(options, field)
	end
	counter
end

"""
    concatenate_choice_related_parameters(model)

Concatenate values of parameters being fitted into a vector of floating point numbers

ARGUMENT
-`model`: the factorial hidden Markov drift-diffusion model

RETURN
-`concatenatedθ`: a vector of the concatenated values of the parameters being fitted
-`indexθ`: a structure indicating the index of each model parameter in the vector of concatenated values

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_27_test/data.mat"; randomize=true)
julia> concatenatedθ, indexθ = FHMDDM.concatenate_choice_related_parameters(model)
```
"""
function concatenate_choice_related_parameters(model::Model)
    @unpack options, θreal, trialsets = model
	concatenatedθ = zeros(0)
    counter = 0
	indexθ = Latentθ(collect(zeros(Int64,1) for i in fieldnames(Latentθ))...)
	for field in fieldnames(Latentθ)
		tofit = is_parameter_fit(options, field) && !any(field .== (:Aᶜ₁₁, :Aᶜ₂₂, :πᶜ₁))
		if tofit
			counter += 1
			getfield(indexθ, field)[1] = counter
			concatenatedθ = vcat(concatenatedθ, getfield(θreal, field)[1])
		else
			getfield(indexθ, field)[1] = 0
		end
	end
	index_glmθ = concatenate_glm_parameters(model, length(concatenatedθ))[2]
    return concatenatedθ, Indexθ(latentθ=indexθ, glmθ = index_glmθ)
end

"""
	countparameters(θ)

Count the number of parameters in the Poisson mixture GLM of one neuron

ARGUMENT
-`θ`: a struct containing the parameters of a GLMs

OPTIONAL ARGUMENT
-`intialization`: whether only the parameters included in the initialization are counted (thereby excluding `b` and `𝛃`)
-`includeunfit`: whether parameters that are not to be fit are to be included

RETURN
-number of parameters in the GLM
"""
function countparameters(θ::GLMθ; initialization::Bool=false, includeunfit::Bool=false)
	counter = (includeunfit || (θ.fit_b && !initialization)) ? 1 : 0
	counter += length(θ.𝐠)-1
	counter += length(θ.𝐮)
	for 𝐯ₖ in θ.𝐯
		counter += length(𝐯ₖ)
	end
	if includeunfit || (θ.fit_𝛃 && !initialization)
		for 𝛃ₖ in θ.𝛃
			counter += length(𝛃ₖ)
		end
	end
	return counter
end

"""
	firstindex(glmθindex)

Index of the first parameter for a mixture of Poisson GLM

ARGUMENT
-`glmθindex`: a structure indexing the parameters of a mixture of Poisson GLM

RETURN
-a positive integer
"""
function firstindex(glmθindex::GLMθ)
	@unpack b, 𝐠, 𝐮, fit_b = glmθindex
	if fit_b
		b[1]
	else
		if length(𝐠) > 1
			𝐠[2]
		else
			𝐮[1]
		end
	end
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
	θnative = Latentθ((zeros(type,1) for field in fieldnames(Latentθ))...)
	real2native!(θnative, model.options, θreal)
	trialsets = map(model.trialsets, indexθ.glmθ) do trialset, glmθindex
					mpGLMs =map(trialset.mpGLMs, glmθindex) do mpGLM, glmθindex
								MixturePoissonGLM(concatenatedθ, mpGLM; offset=firstindex(glmθindex)-1)
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
	sortparameters!(θ, concatenatedθglm; offset=0)

Sort the concatenated parameters from a GLM

MODIFIED ARGUMENT
-`θ`: a struct containing the parameters of the Poisson mixture of a neuron

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector concatenating the parameters of a GLM
"""
function sortparameters!(θ::GLMθ, concatenatedθ::Vector{<:Real}; offset::Integer=0, initialization::Bool=false)
	counter = offset
	if θ.fit_b && !initialization
		counter+=1
		θ.b[1] = concatenatedθ[counter]
	end
	for k = 2:length(θ.𝐠)
		counter+=1
		θ.𝐠[k] = concatenatedθ[counter]
	end
	for q in eachindex(θ.𝐮)
		counter+=1
		θ.𝐮[q] = concatenatedθ[counter]
	end
	for k in eachindex(θ.𝐯)
		for q in eachindex(θ.𝐯[k])
			counter+=1
			θ.𝐯[k][q] = concatenatedθ[counter]
		end
	end
	if θ.fit_𝛃 && !initialization
		for k in eachindex(θ.𝛃)
			for q in eachindex(θ.𝛃[k])
				counter+=1
				θ.𝛃[k][q] = concatenatedθ[counter]
			end
		end
	end
	return nothing
end

"""
	sortparameters!(θ, concatenatedθ, index)

Sort the concatenated parameters of a mixture of Poisson GLM

MODIFIED ARGUMENT
-`θ`: structure organizing the parameters of the GLM, updated with parameters from `concatenatedθ`

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of parameter values
-`index`: an instance of `GLMθ` indexing the parameters
"""
function sortparameters!(θ::GLMθ, concatenatedθ::Vector{<:Real}, index::GLMθ)
	if θ.fit_b
		θ.b[1] = concatenatedθ[index.b[1]]
	end
	for k = 2:length(θ.𝐠)
		θ.𝐠[k] = concatenatedθ[index.𝐠[k]]
	end
	for q in eachindex(θ.𝐮)
		θ.𝐮[q] = concatenatedθ[index.𝐮[q]]
	end
	for k in eachindex(θ.𝐯)
		for q in eachindex(θ.𝐯[k])
			θ.𝐯[k][q] = concatenatedθ[index.𝐯[k][q]]
		end
	end
	if θ.fit_𝛃
		for k in eachindex(θ.𝛃)
			for q in eachindex(θ.𝛃[k])
				θ.𝛃[k][q] = concatenatedθ[index.𝛃[k][q]]
			end
		end
	end
	return nothing
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
	for field in fieldnames(Latentθ) # `Latentθ` is the type of `indexθ.latentθ`
		index = getfield(indexθ, field)[1]
		if index != 0 # an index of 0 indicates that the parameter is not being fit
			getfield(θreal, field)[1] = concatenatedθ[index]
		end
	end
	return nothing
end

"""
	sortparameters!(∇all, index, ∇glm)

Sort the concatenated parameters from a GLM and use them update the values of a vector concatenating all parameters of the model

MODIFIED ARGUMENT
-`θall`: a vector concatenating all parameters of the model

UNMODIFIED ARGUMENT
-`index`: a struct indicating the index of each parameter of GLM in the vector concatenating all parameters of the model
-`θglm`: a vector concatenating the parameters from a GLM
"""
function sortparameters!(θall::Vector{<:Real}, index::GLMθ, θglm::Vector{<:Real})
	counter = 0
	if θ.fit_b
		counter+=1
		θall[index.b[1]] = θglm[counter]
	end
	for k = 2:length(θ.𝐠)
		counter+=1
		θall[index.𝐠[k]] = θglm[counter]
	end
	for q in eachindex(index.𝐮)
		counter+=1
		θall[index.𝐮[q]] = θglm[counter]
	end
	for k in eachindex(θ.𝐯)
		for q in eachindex(index.𝐯[k])
			counter+=1
			θall[index.𝐯[k][q]] = θglm[counter]
		end
	end
	if fit_𝛃
		for k in eachindex(θ.𝛃)
			for q in eachindex(index.𝛃[k])
				counter+=1
				θall[index.𝛃[k][q]] = θglm[counter]
			end
		end
	end
	return nothing
end
