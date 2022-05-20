"""
	sortparameters!(model, concatenatedθ, indexθ)

Sort a vector of concatenated parameter values and convert the values from real space to native space

MODIFIED ARGUMENT
-`model`: a factorial hidden Markov drift-diffusion model

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true);
julia> concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
julia> concatenatedθ = rand(length(concatenatedθ))
julia> FHMDDM.sortparameters!(model, concatenatedθ, indexθ)
```
"""
function sortparameters!(model::Model,
				 		 concatenatedθ::Vector{<:AbstractFloat},
				 		 indexθ::Indexθ)
	sortparameters!(model, concatenatedθ, indexθ.latentθ)
	@unpack options, θnative, θreal, trialsets = model
	for i in eachindex(indexθ.glmθ)
		for n in eachindex(indexθ.glmθ[i])
			@unpack θ = trialsets[i].mpGLMs[n]
			index = indexθ.glmθ[i][n]
			for q in eachindex(θ.𝐮)
				θ.𝐮[q] = concatenatedθ[index.𝐮[q]]
			end
			for k in eachindex(θ.𝐯)
				for q in eachindex(θ.𝐯[k])
					θ.𝐯[k][q] = concatenatedθ[index.𝐯[k][q]]
				end
			end
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
	sortparameters(latentθindex,∇ℓ)

Return the subset of first-order partial derivatives associated with parameters being fitted

ARGUMENT
-`latentθindex`: index of each latent parameter in the gradient
-`∇ℓ`: gradient of the log-likelihood

RETURN
-gradient of the log-likelihood with respect to only the parameters being fitted
"""
function sortparameters(latentθindex::Latentθ, ∇ℓ::Vector{<:Real})
	isfitted = trues(length(∇ℓ))
	latentparameternames = fieldnames(FHMDDM.Latentθ)
	isfitted[1:length(latentparameternames)] .= false
	for j = 1:length(latentparameternames)
		i = getfield(latentθindex, latentparameternames[j])[1]
		if i > 0
			isfitted[j] = true
		end
	end
	if any(isfitted .== false)
		return ∇ℓ[isfitted]
	else
		return ∇ℓ
	end
end

"""
	sortparameters(latentθindex,∇∇ℓ)

Return the subset of second-order partial derivatives associated with parameters being fitted

ARGUMENT
-`latentθindex`: index of each latent parameter in the hessian
-`∇∇ℓ`: hessian of the log-likelihood

RETURN
-hessian of the log-likelihood with respect to only the parameters being fitted
"""
function sortparameters(latentθindex::Latentθ, ∇∇ℓ::Matrix{<:Real})
	isfitted = trues(size(∇∇ℓ,1))
	latentparameternames = fieldnames(Latentθ)
	isfitted[1:length(latentparameternames)] .= false
	for j = 1:length(latentparameternames)
		i = getfield(latentθindex, latentparameternames[j])[1]
		if i > 0
			isfitted[j] = true
		end
	end
	if any(isfitted .== false)
		return ∇∇ℓ[isfitted, isfitted]
	else
		return ∇∇ℓ
	end
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
	concatenatedθ = zeros(0)
	indexθ = Latentθ(collect(zeros(Int64,1) for i in fieldnames(Latentθ))...)
    counter = 0
	tofit = true
	for field in fieldnames(Latentθ)
		if field == :Aᶜ₁₁ || field == :Aᶜ₂₂ || field == :πᶜ₁
			tofit = options.K == 2
		else
			options_field = Symbol("fit_"*String(field))
			if hasfield(typeof(options), options_field)
				tofit = getfield(options, options_field)
			else
				error("Unrecognized field: "*String(field))
			end
		end
		if tofit
			counter += 1
			getfield(indexθ, field)[1] = counter
			concatenatedθ = vcat(concatenatedθ, getfield(θreal, field)[1])
		else
			getfield(indexθ, field)[1] = 0
		end
	end
	return concatenatedθ, indexθ
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
    @unpack options, trialsets = model
	indexθ = map(model.trialsets) do trialset
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
		end
	end
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
function concatenateparameters(θ::GLMθ)
	concatenatedθ = zeros(eltype(θ.𝐮), countparameters(θ))
	counter = 0
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
	return concatenatedθ
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
	tofit = true
	for field in fieldnames(Latentθ)
		if field == :Aᶜ₁₁ || field == :Aᶜ₂₂ || field == :πᶜ₁
			tofit = false
		else
			options_field = Symbol("fit_"*String(field))
			if hasfield(typeof(options), options_field)
				tofit = getfield(options, options_field)
			else
				error("Unrecognized field: "*String(field))
			end
		end
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
	Model(concatenatedθ, indexθ, model)

Create a new model using the concatenated parameter values

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values
-`model`: the model with old parameter values

RETURN
-`model`: the model with new parameter values

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_14_test/data.mat"; randomize=true);
julia> concatenatedθ, indexθ = FHMDDM.concatenateparameters(model)
julia> concatenatedθ = rand(length(concatenatedθ))
julia> model = Model(concatenatedθ, indexθ, model)
```
"""
function Model(concatenatedθ::Vector{type},
	 		   indexθ::Indexθ,
			   model::Model) where {type<:Real}
	θreal = Latentθ(concatenatedθ, indexθ.latentθ, model.θreal)
	θnative = Latentθ((zeros(type,1) for field in fieldnames(Latentθ))...)
	real2native!(θnative, model.options, θreal)
	trialsets = map(model.trialsets, indexθ.glmθ) do trialset, glmθindex
					mpGLMs =map(trialset.mpGLMs, glmθindex) do mpGLM, glmθindex
								MixturePoissonGLM(concatenatedθ, mpGLM; offset=glmθindex.𝐮[1]-1)
							end
					Trialset(mpGLMs=mpGLMs, trials=trialset.trials)
				end
	Model(	options = model.options,
			θnative = θnative,
			θ₀native=model.θ₀native,
			θreal = θreal,
			trialsets=trialsets)
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
function Model(concatenatedθ::Vector{<:Real},
	 		   indexθ::Latentθ,
			   model::Model)
	θreal = Latentθ(concatenatedθ, indexθ, model.θreal)
	Model(	options = model.options,
			θnative = real2native(model.options, θreal),
			θ₀native=model.θ₀native,
			θreal = θreal,
			trialsets=model.trialsets)
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
function Latentθ(concatenatedθ::Vector{T},
				index::Latentθ,
				old::Latentθ) where {T<:Real}
	new = Latentθ((similar(getfield(old, field), T) for field in fieldnames(Latentθ))...)
	for field in fieldnames(Latentθ)
		if getfield(index, field)[1] == 0
			getfield(new, field)[1] = getfield(old, field)[1]
		else
			getfield(new, field)[1] = concatenatedθ[getfield(index, field)[1]]
		end
	end
	new
end

"""
	MixturePoissonGLM(concatenatedθ, glmθindex, mpGLM)

Create a structure for a mixture of Poisson GLM with updated parameters

ARGUMENT
-`concatenatedθ`: a vector of new parameter values
-`glmθindex`: index of each parameter in the vector of values
-`mpGLM`: a structure containing information on the mixture of Poisson GLM for one neuron

OUTPUT
-a new structure for the mixture of Poisson GLM of a neuron with new parameter values
"""
function MixturePoissonGLM(concatenatedθ::Vector{T},
						   mpGLM::MixturePoissonGLM;
						   offset=0) where {T<:Real}
	mpGLM = MixturePoissonGLM(Δt=mpGLM.Δt,
							d𝛏_dB=mpGLM.d𝛏_dB,
							max_spikehistory_lag=mpGLM.max_spikehistory_lag,
							Φ=mpGLM.Φ,
							θ=GLMθ(mpGLM.θ, T),
							𝐕=mpGLM.𝐕,
							𝐗=mpGLM.𝐗,
							𝐲=mpGLM.𝐲)
	sortparameters!(mpGLM.θ, concatenatedθ; offset=offset)
	return mpGLM
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
function sortparameters!(θall::Vector{<:Real},
						 index::GLMθ,
						 θglm::Vector{<:Real})
	counter = 0
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
	return nothing
end

"""
	sortparameters!(θ, concatenatedθglm; offset=0)

Sort the concatenated parameters from a GLM

MODIFIED ARGUMENT
-`θ`: a struct containing the parameters of the Poisson mixture of a neuron

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector concatenating the parameters of a GLM
"""
function sortparameters!(θ::GLMθ, concatenatedθ::Vector{<:Real}; offset=0)
	counter = offset
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
	return nothing
end

"""
	countparameters(θ)

Count the number of parameters in the Poisson mixture GLM of one neuron

ARGUMENT
-`θ`: a struct containing the parameters of a GLM

RETURN
-number of parameters in the GLM
"""
function countparameters(θ::GLMθ)
	counter = 0
	counter = length(θ.𝐮)
	for 𝐯 in θ.𝐯
		counter += length(𝐯)
	end
	return counter
end
