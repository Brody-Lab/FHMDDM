"""
	sortparameters!(model, concatenatedθ, indexθ)

Sort a vector of concatenated parameter values and convert the values from real space to native space

MODIFIED ARGUMENT
-`model`: a factorial hidden Markov drift-diffusion model

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values
"""
function sortparameters!(model::Model,
				 		 concatenatedθ::Vector{<:AbstractFloat},
				 		 indexθ::Indexθ)
	@unpack options, θnative, θreal, trialsets = model
	for field in fieldnames(Latentθ) # `Latentθ` is the type of `indexθ.latentθ`
		index = getfield(indexθ.latentθ, field)[1]
		if index != 0 # an index of 0 indicates that the parameter is not being fit
			getfield(θreal, field)[1] = concatenatedθ[index]
		end
	end
	real2native!(θnative, options, θreal)
	for i in eachindex(indexθ.glmθ)
		for n in eachindex(indexθ.glmθ[i])
			for field in fieldnames(GLMθ)
				indices = getfield(indexθ.glmθ[i][n], field)
				for j in eachindex(indices)
					if indices[j] != 0
						getfield(trialsets[i].mpGLMs[n].θ, field)[j] = concatenatedθ[indices[j]]
					end
				end
			end
		end
	end
	return nothing
end

"""
	sortparameters(concatenatedθ, indexθ, model)

Sort a vector of concatenated parameter values

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values
-`indexθ`: struct indexing of each parameter in the vector of concatenated values
-`model`: the model with old parameter values

RETURN
-`model`: the model with new parameter values
"""
function sortparameters(concatenatedθ::Vector{<:Real},
				 		indexθ::Indexθ,
						model::Model)
	θreal = Latentθ(concatenatedθ, indexθ.latentθ, model.θreal)
	trialsets = map(model.trialsets, indexθ.glmθ) do trialset, glmθindex
					mpGLMs =map(trialset.mpGLMs, glmθindex) do mpGLM, glmθindex
								MixturePoissonGLM(concatenatedθ, glmθindex, mpGLM)
							end
					Trialset(mpGLMs=mpGLMs, trials=trialset.trials)
				end
	Model(	options = model.options,
			θnative = real2native(model.options, θreal),
			θ₀native=model.θ₀native,
			θreal = θreal,
			trialsets=trialsets)
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
						   glmθindex::GLMθ,
						   mpGLM::MixturePoissonGLM) where {T<:Real}
	glmθ = GLMθ((similar(getfield(mpGLM.θ, field), T) for field in fieldnames(GLMθ))...) # instantiate a GLMθ whose fields are uninitialized arrays with element type T
	for field in fieldnames(GLMθ)
		oldparameters = getfield(mpGLM.θ, field)
		newparameters = getfield(glmθ, field)
		indices = getfield(glmθindex, field)
		for i in eachindex(oldparameters)
			if indices[i] == 0
				newparameters[i] = oldparameters[i]
			else
				newparameters[i] = concatenatedθ[indices[i]]
			end
		end
	end
	MixturePoissonGLM(	Δt=mpGLM.Δt,
						K=mpGLM.K,
						𝚽=mpGLM.𝚽,
						Φ=mpGLM.Φ,
						𝐔=mpGLM.𝐔,
						𝐗=mpGLM.𝐗,
						𝛏=mpGLM.𝛏,
						𝐲=mpGLM.𝐲,
						θ=glmθ)
end

"""
    concatenateparameters(model)

Concatenate values of parameters being fitted into a vector of floating point numbers

ARGUMENT
-`model`: the factorial hidden Markov drift-diffusion model

RETURN
-`concatenatedθ`: a vector of the concatenated values of the parameters being fitted
-`indexθ`: a structure indicating the index of each model parameter in the vector of concatenated values
"""
function concatenateparameters(model::Model)
    @unpack options, θreal, trialsets = model
	concatenatedθ = zeros(0)
    counter = 0
	latentθ = Latentθ(collect(zeros(Int64,1) for i in fieldnames(Latentθ))...)
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
			getfield(latentθ, field)[1] = counter
			concatenatedθ = vcat(concatenatedθ, getfield(θreal, field)[1])
		else
			getfield(latentθ, field)[1] = 0
		end
	end
	glmθ = 	map(model.trialsets) do trialset
				map(trialset.mpGLMs) do mpGLM
					GLMθ((zeros(Int64, length(getfield(mpGLM.θ, field))) for field in fieldnames(GLMθ))...)
				end
			end
    for i in eachindex(trialsets)
        for n in eachindex(trialsets[i].mpGLMs)
			@unpack θ = trialsets[i].mpGLMs[n]
			for field in fieldnames(GLMθ)
				if (field == :a && !options.fit_a) ||
				   (field == :b && !options.fit_b)
					getfield(glmθ[i][n], field) .= 0
				else
					parameters = getfield(θ, field)
					p = length(parameters)
					concatenatedθ = vcat(concatenatedθ, parameters)
					getfield(glmθ[i][n], field) .= counter+1:counter+p
					counter += p
				end
			end
        end
    end
    indexθ = Indexθ(latentθ=latentθ, glmθ = glmθ)
    return concatenatedθ, indexθ
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
	latentθ = Latentθ(collect(zeros(Int64,1) for i in fieldnames(Latentθ))...)
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
			getfield(latentθ, field)[1] = counter
			concatenatedθ = vcat(concatenatedθ, getfield(θreal, field)[1])
		else
			getfield(latentθ, field)[1] = 0
		end
	end
	glmθ = 	map(model.trialsets) do trialset
				map(trialset.mpGLMs) do mpGLM
					GLMθ((zeros(Int64, length(getfield(mpGLM.θ, field))) for field in fieldnames(GLMθ))...)
				end
			end
    indexθ = Indexθ(latentθ=latentθ, glmθ=glmθ)
    return concatenatedθ, indexθ
end
