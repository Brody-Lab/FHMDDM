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
	latentparameternames = fieldnames(Latentθ)
	isfitted[1:length(latentparameternames)] .= false
	for parametername in latentparameternames
		i = getfield(latentθindex, parametername)[1]
		if i > 0
			isfitted[i] = true
		end
	end
	if any(isfitted .= false)
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
	for parametername in latentparameternames
		i = getfield(latentθindex, parametername)[1]
		if i > 0
			isfitted[i] = true
		end
	end
	if any(isfitted .= false)
		return ∇∇ℓ[isfitted, isfitted]
	else
		return ∇∇ℓ
	end
end


"""
	sortnativeparameters(concatenatedθ, model)

Sort a vector of concatenated parameter values in their native space

ARGUMENT
-`concatenatedθ`: a vector of concatenated parameter values

RETURN
-`model`: the model with new parameter values
"""
function sort_native_parameters(concatenatedθ::Vector{type}, model::Model) where {type<:Real}
	θnative = Latentθ((similar(getfield(model.θnative, field), type) for field in fieldnames(Latentθ))...)
	counter = 0
	for field in fieldnames(Latentθ)
		counter+=1
		getfield(θnative, field)[1] = concatenatedθ[counter]
	end
	𝐮 = map(model.trialsets) do trialset
			map(trialset.mpGLMs) do mpGLM
				zeros(type, length(mpGLM.θ.𝐮))
			end
		end
	𝐯 = map(model.trialsets) do trialset
			map(trialset.mpGLMs) do mpGLM
				zeros(type, length(mpGLM.θ.𝐯))
			end
		end
	for s in eachindex(model.trialsets)
		for n in eachindex(model.trialsets[s].mpGLMs)
			for q in eachindex(𝐮[s][n])
				counter +=1
				𝐮[s][n][q] = concatenatedθ[counter]
			end
			for q in eachindex(𝐯[s][n])
				counter +=1
				𝐯[s][n][q] = concatenatedθ[counter]
			end
		end
	end
	trialsets = map(model.trialsets, 𝐮, 𝐯) do trialset, 𝐮, 𝐯
					mpGLMs =map(trialset.mpGLMs, 𝐮, 𝐯) do mpGLM, 𝐮, 𝐯
								MixturePoissonGLM(Δt=mpGLM.Δt, K=mpGLM.K, 𝐔=mpGLM.𝐔, 𝚽=mpGLM.𝚽, Φ=mpGLM.Φ, θ=GLMθ(𝐮=𝐮, 𝐯=𝐯), 𝐗=mpGLM.𝐗, 𝛏=mpGLM.𝛏, 𝐲=mpGLM.𝐲, 𝐲! =mpGLM.𝐲!)
							end
					Trialset(mpGLMs=mpGLMs, trials=trialset.trials)
				end
	Model(	options = model.options,
			θnative = θnative,
			θ₀native= model.θ₀native,
			θreal = native2real(model.options, θnative),
			trialsets=trialsets)
end

"""
	concatenatenativeparameters(model)

Concatenated parameters in their native space

ARGUMENT
-`model`: the model with new parameter values

RETURN
-`concatenatedθ`: a vector of concatenated parameter values
"""
function concatenate_native_parameters(model::Model)
	concatenatedθ = zeros(typeof(model.θnative.B[1]), 0)
	for field in fieldnames(Latentθ)
		concatenatedθ = vcat(concatenatedθ, getfield(model.θnative, field))
	end
	for s in eachindex(model.trialsets)
		for n in eachindex(model.trialsets[s].mpGLMs)
			for field = (:𝐮, :𝐯)
				concatenatedθ = vcat(concatenatedθ, getfield(model.trialsets[s].mpGLMs[n].θ, field))
			end
		end
	end
	return concatenatedθ
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
