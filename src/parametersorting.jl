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
function sortparameters(concatenatedθ,
				 		indexθ::Indexθ,
						model::Model)
	T = eltype(concatenatedθ)
	θreal = Latentθ((zeros(T,1) for field in fieldnames(Latentθ))...)
	for field in fieldnames(Latentθ) # `Latentθ` is the type of `indexθ.latentθ`
		index = getfield(indexθ.latentθ, field)[1]
		if index != 0 # an index of 0 indicates that the parameter is not being fit
			getfield(θreal, field)[1] = concatenatedθ[index]
		end
	end
	trialsets = map(model.trialsets, indexθ.glmθ) do trialset, glmθ
					mpGLMs =map(trialset.mpGLMs, glmθ) do mpGLM, glmθ
						if all(glmθ.𝐮 .== 0)
							mpGLM
						else
							θ = GLMθ(𝐮=concatenatedθ[glmθ.𝐮],
									𝐯=concatenatedθ[glmθ.𝐯],
									a=concatenatedθ[glmθ.a],
									b=concatenatedθ[glmθ.b])
							MixturePoissonGLM(	Δt=mpGLM.Δt,
												K=mpGLM.K,
												𝚽=mpGLM.𝚽,
												Φ=mpGLM.Φ,
												𝐔=mpGLM.𝐔,
												𝐗=mpGLM.𝐗,
												𝛏=mpGLM.𝛏,
												𝐲=mpGLM.𝐲,
												θ=θ)
						end
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
					GLMθ(𝐮 = zeros(Int64, length(mpGLM.θ.𝐮)),
						 𝐯 = zeros(Int64, length(mpGLM.θ.𝐯)),
						 a = zeros(Int64, length(mpGLM.θ.a)),
						 b = zeros(Int64, length(mpGLM.θ.b)))
				end
			end
    for i in eachindex(trialsets)
        for n in eachindex(trialsets[i].mpGLMs)
			@unpack θ = trialsets[i].mpGLMs[n]
			for field in fieldnames(GLMθ)
				parameters = getfield(θ, field)
				concatenatedθ = vcat(concatenatedθ, parameters)
	            p = length(parameters)
				getfield(glmθ[i][n], field) .= counter+1:counter+p
	            counter += p
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
					GLMθ(𝐮 = zeros(Int64, length(mpGLM.θ.𝐮)),
						 𝐯 = zeros(Int64, length(mpGLM.θ.𝐯)),
						 a = zeros(Int64, length(mpGLM.θ.a)),
						 b = zeros(Int64, length(mpGLM.θ.b)))
				end
			end
    indexθ = Indexθ(latentθ=latentθ, glmθ=glmθ)
    return concatenatedθ, indexθ
end
