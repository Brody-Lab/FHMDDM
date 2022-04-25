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
			@unpack θ = trialsets[i].mpGLMs[n]
			index = indexθ.glmθ[i][n]
			for q in eachindex(θ.𝐡)
				θ.𝐡[q] = concatenatedθ[index.𝐡[q]]
			end
			for k in eachindex(θ.𝐮)
				for q in eachindex(θ.𝐮[k])
					θ.𝐮[k][q] = concatenatedθ[index.𝐮[k][q]]
				end
			end
			for k in eachindex(θ.𝐯)
				for q in eachindex(θ.𝐯[k])
					θ.𝐯[k][q] = concatenatedθ[index.𝐯[k][q]]
				end
			end
			for q in eachindex(θ.𝐰)
				θ.𝐰[q] = concatenatedθ[index.𝐰[q]]
			end
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
					GLMθ(𝐡 = zeros(Int, length(mpGLM.θ.𝐡)),
						 𝐮 = collect(zeros(Int, length(𝐮)) for 𝐮 in mpGLM.θ.𝐮),
						 𝐯 = collect(zeros(Int, length(𝐯)) for 𝐯 in mpGLM.θ.𝐯),
 						 𝐰 = zeros(Int, length(mpGLM.θ.𝐰)))
				end
			end
	nglmparameters = 0
	for trialset in trialsets
		for mpGLM in trialset.mpGLMs
			nglmparameters += length(mpGLM.θ.𝐡)
			for 𝐮 in mpGLM.θ.𝐮
				nglmparameters += length(𝐮)
			end
			for 𝐯 in mpGLM.θ.𝐯
				nglmparameters += length(𝐯)
			end
			nglmparameters += length(mpGLM.θ.𝐰)
		end
	end
	concatenatedθ = zeros(nglmparameters)
	counter = 0
	for i in eachindex(trialsets)
        for n in eachindex(trialsets[i].mpGLMs)
			@unpack θ = trialsets[i].mpGLMs[n]
			for q in eachindex(θ.𝐡)
				counter += 1
				concatenatedθ[counter] = θ.𝐡[q]
				indexθ[i][n].𝐡[q] = offset + counter
			end
			for k in eachindex(θ.𝐮)
				for q in eachindex(θ.𝐮[k])
					counter += 1
					concatenatedθ[counter] = θ.𝐮[k][q]
					indexθ[i][n].𝐮[k][q] = offset + counter
				end
			end
			for k in eachindex(θ.𝐯)
				for q in eachindex(θ.𝐯[k])
					counter += 1
					concatenatedθ[counter] = θ.𝐯[k][q]
					indexθ[i][n].𝐯[k][q] = offset + counter
				end
			end
			for q in eachindex(θ.𝐰)
				counter += 1
				concatenatedθ[counter] = θ.𝐰[q]
				indexθ[i][n].𝐰[q] = offset + counter
			end
		end
	end
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
function Model(concatenatedθ::Vector{<:Real},
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
	mpGLMnew = MixturePoissonGLM(	Δt=mpGLM.Δt,
									Φ=mpGLM.Φ,
									𝐇=mpGLM.𝐇,
									𝐔=mpGLM.𝐔,
									𝐕=mpGLM.𝐕,
									d𝛏_dB=mpGLM.d𝛏_dB,
									𝐲=mpGLM.𝐲,
									θ=GLMθ(mpGLM.θ, T))
	for q in eachindex(mpGLMnew.θ.𝐡)
		mpGLMnew.θ.𝐡[q] = concatenatedθ[glmθindex.𝐡[q]]
	end
	for k in eachindex(mpGLMnew.θ.𝐮)
		for q in eachindex(mpGLMnew.θ.𝐮[k])
			mpGLMnew.θ.𝐮[k][q] = concatenatedθ[glmθindex.𝐮[k][q]]
		end
	end
	for k in eachindex(mpGLMnew.θ.𝐯)
		for q in eachindex(mpGLMnew.θ.𝐯[k])
			mpGLMnew.θ.𝐯[k][q] = concatenatedθ[glmθindex.𝐯[k][q]]
		end
	end
	for q in eachindex(mpGLMnew.θ.𝐰)
		mpGLMnew.θ.𝐰[q] = concatenatedθ[glmθindex.𝐰[q]]
	end
	return mpGLMnew
end
