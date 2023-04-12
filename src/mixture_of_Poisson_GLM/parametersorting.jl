"""
	concatenateparameters(glmθ)

Concatenate into a vector the parameters of a Poisson mixture generalized linear model

ARGUMENT
-`glmθ`: a composite containing the parameters of a Poisson mixture generalized linear model

OPTIONAL ARGUMENT
-`includeunfit`: whether parameters that are not fit are concatenated
-`initialization`: whether only the parameters that are to be initialized are concatenated

OUTPUT
-a vector of floats
"""
function concatenateparameters(glmθ::GLMθ; includeunfit::Bool=false, initialization::Bool=false)
	emptyvector = eltype(getfield(glmθ,glmθ.concatenationorder[1]))[]
	vcat((	if name==:𝛃
				if includeunfit || glmθ.fit_𝛃
					vcat(glmθ.𝛃...)
				else
					emptyvector
				end
			elseif name==:a
				if (includeunfit || glmθ.fit_overdispersion)
					glmθ.a
				else
					emptyvector
				end
			elseif name==:b
				if !initialization && (includeunfit || glmθ.fit_b)
					glmθ.b
				else
					emptyvector
				end
			else
				vcat(getfield(glmθ, name)...)
			end for name in glmθ.concatenationorder)...)
end

"""
	concatenateparameters(trialset)

Concatenate into a vector the parameters of all the Poisson mixture generalized linear models in a trialset

ARGUMENT
-`trialset`: a structure containing the Poisson mixture generalized linear models of that trialset

OUTPUT
-a vector of floats
"""
function concatenateparameters(trialset::Trialset; includeunfit::Bool=false, initialization::Bool=false)
	vcat((concatenateparameters(mpGLM.θ; includeunfit=includeunfit, initialization=initialization) for mpGLM in trialset.mpGLMs)...)
end

"""
	copy(glmθ)

Duplicate a composite containing the parameters of a Poisson mixture generalized linear model
"""
function FHMDDM.copy(glmθ::GLMθ)
	θnew = GLMθ(eltype(glmθ.𝐮), glmθ)
	sortparameters!(θnew, glmθ)
	return θnew
end

"""
	GLMθ(concatenatedθ, θ)

Create an instance of `GLMθ` by updating a pre-existing instance with new concatenated parameters

ARGUMENT
-`θ`: pre-existing instance of `GLMθ`
-`concatenatedθ`: values of the parameters being fitted, concatenated into a vector

OPTION ARGUMENT
-`offset`: the number of unrelated parameters in `concatenatedθ` preceding the relevant parameters
-`initialization`: whether to purposefully ignore the transformation parameteter `b` and the bound encoding `𝛃`
"""
function GLMθ(concatenatedθ::Vector{elementtype}, glmθ::GLMθ; offset::Integer, initialization::Bool=false) where {elementtype<:Real}
	θnew = GLMθ(elementtype, glmθ)
	sortparameters!(θnew, glmθ)
	sortparameters!(θnew, concatenatedθ; initialization=initialization, offset=offset)
	return θnew
end

"""
	GLMθ(elementtype, glmθ)

Create an uninitialized instance of `GLMθ` with the given element type.

This is for using ForwardDiff

ARGUMENT
-`glmθ`: an instance of GLMθ
-`elementtype`: type of the element in each field of GLMθ

RETURN
-an instance of GLMθ
"""
function GLMθ(elementtype, glmθ::GLMθ)
	values = map(fieldnames(GLMθ)) do fieldname
				if fieldname ∈ glmθ.concatenationorder
					x = getfield(glmθ, fieldname)
					if eltype(x) <: Real
						zeros(elementtype, length(x))
					else
						collect(zeros(elementtype, length(x)) for x in x)
					end
				else
					getfield(glmθ, fieldname)
				end
			end
	return GLMθ(values...)
end

"""
	indexparameters(glmθ)

Identify the parameters of Poisson mixture generalized linear model when they are concatenated in a vector

ARGUMENT
-`glmθ`: a composite containing the parameters of a Poisson mixture generalized linear model

OPTIONAL ARGUMENT
-`includeunfit`: whether parameters that are not fit are indexed
-`initialization`: whether only the parameters that are to be initialized are indexed
-`offset`: an integer added to all indices

OUTPUT
-a composite of the same type as of `glmθ` containing index of each parameter in a concatenated vector
"""
function indexparameters(glmθ::GLMθ; includeunfit::Bool=false, initialization::Bool=false, offset::Integer=0)
	indexθ = GLMθ(typeof(offset), glmθ)
	indexparameters!(indexθ; includeunfit=includeunfit, initialization=initialization, offset=offset)
	indexθ
end

"""
	indexparameters!(glmθ)

Positions of the parameters of a Poisson mixture generalized linear model when they are concatenated into a vector

MODIFIED ARGUMENT
-`indexθ`: a composite containing the index of each parameter of a Poisson mixture generalized linear model when the parameters are concatenated in a vector

OPTIONAL ARGUMENT
-`offset`: an integer added to all indices
"""
function indexparameters!(indexθ::GLMθ; includeunfit::Bool=false, initialization::Bool=false, offset::Integer=0)
	concatenatedθ = concatenateparameters(indexθ; includeunfit=includeunfit, initialization=initialization)
	indices = offset .+ collect(1:length(concatenatedθ))
	sortparameters!(indexθ, indices; includeunfit=includeunfit, initialization=initialization)
	return nothing
end

"""
	indexparameter(trialset)

Positions of the parameters of all Poisson mixture generalized linear models of a trialset when they are concatenated into a vector

ARGUMENT
-`trialset`: a composite containing all the Poisson mixture generalized linear models of a trialset

OPTIONAL ARGUMENT
-`offset`: an integer added to all indices

OUTPUT
-a vector whose each element is a composite containing index of each parameter
"""
function indexparameters(trialset::Trialset; includeunfit::Bool=false, initialization::Bool=false, offset::Integer=0)
	parametercounts = collect(length(concatenateparameters(trialset.mpGLMs[n].θ;includeunfit=includeunfit, initialization=initialization)) for n=1:length(trialset.mpGLMs)-1)
	cumulativeoffsets = cumsum(vcat(offset, parametercounts))
	collect(indexparameters(mpGLM.θ; includeunfit=includeunfit, initialization=initialization, offset=cumulativeoffset) for (mpGLM, cumulativeoffset) in zip(trialset.mpGLMs, cumulativeoffsets))
end

"""
	indexparameters(trialsets)

Positions of the parameters of all Poisson mixture generalized linear models in all trialsets

ARGUMENT
-`trialsets`: a vector of composite containing all the Poisson mixture generalized linear models of a trialset

OPTIONAL ARGUMENT
-`offset`: an integer added to all indices

OUTPUT
-a nested vector `indexθ` whose element `indexθ[i][n]` corresponds to the n-th Poisson mixture generalized linear model of the i-th trialset
"""
function indexparameters(trialsets::Vector{<:Trialset}; includeunfit::Bool=false, initialization::Bool=false, offset::Integer=0)
	indexθ = Vector{GLMθ}[]
	for trialset in trialsets
		indexθ = vcat(indexθ, [indexparameters(trialset; includeunfit=includeunfit, initialization=initialization, offset=offset)])
		concatenatedindices = concatenateparameters(indexθ[end][end]; includeunfit=includeunfit, initialization=initialization)
		offset = concatenatedindices[end]
	end
	indexθ
end

"""
	Indices𝐮(ngain, npostspike, npoststereoclick, npremovement, npostphotostimulus)

Indices of the encoding weights of the temporal basis vectors of each filter that is independent of the accumulator

ARGUMENT
-number of temporal basis vectors for the gain, and the post-spike, post-stereoclick, pre-movement, and post-phostimulus filters

OUTPUT
-a composite containing the indices of the weights of the temporal basis vectors of each filter
"""
function Indices𝐮(ngain::Integer, npostspike::Integer, npoststereoclick::Integer, npremovement::Integer, npostphotostimulus::Integer)
	indices = UnitRange{Int}[]
	k = 0
	for name in fieldnames(Indices𝐮)
		if name ==:gain
			indices = vcat(indices, [k .+ (1:ngain)])
			k += ngain
		elseif name == :postspike
			indices = vcat(indices, [k .+ (1:npostspike)])
			k += npostspike
		elseif name == :poststereoclick
			indices = vcat(indices, [k .+ (1:npoststereoclick)])
			k += npoststereoclick
		elseif name == :premovement
			indices = vcat(indices, [k .+ (1:npremovement)])
			k += npremovement
		elseif name == :postphotostimulus
			indices = vcat(indices, [k .+ (1:npostphotostimulus)])
			k += npostphotostimulus
		end
	end
	Indices𝐮(indices...)
end

"""
	MixturePoissonGLM(concatenatedθ, mpGLM)

Create a structure for a mixture of Poisson GLM with updated parameters

ARGUMENT
-`concatenatedθ`: a vector of new parameter values
-`mpGLM`: a structure containing information on the mixture of Poisson GLM for one neuron

OUTPUT
-a new structure for the mixture of Poisson GLM of a neuron with new parameter values
"""
function MixturePoissonGLM(concatenatedθ::Vector{<:Real}, mpGLM::MixturePoissonGLM; offset::Integer=0, initialization::Bool=false)
	values = map(fieldnames(MixturePoissonGLM)) do fieldname
				if fieldname == :θ
					GLMθ(concatenatedθ, mpGLM.θ; offset=offset, initialization=initialization)
				else
					getfield(mpGLM, fieldname)
				end
			end
	return MixturePoissonGLM(values...)
end

"""
	nameparameter(glmθ)

Name the parameters of a Poisson mixture GLM

ARGUMENT
-`glmθ`: a composite containing the parameters of

RETURN
-a vector of String
"""
function nameparameters(glmθ::GLMθ)
	parameternames = String[]
	for name in glmθ.concatenationorder
		if name == :𝐮
			for field in fieldnames(Indices𝐮)
				for q in eachindex(getfield(glmθ.indices𝐮, field))
					parameternames = vcat(parameternames, string(field)*string(q))
				end
			end
		elseif name == :𝐯
			for k in eachindex(glmθ.𝐯)
				for q in eachindex(glmθ.𝐯[k])
					parameternames = vcat(parameternames, "precommitment_encoding_"*string(k)*"_"*string(q))
				end
			end
		elseif (name == :𝛃) & glmθ.fit_𝛃
			for k in eachindex(glmθ.𝛃)
				for q in eachindex(glmθ.𝛃[k])
					parameternames = vcat(parameternames, "postcommitment_encoding_"*string(k)*"_"*string(q))
				end
			end
		elseif (name == :a) & glmθ.fit_overdispersion
			parameternames = vcat(parameternames, "overdispersion")
		elseif (name == :b) & glmθ.fit_b
			parameternames = vcat(parameternames, "transformation")
		end
	end
	return parameternames
end

"""
	nameparameter(trialsetindex, neuronindex, glmθ)

Name the parameters of the Poisson mixture GLM of a neuron

ARGUMENT
-`trialsetindex`: index of the trialset containing the neuron
-`neuronindex`: order of the neuron in the trialset
-`glmθ`: a composite containing the parameters of the neuron's Poisson mixture GLM

RETURN
-a vector of String
"""
function nameparameters(trialsetindex::Integer, neuronindex::Integer, glmθ::GLMθ)
	map(nameparameters(glmθ)) do name
		"trialset"*string(trialsetindex)*"_neuron"*string(neuronindex)*"_"*name
	end
end

"""
	nameparameter(neuronindex, glmθ)

Name the parameters of the Poisson mixture GLM of a neuron

ARGUMENT
-`neuronindex`: order of the neuron in the trialset
-`glmθ`: a composite containing the parameters of the neuron's Poisson mixture GLM

RETURN
-a vector of String
"""
function nameparameters(neuronindex::Integer, glmθ::GLMθ)
	map(nameparameters(glmθ)) do name
		"neuron"*string(neuronindex)*"_"*name
	end
end

"""
	sortparameters!(θ, concatenatedθ)

Sort the concatenated parameters from a GLM

MODIFIED ARGUMENT
-`θ`: a struct containing the parameters of the Poisson mixture of a neuron

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector concatenating the parameters of a GLM

OPTIONAL ARGUMENT
-`initialization`: whether only the parameters that are to be initialized are concatenated
-`offset`: an integer added to all indices
"""
function sortparameters!(θ::GLMθ, concatenatedθ::Vector{<:Real}; includeunfit::Bool=false, initialization::Bool=false, offset::Integer=0)
	for name in θ.concatenationorder
		if name == :𝛃
			if (includeunfit || θ.fit_𝛃)
				for k in eachindex(θ.𝛃)
					for q in eachindex(θ.𝛃[k])
						offset += 1
						θ.𝛃[k][q] = concatenatedθ[offset]
					end
				end
			end
		elseif name == :a
			if (includeunfit || θ.fit_overdispersion)
				offset += 1
				θ.a[1] = concatenatedθ[offset]
			end
		elseif name == :b
			if !initialization && (includeunfit || θ.fit_b)
				offset += 1
				θ.b[1] = concatenatedθ[offset]
			end
		else
			x = getfield(θ, name)
			if eltype(x) <: Real
				for q in eachindex(x)
					offset += 1
					x[q] = concatenatedθ[offset]
				end
 			else
				for k in eachindex(x)
					for q in eachindex(x[k])
						offset += 1
						x[k][q] = concatenatedθ[offset]
					end
				end
			end
		end
	end
	return nothing
end

"""
	sortparameters!(θ, concatenatedθ, indexθ)

Sort the concatenated parameters of a mixture of Poisson GLM

MODIFIED ARGUMENT
-`θ`: structure organizing the parameters of the GLM, updated with parameters from `concatenatedθ`

UNMODIFIED ARGUMENT
-`concatenatedθ`: a vector of parameter values
-`indexθ`: an instance of `GLMθ` indexing the parameters
"""
function sortparameters!(θ::GLMθ, concatenatedθ::Vector{<:Real}, indexθ::GLMθ; offset::Integer=0)
	for name in θ.concatenationorder
		values = getfield(θ, name)
		indices = getfield(indexθ, name)
		if eltype(values) <: Real
			for i in eachindex(values)
				if indices[i] > 0
					offsetindex = offset + indices[i]
					values[i] = concatenatedθ[offsetindex]
				end
			end
		else
			for j in eachindex(values)
				for i in eachindex(values[j])
					if indices[j][i] > 0
						offsetindex = offset + indices[j][i]
						values[j][i] = concatenatedθ[offsetindex]
					end
				end
			end
		end
	end
	return nothing
end

"""
	sortparameters!(θnew, θold)

Copy the parameters of a Poisson mixture generalized linear model

MODIFIED ARGUMENT
-`θnew`: destination of the parameters

UNMODIFIED ARGUMENT
-`θold`: source of the parameters
"""
function sortparameters!(θnew::GLMθ, θold::GLMθ)
	for name in θold.concatenationorder
		xold = getfield(θold, name)
		xnew = getfield(θnew, name)
		if eltype(xold) <: Real
			xnew .= xold
		else
			for k in eachindex(xold)
				xnew[k] .= xold[k]
			end
		end
	end
end

"""
	update!(glmθ, dict)

Update the parameters of a Poisson mixture generalized linear model

MODIFIED ARGUMENT
-`glmθ`: a composite containing the parameters of a Poisson mixture generalized linear model

UNMODIFIED ARGUMENT
-`dict`: a `Dict` containing the values used to update the parameters
"""
function sortparameters!(glmθ::GLMθ, dict::Dict)
	glmθ.a[1] = dict["a"]
	glmθ.b[1] = dict["b"]
	glmθ.𝐮 .= dict["u"]
	for k in eachindex(glmθ.𝐯)
		glmθ.𝐯[k] .= dict["v"][k]
	end
	for k in eachindex(glmθ.𝛃)
		glmθ.𝛃[k] .= dict["beta"][k]
	end
end
