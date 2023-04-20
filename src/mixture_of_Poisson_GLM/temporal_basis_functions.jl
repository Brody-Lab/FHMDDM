"""
	temporal_basis_functions(filtername, options)

Temporal basis functions used to fit a filter

ARGUMENT
-`filtername`: name of the filter
-`options`: settings of the model

RETURN
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time bin in the kernel
"""
function temporal_basis_functions(filtername::String, options::Options)
	temporal_basis_functions(getfield(options, Symbol("tbf_"*filtername*"_begins0")),
                            options.Δt,
							getfield(options, Symbol("tbf_"*filtername*"_dur_s")),
                            getfield(options, Symbol("tbf_"*filtername*"_ends0")),
                            getfield(options, Symbol("tbf_"*filtername*"_hz")),
							options.sf_mpGLM[1],
                            getfield(options, Symbol("tbf_"*filtername*"_stretch")),
							orthogonal_to_ones = true)
end

"""
	temporal_basis_functions(begins0, Δt, duration_s, ends0, hz, scalefactor, stretch)

Value of each temporal basis at each time bin in a trialset

INPUT
-`begins0`: whether the basis begins at zero
-`Δt`: time bin, in seconds
-`duration_s`: duration in seconds
-`ends0`: whether the basis end at zero
-`hz`: number of temporal basis functions per second
-`scalefactor`: scaling
-`stretch`: nonlinear stretching of time

RETURN
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time step in each trial
"""
function temporal_basis_functions(begins0::Bool, Δt::AbstractFloat, duration_s::Real, ends0::Bool, hz::Real, scalefactor::Real, stretch::Real; orthogonal_to_ones::Bool=false)
	nfunctions = ceil(hz*duration_s)
	if isnan(nfunctions) || (nfunctions < 1)
		return fill(1.0, 0, 0)
	else
		nfunctions = convert(Int, nfunctions)
		ntimesteps = ceil(Int, duration_s/Δt)
		temporal_basis_functions(begins0, ends0, nfunctions, ntimesteps, scalefactor, stretch; orthogonal_to_ones=orthogonal_to_ones)
	end
end

"""
	temporal_basis_functions(begins0, ends0, nfunctions, ntimesteps, scalefactor, stretch)

Value of each temporal basis at each time bin in a trialset

INPUT
-`begins0`: whether the basis begins at zero
-`Δt`: time bin, in seconds
-`ends0`: whether the basis end at zero
-`nfunctions`: number of temporal basis functions
-`ntimesteps`: number of time steps
-`scalefactor`: scaling
-`stretch`: nonlinear stretching of time

RETURN
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time step in each trial
"""
function temporal_basis_functions(begins0::Bool, ends0::Bool, nfunctions::Integer, ntimesteps::Integer, scalefactor::Real, stretch::Real; orthogonal_to_ones::Bool=false)
	Φ = raisedcosines(begins0, ends0, nfunctions, ntimesteps, stretch)
	if orthogonal_to_ones
		Φ = orthogonalize_to_ones(Φ)
	end
	Φ = orthonormalbasis(Φ)
	Φ .*= scalefactor
	return Φ
end

"""
	temporal_basis_functions(Φ, 𝐓)

Value of each temporal basis at each time bin in a trialset

INPUT
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time step in each trial
-`𝐓`: vector of the number of timesteps in each trial

RETURN
-`𝐕`: A matrix whose element 𝐕[t,i] indicates the value of the i-th temporal basis function in the t-th time bin in the trialset
"""
function temporal_basis_functions(Φ::Matrix{<:AbstractFloat}, 𝐓::Vector{<:Integer})
    Tmax = maximum(𝐓)
    D = size(Φ,2)
    𝐕 = zeros(sum(𝐓), D)
	if D > 0
	    k = 0
	    for T in 𝐓
	        for t = 1:T
	            k = k + 1;
	            𝐕[k,:] = Φ[t,:]
	        end
	    end
	end
	return 𝐕
end

"""
	photostimulusbasis(options, 𝐭_onset_s, 𝐭_offset_s, 𝐓)

Temporal basis vectors for learning the photostimulus filter and their values in each time step

ARGUMENT
-`options`: settings of the model
-`𝐭_onset_s`: time of photostimulus onset in each trial
-`𝐭_offset_s`: time of photostimulus offset in each trial
-`𝐓`: number of time steps in each trial

RETURN
-`Φ`: a matrix of floats whose columns correspond to the temporal basis vectors and whose rows correspond to time steps relative to photostimulus onset.
-`Φtimesteps`: a unit range of integers representing the time steps of `Φ` relative to photostimulus onset. Each value of `Φtimesteps` corresponds to a row of `Φ`. A value of `Φtimesteps[i]=1` indicates that the i-th row of `Φ` corresponds to the time step when the photostimulus occured.
-`𝐔`: a matrix of floats whose columns correspond to the temporal basis vectors and whose rows correspond to time steps in a trialset
"""
function photostimulusbasis(options::Options, 𝐭_onset_s::Vector{<:AbstractFloat}, 𝐭_offset_s::Vector{<:AbstractFloat}, 𝐓::Vector{<:Integer})
	indices = map(𝐭_onset_s, 𝐭_offset_s) do t_on, t_off
					!isnan(t_on) && !isnan(t_off)
			  end
	if (sum(indices)==0) || isnan(options.tbf_postphotostimulus_hz)
		Φ = zeros(0, 0)
		Φtimesteps = 1:0
		𝐔 = zeros(sum(𝐓), size(Φ,2))
	else
		duration = round.((𝐭_offset_s[indices] .- 𝐭_onset_s[indices])./options.Δt)
		duration = unique(duration[.!isnan.(duration)])
		@assert length(duration)==1
		@assert duration[1] > 0
		duration = duration[1]
		duration = convert(Int, duration)
		𝐭ₒₙ = 𝐭_onset_s[indices]./options.Δt
		𝐭ₒₙ = collect(tₒₙ < 0.0 ? floor(Int, tₒₙ) : ceil(Int, tₒₙ) for tₒₙ in 𝐭ₒₙ)
		Φ, Φtimesteps = photostimulusbasis(duration, options, 𝐓[indices], 𝐭ₒₙ)
		𝐔 = zeros(sum(𝐓), size(Φ,2))
		photostimulusbasis!(𝐔, indices, Φ, Φtimesteps, 𝐓, 𝐭ₒₙ)
	end
	return Φ, Φtimesteps, 𝐔
end

"""
	photostimulusbasis(duration, options, 𝐓, 𝐭ₒₙ)

Temporal basis vectors for learning the photostimulus filter

ARGUMENT
-`duration`: number of time steps in the photostimulus
-`options`: settings of the model
-`𝐓`: number of time steps in each trial, for only the trials with a photostimulus
-`𝐭ₒₙ`: the time step in each trial when the photostimulus began, for only the trials with a photostimulus

RETURN
-`Φ`: a matrix of floats whose columns correspond to the temporal basis vectors and whose rows correspond to time steps relative to photostimulus onset.
-`Φtimesteps`: a unit range of integers representing the time steps of `Φ` relative to photostimulus onset. Each value of `Φtimesteps` corresponds to a row of `Φ`. A value of `Φtimesteps[i]=1` indicates that the i-th row of `Φ` corresponds to the time step when the photostimulus occured.
"""
function photostimulusbasis(duration::Integer, options::Options, 𝐓::Vector{<:Integer}, 𝐭ₒₙ::Vector{<:Integer})
	nsteps_onset_to_trialend = map((T, tₒₙ)-> tₒₙ < 0 ? T-tₒₙ : T-tₒₙ+1, 𝐓, 𝐭ₒₙ)
	ntimesteps = maximum(nsteps_onset_to_trialend)
	nfunctions = ceil(Int, options.tbf_postphotostimulus_hz*duration*options.Δt)
	scalefactor = options.sf_mpGLM[1]
	Φon = temporal_basis_functions(options.tbf_postphotostimulus_begins0,
									options.tbf_postphotostimulus_ends0,
									nfunctions,
									ntimesteps,
									1.0,
									options.tbf_postphotostimulus_stretch;
									orthogonal_to_ones=true)
	latest_onset = maximum(𝐭ₒₙ)
	if latest_onset < 0
		Φtimesteps = 1-latest_onset:size(Φon,1)
		Φon = Φon[Φtimesteps, :]
	else
		Φtimesteps = 1:size(Φon,1)
	end
	Φon = orthonormalbasis(Φon)
	indexoff = findfirst(Φtimesteps.==(duration+1))
	if indexoff != nothing
		nsteps_offset = length(Φtimesteps) - indexoff + 1
		nfunctions = ceil(Int, options.tbf_postphotostimulus_hz*nsteps_offset*options.Δt)
		Φoff = temporal_basis_functions(options.tbf_postphotostimulus_begins0,
			                           options.tbf_postphotostimulus_ends0,
									   nfunctions,
			                           nsteps_offset,
			                           scalefactor,
			                           options.tbf_postphotostimulus_stretch;
   									   orthogonal_to_ones=true)
		Φoff = vcat(zeros(indexoff-1, size(Φoff,2)), Φoff)
		if !isempty(Φoff)
			Φoff = orthonormalbasis(Φoff)
		end
 		Φ = hcat(Φon, Φoff)
		Φ = orthonormalbasis(Φ)
	else
		Φ = Φon
	end
	Φ .*= scalefactor
	return Φ, Φtimesteps
end

"""
	photostimulusbasis!(𝐔, Φ, Φtimesteps, 𝐓, 𝐭ₒₙ)

Evaluate each temporal basis vector at each time step in a trialset

MODIFIED ARGUMENT
-`𝐔`: a matrix of floats whose columns correspond to the temporal basis vectors and whose rows correspond to time steps in a trialset

ARGUMENT
-`indices`: a bit vector indicating which trial in the trialset has a photostimulus
-`Φ`: a matrix of floats whose columns correspond to the temporal basis vectors and whose rows correspond to time steps relative to photostimulus onset.
-`Φtimesteps`: a unit range of integers representing the time steps of `Φ` relative to photostimulus onset. Each value of `Φtimesteps` corresponds to a row of `Φ`. A value of `Φtimesteps[i]=1` indicates that the i-th row of `Φ` corresponds to the time step when the photostimulus occured.
-`𝐓`: a vector of integers representing the number of time steps in each trial in the trialset
-`𝐭ₒₙ`: a vector of integers representing the time step when the photostimulus began, for trials with a photostimulus.
"""
function photostimulusbasis!(𝐔::Matrix{<:AbstractFloat}, indices::Vector{Bool}, Φ::Matrix{<:AbstractFloat}, Φtimesteps::UnitRange{<:Integer}, 𝐓::Vector{<:Integer}, 𝐭ₒₙ::Vector{<:Integer})
	D = size(Φ,2)
	τ = 0
	k = 0
	for m in eachindex(𝐓)
		T = 𝐓[m]
		if indices[m]
			k += 1
			tₒₙ = 𝐭ₒₙ[k]
			if tₒₙ < 0
				i = 1 - Φtimesteps[1] - tₒₙ
				for t in 1:T
					τ += 1
					i += 1
					for d = 1:D
						𝐔[τ,d] = Φ[i,d]
					end
				end
			else
				i = 1 - tₒₙ
				for t in 1:T
					τ += 1
					i += 1
					if i > 0
						for d = 1:D
							𝐔[τ,d] = Φ[i,d]
						end
					end
				end
			end
		else
			τ += T
		end
	end
	@assert τ == size(𝐔,1)
	return nothing
end

"""
	premovementbasis(movementtimesteps, Φ, 𝐓)

Temporal basis functions for the premovement kernel

ARGUMENT
-`movementtimes_s`: time of movement relative to the stereoclick, in seconds
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time bin in the kernel
-`𝐓`: number of timesteps

RETURN
-`𝐔`: A matrix whose element 𝐔[t,i] indicates the value of the i-th temporal basis function in the t-th time bin in the trialset
"""
function premovementbasis(movementtimesteps::Vector{<:Integer}, Φ::Matrix{<:AbstractFloat}, 𝐓::Vector{<:Integer})
	nbins, D = size(Φ)
	𝐔 = zeros(sum(𝐓), D)
	if D > 0
		τ = 0
		for i=1:length(𝐓)
			T = 𝐓[i]
			if movementtimesteps[i] < nbins
				j₀ = nbins - movementtimesteps[i] + 1
				for (t,j) in zip(1:T, j₀:nbins)
					𝐔[τ+t,:] = Φ[j,:]
				end
			else
				t₀ = movementtimesteps[i] - nbins + 1
				for (t,j) in zip(t₀:T, 1:nbins)
					𝐔[τ+t,:] = Φ[j,:]
				end
			end
			τ += T
		end
	end
	return 𝐔
end

"""
	spikehistorybasis(Φ, 𝐓, 𝐲)

Response of each temporal basis function parametrizing the postspike filter at each time step in the trialset

ARGUMENT
-`Φ`: Value of each temporal basis function parametrizing the postspike filter at each time step in the filter
-`𝐓`: number of time step in each trial for all trials in the trialset
-`𝐲`: number of spikes of one neuron in each time step, concatenated across trials

RETURN
-`𝐔`: a matrix whose element 𝐔[t,i] corresponds to the response of the i-th temporal basis function at the t-th time step in the trialset.
"""
function spikehistorybasis(Φ::Matrix{<:AbstractFloat}, 𝐓::Vector{<:Integer}, 𝐲::Vector{<:Integer})
	filterlength, D = size(Φ)
	𝐔 = zeros(sum(𝐓), D)
	if D > 0
		τ = 0
		for T in 𝐓
			indices = τ .+ (1:T)
			spiketimesteps = findall(𝐲[indices] .> 0)
			for tₛₚₖ in spiketimesteps
				y = 𝐲[tₛₚₖ+τ]
				indices𝐔 = τ .+ (tₛₚₖ+1:T)
				for (i,j) in zip(indices𝐔, 1:filterlength)
					for p = 1:D
						𝐔[i,p] += y*Φ[j,p]
					end
				end
			end
			τ = τ + T;
		end
	end
	return 𝐔
end

"""
	poststereoclickbasis(Φ, 𝐓)

Value of each temporal basis vector of the post-stereoclick filter at each time bin in a trialset

INPUT
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time step in each trial
-`𝐓`: vector of the number of timesteps in each trial

RETURN
-`𝐔`: A matrix whose element 𝐔[t,i] indicates the value of the i-th temporal basis function in the t-th time bin in the trialset
"""
function poststereoclickbasis(Φ::Matrix{<:AbstractFloat}, 𝐓::Vector{<:Integer})
    ntimesteps, nfunctions = size(Φ)
    𝐔 = zeros(sum(𝐓), nfunctions)
    τ = 0
    for T in 𝐓
        for t = 1:min(T,ntimesteps)
			for i = 1:nfunctions
	            𝐔[τ+t,i] = Φ[t,i]
			end
        end
		τ+=T
    end
	return 𝐔
end

"""
	orthogonalize_to_ones(Φ)

Orthogonalize the columns of a matrix to a vector of ones

RETURN
-A matrix whose columns are orthogonal to any vector whose elements have the same value
"""
function orthogonalize_to_ones(Φ::Matrix{<:AbstractFloat})
	nrows = size(Φ,1)
	(I - fill(1/nrows,nrows,nrows))*Φ
end

"""
	orthonormalbasis(X)

Orthonormal basis for the real vector space spanned by the columns of `X`

OPTIONAL ARGUMENT
-`min_relative_singular_value`: dimensions whose singular value, relative to the maximum singular value across dimensions, is less than `min_relative_singular_value` are omitted

RETURN
-A unitary matrix whose columns span the same real vector space spanned by the columns of `X`
"""
function orthonormalbasis(Φ::Matrix{<:AbstractFloat}; min_relative_singular_value::AbstractFloat=1e-2)
	factorization = svd(Φ)
	relative_singular_values = factorization.S./maximum(factorization.S)
	indices = relative_singular_values .> min_relative_singular_value
	return factorization.U[:,indices]
end

"""
    raisedcosines(begins0, ends0, D, nbins, stretch)

Values of raised cosine temporal basis functions (tbf's)

ARGUMENT
-`begins0`: whether the first temporal basis function begins at the trough or at the peak
-`ends0`: whether the last temporal basis function begins at the trough or at the peak
-`D`: number of bases
-`nbins`: number of bins in the time window tiled by the bases
-`stretch`: an index of the stretching of the cosines

OPTIONAL ARGUMENT
-`period`: period of the raised cosine, in units of the inter-center distance

RETURN
-`Φ`: Matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis function at the i-th timestep from beginning of the trial
"""
function raisedcosines(begins0::Bool, ends0::Bool, D::Integer, nbins::Integer, stretch::Real; period::Real=4)
    if isnan(stretch) || stretch < eps()
        a = 1
        b = nbins
        t = collect(1:nbins)
    else
        λ = 1/stretch
        a = log(1+λ)
        b = log(nbins+λ)
        t = log.(collect(1:nbins) .+ λ)
    end
    if begins0
        if ends0
            Δcenter = (b-a) / (D+3)
        else
            Δcenter = (b-a) / (D+1)
        end
        centers = a .+ 2Δcenter .+ collect(0:max(1,D-1)).*Δcenter
    else
        if ends0
            Δcenter = (b-a) / (D+1)
        else
            Δcenter = (b-a) / (D-1)
        end
        centers = a .+ collect(0:max(1,D-1)).*Δcenter
    end
    ω = 2π/Δcenter/period
    Φ = raisedcosines(centers, ω, t)
    if !begins0 && !ends0 # allow temporal basis functions to parametrize a constant function
        lefttail = raisedcosines([centers[1]-Δcenter], ω, t)
        righttail = raisedcosines([centers[end]+Δcenter], ω, t)
        Φ[:,1] += lefttail
        Φ[:,end] += righttail
        indices = t .<= centers[1] + period/2*Δcenter
        deviations = 2.0 .- sum(Φ,dims=2) # introduced by time compression
        Φ[indices,1] .+= deviations[indices]
    end
    return Φ
end

"""
    raisedcosines(centers, ω, t)

Values of raised cosine temporal basis functions

ARGUMENT
-`centers`: Vector of the centers of the raised cosines
-`ω`: angular frequency
-`t`: values at which the temporal basis functions are evaluated

RETURN
-`Φ`: Matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis function at the i-th timestep from beginning of the trial
"""
function raisedcosines(centers::Vector{<:AbstractFloat}, ω::AbstractFloat, t::Vector{<:AbstractFloat})
    T = t .- centers'
    (cos.(max.(-π, min.(π, ω.*T))) .+ 1)/2
end
