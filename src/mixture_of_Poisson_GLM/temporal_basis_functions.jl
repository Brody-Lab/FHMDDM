"""
	accumulatorbasis(options, 𝐓)

Temporal basis functions for the accumulator kernel

ARGUMENT
-`maxtimesteps`: maximum number of time steps across all trials in a trialset
-`options`: settings of the model

RETURN
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time bin in the kernel
"""
function accumulatorbasis(maxtimesteps::Integer, options::Options)
    temporal_basis_functions(options.tbf_accu_begins0,
                            options.Δt,
                            options.tbf_accu_ends0,
                            options.tbf_accu_hz,
                            options.tbf_period,
                            options.tbf_accu_scalefactor,
                            options.tbf_accu_stretch,
							maxtimesteps)
end

"""
	timebasis(options, 𝐓)

Temporal basis functions for the time kernel

ARGUMENT
-`maxtimesteps`: maximum number of time steps across all trials in a trialset
-`options`: settings of the model

RETURN
-`𝐔`: A matrix whose element 𝐔[t,i] indicates the value of the i-th temporal basis function in the t-th time bin in the trialset
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time bin in the kernel
"""
function timebasis(maxtimesteps::Integer, options::Options)
    temporal_basis_functions(options.tbf_time_begins0,
                            options.Δt,
                            options.tbf_time_ends0,
                            options.tbf_time_hz,
                            options.tbf_period,
                            options.tbf_time_scalefactor,
                            options.tbf_time_stretch,
							maxtimesteps)
end

"""
	premovementbasis(options)

Temporal basis functions for the premovement kernel

ARGUMENT
-`options`: settings of the model

RETURN
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time bin in the kernel
"""
function premovementbasis(options::Options)
    temporal_basis_functions(options.tbf_move_begins0,
                            options.Δt,
                            options.tbf_move_ends0,
                            options.tbf_move_hz,
                            options.tbf_period,
                            options.tbf_move_scalefactor,
                            options.tbf_move_stretch,
							ceil(Int, options.tbf_move_dur_s/options.Δt))
end

"""
	spikehistorybasis(options)

Values of temporal basis functions parametrizing a postspike filter

ARGUMENT
-`options`: settings of the model

OUTPUT
-`Φ`: a matrix whose element Φ[τ,i] corresponds to the value of i-th temporal basis function in the τ-th time bin after the spike
"""
function spikehistorybasis(options::Options)
	temporal_basis_functions(options.tbf_hist_begins0,
                            options.Δt,
                            options.tbf_hist_ends0,
                            options.tbf_hist_hz,
                            options.tbf_period,
                            options.tbf_hist_scalefactor,
                            options.tbf_hist_stretch,
							ceil(Int, options.tbf_hist_dur_s/options.Δt))
end

"""
	temporal_basis_functions(begins0, ends0, hz, period, scalefactor, stretch, 𝐓)

Value of each temporal basis at each time bin in a trialset

INPUT
-`begins0`: whether the basis begins at zero
-`Δt`: time bin, in seconds
-`ends0`: whether the basis end at zero
-`hz`: number of temporal basis functions per second
-`period`: width of each temporal basis function, in terms of the inter-center distance
-`scalefactor`: scaling
-`stretch`: nonlinear stretching of time
-`𝐓`: vector of the number of timesteps in each trial

RETURN
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time step in each trial
"""
function temporal_basis_functions(begins0::Bool, Δt::AbstractFloat, ends0::Bool, hz::Real, period::Real, scalefactor::Real, stretch::Real, Tmax::Integer)
	if isnan(hz) || (Tmax < 1)
		return fill(1.0, 0, 0)
	else
		D = max(1, ceil(Int, hz*(Tmax*Δt)))
	    if D == 1
			x = scalefactor/sqrt(Tmax)
	        Φ = fill(x,Tmax,1)
	    else
	        Φ = unitarybasis(begins0, ends0, D, Tmax, period, stretch).*scalefactor
	    end
	    return Φ
	end
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
	if (sum(indices)==0) || isnan(options.tbf_phot_hz)
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
	Φon = temporal_basis_functions(options.tbf_phot_begins0,
									options.Δt,
									options.tbf_phot_ends0,
									options.tbf_phot_hz,
									options.tbf_period,
									1.0,
									options.tbf_phot_stretch,
									maximum(nsteps_onset_to_trialend))
	latest_onset = maximum(𝐭ₒₙ)
	if latest_onset < 0
		Φtimesteps = 1-latest_onset:size(Φon,1)
		Φon = Φon[Φtimesteps, :]
	else
		Φtimesteps = 1:size(Φon,1)
	end
	Φon = unitarybasis(Φon)
	indexoff = findfirst(Φtimesteps.==(duration+1))
	if indexoff != nothing
		nsteps_offset = length(Φtimesteps) - indexoff + 1
		Φoff = temporal_basis_functions(options.tbf_phot_begins0,
	                            	   options.Δt,
			                           options.tbf_phot_ends0,
			                           options.tbf_phot_hz,
			                           options.tbf_period,
			                           options.tbf_phot_scalefactor,
			                           options.tbf_phot_stretch,
									   nsteps_offset)
		Φoff = vcat(zeros(indexoff-1, size(Φoff,2)), Φoff)
		Φoff = unitarybasis(Φoff)
 		Φ = hcat(Φon, Φoff)
		Φ = unitarybasis(Φ)
	else
		Φ = Φon
	end
	Φ .*= options.tbf_phot_scalefactor
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
	premovementbasis(options, movementtimes_s, Φ, 𝐓)

Temporal basis functions for the premovement kernel

ARGUMENT
-`options`: settings of the model
-`movementtimes_s`: time of movement relative to the stereoclick, in seconds
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time bin in the kernel
-`𝐓`: number of timesteps

RETURN
-`𝐔`: A matrix whose element 𝐔[t,i] indicates the value of the i-th temporal basis function in the t-th time bin in the trialset
"""
function premovementbasis(movementtimes_s::Vector{<:AbstractFloat}, options::Options, Φ::Matrix{<:AbstractFloat}, 𝐓::Vector{<:Integer})
	nbins, D = size(Φ)
	𝐔 = zeros(sum(𝐓), D)
	if D > 0
		movementbin = ceil.(Int, movementtimes_s./options.Δt) # movement times are always positive
		τ = 0
		for i=1:length(𝐓)
			T = 𝐓[i]
			if movementbin[i] < nbins
				j₀ = nbins - movementbin[i] + 1
				for (t,j) in zip(1:T, j₀:nbins)
					𝐔[τ+t,:] = Φ[j,:]
				end
			else
				t₀ = movementbin[i] - nbins + 1
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
	unitarybasis(begins0, ends0, D, nbins, period, stretch)

A matrix of values from orthogonal temporal basis functions that each has an L2 norm of one.

The raised cosines temporal basis functions are used as the starting point.

ARGUMENT
-`begins0`: whether the raised cosines begin at the trough or at the peak
-`ends0`: whether the raised cosines end at the trough or at the peak
-`D`: number of temporal basis functions
-`nbins`: number of time steps
-`period`: width of the cosines, in terms of inter-center distance
-`stretch`: degree to which later cosines are stretched

RETURN
-`Φ`: A unitary matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis function at the i-th timestep from beginning of the trial

EXAMPLE
```julia-repl
julia> using FHMDDM, LinearAlgebra
julia> Φ = FHMDDM.unitarybasis(true, true, true, 4, 121, 4, 0.1)
julia> maximum(abs.(Φ'*Φ - I))
8.881784197001252e-16
```
"""
function unitarybasis(begins0::Bool, ends0::Bool, D::Integer, nbins::Integer, period::Real, stretch::Real)
	if D == 1
		fill(1/√nbins, nbins, 1)
	else
		Φ = raisedcosines(begins0, ends0, D, nbins, period, stretch)
		F = svd(Φ)
		F.U[:,1:D]
	end
end

"""
	unitarybasis(X)

Unitary basis for the real vector space spanned by the columns of `X`

OPTIONAL ARGUMENT
-`min_relative_singular_value`: dimensions whose singular value, relative to the maximum singular value across dimensions, is less than `min_relative_singular_value` are omitted

RETURN
-A unitary matrix whose columns span the real vector space span by the columns of `X`
"""
function unitarybasis(X::Matrix{<:AbstractFloat}; min_relative_singular_value::AbstractFloat=0.0)
	factorization = svd(X)
	relative_singular_values = factorization.S./maximum(factorization.S)
	indices = relative_singular_values .> min_relative_singular_value
	return factorization.U[:,indices]
end

"""
    raisedcosines(begins0, ends0, D, nbins, period, stretch)

Values of raised cosine temporal basis functions (tbf's)

ARGUMENT
-`begins0`: whether the first temporal basis function begins at the trough or at the peak
-`ends0`: whether the last temporal basis function begins at the trough or at the peak
-`D`: number of bases
-`nbins`: number of bins in the time window tiled by the bases
-`period`: period of the raised cosine, in units of the inter-center distance
-`stretch`: an index of the stretching of the cosines

RETURN
-`Φ`: Matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis function at the i-th timestep from beginning of the trial
"""
function raisedcosines(begins0::Bool, ends0::Bool, D::Integer, nbins::Integer, period::Real, stretch::Real)
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
