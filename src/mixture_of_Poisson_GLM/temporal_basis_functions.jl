"""
	accumulatorbases(options, 𝐓)

Temporal basis functions for the accumulator kernel

ARGUMENT
-`options`: settings of the model
-`𝐓`: number of timesteps

RETURN
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time bin in the kernel
"""
function accumulatorbases(options::Options, 𝐓::Vector{<:Integer})
    temporal_bases_values(  options.tbf_accu_begins0,
                            options.Δt,
                            options.tbf_accu_ends0,
                            options.tbf_accu_hz,
                            options.tbf_accu_period,
                            options.tbf_accu_scalefactor,
                            options.tbf_accu_stretch,
                            𝐓)
end

"""
	timebases(options, 𝐓)

Temporal basis functions for the time kernel

ARGUMENT
-`options`: settings of the model
-`𝐓`: number of timesteps

RETURN
-`𝐔`: A matrix whose element 𝐔[t,i] indicates the value of the i-th temporal basis function in the t-th time bin in the trialset
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time bin in the kernel
"""
function timebases(options::Options, 𝐓::Vector{<:Integer})
    temporal_bases_values(  options.tbf_time_begins0,
                            options.Δt,
                            options.tbf_time_ends0,
                            options.tbf_time_hz,
                            options.tbf_time_period,
                            options.tbf_time_scalefactor,
                            options.tbf_time_stretch,
                            𝐓)
end

"""
	temporal_bases_values(begins0, ends0, hz, period, scalefactor, stretch, 𝐓)

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
function temporal_bases_values(begins0::Bool, Δt::AbstractFloat, ends0::Bool, hz::Real, period::Real, scalefactor::Real, stretch::Real, 𝐓::Vector{<:Integer})
	if isnan(hz)
		return fill(1.0, 0, 0)
	else
		Tmax = maximum(𝐓)
		nbases = max(1, ceil(Int, hz*(Tmax*Δt)))
	    if nbases == 1
			x = scalefactor/sqrt(Tmax)
	        Φ = fill(x,Tmax,1)
	    else
	        Φ = unitarybases(begins0, ends0, nbases, Tmax, period, stretch).*scalefactor
	    end
	    return Φ
	end
end

"""
	temporal_bases_values(Φ, 𝐓)

Value of each temporal basis at each time bin in a trialset

INPUT
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time step in each trial
-`𝐓`: vector of the number of timesteps in each trial

RETURN
-`𝐕`: A matrix whose element 𝐕[t,i] indicates the value of the i-th temporal basis function in the t-th time bin in the trialset
"""
function temporal_bases_values(Φ::Matrix{<:AbstractFloat}, 𝐓::Vector{<:Integer})
    Tmax = maximum(𝐓)
    nbases = size(Φ,2)
    𝐕 = zeros(sum(𝐓), nbases)
	if nbases > 0
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
	premovementbases(options)

Temporal basis functions for the premovement kernel

ARGUMENT
-`options`: settings of the model

RETURN
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time bin in the kernel
"""
function premovementbases(options::Options)
	if isnan(options.tbf_move_hz)
		return fill(1.0, 0, 0)
	else
		nbases = max(1, ceil(Int, options.tbf_move_dur_s*options.tbf_move_hz))
		nbins = ceil(Int, options.tbf_move_dur_s/options.Δt)
		Φ = unitarybases(options.tbf_move_begins0, options.tbf_move_ends0, nbases, nbins, options.tbf_move_period, options.tbf_move_stretch)
		Φ .*= options.tbf_move_scalefactor
		return Φ
	end
end

"""
	premovementbases(options, movementtimes_s, Φ, 𝐓)

Temporal basis functions for the premovement kernel

ARGUMENT
-`options`: settings of the model
-`movementtimes_s`: time of movement relative to the stereoclick, in seconds
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time bin in the kernel
-`𝐓`: number of timesteps

RETURN
-`𝐔`: A matrix whose element 𝐔[t,i] indicates the value of the i-th temporal basis function in the t-th time bin in the trialset
"""
function premovementbases(movementtimes_s::Vector{<:AbstractFloat}, options::Options, Φ::Matrix{<:AbstractFloat}, 𝐓::Vector{<:Integer})
	nbins, nbases = size(Φ)
	𝐔 = zeros(sum(𝐓), nbases)
	if nbases > 0
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
	spikehistorybases(options)

Values of temporal basis functions parametrizing a postspike filter

ARGUMENT
-`options`: settings of the model

OUTPUT
-`Φ`: a matrix whose element Φ[τ,i] corresponds to the value of i-th temporal basis function in the τ-th time bin after the spike
"""
function spikehistorybases(options::Options)
	if isnan(options.tbf_hist_hz)
		return fill(1.0, 0, 0)
	else
		nbases = max(1, ceil(Int, options.tbf_hist_dur_s*options.tbf_hist_hz))
		nbins = ceil(Int, options.tbf_hist_dur_s/options.Δt)
		Φ = unitarybases(options.tbf_hist_begins0, options.tbf_hist_ends0, nbases, nbins, options.tbf_hist_period, options.tbf_hist_stretch)
		Φ .*= options.tbf_hist_scalefactor
		return Φ
	end
end

"""
	spikehistorybases(Φ, 𝐓, 𝐲)

Response of each temporal basis function parametrizing the postspike filter at each time step in the trialset

ARGUMENT
-`Φ`: Value of each temporal basis function parametrizing the postspike filter at each time step in the filter
-`𝐓`: number of time step in each trial for all trials in the trialset
-`𝐲`: number of spikes of one neuron in each time step, concatenated across trials

RETURN
-`𝐔`: a matrix whose element 𝐔[t,i] corresponds to the response of the i-th temporal basis function at the t-th time step in the trialset.
"""
function spikehistorybases(Φ::Matrix{<:AbstractFloat}, 𝐓::Vector{<:Integer}, 𝐲::Vector{<:Integer})
	filterlength, nbases = size(Φ)
	𝐔 = zeros(sum(𝐓), nbases)
	if nbases > 0
		τ = 0
		for T in 𝐓
			indices = τ .+ (1:T)
			spiketimesteps = findall(𝐲[indices] .> 0)
			for tₛₚₖ in spiketimesteps
				y = 𝐲[tₛₚₖ+τ]
				indices𝐔 = τ .+ (tₛₚₖ+1:T)
				for (i,j) in zip(indices𝐔, 1:filterlength)
					for p = 1:nbases
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
	unitarybases(begins0, ends0, nbases, nbins, period, stretch)

A matrix of values from orthogonal temporal basis functions that each has an L2 norm of one.

The raised cosines temporal basis functions are used as the starting point.

ARGUMENT
-`begins0`: whether the raised cosines begin at the trough or at the peak
-`ends0`: whether the raised cosines end at the trough or at the peak
-`nbases`: number of temporal basis functions
-`nbins`: number of time steps
-`period`: width of the cosines, in terms of inter-center distance
-`stretch`: degree to which later cosines are stretched

RETURN
-`Φ`: A unitary matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis function at the i-th timestep from beginning of the trial

EXAMPLE
```julia-repl
julia> using FHMDDM, LinearAlgebra
julia> Φ = FHMDDM.unitarybases(true, true, true, 4, 121, 4, 0.1)
julia> maximum(abs.(Φ'*Φ - I))
8.881784197001252e-16
```
"""
function unitarybases(begins0::Bool, ends0::Bool, nbases::Integer, nbins::Integer, period::Real, stretch::Real)
	if nbases == 1
		fill(1/√nbins, nbins, 1)
	else
		Φ = raisedcosines(begins0, ends0, nbases, nbins, period, stretch)
		F = svd(Φ)
		F.U[:,1:nbases]
	end
end

"""
    raisedcosines(begins0, ends0, nbases, nbins, period, stretch)

Values of raised cosine temporal basis functions (tbf's)

ARGUMENT
-`begins0`: whether the first temporal basis function begins at the trough or at the peak
-`ends0`: whether the last temporal basis function begins at the trough or at the peak
-`nbases`: number of bases
-`nbins`: number of bins in the time window tiled by the bases
-`period`: period of the raised cosine, in units of the inter-center distance
-`stretch`: an index of the stretching of the cosines

RETURN
-`Φ`: Matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis function at the i-th timestep from beginning of the trial
"""
function raisedcosines(begins0::Bool, ends0::Bool, nbases::Integer, nbins::Integer, period::Real, stretch::Real)
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
            Δcenter = (b-a) / (nbases+3)
        else
            Δcenter = (b-a) / (nbases+1)
        end
        centers = a .+ 2Δcenter .+ collect(0:max(1,nbases-1)).*Δcenter
    else
        if ends0
            Δcenter = (b-a) / (nbases+1)
        else
            Δcenter = (b-a) / (nbases-1)
        end
        centers = a .+ collect(0:max(1,nbases-1)).*Δcenter
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
