"""
	accumulatorbases(options, 𝐓)

Temporal basis functions for the accumulator kernel

ARGUMENT
-`options`: settings of the model
-`𝐓`: number of timesteps

RETURN
-`𝐔`: A matrix whose element 𝐔[t,i] indicates the value of the i-th temporal basis function in the t-th time bin in the trialset
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time bin in the kernel
"""
function accumulatorbases(options::Options, 𝐓::Vector{<:Integer})
    temporal_bases_values(  options.tbf_accu_begins0,
                            options.tbf_accu_constantfunction,
                            options.Δt,
                            options.tbf_accu_ends0,
                            options.tbf_accu_hz,
                            options.tbf_accu_period,
                            options.glminputscaling,
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
                            options.tbf_time_constantfunction,
                            options.Δt,
                            options.tbf_time_ends0,
                            options.tbf_time_hz,
                            options.tbf_time_period,
                            options.glminputscaling,
                            options.tbf_time_stretch,
                            𝐓)
end

"""
	premovementbases(options, movementtimes_s, 𝐓)

Temporal basis functions for the premovement kernel

ARGUMENT
-`options`: settings of the model
-`movementtimes_s`: time of movement relative to the stereoclick, in seconds
-`𝐓`: number of timesteps

RETURN
-`𝐔`: A matrix whose element 𝐔[t,i] indicates the value of the i-th temporal basis function in the t-th time bin in the trialset
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time bin in the kernel
"""
function premovementbases(options::Options, movementtimes_s::Vector{<:AbstractFloat}, 𝐓::Vector{<:Integer})
	nbases = ceil(Int, options.tbf_move_dur_s*options.tbf_move_hz)
	nbins = ceil(Int, options.tbf_move_dur_s/options.Δt)
	Φ = unitarybases(options.tbf_move_begins0, options.tbf_move_constantfunction, options.tbf_move_ends0, nbases, nbins, options.tbf_move_period, options.tbf_move_stretch)
	Φ .*= options.glminputscaling
	nbases = size(Φ,2)
	movementbin = ceil.(Int, movementtimes_s./options.Δt) # movement times are always positive
	𝐔 = zeros(sum(𝐓), nbases)
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
	return 𝐔, Φ
end

"""
    temporal_bases_values(begins0, constantfunction, ends0, hz, period, scaling, stretch, 𝐓)

Value of each temporal basis at each time bin in a trialset

INPUT
-`options`: model settings
-`𝐓`: vector of the number of timesteps in each trial

RETURN
-`𝐕`: A matrix whose element 𝐕[t,i] indicates the value of the i-th temporal basis function in the t-th time bin in the trialset
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time step in each trial
"""
function temporal_bases_values(begins0::Bool, constantfunction::Bool, Δt::AbstractFloat, ends0::Bool, hz::Real, period::Real, scaling::Real, stretch::Real, 𝐓::Vector{<:Integer})
    Tmax = maximum(𝐓)
    nbases = max(1, ceil(Int, hz*(Tmax*Δt)))
    if nbases == 1
		x = scaling/sqrt(Tmax)
        Φ = fill(x,Tmax)
        𝐕 = fill(x, sum(𝐓), 1)
    else
        Φ = unitarybases(begins0, constantfunction, ends0, nbases, Tmax, period, stretch)
		Φ .*= scaling
        𝐕 = zeros(sum(𝐓), nbases)
        k = 0
        for T in 𝐓
            for t = 1:T
                k = k + 1;
                𝐕[k,:] = Φ[t,:]
            end
        end
    end
    return 𝐕, Φ
end

"""
	unitarybases(begins0, constantfunction, ends0, nbases, nbins, period, stretch)

A matrix of values from orthogonal temporal basis functions that each has an L2 norm of one.

The raised cosines temporal basis functions are used as the starting point.

ARGUMENT
-`begins0`: whether the raised cosines begin at the trough or at the peak
-`constantfunction`: whether the bases can parametrize a flat line
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
function unitarybases(begins0::Bool, constantfunction::Bool, ends0::Bool, nbases::Integer, nbins::Integer, period::Real, stretch::Real)
	Φ = raisedcosines(begins0, constantfunction, ends0, nbases, nbins, period, stretch)
	F = svd(Φ)
	F.U[:,1:nbases]
end

"""
    orthogonal_wellconditioned_tbf(begins0, constantfunction, ends0, nbases, nbins, period, stretch)

Temporal basis functions that are orthogonal to each other and well-conditioned

The raised cosines temporal basis functions are used as the starting point

ARGUMENT
-`nbases`: starting number of temporal basis functions. The ultimate temporal basis functions may have fewer.
-`nbins`: number of bins in the time window tiled by the bases
-`options`: Settings of the model

RETURN
-`Φ`: Matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis function at the i-th timestep from beginning of the trial
"""
function orthogonal_wellconditioned_tbf(begins0::Bool, constantfunction::Bool, ends0::Bool, nbases::Integer, nbins::Integer, period::Real, stretch::Real; max_condition_number::Real=10)
    Φ = raisedcosines(begins0, constantfunction, ends0, nbases, nbins, period, stretch)
    transformtbf(Φ, constantfunction, max_condition_number)
end

"""
    transformtbf(Φ, max_condition_number)

Transform temporal basis functions so that they are orthogonal to each other and well-conditioned

ARGUMENT
-`Φ`: values of the temporal basis functions--a matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis function at the i-th timestep from beginning of the kernel
-`max_condition_number`: Maximum condition number of `Φ`

RETURN
-`Φ`: Matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis function at the i-th timestep from beginning of the trial
```
"""
function transformtbf(Φ::Matrix{<:AbstractFloat}, constantfunction::Bool, max_condition_number::Real)
    U, s, Vᵀ = svd(Φ)
    abss = abs.(s)
    isflat = (maximum(abss)./abss) .> max_condition_number
    if sum(isflat)==0
        Φ = Φ*Vᵀ
    else
        Φ = Φ*Vᵀ[:, .!isflat]
		if constantfunction
		    𝟏 = ones(size(Φ,1))
			𝐰 = Φ \ 𝟏
			𝛆 = 𝟏 - Φ*𝐰
            Φ[:,end] .+= 𝛆./𝐰[end] # the condition number is slightly modified
            U, s, Vᵀ = svd(Φ)
            Φ = Φ*Vᵀ
        end
    end
    return Φ
end

"""
    raisedcosines(begins0, constantfunction, ends0, nbases, nbins, period, stretch)

Values of raised cosine temporal basis functions (tbf's)

ARGUMENT
-`begins0`: whether the first temporal basis function begins at the trough or at the peak
-`constantfunction`: whether the temporal basis functions can parametrize a flat line
-`ends0`: whether the last temporal basis function begins at the trough or at the peak
-`nbases`: number of bases
-`nbins`: number of bins in the time window tiled by the bases
-`period`: period of the raised cosine, in units of the inter-center distance
-`stretch`: an index of the stretching of the cosines

RETURN
-`Φ`: Matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis function at the i-th timestep from beginning of the trial
"""
function raisedcosines(begins0::Bool, constantfunction::Bool, ends0::Bool, nbases::Integer, nbins::Integer, period::Real, stretch::Real)
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
    if constantfunction
        begins0 = false
        ends0 = false
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
    if constantfunction
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
