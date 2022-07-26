"""
    temporal_bases_values(options, trialset)

Value of each temporal basis at each time bin in a trialset

INPUT
-`options`: model settings
-`𝐓`: vector of the number of timesteps in each trial

RETURN
-`𝐕`: A matrix whose element 𝐕[t,i] indicates the value of the i-th temporal basis function in the t-th time bin in the trialset
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time step in each trial

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_07_26a_test/T176_2018_05_03/data.mat"
julia> model = Model(datapath)
julia> model.trialsets[1].mpGLMs[1].𝐕
julia> model.trialsets[1].mpGLMs[1].Φ
julia> save(model)
julia>
```
"""
function temporal_bases_values(options::Options, 𝐓::Vector{<:Integer})
    Tmax = maximum(𝐓)
    nbases = max(1, ceil(Integer, options.atbf_hz*(Tmax*options.Δt)))
    if nbases == 1
        Φ = ones(Tmax,1)
        𝐕 = ones(sum(𝐓), 1)
    else
        Φ = raisedcosines(nbases, Tmax, options)
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
    raisedcosines(nbases, nbins, options)

Values of raised cosine temporal basis functions (tbf's)

ARGUMENT
-`nbases`: number of bases
-`nbins`: number of bins in the time window tiled by the bases
-`options`: Settings of the model

RETURN
-`Φ`: Matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis function at the i-th timestep from beginning of the trial
"""
function raisedcosines(nbases::Integer, nbins::Integer, options::Options)
    if isnan(options.atbf_stretch) || options.atbf_stretch < eps()
        a = 1
        b = nbins
        t = collect(1:nbins)
    else
        λ = 1/options.atbf_stretch
        a = log(1+λ)
        b = log(nbins+λ)
        t = log.(collect(1:nbins) .+ λ)
    end
    if options.atbf_constantfunction
        begins0 = false
        ends0 = false
    else
        begins0 = options.atbf_begins0
        ends0 = options.atbf_ends0
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
    ω = 2π/Δcenter/options.atbf_period
    Φ = raisedcosines(centers, ω, t)
    if options.atbf_constantfunction
        lefttail = raisedcosines([centers[1]-Δcenter], ω, t)
        righttail = raisedcosines([centers[end]+Δcenter], ω, t)
        Φ[:,1] += lefttail
        Φ[:,end] += righttail
        indices = t .<= centers[1] + options.atbf_period/2*Δcenter
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

"""
    raisedcosinebases(nbases,nbins)

Construct smooth raised cosine bases

The spacing between the centers is 1/4 of the width (period)

ARGUMENT
-`begins_at_0`: whether the value of the first temporal basis function at the first time step is equal to zero or equal to 1
-`ends_at_0`: whether the value of the last temporal basis function at the last time step is equal 0 or equal to 1.
-`nbases`: number of bases
-`nbins`: number of bins in the time window tiled by the bases

RETURN
-`Φ`: Matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis at the i-th timestep from beginning of the trial
"""
function raisedcosinebases(begins_at_0::Bool, ends_at_0::Bool, nbases::Integer, nbins::Integer)
    if begins_at_0
        if ends_at_0
            Δcenter = (nbins-1) / (nbases+1)
            centers = collect(1+Δcenter:Δcenter:nbins-Δcenter)
        else
            Δcenter = (nbins-1) / nbases
            centers = collect(1+Δcenter:Δcenter:nbins)
        end
    else
        if ends_at_0
            Δcenter = (nbins-1) / nbases
            centers = collect(1:Δcenter:nbin-Δcenter)
        else
            Δcenter = (nbins-1) / max(1,nbases-1)
            centers = collect(1:Δcenter:nbins)
        end
    end
    timefromcenter = collect(1:nbins) .- transpose(centers)
    period = 4Δcenter
    (abs.(timefromcenter) .< period/2).*(cos.(timefromcenter*2π/period)*0.5 .+ 0.5)
end

"""
    stretched_raised_cosines(nbases, nbins; stretch)

Raised cosines that are stretched in time.

The first cosine is equal to zero in the first time bin, and the last cosine is equal to its peak at the last time bin

ARGUMENT
-`begins_at_0`: whether the value of the first temporal basis function at the first time step is equal to zero or equal to 1
-`ends_at_0`: whether the value of the last temporal basis function at the last time step is equal 0 or equal to 1.
-`nbases`: number of bases
-`nbins`: number of bins in the time window tiled by the bases

OPTIONAL ARGUMENT
-`stretch`: a positive number that indexes the amount of nonlinear stretch of the basis functions. Larger value indicates greater stretching.

RETURN
-`Φ`: Matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis at the i-th timestep from beginning of the trial
"""
function stretched_raised_cosines(begins_at_0::Bool, ends_at_0::Bool, nbases::Integer, nbins::Integer; stretch::AbstractFloat=1e3)
    λ = max(1/stretch, eps())
    a = log(1+λ)
    b = log(nbins+λ)
    if begins_at_0
        if ends_at_0
            Δcenter = (b-a) / (nbases+3)
        else
            Δcenter = (b-a) / (nbases+1)
        end
        centers = a .+ 2Δcenter .+ collect(0:max(1,nbases-1)).*Δcenter
    else
        if ends_at_0
            Δcenter = (b-a) / (nbases+1)
        else
            Δcenter = (b-a) / max(1,nbases-1)
        end
        centers = a .+ collect(0:max(1,nbases-1)).*Δcenter
    end
    x = log.(collect(1:nbins) .+ λ) .- transpose(centers)
    tbf = (cos.(max.(-π, min.(π, x/Δcenter/2*π))) .+ 1)/2
    tbf./maximum(tbf, dims=1)
end
