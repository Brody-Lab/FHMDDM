"""
    temporal_bases_values(options, trialset)

Value of each temporal basis at each time bin in a trialset

INPUT
-`options`: model settings
-`𝐓`: vector of the number of timesteps in each trial

RETURN
-`𝐕`: A matrix whose element 𝐕[t,i] indicates the value of the i-th temporal basis function in the t-th time bin in the trialset
-`Φ`: temporal basis functions. Element Φ[τ,i] corresponds to the value of  i-th temporal basis function in the τ-th time step in each trial
"""
function temporal_bases_values(options::Options, 𝐓::Vector{<:Integer})
    Tmax = maximum(𝐓)
    nbases = max(1, ceil(Integer, options.a_basis_per_s*(Tmax*options.Δt)))
    if options.basistype == "none"
        Φ = ones(Tmax,1)
        nbases = 1
    elseif options.basistype == "raised_cosine"
        Φ = raisedcosinebases(false, false, nbases, Tmax)
    elseif options.basistype == "Chebyshev_polynomial"
        Φ = chebyshevbases(nbases, Tmax)
    elseif options.basistype == "stretched_raised_cosine"
        Φ = stretched_raised_cosines(nbases, Tmax)
    else
        error("unrecognized type for temporal basis function: ", options.basistype)
    end
    𝐕 = zeros(sum(𝐓), nbases)
    k = 0
    for T in 𝐓
        for t = 1:T
            k = k + 1;
            𝐕[k,:] = Φ[t,:]
        end
    end
    return 𝐕, Φ
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
    chebyshevT(x,n)

Construct Chebyshev polynomial of the first kind

ARGUMENT
-`x`: argument to the polynomial
-`n`: the n-th polynomial

RETURN
-`T_n(x)`: the output of the n-th Chebyshev polynomial evaluated at x
"""
function chebyshevT(x::AbstractFloat, n::Integer)
    if n == 0
        return 1.
    elseif n == 1
        return x
    else
        return 2x * chebyshevT(x, n - 1) - chebyshevT(x, n - 2)
    end
end

"""
    chebyshevbases(nbases,nbins)

Construct Chebyshev bases of (nbases)-th order

ARGUMENT
-`nbases`: number of bases
-`nbins`: number of bins in the time window tiled by the bases

RETURN
-`Φ`: Matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis at the i-th timestep from beginning of the trial
"""
function chebyshevbases(nbases::Integer, nbins::Integer)
    trange = range(-1, 1, length=nbins)
    Φ = chebyshevT.(trange, 0)
    for ibasis = 1:nbases-1
        Φ = hcat(Φ, chebyshevT.(trange, ibasis))
    end
    Φ
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
