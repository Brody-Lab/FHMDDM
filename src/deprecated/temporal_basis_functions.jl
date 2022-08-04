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
