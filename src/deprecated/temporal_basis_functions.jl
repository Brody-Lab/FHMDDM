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
