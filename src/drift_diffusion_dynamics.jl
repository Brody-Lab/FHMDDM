"""
    adapt(clicks, ϕ, k)

Compute the adapted input strength of auditory clicks.

Assumes that adaptation is across-stream: i.e., a click from either side is affected by preceding clicks from both sides.

ARGUMENT
-`clicks`: information on all the clicks in one trial. The stereoclick is excluded.
-`ϕ`: a parameter indicating whether each click is facilitated (ϕ>0) or depressed (ϕ<0) by preceding clicks.
-`k`: a parameter indicating the exponential change rate of the sensory adaptation. Must be in the range of k ∈ (0, ∞).For a fixed non-zero value of ϕ, a smaller k indicates that preceding clicks exert a greater effect.

RETURN
-`C`: the post-adaptation input magnitude of each click. It is a vector of floats that has the same size as field `time` in the argument `clicks`
"""
function adapt(clicks::Clicks, k::T, ϕ::T) where {T<:Real}
    nclicks = length(clicks.time)
	@assert nclicks > 0
    C = zeros(T, nclicks)
	C[1] = 1.0 - (1.0-ϕ)*exp(-k*clicks.time[1])
    for i = 2:nclicks
        Δt = clicks.time[i] - clicks.time[i-1]
        C[i] = 1.0 - (1.0-ϕ*C[i-1])*exp(-k*Δt)
    end
    return C
end

"""
    ∇adapt(clicks, k, ϕ)

Adapt the clicks and compute the first-order partial derivative of the adapted strengths with respect to the parameters

It

ARGUMENT
-`clicks`: structure containing information about the auditory clicks in one trial. Stereoclick excluded.
-`k`: exponential change of the adaptation dynamics
-`ϕ`: strength and sign of the adaptation (facilitation: ϕ > 0; depression: ϕ < 0)

RETURN
-`C`: adapted strengths of the clicks
-`dCdk`: first-order partial derivative of `C` with respect to `k`
-`dCdϕ`: first-order partial derivative of `C` with respect to `ϕ`
"""
function ∇adapt(clicks::Clicks, k::AbstractFloat, ϕ::AbstractFloat)
	nclicks = length(clicks.time)
	@assert nclicks > 0
    C, dCdk, dCdϕ = zeros(nclicks), zeros(nclicks), zeros(nclicks)
    e⁻ᵏᵈᵗ = exp(-k*clicks.time[1])
    C[1] = 1.0 - (1.0-ϕ)*e⁻ᵏᵈᵗ
    dCdϕ[1] = e⁻ᵏᵈᵗ
    dCdk[1] = (1.0-C[1])*clicks.time[1]
    for i = 2:nclicks
        Δt = clicks.time[i] - clicks.time[i-1]
        e⁻ᵏᵈᵗ = exp(-k*Δt)
        C[i] = 1.0 - (1.0 - ϕ*C[i-1])*e⁻ᵏᵈᵗ
        dCdϕ[i] = e⁻ᵏᵈᵗ*(C[i-1] + ϕ*dCdϕ[i-1])
        dCdk[i] = ϕ*e⁻ᵏᵈᵗ*dCdk[i-1] + (1.0-C[i])*Δt
    end
    return C, dCdk, dCdϕ
end

"""
    stochasticmatrix!(A, 𝛍, σ, 𝛏)

In-place computation of the stochastic matrix for the discretized Fokker-Planck system for a single time step

MODIFIED ARGUMENT
-`A`: a square matrix describing the transitions of the accumulator variable at a single time step

UNMODIFIED ARGUMENT
-`𝛍`: mean of the Gaussian PDF of the accumulator variable conditioned on its value in the previous time step
-`σ`: standard deviation of the Weiner process at this time step
-`𝛏`: a vector specifying the equally-spaced values into which the accumulator variable is discretized

RETURN
-nothing
"""
function stochasticmatrix!(A,
                           𝛍,
                           σ::T,
                           𝛏) where {T<:Real}
	Ξ = length(𝛏)
	Ξ_1 = Ξ-1
	σ_Δξ = σ/(𝛏[2]-𝛏[1])
    ΔΦ = zeros(T, Ξ_1)
	A[1,1] = 1.0
	A[Ξ,Ξ] = 1.0
    @inbounds for j = 2:Ξ_1
        𝐳 = (𝛏 .- 𝛍[j])./σ
        Δf = diff(normpdf.(𝐳))
        Φ = normcdf.(𝐳)
        C = normccdf.(𝐳) # complementary cumulative distribution function
        for i = 1:Ξ_1
            if 𝛍[j] <= 𝛏[i]
                ΔΦ[i] = C[i] - C[i+1]
            else
                ΔΦ[i] = Φ[i+1] - Φ[i]
            end
        end
        A[1,j] = Φ[1] + σ_Δξ*(Δf[1] + 𝐳[2]*ΔΦ[1])
        for i = 2:Ξ_1
            A[i,j] = σ_Δξ*(Δf[i] - Δf[i-1] + 𝐳[i+1]*ΔΦ[i] - 𝐳[i-1]*ΔΦ[i-1])
        end
        A[Ξ,j] = C[Ξ] - σ_Δξ*(Δf[Ξ_1] + 𝐳[Ξ_1]*ΔΦ[Ξ_1])
    end
    return nothing
end


"""
    stochasticmatrix!(A, cL, cR, trialinvariant, θnative)

In-place computation of a transition matrix for a single time-step

MODIFIED ARGUMENT
-`A`: the transition matrix. Expects the `A[2:end,1] .== 0` and `A[1:end-1,end] .== 0`

UNMODIFIED ARGUMENT
-`cL`: input from the left
-`cR`: input from the right
-`trialinvariant`: structure containing quantities used for computations for each trial
-`θnative`: model parameters in native space
"""
function stochasticmatrix!(A,
                           cL::Real,
						   cR::Real,
						   trialinvariant::Trialinvariant,
						   θnative::Latentθ)
    @unpack Δt, 𝛏 = trialinvariant
	𝛍 = conditionedmean(cR-cL, Δt, θnative.λ[1], 𝛏)
	σ = √( (cL+cR)*θnative.σ²ₛ[1] + θnative.σ²ₐ[1]*Δt )
	stochasticmatrix!(A, 𝛍, σ, 𝛏)
    return nothing
end

"""
    conditionedmean(Δc, Δt, λ, 𝛏)

Mean of the Gaussian PDF of the accumulator variable conditioned on its value in the previous time step

ARGUMENT
-`Δc`: right input minus left input
-`Δt`: size of the time step
-`λ`: leak or instability
-`𝛏`: values of the accumulator variable in the previous time step
"""
function conditionedmean(Δc::Real, Δt::AbstractFloat, λ::Real, 𝛏)
    if λ==1.0
		𝛏 .+ Δc
	else
		λΔt = λ*Δt
		expλΔt = exp(λΔt)
		c̃ = Δc*(expλΔt- 1.0)/λΔt
	    expλΔt.*𝛏 .+ c̃
	end
end

"""
    stochasticmatrix!(A, ∂μ, ∂σ², ∂B, 𝛍, σ, 𝛚, 𝛏)

Compute the transition matrix and partial derivatives with respect to the means, variance, and the bound parameter (in real space)

MODIFIED ARGUMENT
-`A`: the transition matrix. Expects the `A[2:end,1] .== 0` and `A[1:end-1,end] .== 0`
-`∂μ`: the first order partial derivative of the transition matrix with respect to the mean in each column.
-`∂σ²`: the first order partial derivative of the transition matrix with respect to the variance.
-`∂B`: the first order partial derivative of the transition matrix with respect to the bound height.

UNMODIFIED ARGUMENT
-`𝛍`: mean of the Gaussian PDF's
-`σ`: standard deviation of the Gaussian PDF's
-`𝛚`: temporary quantity used to compute the partial derivative with respect to the bound parameter (in real space)
-`𝛏`: value of the accumulator variable in the previous time step
"""
function stochasticmatrix!(	A::Matrix{<:AbstractFloat},
							∂μ::Matrix{<:AbstractFloat},
							∂σ²::Matrix{<:AbstractFloat},
							∂B::Matrix{<:AbstractFloat},
							𝛍::Vector{<:AbstractFloat},
							σ::AbstractFloat,
							Ω::Matrix{<:AbstractFloat},
							𝛏::Vector{<:AbstractFloat})
	Ξ = length(𝛏)
	Ξ_1 = Ξ-1
	B = 𝛏[end]*(Ξ-2)/Ξ_1
	Δξ = 𝛏[2]-𝛏[1]
	σ_Δξ = σ/Δξ
    σ2Δξ = 2σ*Δξ
    A[1,1] = 1.0
    A[Ξ,Ξ] = 1.0
	ΔΦ = zeros(Ξ_1)
    @inbounds for j = 2:Ξ_1
        𝐳 = (𝛏 .- 𝛍[j])./σ
        Δf = diff(normpdf.(𝐳))
        Φ = normcdf.(𝐳)
        C = normccdf.(𝐳) # complementary cumulative distribution function
        for i = 1:Ξ_1
            if 𝛍[j] <= 𝛏[i]
                ΔΦ[i] = C[i] - C[i+1]
            else
                ΔΦ[i] = Φ[i+1] - Φ[i]
            end
        end
        A[1,j] = Φ[1] + σ_Δξ*(Δf[1] + 𝐳[2]*ΔΦ[1])
        ∂μ[1,j] = -ΔΦ[1]/Δξ
        ∂σ²[1,j] = Δf[1]/σ2Δξ
        ∂B[1,j] = (Φ[1] - A[1,j] + Ω[2,j]*ΔΦ[1])/B
        for i = 2:Ξ_1
            A[i,j] = σ_Δξ*(Δf[i] - Δf[i-1] + 𝐳[i+1]*ΔΦ[i] - 𝐳[i-1]*ΔΦ[i-1])
            ∂μ[i,j] = (ΔΦ[i-1] - ΔΦ[i])/Δξ
            ∂σ²[i,j] = (Δf[i]-Δf[i-1])/σ2Δξ
            ∂B[i,j] = (Ω[i+1,j]*ΔΦ[i] - Ω[i-1,j]*ΔΦ[i-1] - A[i,j])/B
        end
        A[Ξ,j] = C[Ξ] - σ_Δξ*(Δf[Ξ_1] + 𝐳[Ξ_1]*ΔΦ[Ξ_1])
        ∂μ[Ξ,j] = ΔΦ[Ξ_1]/Δξ
        ∂σ²[Ξ,j] = -Δf[Ξ_1]/σ2Δξ
        ∂B[Ξ,j] = (C[Ξ] - A[Ξ,j] - Ω[Ξ_1,j]*ΔΦ[Ξ_1])/B
    end
    return nothing
end

"""
    stochasticmatrix!(A, ∂μ, ∂σ², ∂B, cL, cR, trialinvariant, θnative)

Compute the transition matrix and partial derivatives with respect to the means, variance, and the bound parameter (in real space)

MODIFIED ARGUMENT
-`A`: the transition matrix. Expects the `A[2:end,1] .== 0` and `A[1:end-1,end] .== 0`
-`∂μ`: the first order partial derivative of the transition matrix with respect to the mean in each column.
-`∂σ²`: the first order partial derivative of the transition matrix with respect to the variance.
-`∂B`: the first order partial derivative of the transition matrix with respect to the bound height.

UNMODIFIED ARGUMENT
-`cL`: input from the left
-`cR`: input from the right
-`trialinvariant`: structure containing quantities used for computations for each trial
-`θnative`: model parameters in native space
"""
function stochasticmatrix!(	A::Matrix{<:AbstractFloat},
							∂μ::Matrix{<:AbstractFloat},
							∂σ²::Matrix{<:AbstractFloat},
							∂B::Matrix{<:AbstractFloat},
							cL::Real,
							cR::Real,
							trialinvariant::Trialinvariant,
							θnative::Latentθ)
    @unpack Δt, Ω, 𝛏 = trialinvariant
	𝛍 = conditionedmean(cR-cL, Δt, θnative.λ[1], 𝛏)
	σ = √( (cL+cR)*θnative.σ²ₛ[1] + θnative.σ²ₐ[1]*Δt )
	stochasticmatrix!(A, ∂μ, ∂σ², ∂B, 𝛍, σ, Ω, 𝛏)
	return nothing
end

"""
    probabilityvector(μ, σ, 𝛏)

Discrete representation of a Gaussian PDF

ARGUMENT
-`μ`: mean
-`σ`: standard deviation
-`𝛏`: discrete values used for representation

RETURN
-`𝐩`: probability vector

EXAMPLE
```julia-repl
julia> μ=1.0; σ=2.0; Ξ=7; B=10.0; 𝛏 = B*(2collect(1:Ξ) .- Ξ .- 1)/(Ξ-2); probabilityvector(μ,σ,𝛏)
7-element Array{Float64,1}:
 3.471030649983585e-7
 0.0010013743804762956
 0.09689448862754767
 0.5678589080695604
 0.31962072539725905
 0.014594917590384344
 2.9238831707279765e-5
```
"""
function probabilityvector(μ::T,
						   σ::T,
						   𝛏) where {T<:Real}
    Ξ = length(𝛏)
    Ξ_1 = Ξ-1
    σ_Δξ = σ/(𝛏[2]-𝛏[1])
    𝐳 = (𝛏 .- μ)./σ
    Δf = diff(normpdf.(𝐳))
    Φ = normcdf.(𝐳)
    C = normccdf.(𝐳) # complementary cumulative distribution function
    ΔΦ = zeros(T, Ξ_1)
    for i = 1:Ξ_1
        if μ <= 𝛏[i]
            ΔΦ[i] = C[i] - C[i+1]
        else
            ΔΦ[i] = Φ[i+1] - Φ[i]
        end
    end
    𝐩 = Φ # reuse the memory
    𝐩[1] = Φ[1] + σ_Δξ*(Δf[1] + 𝐳[2]*ΔΦ[1])
    for i = 2:Ξ_1
        𝐩[i] = σ_Δξ*(Δf[i] - Δf[i-1] + 𝐳[i+1]*ΔΦ[i] - 𝐳[i-1]*ΔΦ[i-1])
    end
    𝐩[Ξ] = C[Ξ] - σ_Δξ*(Δf[Ξ_1] + 𝐳[Ξ_1]*ΔΦ[Ξ_1])
    return 𝐩
end

"""
    probabilityvector(π, ∂μ, ∂σ², ∂B, μ, σ, 𝛏)

Discrete representation of a Gaussian PDF and its partial derivative with respect to the mean, variance, and bound (in real space)

MODIFIED ARGUMENT
-`π`: probability vector
-`∂μ`: the first order partial derivative with respect to the mean in each column.
-`∂σ²`: the first order partial derivative with respect to the variance.
-`∂B`: the first order partial derivative of the transition matrix with respect to the bound height.

UNMODIFIED ARGUMENT
-`μ`: mean
-`σ`: standard deviation
-`𝛏`: discrete values used for representation

"""
function probabilityvector!(π::Vector{<:AbstractFloat},
							∂μ::Vector{<:AbstractFloat},
							∂σ²::Vector{<:AbstractFloat},
							∂B::Vector{<:AbstractFloat},
							μ::AbstractFloat,
							𝛚::Vector{<:AbstractFloat},
							σ::AbstractFloat,
							𝛏::Vector{<:AbstractFloat})
    Ξ = length(𝛏)
    Ξ_1 = Ξ-1
	B = 𝛏[end]*(Ξ-2)/Ξ_1
    Δξ=𝛏[2]-𝛏[1]
    σ_Δξ = σ/Δξ
	σ2Δξ = 2σ*Δξ
    𝐳 = (𝛏 .- μ)./σ
    Δf = diff(normpdf.(𝐳))
    Φ = normcdf.(𝐳)
    C = normccdf.(𝐳) # complementary cumulative distribution function
    ΔΦ = zeros(Ξ_1)
    for i = 1:Ξ_1
        if μ <= 𝛏[i]
            ΔΦ[i] = C[i] - C[i+1]
        else
            ΔΦ[i] = Φ[i+1] - Φ[i]
        end
    end
    π[1] = Φ[1] + σ_Δξ*(Δf[1] + 𝐳[2]*ΔΦ[1])
	∂μ[1] = -ΔΦ[1]/Δξ
	∂σ²[1] = Δf[1]/σ2Δξ
	∂B[1] = (Φ[1] - π[1] + 𝛚[2]*ΔΦ[1])/B
	for i = 2:Ξ_1
        π[i] = σ_Δξ*(Δf[i] - Δf[i-1] + 𝐳[i+1]*ΔΦ[i] - 𝐳[i-1]*ΔΦ[i-1])
		∂μ[i] = (ΔΦ[i-1] - ΔΦ[i])/Δξ
		∂σ²[i] = (Δf[i]-Δf[i-1])/σ2Δξ
		∂B[i] = (𝛚[i+1]*ΔΦ[i] - 𝛚[i-1]*ΔΦ[i-1] - π[i])/B
    end
    π[Ξ] = C[Ξ] - σ_Δξ*(Δf[Ξ_1] + 𝐳[Ξ_1]*ΔΦ[Ξ_1])
	∂μ[Ξ] = ΔΦ[Ξ_1]/Δξ
	∂σ²[Ξ] = -Δf[Ξ_1]/σ2Δξ
	∂B[Ξ] = (C[Ξ] - π[Ξ] - 𝛚[Ξ_1]*ΔΦ[Ξ_1])/B
    return nothing
end


"""
    approximatetransition!(Aᵃ, dt, dx, λ, μ, n, σ², xc)

Compute the approximate transition matrix ``𝑝(𝑎ₜ ∣ 𝑎ₜ₋₁, clicks(𝑡), 𝜃)`` for a single time bin and store it in `Aᵃ`.

The computation makes use of the `λ`, a scalar indexing leakiness or instability; `μ` and `σ²`, mean and variance of the Gaussian noise added, time bin size `dt`, size of bins of the accumulator variable `dx`, number of bins of the accumulator variables `n`, and bin centers `xc`
"""
function approximatetransition!(Aᵃ,
	                           dt::AbstractFloat,
	                           dx::T,
	                           λ::T,
	                           μ::T,
	                           n::Integer,
	                           σ²::T,
	                           xc;
	                           minAᵃ=zero(T)) where {T<:Real}
    Aᵃ[1,1] = one(T)
    Aᵃ[end,end] = one(T)
    Aᵃ[2:end,1] .= zero(T)
    Aᵃ[1:end-1,end] .= zero(T)
    Aᵃ[:,2:n-1] .= minAᵃ
    ndeltas = max(70,ceil(Int, 10. *sqrt(σ²)/dx))
    deltaidx = collect(-ndeltas:ndeltas)
    deltas = deltaidx * (5. *sqrt(σ²))/ndeltas
    p̃s = exp.(-0.5 * (5*deltaidx./ndeltas).^2) # p(s) is not yet normalized
    sqrt2πσ² = √(2π*σ²)
    @inbounds for j = 2:n-1
        mu = exp(λ*dt)*xc[j] + μ * expm1_div_x(λ*dt)
        # set minimum values
        s_lower = mu + deltas[1] - dx
        s_upper = mu + deltas[end] + dx
        ∑ = 1.0
        for i = 1:n
            if xc[i]<s_lower || xc[i]>s_upper
                Aᵃ[i,j] += exp(-(xc[i]-mu)^2/2σ²)/sqrt2πσ²
            end
            ∑ -= Aᵃ[i,j]
        end
        ps = p̃s/sum(p̃s)/∑ # now p(s) is normalized
        #now we're going to look over all the slices of the gaussian
        for k = 1:2*ndeltas+1
            s = mu + deltas[k]
            if s <= xc[1]
                Aᵃ[1,j] += ps[k]
            elseif s >= xc[end]
                Aᵃ[end,j] += ps[k]
            else
                if (xc[1] < s) && (xc[2] > s)
                    lp,hp = 1,2
                elseif (xc[end-1] < s) && (xc[end] > s)
                    lp,hp = n-1,n
                else
                    hp,lp = ceil(Int, (s-xc[2])/dx) + 2, floor(Int, (s-xc[2])/dx) + 2
                end
                if hp == lp
                    Aᵃ[lp,j] += ps[k]
                else
                    dd = xc[hp] - xc[lp]
                    Aᵃ[hp,j] += ps[k]*(s-xc[lp])/dd
                    Aᵃ[lp,j] += ps[k]*(xc[hp]-s)/dd
                end
            end
        end
    end
    return nothing
end

"""
    expm1_div_x(x)
"""
function expm1_div_x(x)

    y = exp(x)
    y == 1. ? one(y) : (y-1.)/log(y)

end
