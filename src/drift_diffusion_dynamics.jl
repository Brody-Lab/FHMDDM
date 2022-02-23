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
function adapt(clicks::Clicks, k::T1, ϕ::T2) where {T1<:Real, T2<:Real}
	T = T1<:T2 ? T2 : T1
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

ARGUMENT
-`clicks`: structure containing information about the auditory clicks in one trial. Stereoclick excluded.
-`k`: exponential change of the adaptation dynamics
-`ϕ`: strength and sign of the adaptation (facilitation: ϕ > 0; depression: ϕ < 0)

RETURN
-`C`: adapted strengths of the clicks
-`dCdk`: first-order partial derivative of `C` with respect to `k`
-`dCdϕ`: first-order partial derivative of `C` with respect to `ϕ`
"""
function ∇adapt(clicks::Clicks, k::T1, ϕ::T2) where {T1<:Real, T2<:Real}
	T = T1<:T2 ? T2 : T1
	nclicks = length(clicks.time)
	@assert nclicks > 0
    C, dCdk, dCdϕ = zeros(T, nclicks), zeros(T, nclicks), zeros(T, nclicks)
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
function stochasticmatrix!(A::Matrix{T},
                           𝛍::Vector{<:Real},
                           σ::Real,
                           𝛏::Vector{<:Real}) where {T<:Real}
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
function stochasticmatrix!(A::Matrix{<:Real},
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
function conditionedmean(Δc::Real, Δt::AbstractFloat, λ::T, 𝛏::Vector{<:Real}) where {T<:Real}
    if λ==zero(T)
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
function stochasticmatrix!(	A::Matrix{T},
							∂μ::Matrix{<:Real},
							∂σ²::Matrix{<:Real},
							∂B::Matrix{<:Real},
							𝛍::Vector{<:Real},
							σ::Real,
							Ω::Matrix{<:Real},
							𝛏::Vector{<:Real}) where {T<:Real}
	Ξ = length(𝛏)
	Ξ_1 = Ξ-1
	B = 𝛏[end]*(Ξ-2)/Ξ_1
	Δξ = 𝛏[2]-𝛏[1]
	σ_Δξ = σ/Δξ
    σ2Δξ = 2σ*Δξ
    A[1,1] = 1.0
    A[Ξ,Ξ] = 1.0
	ΔΦ = zeros(T, Ξ_1)
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
function stochasticmatrix!(	A::Matrix{<:Real},
							∂μ::Matrix{<:Real},
							∂σ²::Matrix{<:Real},
							∂B::Matrix{<:Real},
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

RETURN
-`C`: complementary cumulative distribution function evaluated at each value z-scored value of the accumulator
-`Δf`: Difference between the probability densitiy function evaluated at consecutive z-scored values of the accumulator
-`ΔΦ`: Difference between the cumulative distribution function evaluated at consecutive z-scored valuse of the accumulator
-`f`: probability densitiy function evaluated at z-scored values of the accumulator
-`Φ`: cumulative distribution function evaluated at z-scored values of the accumulator
-`𝐳`: z-scored value of the accumulator
"""
function probabilityvector!(π::Vector{T},
							∂μ::Vector{<:Real},
							∂σ²::Vector{<:Real},
							∂B::Vector{<:Real},
							μ::Real,
							𝛚::Vector{<:Real},
							σ::Real,
							𝛏::Vector{<:Real}) where {T<:Real}
    Ξ = length(𝛏)
    Ξ_1 = Ξ-1
	B = 𝛏[end]*(Ξ-2)/Ξ_1
    Δξ=𝛏[2]-𝛏[1]
    σ_Δξ = σ/Δξ
	σ2Δξ = 2σ*Δξ
    𝐳 = (𝛏 .- μ)./σ
	f = normpdf.(𝐳)
    Δf = diff(f)
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
    return C, Δf, ΔΦ, f, Φ, 𝐳
end

"""
	expectatedHessian!

Expectation of second-derivatives of the log of the initial probability of the accumulator variable.

Computes the following for a single trial:

	`∑_𝐚₁ p(𝐚₁ ∣ 𝐘, d, θ) ⋅ ∇∇log(𝐚₁ ∣ B, μ₀, σᵢ², wₕ)`

ARGUMENT:
-`γᵃ₁`: a vector of floating-point numbers whose i-th element is the posterior probability of the initial value of accumulator in the i-th state: `γᵃ₁[i] ≡ p(a₁ᵢ=1 ∣ 𝐘, d, θ)`
-`μ₀`: a floating-point number representing the an offset to the mean of the initial distribution of the accumulator; this offset is fixed across all trials
-`previousanswer`: an integer representing whether the previous answer is on the left (-1), on the right (1), or unknown (0)
-`σ`: a floating-point number representing the standard deviation of the initial value of the accumulator
-`wₕ`: a floating-point number representing the weight of the previous answer on the mean of the initial distribution of the accumulator
-`𝛏`: a vector of floating-point numbers representing the values into which the accumulator is discretized

RETURN:
-`EH`: a four-by-four matrix whose columns and rows correspond to the second-order partial derivatives with respect to B, μ₀, σᵢ², and wₕ, in this order.

EXAMPLE
```julia-repo
> Ξ = 53
> γᵃ₁ = rand(Ξ)
> γᵃ₁ ./= sum(γᵃ₁)
> 𝛏 = (2.*collect(1:Ξ) .- Ξ .- 1)./(Ξ - 2)
> μ₀ = 0.5
> wₕ = 0.1
> σ = 0.8
> previousanswer = -1
> EH = expectatedHessian(γᵃ₁, μ₀, previousanswer, σ, wₕ, 𝛏)
```
"""
function expectatedHessian(γᵃ₁::Vector{<:AbstractFloat},
							μ₀::AbstractFloat,
							previousanswer::Integer,
							σ::AbstractFloat,
							wₕ::AbstractFloat,
							𝛏::Vector{<:AbstractFloat})
    Ξ = length(𝛏)
    Ξ_1 = Ξ-1
	B = 𝛏[end]*(Ξ-2)/Ξ_1
    Δξ=𝛏[2]-𝛏[1]
    𝛚 = 𝛏./Δξ
	μ = μ₀ + wₕ*previousanswer
	𝛑, ∂μ, ∂σ², ∂B = zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ)
	C, Δf, ΔΦ, 𝐟, Φ, 𝐳 = probabilityvector!(𝛑, ∂μ, ∂σ², ∂B, μ, 𝛚, σ, 𝛏)
	Δζ = diff(𝐟.*(𝐳.^2 .- 1.0)./4.0./σ.^3.0./Δξ)
	Δfωξ = diff(𝐟.*𝛚.*𝛏)
	Δfωz = diff(𝐟.*𝛚.*𝐳)
	Δfξ = diff(𝐟.*𝛏)
	Δfz = diff(𝐟.*𝐳)
	B²σ = B^2*σ
	BΔξσ = B*Δξ*σ
	Bσ²2 = B*σ^2*2
	Δξσ²2 = Δξ*σ^2*2
	EH = zeros(4,4)
	for i=1:Ξ
		if i == 1
			∂B∂B = Δfωξ[1]/B²σ - 2∂B[1]/B
			∂B∂μ = -Δfξ[1]/BΔξσ - ∂μ[1]/B
			∂B∂σ² = -Δfωz[1]/Bσ²2 - ∂σ²[1]/B
			∂μ∂σ² = Δfz[1]/Δξσ²2
			∂σ²∂σ² = Δζ[1]
		elseif i < Ξ
			∂B∂B = (Δfωξ[i] - Δfωξ[i-1])/B²σ - 2∂B[i]/B
			∂B∂μ = (Δfξ[i-1]-Δfξ[i])/BΔξσ - ∂μ[i]/B
			∂B∂σ² = (Δfωz[i-1]-Δfωz[i])/Bσ²2 - ∂σ²[i]/B
			∂μ∂σ² = (Δfz[i]-Δfz[i-1])/Δξσ²2
			∂σ²∂σ² = Δζ[i] - Δζ[i-1]
		else
			∂B∂B = -Δfωξ[Ξ_1]/B²σ - 2∂B[Ξ]/B
			∂B∂μ = Δfξ[Ξ_1]/BΔξσ - ∂μ[Ξ]/B
			∂B∂σ² = Δfωz[Ξ_1]/Bσ²2 - ∂σ²[Ξ]/B
			∂μ∂σ² = -Δfz[Ξ_1]/Δξσ²2
			∂σ²∂σ² = -Δζ[Ξ_1]
		end
		∂μ∂μ = ∂σ²[i]*2
		EH[1,1] += γᵃ₁[i]*∂B∂B
		EH[1,2] += γᵃ₁[i]*∂B∂μ
		EH[1,3] += γᵃ₁[i]*∂B∂σ²
		EH[2,2] += γᵃ₁[i]*∂μ∂μ
		EH[2,3] += γᵃ₁[i]*∂μ∂σ²
		EH[3,3] += γᵃ₁[i]*∂σ²∂σ²
	end
	EH[2,1] = EH[1,2]
	EH[3,1] = EH[1,3]
	EH[4,1] = EH[1,4] = EH[1,2]*previousreward #𝔼(∂wₕ∂B) = 𝔼(∂B∂wₕ) = 𝔼(∂μ₀∂B)*previousreward
	EH[3,2] = EH[2,3]
	EH[4,2] = EH[2,4] = EH[2,2]*previousreward #𝔼(∂wₕ∂μ₀) = 𝔼(∂μ₀∂wₕ) = 𝔼(∂μ₀∂μ₀)*previousreward
	EH[4,3] = EH[3,4] = EH[2,3]*previousreward #𝔼(∂wₕ∂σ²) = 𝔼(∂σ²∂wₕ) = 𝔼(∂μ₀∂σ²)*previousreward
	EH[4,4] = EH[2,2]*previousreward^2 #𝔼(∂wₕ∂wₕ) = 𝔼(∂μ₀∂μ₀)*previousreward^2
	return EH
end

"""
	Hessian

Expectation of second-derivatives of the log of the initial probability of the accumulator variable.

Computes the following for a single trial:

	`∇∇log(𝐚₁ ∣ B, μ₀, σᵢ², wₕ)`

ARGUMENT:
-`μ`: a floating-point number representing the mean of the initial distribution of the accumulator
-`σ`: a floating-point number representing the standard deviation of the initial value of the accumulator
-`𝛏`: a vector of floating-point numbers representing the values into which the accumulator is discretized

RETURN:
-`EH`: a four-by-four matrix whose columns and rows correspond to the second-order partial derivatives with respect to B, μ₀, σᵢ², and wₕ, in this order.

EXAMPLE
```julia-repo
Ξ = 53
B = 10.0
𝛏 = B.*(2.0.*collect(1:Ξ) .- Ξ .- 1)./(Ξ - 2)
μ = 0.5
σ = 0.8
i = 28
EH = Hessian(i, μ, σ, 𝛏)
```
"""
function Hessian(i::Integer,
				 μ::AbstractFloat,
				 σ::AbstractFloat,
				 𝛏::Vector{<:AbstractFloat})
    Ξ = length(𝛏)
    Ξ_1 = Ξ-1
	B = 𝛏[end]*(Ξ-2)/Ξ_1
    Δξ=𝛏[2]-𝛏[1]
    𝛚 = 𝛏./Δξ
	𝛑, ∂μ, ∂σ², ∂B = zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ)
	C, Δf, ΔΦ, 𝐟, Φ, 𝐳 = probabilityvector!(𝛑, ∂μ, ∂σ², ∂B, μ, 𝛚, σ, 𝛏)
	Δζ = diff(𝐟.*(𝐳.^2 .- 1.0)./4.0./σ.^3.0./Δξ)
	Δfωξ = diff(𝐟.*𝛚.*𝛏)
	Δfωz = diff(𝐟.*𝛚.*𝐳)
	Δfξ = diff(𝐟.*𝛏)
	Δfz = diff(𝐟.*𝐳)
	B²σ = B^2*σ
	BΔξσ = B*Δξ*σ
	Bσ²2 = B*σ^2*2
	Δξσ²2 = Δξ*σ^2*2
	EH = zeros(3,3)
	if i == 1
		∂B∂B = Δfωξ[1]/B²σ - 2∂B[1]/B
		∂B∂μ = -Δfξ[1]/BΔξσ - ∂μ[1]/B
		∂B∂σ² = -Δfωz[1]/Bσ²2 - ∂σ²[1]/B
		∂μ∂σ² = Δfz[1]/Δξσ²2
		∂σ²∂σ² = Δζ[1]
	elseif i < Ξ
		∂B∂B = (Δfωξ[i] - Δfωξ[i-1])/B²σ - 2∂B[i]/B
		∂B∂μ = (Δfξ[i-1]-Δfξ[i])/BΔξσ - ∂μ[i]/B
		∂B∂σ² = (Δfωz[i-1]-Δfωz[i])/Bσ²2 - ∂σ²[i]/B
		∂μ∂σ² = (Δfz[i]-Δfz[i-1])/Δξσ²2
		∂σ²∂σ² = Δζ[i] - Δζ[i-1]
	else
		∂B∂B = -Δfωξ[Ξ_1]/B²σ - 2∂B[Ξ]/B
		∂B∂μ = Δfξ[Ξ_1]/BΔξσ - ∂μ[Ξ]/B
		∂B∂σ² = Δfωz[Ξ_1]/Bσ²2 - ∂σ²[Ξ]/B
		∂μ∂σ² = -Δfz[Ξ_1]/Δξσ²2
		∂σ²∂σ² = -Δζ[Ξ_1]
	end
	EH[1,1] = ∂B∂B
	EH[1,2] = ∂B∂μ
	EH[1,3] = ∂B∂σ²
	EH[2,2] = ∂σ²[i]*2 #∂μ∂μ
	EH[2,3] = ∂μ∂σ²
	EH[3,3] = ∂σ²∂σ²
	EH[2,1] = EH[1,2]
	EH[3,1] = EH[1,3]
	EH[3,2] = EH[2,3]
	return EH
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
