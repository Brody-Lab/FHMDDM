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
	Δt = clicks.time[1]
    e⁻ᵏᵈᵗ = exp(-k*Δt)
    C[1] = 1.0 - (1.0-ϕ)*e⁻ᵏᵈᵗ
    dCdϕ[1] = e⁻ᵏᵈᵗ
    dCdk[1] = e⁻ᵏᵈᵗ*(1.0-ϕ)*Δt
    for i = 2:nclicks
        Δt = clicks.time[i] - clicks.time[i-1]
        e⁻ᵏᵈᵗ = exp(-k*Δt)
        C[i] = 1.0 - (1.0 - ϕ*C[i-1])*e⁻ᵏᵈᵗ
        dCdϕ[i] = e⁻ᵏᵈᵗ*(C[i-1] + ϕ*dCdϕ[i-1])
        dCdk[i] = e⁻ᵏᵈᵗ*(ϕ*dCdk[i-1] + Δt*(1.0-ϕ*C[i-1]))
    end
    return C, dCdk, dCdϕ
end

"""
    ∇∇adapt(clicks, k, ϕ)

Compute the adapted impact of each click in a trial as well as the first- and second-order partial derivatives

ARGUMENT
-`clicks`: structure containing information about the auditory clicks in one trial. Stereoclick excluded.
-`k`: exponential change of the adaptation dynamics
-`ϕ`: strength and sign of the adaptation (facilitation: ϕ > 0; depression: ϕ < 0)

RETURN
-`C`: adapted strengths of the clicks
-`dCdk`: first-order partial derivative of `C` with respect to `k`
-`dCdϕ`: first-order partial derivative of `C` with respect to `ϕ`
-`dCdkdk`: second-order partial derivative of `C` with respect to `k`
-`dCdkdϕ`: second-order partial derivative of `C` with respect to `k` and `ϕ`
-`dCdϕdϕ`: second-order partial derivative of `C` with respect to `ϕ` and `ϕ`

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> clicks = FHMDDM.sampleclicks(0.01, 40, 0.01, 100, 30; rng=MersenneTwister(1234))
julia> C, dCdk, dCdϕ, dCdkdk, dCdkdϕ, dCdϕdϕ = FHMDDM.∇∇adapt(clicks, 0.5, 0.8);
julia> dCdkdk[1]
	-0.0004489135110232355
```
"""
function ∇∇adapt(clicks::Clicks, k::Real, ϕ::Real)
	nclicks = length(clicks.time)
	@assert nclicks > 0
    C, dCdk, dCdϕ, dCdkdk, dCdkdϕ, dCdϕdϕ = zeros(nclicks), zeros(nclicks), zeros(nclicks), zeros(nclicks), zeros(nclicks), zeros(nclicks)
	Δt = clicks.time[1]
    e⁻ᵏᵈᵗ = exp(-k*Δt)
    C[1] = 1.0 - (1.0-ϕ)*e⁻ᵏᵈᵗ
    dCdϕ[1] = e⁻ᵏᵈᵗ
    dCdk[1] = e⁻ᵏᵈᵗ*(1.0-ϕ)*Δt
    dCdkdk[1] = -Δt*dCdk[1]
	dCdkdϕ[1] = -Δt*dCdϕ[1]
    for i = 2:nclicks
        Δt = clicks.time[i] - clicks.time[i-1]
        e⁻ᵏᵈᵗ = exp(-k*Δt)
        C[i] = 1.0 - (1.0 - ϕ*C[i-1])*e⁻ᵏᵈᵗ
        dCdϕ[i] = e⁻ᵏᵈᵗ*(C[i-1] + ϕ*dCdϕ[i-1])
        dCdk[i] = e⁻ᵏᵈᵗ*(ϕ*dCdk[i-1] + Δt*(1.0-ϕ*C[i-1]))
		dCdkdk[i] = -Δt*dCdk[i] + ϕ*e⁻ᵏᵈᵗ*(dCdkdk[i-1] - Δt*dCdk[i-1])
		dCdkdϕ[i] = -Δt*dCdϕ[i] + e⁻ᵏᵈᵗ*(dCdk[i-1] + ϕ*dCdkdϕ[i-1])
		dCdϕdϕ[i] = e⁻ᵏᵈᵗ*(2*dCdϕ[i-1] + ϕ*dCdϕdϕ[i-1])
    end
    return C, dCdk, dCdϕ, dCdkdk, dCdkdϕ, dCdϕdϕ
end

"""
	compareHessians(clicks, k, ϕ)

Compare the automatically differentiated and hand-coded second-order partial derivatives of the adapted click magnitude with respect to k and ϕ

INPUT
-`clicks`: a structure containing the times, sources, and time steps of the clicks in one trial
-`k`: change rate of adaptation
-`ϕ`: strength of adaptation

RETURN
-`maxabsdiff`: maximum absolute difference between the automatically computed and hand-coded Hessians of the adapated impacts
-`automatic_Hessians`: a vector of matrices whose i-th element is the automatically computed Hessian matrix of the adapted strength of the i-th click
-`handcoded_Hessians`: a vector of matrices whose i-th element is the hand-coded Hessian matrix of the adapted strength of the i-th click

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> clicks = FHMDDM.sampleclicks(0.01, 40, 0.01, 100, 30; rng=MersenneTwister(1234))
julia> maxabsdiff, automatic_Hessians, handcoded_Hessians = FHMDDM.compareHessians(clicks, 0.5, 0.8)
julia> maxabsdiff
	5.329070518200751e-15
```
"""
function compareHessians(clicks::Clicks, k::Real, ϕ::Real)
	C, dCdk, dCdϕ, dCdkdk, dCdkdϕ, dCdϕdϕ = FHMDDM.∇∇adapt(clicks, k, ϕ)
	x₀ = [k,ϕ]
	nclicks = length(clicks.time)
	automatic_Hessians, handcoded_Hessians = collect(zeros(2,2) for i=1:nclicks), collect(zeros(2,2) for i=1:nclicks)
	for i = 1:nclicks
		f(x) = adapt(clicks, x[1], x[2])[i]
		ForwardDiff.hessian!(automatic_Hessians[i], f, x₀)
		handcoded_Hessians[i][1,1] = dCdkdk[i]
		handcoded_Hessians[i][1,2] = handcoded_Hessians[i][2,1] = dCdkdϕ[i]
		handcoded_Hessians[i][2,2] = dCdϕdϕ[i]
	end
	maxabsdiff = maximum(map((x,y)->maximum(abs.(x.-y)), automatic_Hessians, handcoded_Hessians))
	return maxabsdiff, automatic_Hessians, handcoded_Hessians
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
-`𝛏`: conditional values of the accumulator variable in the previous time step

RETURN
-a vector whose j-th element represents the mean of the accumulator conditioned on the accumulator in the previous time step equal to 𝛏[j]
"""
function conditionedmean(Δc::Real, Δt::AbstractFloat, λ::Real, 𝛏::Vector{<:Real})
	dμ_dΔc = differentiate_μ_wrt_Δc(Δt, λ)
	exp(λ*Δt).*𝛏 .+ Δc*dμ_dΔc
end

"""
    conditionedmean(Δc, Δt, λ, ξ)

Mean of the Gaussian PDF of the accumulator variable conditioned on its value in the previous time step

ARGUMENT
-`Δc`: right input minus left input
-`Δt`: size of the time step
-`λ`: leak or instability
-`ξ`: conditional value of the accumulator variable in the previous time step

RETURN
-the mean of the accumulator conditioned on the accumulator in the previous time step equal to 𝛏[j]
"""
function conditionedmean(Δc::Real, Δt::AbstractFloat, λ::Real, ξ::Real)
	dμ_dΔc = differentiate_μ_wrt_Δc(Δt, λ)
	exp(λ*Δt)*ξ + Δc*dμ_dΔc
end

"""
	differentiate_μ_wrt_Δc(Δt, λ)

Partial derivative of the mean of the accumulator with respect to the auditory input

ARGUMENT
-`Δt`: size of the time step
-`λ`: feedback of the accumulator onto itself

RETURN
-the partial derivative
"""
function differentiate_μ_wrt_Δc(Δt::AbstractFloat, λ::Real)
	if abs(λ) > 1e-8
		λΔt = λ*Δt
		(exp(λΔt) - 1.0)/λΔt
	else
		1.0
	end
end

"""
	differentiate_μ_wrt_Δcλ(Δt, λ)

Third-order partial derivative of the mean of the accumulator with respect to the auditory input and λ

ARGUMENT
-`Δt`: size of the time step
-`λ`: feedback of the accumulator onto itself

RETURN
-the partial derivative
"""
function differentiate_μ_wrt_Δcλ(Δt::AbstractFloat, λ::Real)
	if abs(λ) > 1e-3
		λΔt = λ*Δt
		(exp(λΔt)*(λΔt-1.0)+1.0)/λΔt/λ
	else
		Δt/2
	end
end

"""
	differentiate_μ_wrt_Δcλλ(Δt, λ)

Third-order partial derivative of the mean of the accumulator with respect to the auditory input (once) and λ (twice)

ARGUMENT
-`Δt`: size of the time step
-`λ`: feedback of the accumulator onto itself

RETURN
-the partial derivative
"""
function differentiate_μ_wrt_Δcλλ(Δt::AbstractFloat, λ::Real)
	if abs(λ) > 1e-2
		λΔt = λ*Δt
		(exp(λΔt)*(λΔt^2 - 2λΔt + 2)-2)/λΔt/λ^2
	else
		Δt^2/3
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
						   𝛏::Vector{T}) where {T<:Real}
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
