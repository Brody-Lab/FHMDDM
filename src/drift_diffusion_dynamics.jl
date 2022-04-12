"""
    adapt(clicks, ϕ, k)

Compute the adapted input strength of auditory clicks.

Assumes that adaptation is across-stream: i.e., a click from either side is affected by preceding clicks from both sides.

ARGUMENT
-`clicks`: information on all the clicks in one trial. The stereoclick is excluded.
-`ϕ`: a parameter indicating whether each click is facilitated (ϕ>0) or depressed (ϕ<0) by preceding clicks.
-`k`: a parameter indicating the exponential change rate of the sensory adaptation. Must be in the range of k ∈ (0, ∞).For a fixed non-zero value of ϕ, a smaller k indicates that preceding clicks exert a greater effect.

RETURN
-a structure containing the adapted magnitude of each click
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
    Adaptedclicks(C=C)
end

"""
    ∇adapt(clicks, k, ϕ)

Adapt the clicks and compute the first-order partial derivative of the adapted strengths with respect to the parameters

ARGUMENT
-`clicks`: structure containing information about the auditory clicks in one trial. Stereoclick excluded.
-`k`: exponential change of the adaptation dynamics
-`ϕ`: strength and sign of the adaptation (facilitation: ϕ > 0; depression: ϕ < 0)

RETURN
-a structure containing the adapted magnitude of each click and its partial derivatives
"""
function ∇adapt(clicks::Clicks, k::T1, ϕ::T2) where {T1<:Real, T2<:Real}
	T = T1<:T2 ? T2 : T1
	nclicks = length(clicks.time)
	@assert nclicks > 0
    C, dC_dk, dC_dϕ = zeros(T, nclicks), zeros(T, nclicks), zeros(T, nclicks)
	Δt = clicks.time[1]
    e⁻ᵏᵈᵗ = exp(-k*Δt)
    C[1] = 1.0 - (1.0-ϕ)*e⁻ᵏᵈᵗ
    dC_dϕ[1] = e⁻ᵏᵈᵗ
    dC_dk[1] = e⁻ᵏᵈᵗ*(1.0-ϕ)*Δt
    for i = 2:nclicks
        Δt = clicks.time[i] - clicks.time[i-1]
        e⁻ᵏᵈᵗ = exp(-k*Δt)
        C[i] = 1.0 - (1.0 - ϕ*C[i-1])*e⁻ᵏᵈᵗ
        dC_dϕ[i] = e⁻ᵏᵈᵗ*(C[i-1] + ϕ*dC_dϕ[i-1])
        dC_dk[i] = e⁻ᵏᵈᵗ*(ϕ*dC_dk[i-1] + Δt*(1.0-ϕ*C[i-1]))
    end
    Adaptedclicks(C=C, dC_dk=dC_dk, dC_dϕ=dC_dϕ)
end

"""
    ∇∇adapt(clicks, k, ϕ)

Compute the adapted impact of each click in a trial as well as the first- and second-order partial derivatives

ARGUMENT
-`clicks`: structure containing information about the auditory clicks in one trial. Stereoclick excluded.
-`k`: exponential change of the adaptation dynamics
-`ϕ`: strength and sign of the adaptation (facilitation: ϕ > 0; depression: ϕ < 0)

RETURN
-a structure containing the adapted magnitude of each click and its first- and second-order partial derivatives

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> clicks = FHMDDM.sampleclicks(0.01, 40, 0.01, 100, 30; rng=MersenneTwister(1234));
julia> adaptedclicks = FHMDDM.∇∇adapt(clicks, 0.5, 0.8);
julia> adaptedclicks.d²C_dkdk[1]
	-0.0004489135110232355
```
"""
function ∇∇adapt(clicks::Clicks, k::Real, ϕ::Real)
	nclicks = length(clicks.time)
	@assert nclicks > 0
    C, dC_dk, dC_dϕ, d²C_dkdk, d²C_dkdϕ, d²C_dϕdϕ = zeros(nclicks), zeros(nclicks), zeros(nclicks), zeros(nclicks), zeros(nclicks), zeros(nclicks)
	Δt = clicks.time[1]
    e⁻ᵏᵈᵗ = exp(-k*Δt)
    C[1] = 1.0 - (1.0-ϕ)*e⁻ᵏᵈᵗ
    dC_dϕ[1] = e⁻ᵏᵈᵗ
    dC_dk[1] = e⁻ᵏᵈᵗ*(1.0-ϕ)*Δt
    d²C_dkdk[1] = -Δt*dC_dk[1]
	d²C_dkdϕ[1] = -Δt*dC_dϕ[1]
    for i = 2:nclicks
        Δt = clicks.time[i] - clicks.time[i-1]
        e⁻ᵏᵈᵗ = exp(-k*Δt)
        C[i] = 1.0 - (1.0 - ϕ*C[i-1])*e⁻ᵏᵈᵗ
        dC_dϕ[i] = e⁻ᵏᵈᵗ*(C[i-1] + ϕ*dC_dϕ[i-1])
        dC_dk[i] = e⁻ᵏᵈᵗ*(ϕ*dC_dk[i-1] + Δt*(1.0-ϕ*C[i-1]))
		d²C_dkdk[i] = -Δt*dC_dk[i] + ϕ*e⁻ᵏᵈᵗ*(d²C_dkdk[i-1] - Δt*dC_dk[i-1])
		d²C_dkdϕ[i] = -Δt*dC_dϕ[i] + e⁻ᵏᵈᵗ*(dC_dk[i-1] + ϕ*d²C_dkdϕ[i-1])
		d²C_dϕdϕ[i] = e⁻ᵏᵈᵗ*(2*dC_dϕ[i-1] + ϕ*d²C_dϕdϕ[i-1])
    end
    Adaptedclicks(C=C, dC_dk=dC_dk, dC_dϕ=dC_dϕ, d²C_dkdk=d²C_dkdk, d²C_dkdϕ=d²C_dkdϕ, d²C_dϕdϕ=d²C_dϕdϕ)
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
	adaptedclicks = FHMDDM.∇∇adapt(clicks, k, ϕ)
	x₀ = [k,ϕ]
	nclicks = length(clicks.time)
	automatic_Hessians, handcoded_Hessians = collect(zeros(2,2) for i=1:nclicks), collect(zeros(2,2) for i=1:nclicks)
	for i = 1:nclicks
		f(x) = adapt(clicks, x[1], x[2]).C[i]
		ForwardDiff.hessian!(automatic_Hessians[i], f, x₀)
		handcoded_Hessians[i][1,1] = adaptedclicks.d²C_dkdk[i]
		handcoded_Hessians[i][1,2] = handcoded_Hessians[i][2,1] = adaptedclicks.d²C_dkdϕ[i]
		handcoded_Hessians[i][2,2] = adaptedclicks.d²C_dϕdϕ[i]
	end
	maxabsdiff = maximum(map((x,y)->maximum(abs.(x.-y)), automatic_Hessians, handcoded_Hessians))
	return maxabsdiff, automatic_Hessians, handcoded_Hessians
end

"""
	Probabilityvector(Δt, θnative, Ξ)

Makes a struct that contains quantities for computing the prior or transition probabilities of the accumulator and the first- and second-order partial derivatives of these probabilities

Takes about 7 μs to construct the struct.

ARGUMENT
-`Δt`: size of the time step
-`θnative`: a struct containing the parameters specifying the prior and transition probabilities of the accumulator
-`Ξ`: number of values into which the accumulator is discretized

OUTPUT
-an instance of the type `Probabilityvector`

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat");
julia> P = FHMDDM.Probabilityvector(model.options.Δt, model.θnative, model.options.Ξ)
"""
function Probabilityvector(Δt::Real, θnative::Latentθ, Ξ::Integer)
	Probabilityvector(Δt=Δt, Ξ=Ξ, B=θnative.B[1], k=θnative.k[1], λ=θnative.λ[1], μ₀=θnative.μ₀[1], ϕ=θnative.ϕ[1], σ²ₐ=θnative.σ²ₐ[1], σ²ᵢ=θnative.σ²ᵢ[1], σ²ₛ=θnative.σ²ₛ[1], wₕ=θnative.wₕ[1])
end

"""
	∇∇transitionmatrix!(∇∇A, ∇A, A, P)

Computes the second derivatives of the accumulator's transition at one time step

The gradient of the transition probabilities in the transition matrix, as well as the transition matrix itself, are also computed.

MODIFIED ARGUMENT
-`∇∇A`: Hessian of each transition probability of the accumulator. The element `∇∇A[m,n][i,j]` corresponds to `∂²p{a(t) = ξ(i) ∣ a(t-1) = ξ(j)} / ∂θ[m]∂θ[n]`. The parameters are order alphabetically:
	θ[1] = B, bound height
	θ[2] = k, adaptation change rate
	θ[3] = λ, feedback
	θ[4] = σ²ₐ, variance of diffusion noise
	θ[5] = σ²ₛ, variance of per-click noise
	θ[6] = ϕ, adaptation strength
-`∇A`: Gradient of each transition probability of the accumulator. The element `∇A[m][i,j]` corresponds to `∂p{a(t) = ξ(i) ∣ a(t-1) = ξ(j)} / ∂θ[m]`
-`A`: Transition matrix of the accumulator. The element `A[i,j]` corresponds to `p{a(t) = ξ(i) ∣ a(t-1) = ξ(j)}`
-'P': a structure containing the first and second partial derivatives of a probability vector of the accumulator and quantities used for computing these derivatives

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat");
julia> Ξ = model.options.Ξ
julia> P = FHMDDM.Probabilityvector(model.options.Δt, model.θnative, Ξ);
julia> ∇∇A = map(i->zeros(Ξ,Ξ), CartesianIndices((6,6)));
julia> ∇A = map(i->zeros(Ξ,Ξ), 1:6);
julia> A = zeros(Ξ,Ξ);
julia> A[1,1] = A[Ξ, Ξ] = 1.0;
julia> clicks = model.trialsets[1].trials[1].clicks;
julia> adaptedclicks = FHMDDM.∇∇adapt(clicks, model.θnative.k[1], model.θnative.ϕ[1]);
julia> t = 3
julia> FHMDDM.update_for_∇∇transition_probabilities!(P, adaptedclicks, clicks, t)
julia> FHMDDM.∇∇transitionmatrix!(∇∇A, ∇A, A, P)
```
"""
function ∇∇transitionmatrix!(∇∇A::Matrix{<:Matrix{<:Real}},
							 ∇A::Vector{<:Matrix{<:Real}},
							 A::Matrix{<:Real},
							 P::Probabilityvector)
	for j = 2:P.Ξ-1
		differentiate_twice_wrt_Bμσ²!(P, j)
		differentiate_twice_wrt_transition_parameters!(P, j)
		differentiate_wrt_transition_parameters!(P,j)
		assign!(∇∇A, P, j)
		assign!(∇A, P, j)
		assign!(A, P, j)
	end
	return nothing
end

"""
	∇transitionmatrix!(∇A, A, P)

Computes the first derivatives of the accumulator's transition at one time step

The transition matrix itself is also computed.

MODIFIED ARGUMENT
-`∇A`: Gradient of each transition probability of the accumulator. The element `∇A[m][i,j]` corresponds to `∂p{a(t) = ξ(i) ∣ a(t-1) = ξ(j)} / ∂θ[m]`. The parameters are order alphabetically:
	θ[1] = B, bound height
	θ[2] = k, adaptation change rate
	θ[3] = λ, feedback
	θ[4] = σ²ₐ, variance of diffusion noise
	θ[5] = σ²ₛ, variance of per-click noise
	θ[6] = ϕ, adaptation strength
-`A`: Transition matrix of the accumulator. The element `A[i,j]` corresponds to `p{a(t) = ξ(i) ∣ a(t-1) = ξ(j)}`
-'P': a structure containing the first and second partial derivatives of a probability vector of the accumulator and quantities used for computing these derivatives
"""
function ∇transitionmatrix!(∇A::Vector{<:Matrix{<:Real}},
							A::Matrix{<:Real},
							P::Probabilityvector)
	for j = 2:P.Ξ-1
		differentiate_wrt_Bμσ²!(P, j)
		differentiate_wrt_transition_parameters!(P, j)
		assign!(∇A, P, j)
		assign!(A, P, j)
	end
	return nothing
end

"""
	transitionmatrix!(A, P)

Computes the the accumulator's transition matrix at one time step

MODIFIED ARGUMENT
-`A`: Transition matrix of the accumulator. The element `A[i,j]` corresponds to `p{a(t) = ξ(i) ∣ a(t-1) = ξ(j)}`
-'P': a structure containing the first and second partial derivatives of a probability vector of the accumulator and quantities used for computing these derivatives

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat");
julia> P = FHMDDM.Probabilityvector(model.options.Δt, model.θnative, model.options.Ξ);
julia> A = zeros(P.Ξ,P.Ξ);
julia> A[1,1] = A[P.Ξ, P.Ξ] = 1.0;
julia> clicks = model.trialsets[1].trials[1].clicks;
julia> adaptedclicks = FHMDDM.∇∇adapt(clicks, model.θnative.k[1], model.θnative.ϕ[1]);
julia> t = 3
julia> FHMDDM.update_for_transition_probabilities!(P, adaptedclicks, clicks, t)
julia> FHMDDM.transitionmatrix!(A, P)
julia> Aapprox = zeros(P.Ξ,P.Ξ);
julia> FHMDDM.approximatetransition!(Aapprox, P.Δt, P.Δξ, model.θnative.λ[1], P.Δc[1], P.Ξ, P.σ²[1], P.𝛏)
julia> maximum(abs.(A .- Aapprox))
```
"""
function transitionmatrix!(A::Matrix{<:Real},
						   P::Probabilityvector)
	for j = 2:P.Ξ-1
		evaluate_using_Bμσ²!(P, j)
		assign!(A, P, j)
	end
	return nothing
end

"""
	assign!(∇∇A, P, j)

Assign second derivatives of a probability vector to elements in a nested array corresponding to the second derivatives of the j-th column of the transition matrix

MODIFIED ARGUMENT
-`∇∇A`: a nested array representing the second-order partial derivatives of each transition probability of the accumulator. The element `∇∇A[m,n][i,j]` corresponds to `∂²p{a(t) = ξ(i) ∣ a(t-1) = ξ(j)} / ∂θ[m]∂θ[n]`. The parameters are order alphabetically:
	θ[1] = B, bound height
	θ[2] = k, adaptation change rate
	θ[3] = λ, feedback
	θ[4] = σ²ₐ, variance of diffusion noise
	θ[5] = σ²ₛ, variance of per-click noise
	θ[6] = ϕ, adaptation strength

UNMODIFIED ARGUMENT
-`P`: structure containing second-order partial derivatives
-`j`: column of the transition matrix
"""
function assign!(∇∇A::Matrix{<:Matrix{<:Real}},
				 P::Probabilityvector,
			     j::Integer)
	for i = 1:P.Ξ
		∇∇A[1,1][i,j] = P.d²𝛑_dBdB[i]
		∇∇A[1,2][i,j] = P.d²𝛑_dBdk[i]
		∇∇A[1,3][i,j] = P.d²𝛑_dBdλ[i]
		∇∇A[1,4][i,j] = P.d²𝛑_dBdϕ[i]
		∇∇A[1,5][i,j] = P.d²𝛑_dBdσ²ₐ[i]
		∇∇A[1,6][i,j] = P.d²𝛑_dBdσ²ₛ[i]
		∇∇A[2,2][i,j] = P.d²𝛑_dkdk[i]
		∇∇A[2,3][i,j] = P.d²𝛑_dkdλ[i]
		∇∇A[2,4][i,j] = P.d²𝛑_dkdϕ[i]
		∇∇A[2,5][i,j] = P.d²𝛑_dkdσ²ₐ[i]
		∇∇A[2,6][i,j] = P.d²𝛑_dkdσ²ₛ[i]
		∇∇A[3,3][i,j] = P.d²𝛑_dλdλ[i]
		∇∇A[3,4][i,j] = P.d²𝛑_dλdϕ[i]
		∇∇A[3,5][i,j] = P.d²𝛑_dλdσ²ₐ[i]
		∇∇A[3,6][i,j] = P.d²𝛑_dλdσ²ₛ[i]
		∇∇A[4,4][i,j] = P.d²𝛑_dϕdϕ[i]
		∇∇A[4,5][i,j] = P.d²𝛑_dϕdσ²ₐ[i]
		∇∇A[4,6][i,j] = P.d²𝛑_dϕdσ²ₛ[i]
		∇∇A[5,5][i,j] = P.d²𝛑_dσ²ₐdσ²ₐ[i]
		∇∇A[5,6][i,j] = P.d²𝛑_dσ²ₐdσ²ₛ[i]
		∇∇A[6,6][i,j] = P.d²𝛑_dσ²ₛdσ²ₛ[i]
	end
end

"""
	assign!(∇A, P, j)

Assign first derivatives of a probability vector to elements in a nested array corresponding to the first derivatives of the j-th column of the transition matrix

MODIFIED ARGUMENT
-`∇A`: a nested array representing the first-order partial derivatives of each transition probability of the accumulator. The element `∇∇A[m][i,j]` corresponds to `∂p{a(t) = ξ(i) ∣ a(t-1) = ξ(j)} / ∂θ[m]`. The parameters are order alphabetically:
	θ[1] = B, bound height
	θ[2] = k, adaptation change rate
	θ[3] = λ, feedback
	θ[4] = σ²ₐ, variance of diffusion noise
	θ[5] = σ²ₛ, variance of per-click noise
	θ[6] = ϕ, adaptation strength

UNMODIFIED ARGUMENT
-`P`: structure containing second-order partial derivatives
-`j`: column of the transition matrix
"""
function assign!(∇A::Vector{<:Matrix{<:Real}},
				 P::Probabilityvector,
		 		 j::Integer)
	for i = 1:P.Ξ
		∇A[1][i,j] = P.d𝛑_dB[i]
		∇A[2][i,j] = P.d𝛑_dk[i]
		∇A[3][i,j] = P.d𝛑_dλ[i]
		∇A[4][i,j] = P.d𝛑_dϕ[i]
		∇A[5][i,j] = P.d𝛑_dσ²ₐ[i]
		∇A[6][i,j] = P.d𝛑_dσ²ₛ[i]
	end
	return nothing
end

"""
	assign!(A, P, j)

Assign elements of probability vector to elements in a matrix corresponding to the j-th column of the transition matrix

MODIFIED ARGUMENT
-`A`: transition matrix of the accumulator.

UNMODIFIED ARGUMENT
-`P`: structure containing the probability vector
-`j`: column of the transition matrix
"""
function assign!(A::Matrix{<:Real},
				 P::Probabilityvector,
		 		 j::Integer)
	for i = 1:P.Ξ
		A[i,j] = P.𝛑[i]
	end
	return nothing
end

"""
	assign!(∇∇𝛑, P)

Assign second derivatives of a probability vector to elements in a nested array corresponding to the second derivatives of the prior probability

MODIFIED ARGUMENT
-`∇∇𝛑`: a nested array representing the second-order partial derivatives of the prior probability of the accumulator. The element `∇∇𝛑[m,n][i]` corresponds to `∂²p{a(t=1) = ξ(i)} / ∂θ[m]∂θ[n]`. The parameters are order alphabetically:
	θ[1] = B, bound height
	θ[2] = μ₀, additive offset to the mean that is constant across trials
	θ[3] = σ²ᵢ, variance
	θ[4] = wₕ, weight of the location of the previous reward

UNMODIFIED ARGUMENT
-`P`: structure containing second-order partial derivatives
"""
function assign!(∇∇𝛑::Matrix{<:Vector{<:Real}},
				 P::Probabilityvector)
	for i = 1:P.Ξ
		∇∇𝛑[1,1][i] = P.d²𝛑_dBdB[i]
		∇∇𝛑[1,2][i] = P.d²𝛑_dBdμ₀[i]
		∇∇𝛑[1,3][i] = P.d²𝛑_dBdσ²ᵢ[i]
		∇∇𝛑[1,4][i] = P.d²𝛑_dBdwₕ[i]
		∇∇𝛑[2,2][i] = P.d²𝛑_dμ₀dμ₀[i]
		∇∇𝛑[2,3][i] = P.d²𝛑_dμ₀dσ²ᵢ[i]
		∇∇𝛑[2,4][i] = P.d²𝛑_dμ₀dwₕ[i]
		∇∇𝛑[3,3][i] = P.d²𝛑_dσ²ᵢdσ²ᵢ[i]
		∇∇𝛑[3,4][i] = P.d²𝛑_dσ²ᵢdwₕ[i]
		∇∇𝛑[4,4][i] = P.d²𝛑_dwₕdwₕ[i]
	end
end

"""
	assign!(∇𝛑, P)

Assign first-order derivatives of a probability vector to elements in a nested array corresponding to the first-order derivatives of the prior probability

MODIFIED ARGUMENT
-`∇𝛑`: a nested array representing the second-order partial derivatives of the prior probability of the accumulator. The element `∇𝛑[m][i]` corresponds to `∂p{a(t=1) = ξ(i)} / ∂θ[m]`. The parameters are order alphabetically:
	θ[1] = B, bound height
	θ[2] = μ₀, additive offset to the mean that is constant across trials
	θ[3] = σ²ᵢ, variance
	θ[4] = wₕ, weight of the location of the previous reward

UNMODIFIED ARGUMENT
-`P`: structure containing second-order partial derivatives
"""
function assign!(∇𝛑::Vector{<:Vector{<:Real}},
				 P::Probabilityvector)
	for i = 1:P.Ξ
		∇𝛑[1][i] = P.d𝛑_dB[i]
		∇𝛑[2][i] = P.d𝛑_dμ₀[i]
		∇𝛑[3][i] = P.d𝛑_dσ²ᵢ[i]
		∇𝛑[4][i] = P.d𝛑_dwₕ[i]
	end
end

"""
	update_for_second_derivatives!(P, adaptedclicks, clicks, t)

Compute the intermediate quantities that are updated at each time step for obtaining the second order partial derivatives of a probability vector

Refer to the definition of the types 'Adaptedclicks` and  `Probabilityvector` in `types.jl` for the meaning of each term

MODIFIED ARGUMENT
-`P`: structure containing derivatives with respect to the parameters of the accumulator

UNMODIFIED ARGUMENT
-`clicks`: structure containing information about the auditory clicks in one trial. Stereoclick excluded.
-`adaptedclicks': structure containing the adapted magnitude of each click and the first- and second-order partial derivatives of the adapted magnitude
-`t`: time step

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat");
julia> P = FHMDDM.Probabilityvector(model.options.Δt, model.θnative, model.options.Ξ);
julia> clicks = model.trialsets[1].trials[1].clicks;
julia> adaptedclicks = FHMDDM.∇∇adapt(clicks, model.θnative.k[1], model.θnative.ϕ[1]);
julia> FHMDDM.update_for_∇∇transition_probabilities!(P, adaptedclicks, clicks, 3);
```
"""
function update_for_∇∇transition_probabilities!(P::Probabilityvector,
											    adaptedclicks::Adaptedclicks,
											    clicks::Clicks,
											    t::Integer)
	update_for_∇transition_probabilities!(P, adaptedclicks, clicks, t)
	d²cR_dkdk = sum(adaptedclicks.d²C_dkdk[clicks.right[t]])
	d²cL_dkdk = sum(adaptedclicks.d²C_dkdk[clicks.left[t]])
	d²cR_dkdϕ = sum(adaptedclicks.d²C_dkdϕ[clicks.right[t]])
	d²cL_dkdϕ = sum(adaptedclicks.d²C_dkdϕ[clicks.left[t]])
	d²cR_dϕdϕ = sum(adaptedclicks.d²C_dϕdϕ[clicks.right[t]])
	d²cL_dϕdϕ = sum(adaptedclicks.d²C_dϕdϕ[clicks.left[t]])
	P.d²Δc_dkdk[1] = d²cR_dkdk - d²cL_dkdk
	P.d²∑c_dkdk[1] = d²cR_dkdk + d²cL_dkdk
	P.d²Δc_dkdϕ[1] = d²cR_dkdϕ - d²cL_dkdϕ
	P.d²∑c_dkdϕ[1] = d²cR_dkdϕ + d²cL_dkdϕ
	P.d²Δc_dϕdϕ[1] = d²cR_dϕdϕ - d²cL_dϕdϕ
	P.d²∑c_dϕdϕ[1] = d²cR_dϕdϕ + d²cL_dϕdϕ
	P.Δξσ²2[1] = 2P.Δξ*P.σ[1]^2
	P.d²μ_dkdk[1] = P.d²Δc_dkdk[1]*P.dμ_dΔc
	P.d²μ_dkdϕ[1] = P.d²Δc_dkdϕ[1]*P.dμ_dΔc
	P.d²μ_dϕdϕ[1] = P.d²Δc_dϕdϕ[1]*P.dμ_dΔc
	P.d²σ²_dkdk[1] = P.d²∑c_dkdk[1]*P.dσ²_d∑c
	P.d²σ²_dkdϕ[1] = P.d²∑c_dkdϕ[1]*P.dσ²_d∑c
	P.d²σ²_dϕdϕ[1] = P.d²∑c_dϕdϕ[1]*P.dσ²_d∑c
	P.d²𝛍_dλdλ .= P.Δt^2 .* P.expλΔt .* P.𝛏 .+ P.Δc[1]*P.d³μ_dΔcdλdλ
	return nothing
end

"""
	update_for_∇∇transition_probabilities!(P)

Compute the intermediate quantities that are updated for obtaining the second order partial derivatives of a probability vector, at a time step when no click occured

MODIFIED ARGUMENT
-`P`: structure containing derivatives with respect to the parameters of the accumulator
"""
function update_for_∇∇transition_probabilities!(P::Probabilityvector)
	update_for_∇transition_probabilities!(P)
	P.d²Δc_dkdk[1] = 0.0
	P.d²∑c_dkdk[1] = 0.0
	P.d²Δc_dkdϕ[1] = 0.0
	P.d²∑c_dkdϕ[1] = 0.0
	P.d²Δc_dϕdϕ[1] = 0.0
	P.d²∑c_dϕdϕ[1] = 0.0
	P.Δξσ²2[1] = 2P.Δξ*P.σ[1]^2
	P.d²μ_dkdk[1] = 0.0
	P.d²μ_dkdϕ[1] = 0.0
	P.d²μ_dϕdϕ[1] = 0.0
	P.d²σ²_dkdk[1] = 0.0
	P.d²σ²_dkdϕ[1] = 0.0
	P.d²σ²_dϕdϕ[1] = 0.0
	P.d²𝛍_dλdλ .= P.Δt^2 .* P.expλΔt .* P.𝛏 .+ P.Δc[1]*P.d³μ_dΔcdλdλ
end

"""
	update_for_∇transition_probabilities!(P, adaptedclicks, clicks, t)

Compute the intermediate quantities that are updated at each time step for obtaining the first order partial derivatives of a probability vector

Refer to the definition of the types 'Adaptedclicks` and  `Probabilityvector` in `types.jl` for the meaning of each term

MODIFIED ARGUMENT
-`P`: structure containing derivatives with respect to the parameters of the accumulator

UNMODIFIED ARGUMENT
-`clicks`: structure containing information about the auditory clicks in one trial. Stereoclick excluded.
-`adaptedclicks': structure containing the adapted magnitude of each click and the first- and second-order partial derivatives of the adapted magnitude
-`t`: time step
"""
function update_for_∇transition_probabilities!(P::Probabilityvector,
									   		  adaptedclicks::Adaptedclicks,
											  clicks::Clicks,
											  t::Integer)
	update_for_transition_probabilities!(P, adaptedclicks, clicks, t)
	dcR_dk = sum(adaptedclicks.dC_dk[clicks.right[t]])
	dcL_dk = sum(adaptedclicks.dC_dk[clicks.left[t]])
	dcR_dϕ = sum(adaptedclicks.dC_dϕ[clicks.right[t]])
	dcL_dϕ = sum(adaptedclicks.dC_dϕ[clicks.left[t]])
	P.dΔc_dk[1] = dcR_dk - dcL_dk
	P.d∑c_dk[1] = dcR_dk + dcL_dk
	P.dΔc_dϕ[1] = dcR_dϕ - dcL_dϕ
	P.d∑c_dϕ[1] = dcR_dϕ + dcL_dϕ
	P.σ2Δξ[1] = 2*P.σ[1]*P.Δξ[1]
	P.dμ_dk[1] = P.dΔc_dk[1]*P.dμ_dΔc
	P.dμ_dϕ[1] = P.dΔc_dϕ[1]*P.dμ_dΔc
	P.dσ²_dk[1] = P.d∑c_dk[1]*P.dσ²_d∑c
	P.dσ²_dϕ[1] = P.d∑c_dϕ[1]*P.dσ²_d∑c
	P.d𝛍_dλ .= P.Δt .* P.expλΔt .* P.𝛏 .+ P.Δc[1]*P.d²μ_dΔcdλ
	return nothing
end

"""
	update_for_∇transition_probabilities!(P, adaptedclicks, clicks, t)

Compute the intermediate quantities that are updated for obtaining the first order partial derivatives of a probability vector, at a time step when no click occured

MODIFIED ARGUMENT
-`P`: structure containing derivatives with respect to the parameters of the accumulator
"""
function update_for_∇transition_probabilities!(P::Probabilityvector)
	update_for_transition_probabilities!(P)
	P.dΔc_dk[1] = 0.0
	P.d∑c_dk[1] = 0.0
	P.dΔc_dϕ[1] = 0.0
	P.d∑c_dϕ[1] = 0.0
	P.σ2Δξ[1] = 2*P.σ[1]*P.Δξ[1]
	P.dμ_dk[1] = 0.0
	P.dμ_dϕ[1] = 0.0
	P.dσ²_dk[1] = 0.0
	P.dσ²_dϕ[1] = 0.0
	P.d𝛍_dλ .= P.Δt .* P.expλΔt .* P.𝛏
	return nothing
end

"""
	update_for_transition_probabilities!(P, adaptedclicks, clicks, t)

Compute the intermediate quantities that are updated at each time step for obtaining the values of a probability vector

Refer to the definition of the types 'Adaptedclicks` and  `Probabilityvector` in `types.jl` for the meaning of each term

MODIFIED ARGUMENT
-`P`: structure containing derivatives with respect to the parameters of the accumulator

UNMODIFIED ARGUMENT
-`clicks`: structure containing information about the auditory clicks in one trial. Stereoclick excluded.
-`adaptedclicks': structure containing the adapted magnitude of each click and the first- and second-order partial derivatives of the adapted magnitude
-`t`: time step
"""
function update_for_transition_probabilities!(P::Probabilityvector,
								   			 adaptedclicks::Adaptedclicks,
								   			 clicks::Clicks,
								   			 t::Integer)
	cL = sum(adaptedclicks.C[clicks.left[t]])
	cR = sum(adaptedclicks.C[clicks.right[t]])
	P.Δc[1] = cR-cL
	P.∑c[1] = cR+cL
	P.σ²[1] = P.∑c[1]*P.σ²ₛ + P.Δt*P.σ²ₐ
	P.σ[1] = √P.σ²[1]
	P.σ_Δξ[1] = P.σ[1]/P.Δξ[1]
	P.𝛍 .= P.expλΔt.*P.𝛏 .+ P.Δc[1]*P.dμ_dΔc
	return nothing
end

"""
	update_for_transition_probabilities!(P)

Compute the intermediate quantities that are updated for obtaining the values of a probability vector, at time step when no click occured

MODIFIED ARGUMENT
-`P`: structure containing derivatives with respect to the parameters of the accumulator
"""
function update_for_transition_probabilities!(P::Probabilityvector)
	P.Δc[1] = 0.0
	P.∑c[1] = 0.0
	P.σ²[1] = P.Δt*P.σ²ₐ
	P.σ[1] = √P.σ²[1]
	P.σ_Δξ[1] = P.σ[1]/P.Δξ[1]
	P.𝛍 .= P.expλΔt.*P.𝛏
	return nothing
end

"""
	∇∇priorprobability(∇∇𝛑, ∇𝛑, P, previousanswer)

Compute the second-order partial derivatives of the prior probability vector

MODIFIED ARGUMENT
-`∇∇𝛑`: a nested array representing the second-order partial derivatives of the prior probability of the accumulator. The element `∇∇𝛑[m,n][i]` corresponds to `∂²p{a(t=1) = ξ(i)} / ∂θ[m]∂θ[n]`. The parameters are order alphabetically:
	θ[1] = B, bound height
	θ[2] = μ₀, additive offset to the mean that is constant across trials
	θ[3] = σ²ᵢ, variance
	θ[4] = wₕ, weight of the location of the previous reward
-`∇𝛑`: a nested array representing the second-order partial derivatives of the prior probability of the accumulator. The element `∇𝛑[m][i]` corresponds to `∂p{a(t=1) = ξ(i)} / ∂θ[m]`.
-`P`: structure containing quantities for computing the prior and transition probabilities of the accumulator variable and the first- and second-order derivatives of these probabilities

UNMODIFIED ARGUMENT
-`previousanswer`: location of the reward in the previous trial

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat");
julia> P = FHMDDM.Probabilityvector(model.options.Δt, model.θnative, model.options.Ξ);
julia> ∇∇𝛑 = map(i->zeros(P.Ξ), CartesianIndices((4,4)))
julia> ∇𝛑 = map(i->zeros(P.Ξ), 1:4)
julia> FHMDDM.∇∇priorprobability!(∇∇𝛑, ∇𝛑, P, -1)
```
"""
function ∇∇priorprobability!(∇∇𝛑::Matrix{<:Vector{<:Real}}, ∇𝛑::Vector{<:Vector{<:Real}}, P::Probabilityvector, previousanswer::Integer)
	update_for_prior_probabilities!(P, previousanswer)
	differentiate_twice_wrt_Bμσ²!(P, cld(P.Ξ,2))
	differentiate_twice_wrt_prior_parameters!(P)
	assign!(∇∇𝛑, P)
	assign!(∇𝛑, P)
end

"""
	∇priorprobability!(∇𝛑, P, previousanswer)

Compute the first-order partial derivatives of the prior probability vector

MODIFIED ARGUMENT
-`∇𝛑`: a nested array representing the second-order partial derivatives of the prior probability of the accumulator. The element `∇𝛑[m][i]` corresponds to `∂p{a(t=1) = ξ(i)} / ∂θ[m]`.The parameters are order alphabetically:
	θ[1] = B, bound height
	θ[2] = μ₀, additive offset to the mean that is constant across trials
	θ[3] = σ²ᵢ, variance
	θ[4] = wₕ, weight of the location of the previous reward
-`P`: structure containing quantities for computing the prior and transition probabilities of the accumulator variable and the first- and second-order derivatives of these probabilities

UNMODIFIED ARGUMENT
-`previousanswer`: location of the reward in the previous trial
"""
function ∇priorprobability!(∇𝛑::Vector{<:Vector{<:Real}}, P::Probabilityvector, previousanswer::Integer)
	update_for_prior_probabilities!(P, previousanswer)
	differentiate_wrt_Bμσ²!(P, cld(P.Ξ,2))
	differentiate_wrt_prior_parameters!(P)
	assign!(∇𝛑, P)
end

"""
	priorprobability(P, previousanswer)

Compute the prior probability vector

MODIFIED ARGUMENT
-`P`: structure containing quantities for computing the prior and transition probabilities of the accumulator variable and the first- and second-order derivatives of these probabilities

UNMODIFIED ARGUMENT
-`previousanswer`: location of the reward in the previous trial
"""
function priorprobability!(P::Probabilityvector, previousanswer::Integer)
	update_for_prior_probabilities!(P, previousanswer)
	evaluate_using_Bμσ²!(P, cld(P.Ξ,2))
end

"""
	update_for_prior_probabilities!(P, previousanswer)

Compute the mean, variance, and

MODIFIED ARGUMENT
-`P`: structure containing quantities for computing the prior and transition probabilities of the accumulator variable and the first- and second-order derivatives of these probabilities

UNMODIFIED ARGUMENT
-`previousanswer`: location of the reward in the previous trial
"""
function update_for_prior_probabilities!(P::Probabilityvector, previousanswer::Integer)
	P.previousanswer[1] = previousanswer
	P.𝛍[cld(P.Ξ,2)] = P.μ₀ + P.wₕ*P.previousanswer[1]
	P.σ²[1] = P.σ²ᵢ
	P.σ[1] = √P.σ²[1]
	P.σ_Δξ[1] = P.σ[1]/P.Δξ[1]
	P.σ2Δξ[1] = 2*P.σ[1]*P.Δξ[1]
	P.Δξσ²2[1] = 2P.Δξ*P.σ[1]^2
end

"""
	differentiate_twice_wrt_prior_parameters!(P)

Compute the second- (and first-) order partial derivatives of the prior probabilities of the accumulator

MODIFIED ARGUMENT
-`P`: a structure containing first and second order partial derivatives of a probability vector of the accumulator variable. The probability vector represents the j-th column of the transition matrix: `p(𝐚(t) ∣ a(t-1) = j)`
"""
function differentiate_twice_wrt_prior_parameters!(P::Probabilityvector)
	differentiate_wrt_prior_parameters!(P)
	for i = 1:P.Ξ
		P.d𝛑_dwₕ[i] = P.previousanswer[1]*P.d𝛑_dμ[i]
		P.d²𝛑_dBdwₕ[i] = P.previousanswer[1]*P.d²𝛑_dBdμ[i]
		P.d²𝛑_dμ₀dwₕ[i] = P.previousanswer[1]*P.d²𝛑_dμdμ[i]
		P.d²𝛑_dσ²ᵢdwₕ[i] = P.previousanswer[1]*P.d²𝛑_dμdσ²[i]
		P.d²𝛑_dwₕdwₕ[i] = P.previousanswer[1]^2*P.d²𝛑_dμdμ[i]
	end
	return nothing
end

"""
	differentiate_wrt_prior_parameters!(P)

Compute the first-order partial derivatives of the prior probabilities of the accumulator

MODIFIED ARGUMENT
-`P`: a structure containing first and second order partial derivatives of a probability vector of the accumulator variable. The probability vector represents the j-th column of the transition matrix: `p(𝐚(t) ∣ a(t-1) = j)`
"""
function differentiate_wrt_prior_parameters!(P::Probabilityvector)
	for i = 1:P.Ξ
		P.d𝛑_dwₕ[i] = P.previousanswer[1]*P.d𝛑_dμ[i]
	end
	return nothing
end

"""
	differentiate_twice_wrt_transition_parameters!(P,j)

Compute the second- (and first-) order partial derivatives of the j-th column of the transition matrix

MODIFIED ARGUMENT
-`P`: a structure containing first and second order partial derivatives of a probability vector of the accumulator variable. The probability vector represents the j-th column of the transition matrix: `p(𝐚(t) ∣ a(t-1) = j)`

UNMODIFIED ARGUMENT
-`j`: the index of the state of the accumulator variable in the previous time step

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat");
julia> P = FHMDDM.Probabilityvector(model.options.Δt, model.θnative, model.options.Ξ);
julia> clicks = model.trialsets[1].trials[1].clicks;
julia> adaptedclicks = FHMDDM.∇∇adapt(clicks, model.θnative.k[1], model.θnative.ϕ[1]);
julia> FHMDDM.update_for_∇∇transition_probabilities!(P, adaptedclicks, clicks, 3)
julia> FHMDDM.differentiate_twice_wrt_Bμσ²!(P, 2)
julia> FHMDDM.differentiate_twice_wrt_transition_parameters!(P,2)
```
"""
function differentiate_twice_wrt_transition_parameters!(P::Probabilityvector, j::Integer)
	for i = 1:P.Ξ
		P.d²𝛑_dBdk[i] = P.dμ_dk[1]*P.d²𝛑_dBdμ[i] + P.dσ²_dk[1]*P.d²𝛑_dBdσ²[i]
		P.d²𝛑_dBdλ[i] = P.d²𝛍_dBdλ[j]*P.d𝛑_dμ[i] + P.d𝛍_dλ[j]*P.d²𝛑_dBdμ[i]
		P.d²𝛑_dBdϕ[i] = P.dμ_dϕ[1]*P.d²𝛑_dBdμ[i] + P.dσ²_dϕ[1]*P.d²𝛑_dBdσ²[i]
		P.d²𝛑_dBdσ²ₐ[i] = P.Δt*P.d²𝛑_dBdσ²[i]
		P.d²𝛑_dBdσ²ₛ[i] = P.∑c[1]*P.d²𝛑_dBdσ²[i]
		P.d²𝛑_dkdk[i] = P.d²μ_dkdk[1]*P.d𝛑_dμ[i] + P.d²σ²_dkdk[1]*P.d𝛑_dσ²[i] + P.dμ_dk[1]^2*P.d²𝛑_dμdμ[i] + 2P.dμ_dk[1]*P.dσ²_dk[1]*P.d²𝛑_dμdσ²[i] + P.dσ²_dk[1]^2*P.d²𝛑_dσ²dσ²[i]
		d²πᵢ_dλdΔc = P.d²μ_dΔcdλ*P.d𝛑_dμ[i] + P.dμ_dΔc[1]*P.d𝛍_dλ[j]*P.d²𝛑_dμdμ[i]
		d²πᵢ_dλd∑c = P.dσ²_d∑c[1]*P.d𝛍_dλ[j]*P.d²𝛑_dμdσ²[i]
		P.d²𝛑_dkdλ[i] = P.dΔc_dk[1]*d²πᵢ_dλdΔc + P.d∑c_dk[1]*d²πᵢ_dλd∑c
		P.d²𝛑_dkdϕ[i] = P.d²μ_dkdϕ[1]*P.d𝛑_dμ[i] + P.d²σ²_dkdϕ[1]*P.d𝛑_dσ²[i] + P.dμ_dk[1]*P.dμ_dϕ[1]*P.d²𝛑_dμdμ[i] + P.d²𝛑_dμdσ²[i]*(P.dμ_dϕ[1]*P.dσ²_dk[1] + P.dσ²_dϕ[1]*P.dμ_dk[1]) + P.dσ²_dk[1]*P.dσ²_dϕ[1]*P.d²𝛑_dσ²dσ²[i]
		d²πᵢ_dkdσ² = P.dμ_dk[1]*P.d²𝛑_dμdσ²[i] + P.dσ²_dk[1]*P.d²𝛑_dσ²dσ²[i]
		P.d²𝛑_dkdσ²ₐ[i] = P.Δt*d²πᵢ_dkdσ²
		P.d²𝛑_dkdσ²ₛ[i] = P.∑c[1]*d²πᵢ_dkdσ² + P.d∑c_dk[1]*P.d𝛑_dσ²[i]
		P.d²𝛑_dλdλ[i] = P.d²𝛍_dλdλ[j]*P.d𝛑_dμ[i] + P.d𝛍_dλ[j]^2*P.d²𝛑_dμdμ[i]
		P.d²𝛑_dλdϕ[i] = P.dΔc_dϕ[1]*d²πᵢ_dλdΔc + P.d∑c_dϕ[1]*d²πᵢ_dλd∑c
		d²πᵢ_dλdσ² = P.d𝛍_dλ[j]*P.d²𝛑_dμdσ²[i]
		P.d²𝛑_dλdσ²ₐ[i] = P.Δt*d²πᵢ_dλdσ²
		P.d²𝛑_dλdσ²ₛ[i] = P.∑c[1]*d²πᵢ_dλdσ²
		P.d²𝛑_dϕdϕ[i] = P.d²μ_dϕdϕ[1]*P.d𝛑_dμ[i] + P.d²σ²_dϕdϕ[1]*P.d𝛑_dσ²[i] + P.dμ_dϕ[1]^2*P.d²𝛑_dμdμ[i] + 2P.dμ_dϕ[1]*P.dσ²_dϕ[1]*P.d²𝛑_dμdσ²[i] + P.dσ²_dϕ[1]^2*P.d²𝛑_dσ²dσ²[i]
		d²πᵢ_dϕdσ² = P.dμ_dϕ[1]*P.d²𝛑_dμdσ²[i] + P.dσ²_dϕ[1]*P.d²𝛑_dσ²dσ²[i]
		P.d²𝛑_dϕdσ²ₐ[i] = P.Δt*d²πᵢ_dϕdσ²
		P.d²𝛑_dϕdσ²ₛ[i] = P.∑c[1]*d²πᵢ_dϕdσ² + P.d∑c_dϕ[1]*P.d𝛑_dσ²[i]
		P.d²𝛑_dσ²ₐdσ²ₐ[i] = P.Δt^2*P.d²𝛑_dσ²dσ²[i]
		P.d²𝛑_dσ²ₐdσ²ₛ[i] = P.Δt*P.∑c[1]*P.d²𝛑_dσ²dσ²[i]
		P.d²𝛑_dσ²ₛdσ²ₛ[i] = P.∑c[1]^2*P.d²𝛑_dσ²dσ²[i]
	end
	return nothing
end

"""
	differentiate_wrt_transition_parameters!(P, j)

Compute the first-order partial derivatives of the j-th column of the transition matrix

MODIFIED ARGUMENT
-`P`: a structure containing first partial derivatives of a probability vector of the accumulator variable. The probability vector represents the j-th column of the transition matrix: `p(𝐚(t) ∣ a(t-1) = j)`

UNMODIFIED ARGUMENT
-`j`: the index of the state of the accumulator variable in the previous time step
"""
function differentiate_wrt_transition_parameters!(P::Probabilityvector, j::Integer)
	for i = 1:P.Ξ
		P.d𝛑_dk[i] = P.dσ²_dk[1]*P.d𝛑_dσ²[i] + P.dμ_dk[1]*P.d𝛑_dμ[i]
		P.d𝛑_dλ[i] = P.d𝛍_dλ[j]*P.d𝛑_dμ[i]
		P.d𝛑_dϕ[i] = P.dσ²_dϕ[1]*P.d𝛑_dσ²[i] + P.dμ_dϕ[1]*P.d𝛑_dμ[i]
		P.d𝛑_dσ²ₐ[i] = P.Δt*P.d𝛑_dσ²[i]
		P.d𝛑_dσ²ₛ[i] = P.∑c[1]*P.d𝛑_dσ²[i]
	end
	return nothing
end

"""
	differentiate_twice_wrt_Bμσ²!(P, j)

Compute the second- (and first-) order partial derivatives of a probability vector of the accumulator with respect to the bound height, mean, and variance

The probability vector can represent a column of the transition matrix or the prior probability of the accumulator

MODIFIED ARGUMENT
-`P`: a structure containing first and second order partial derivatives of a probability vector of the accumulator variable.

UNMODIFIED ARGUMENT
-`j`: the index of the state of the accumulator variable in the previous time step. For computing prior probability, set j to be (P.Ξ + 1)/2

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat");
julia> P = FHMDDM.Probabilityvector(model.options.Δt, model.θnative, model.options.Ξ);
julia> clicks = model.trialsets[1].trials[1].clicks;
julia> adaptedclicks = FHMDDM.∇∇adapt(clicks, model.θnative.k[1], model.θnative.ϕ[1]);
julia> FHMDDM.update_for_∇∇transition_probabilities!(P, adaptedclicks, clicks, 3)
julia> FHMDDM.differentiate_twice_wrt_Bμσ²!(P, 2)
```
"""
function differentiate_twice_wrt_Bμσ²!(P::Probabilityvector, j::Integer)
	differentiate_wrt_Bμσ²!(P, j)
	Ξ = P.Ξ
	fη = P.𝐟 .* P.𝛈
	Δfη = diff(fη)
	Δfω = diff(P.𝐟 .* P.𝛚)
	Δfωz = diff(P.𝐟 .* P.𝛚 .* P.𝐳)
	Δfz = diff(P.𝐟 .* P.𝐳)
	Δζ = diff(P.𝐟 .* (P.𝐳.^2 .- 1.0) ./ 4.0 ./ P.σ[1].^3 ./ P.Δξ)
	P.d²𝛑_dBdB[1] 	= ((fη[1] + P.𝛚[2]*Δfη[1])/P.σ[1] - 2P.d𝛑_dB[1])/P.B
	P.d²𝛑_dBdμ[1] 	= (-Δfω[1]/P.σ[1] - P.d𝛑_dμ[1])/P.B
	P.d²𝛑_dBdσ²[1] = (-Δfωz[1]/2/P.σ²[1] - P.d𝛑_dσ²[1])/P.B
	P.d²𝛑_dμdσ²[1] = Δfz[1]/P.Δξσ²2[1]
	P.d²𝛑_dσ²dσ²[1]= Δζ[1]
	for i=2:Ξ-1
		P.d²𝛑_dBdB[i] 	= ((P.𝛚[i+1]*Δfη[i] - P.𝛚[i-1]*Δfη[i-1])/P.σ[1] - 2P.d𝛑_dB[i])/P.B
		P.d²𝛑_dBdμ[i] 	= ((Δfω[i-1]-Δfω[i])/P.σ[1] - P.d𝛑_dμ[i])/P.B
		P.d²𝛑_dBdσ²[i] = ((Δfωz[i-1]-Δfωz[i])/2/P.σ²[1] - P.d𝛑_dσ²[i])/P.B
		P.d²𝛑_dμdσ²[i] = (Δfz[i]-Δfz[i-1])/P.Δξσ²2[1]
		P.d²𝛑_dσ²dσ²[i] = Δζ[i] - Δζ[i-1]
	end
	P.d²𝛑_dBdB[Ξ]	= -((fη[Ξ] + P.𝛚[Ξ-1]*Δfη[Ξ-1])/P.σ[1] + 2P.d𝛑_dB[Ξ])/P.B
	P.d²𝛑_dBdμ[Ξ]	= (Δfω[Ξ-1]/P.σ[1] - P.d𝛑_dμ[Ξ])/P.B
	P.d²𝛑_dBdσ²[Ξ] = (Δfωz[Ξ-1]/2/P.σ²[1] - P.d𝛑_dσ²[Ξ])/P.B
	P.d²𝛑_dμdσ²[Ξ] = -Δfz[Ξ-1]/P.Δξσ²2[1]
	P.d²𝛑_dσ²dσ²[Ξ] = -Δζ[Ξ-1]
	for i = 1:Ξ
		P.d²𝛑_dμdμ[i] = 2P.d𝛑_dσ²[i]
	end
	return nothing
end

"""
	differentiate_wrt_Bμσ²!(P, j)

Compute the first-order partial derivatives of a probability vector of the accumulator with respect to bound height, mean, and variance

The probability vector can represent a column of the transition matrix or the prior probability of the accumulator

MODIFIED ARGUMENT
-`P`: a structure containing first and second order partial derivatives of a probability vector of the accumulator variable.

UNMODIFIED ARGUMENT
-`j`: the index of the state of the accumulator variable in the previous time step. For computing prior probability, set j to be (P.Ξ + 1)/2
"""
function differentiate_wrt_Bμσ²!(P::Probabilityvector, j::Integer)
	evaluate_using_Bμσ²!(P, j)
	Ξ = P.Ξ
	P.d𝛑_dB[1] = (P.Φ[1] - P.𝛑[1] + P.𝛚[2]*P.ΔΦ[1])/P.B
	P.d𝛑_dμ[1] = -P.ΔΦ[1]/P.Δξ
	P.d𝛑_dσ²[1] = P.Δf[1]/P.σ2Δξ[1]
	for i = 2:P.Ξ-1
		P.d𝛑_dB[i] = (P.𝛚[i+1]*P.ΔΦ[i] - P.𝛚[i-1]*P.ΔΦ[i-1] - P.𝛑[i])/P.B
		P.d𝛑_dμ[i] = (P.ΔΦ[i-1] - P.ΔΦ[i])/P.Δξ
		P.d𝛑_dσ²[i] = (P.Δf[i]-P.Δf[i-1])/P.σ2Δξ[1]
    end
	P.d𝛑_dB[Ξ] = (P.Ψ[Ξ] - P.𝛑[Ξ] - P.𝛚[Ξ-1]*P.ΔΦ[Ξ-1])/P.B
	P.d𝛑_dμ[Ξ] = P.ΔΦ[Ξ-1]/P.Δξ
	P.d𝛑_dσ²[Ξ] = -P.Δf[Ξ-1]/P.σ2Δξ[1]
	return nothing
end

"""
	evaluate_using_Bμσ²!(P, j)

Evaluate the probabilities of the accumulator using the bound height, mean, and variance

The integer j indicates the state of the accumulator at the previous time step on which the probabilities are conditioned. To compute the prior probabilities, set j to equal (Ξ+1)/2

MODIFIED ARGUMENT
-`P`: a structure containing first and second order partial derivatives of a probability vector of the accumulator variable.

UNMODIFIED ARGUMENT
-`j`: the index of the state of the accumulator variable in the previous time step. For computing prior probability, set j to be (P.Ξ + 1)/2
"""
function evaluate_using_Bμσ²!(P::Probabilityvector, j::Integer)
	Ξ = P.Ξ
	expλΔt_dξⱼ_dB = P.expλΔt*P.d𝛏_dB[j]
	Ξd2m1 = (P.Ξ-2)/2
	for i = 1:Ξ
		P.𝛈[i] = P.d𝛏_dB[i] - expλΔt_dξⱼ_dB
		P.𝛚[i] = P.𝛈[i]*Ξd2m1
		P.𝐳[i] = (P.𝛏[i] - P.𝛍[j])/P.σ[1]
		P.𝐟[i] = normpdf(P.𝐳[i])
		P.Φ[i] = normcdf(P.𝐳[i])
		P.Ψ[i] = normccdf(P.𝐳[i])
	end
	for i = 1:Ξ-1
		P.Δf[i] = P.𝐟[i+1] - P.𝐟[i]
		if P.𝛍[j] <= P.𝛏[i]
			P.ΔΦ[i] = P.Ψ[i] - P.Ψ[i+1]
		else
			P.ΔΦ[i] = P.Φ[i+1] - P.Φ[i]
		end
	end
	P.𝛑[1] = P.Φ[1] + P.σ_Δξ[1]*(P.Δf[1] + P.𝐳[2]*P.ΔΦ[1])
	for i = 2:Ξ-1
		P.𝛑[i] = P.σ_Δξ[1]*(P.Δf[i] - P.Δf[i-1] + P.𝐳[i+1]*P.ΔΦ[i] - P.𝐳[i-1]*P.ΔΦ[i-1])
	end
	P.𝛑[Ξ] = P.Ψ[Ξ] - P.σ_Δξ[1]*(P.Δf[Ξ-1] + P.𝐳[Ξ-1]*P.ΔΦ[Ξ-1])
	return nothing
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
	check∇∇transitionmatrix(model)

Maximum absolute difference between the automatically computed and hand-coded first and second order partial derivatives of the transition probabilities of the accumulator

ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters for a factorial hidden-Markov drift-diffusion model

RETURN
-`maxabsdiff∇∇`: maximum absolute difference between the automatically and hand-coded second-order partial derivatives, across of all elements of the transition matrix at each time step, across all time steps, trials, and trialsets. Element (i,j) corresponds to the derivative with respect to the i-th and j-th parameter. The parameters that determine the transition probabilties are ordered alphabetically:
	θ[1] = B, bound height
	θ[2] = k, adaptation change rate
	θ[3] = λ, feedback
	θ[4] = σ²ₐ, variance of diffusion noise
	θ[5] = σ²ₛ, variance of per-click noise
	θ[6] = ϕ, adaptation strength
-`maxabsdiff∇`: maximum absolute difference between the automatically and hand-coded first-order partial derivatives. The i-th element corresponds to the derivative with respect to the i-th parameter.

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_09_test/data.mat"; randomize=true);
julia> maxabsdiff∇∇, maxabsdiff∇ = FHMDDM.check∇∇transitionmatrix(model)
```
"""
function check∇∇transitionmatrix(model::Model)
	@unpack Δt, Ξ = model.options
	@unpack θnative = model
	x₀ = [θnative.B[1], θnative.k[1], θnative.λ[1], θnative.ϕ[1], θnative.σ²ₐ[1], θnative.σ²ₛ[1]]
	nparameters = length(x₀)
	maxabsdiff∇∇, ∇∇auto = zeros(nparameters,nparameters), zeros(nparameters,nparameters)
	maxabsdiff∇, ∇auto = zeros(nparameters), zeros(nparameters)
	∇∇hand = map(i->zeros(Ξ,Ξ), CartesianIndices((nparameters,nparameters)));
	∇hand = map(i->zeros(Ξ,Ξ), 1:nparameters);
	P = Probabilityvector(model.options.Δt, model.θnative, model.options.Ξ)
	A = zeros(Ξ,Ξ);
	A[1,1] = A[Ξ,Ξ] = 1.0
	P = FHMDDM.Probabilityvector(model.options.Δt, model.θnative, model.options.Ξ);
	for s in eachindex(model.trialsets)
		for m in eachindex(model.trialsets[s].trials)
			trial = model.trialsets[s].trials[m]
			adaptedclicks = ∇∇adapt(trial.clicks, model.θnative.k[1], model.θnative.ϕ[1])
			for t = 2:trial.ntimesteps
				update_for_∇∇transition_probabilities!(P, adaptedclicks, trial.clicks, t)
				∇∇transitionmatrix!(∇∇hand, ∇hand, A, P)
				for j = 2:Ξ-1
					for i = 1:Ξ
						f(x) = accumulatorprobability(trial.clicks,Δt,i,j,t,Ξ,x)
						ForwardDiff.hessian!(∇∇auto, f, x₀)
						ForwardDiff.gradient!(∇auto, f, x₀)
						for q = 1:nparameters
							maxabsdiff∇[q] = max(maxabsdiff∇[q], abs(∇auto[q] - ∇hand[q][i,j]))
							for r = q:nparameters
								maxabsdiff∇∇[q,r] = maxabsdiff∇∇[r,q] = max(maxabsdiff∇∇[q,r], abs(∇∇auto[q,r] - ∇∇hand[q,r][i,j]))
							end
						end
					end
				end
			end
		end
	end
	return maxabsdiff∇∇, maxabsdiff∇
end

"""
    accumulatorprobability(clicktimes,i,j,t,x)

Compute the transition probability of the accumulator variable `p(aₜ=i ∣ aₜ₋₁=j)`

INPUT
-`clicks`: a structure containing the times and origin of each auditory click played during a trial
-`Δt`: duration of each time step
-`i`: state of the accumulator at time step t
-`j`: state of the accumulator at time step t-1
-'t': time step
-`Ξ`: number of states into which the accumulator is discretized
-`x`: vector containing the alphabetically concatenated values of the parameters

RETURN
-transition probability `p(aₜ=i ∣ aₜ₋₁=j)`

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> clicks = FHMDDM.sampleclicks(0.01, 40, 0.01, 100, 30);
julia> x = [10.0, 0.5, -0.5, 0.8, 2.0, 0.4];
julia> p = FHMDDM.transitionprobability(clicks,0.01,4,10,20,53,x)
```
"""
function accumulatorprobability(clicks::Clicks,
							   Δt::AbstractFloat,
                               i::Integer,
                               j::Integer,
                               t::Integer,
							   Ξ::Integer,
                               x::Vector{<:Real})
	@assert t > 1
	@assert length(x)==6
	B = x[1]
    k = x[2]
    λ = x[3]
    ϕ = x[4]
    σ²ₐ = x[5]
    σ²ₛ = x[6]
    C = adapt(clicks, k, ϕ).C
    cL = sum(C[clicks.left[t]])
    cR = sum(C[clicks.right[t]])
	𝛏 = B.*(2 .*collect(1:Ξ) .- Ξ .- 1)./(Ξ-2)
	μ = exp(λ*Δt)*𝛏[j] + (cR-cL)*differentiate_μ_wrt_Δc(Δt, λ)
	σ = √( (cL+cR)*σ²ₛ + Δt*σ²ₐ )
	probabilityvector(μ, σ, 𝛏)[i]
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
	check∇∇priorprobability(model)

Maximum absolute difference between the automatically computed and hand-coded first and second order partial derivatives of the prior probabilities of the accumulator

ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters for a factorial hidden-Markov drift-diffusion model

RETURN
-`maxabsdiff∇∇`: maximum absolute difference between the automatically and hand-coded second-order partial derivatives, across of all elements of the prior probability vector of each trial and across all trials and trialsets. Element (i,j) corresponds to the derivative with respect to the i-th and j-th parameter. The parameters that determine the transition probabilties are ordered alphabetically:
	θ[1] = B, bound height
	θ[2] = μ₀, additive offset to the mean that is constant across trials
	θ[3] = σ²ᵢ, variance
	θ[4] = wₕ, weight of the location of the previous reward
-`maxabsdiff∇`: maximum absolute difference between the automatically and hand-coded first-order partial derivatives. The i-th element corresponds to the derivative with respect to the i-th parameter.

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_09_test/data.mat"; randomize=true);
julia> maxabsdiff∇∇, maxabsdiff∇ = FHMDDM.check∇∇priorprobability(model)
```
"""
function check∇∇priorprobability(model::Model)
	@unpack Δt, Ξ = model.options
	@unpack θnative = model
	x₀ = [θnative.B[1], θnative.μ₀[1], θnative.σ²ᵢ[1], θnative.wₕ[1]]
	nparameters = length(x₀)
	maxabsdiff∇∇, ∇∇auto = zeros(nparameters,nparameters), zeros(nparameters,nparameters)
	maxabsdiff∇, ∇auto = zeros(nparameters), zeros(nparameters)
	P = Probabilityvector(model.options.Δt, model.θnative, model.options.Ξ)
	∇∇hand = map(i->zeros(Ξ), CartesianIndices((nparameters,nparameters)))
	∇hand = map(i->zeros(Ξ), 1:nparameters)
	P = FHMDDM.Probabilityvector(model.options.Δt, model.θnative, model.options.Ξ)
	for s in eachindex(model.trialsets)
		for m in eachindex(model.trialsets[s].trials)
			@unpack previousanswer = model.trialsets[s].trials[m]
			∇∇priorprobability!(∇∇hand, ∇hand, P, previousanswer)
			for i = 1:Ξ
				f(x) = accumulatorprobability(Δt,i,previousanswer,Ξ,x)
				ForwardDiff.hessian!(∇∇auto, f, x₀)
				ForwardDiff.gradient!(∇auto, f, x₀)
				for q = 1:nparameters
					maxabsdiff∇[q] = max(maxabsdiff∇[q], abs(∇auto[q] - ∇hand[q][i]))
					for r = q:nparameters
						maxabsdiff∇∇[q,r] = maxabsdiff∇∇[r,q] = max(maxabsdiff∇∇[q,r], abs(∇∇auto[q,r] - ∇∇hand[q,r][i]))
					end
				end
			end
		end
	end
	return maxabsdiff∇∇, maxabsdiff∇
end

"""
    accumulatorprobability(Δt, i, previousanswer, Ξ, x)

Compute the transition probability of the accumulator variable `p(aₜ=i ∣ aₜ₋₁=j)`

INPUT
-`clicks`: a structure containing the times and origin of each auditory click played during a trial
-`Δt`: duration of each time step
-`i`: state of the accumulator at time step t
-`j`: state of the accumulator at time step t-1
-'t': time step
-`Ξ`: number of states into which the accumulator is discretized
-`x`: vector containing the alphabetically concatenated values of the parameters

RETURN
-transition probability `p(aₜ=i ∣ aₜ₋₁=j)`

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> x = [10.0, 0.5, 2.0, 0.8];
julia> FHMDDM.accumulatorprobability(0.01, 26, 1, 53, x)
	0.0542176221212666
```
"""
function accumulatorprobability(Δt::AbstractFloat,
                                i::Integer,
								previousanswer::Real,
								Ξ::Integer,
                                x::Vector{<:Real})
	@assert length(x)==4
	B = x[1]
    μ₀ = x[2]
    σ²ᵢ = x[3]
    wₕ = x[4]
	𝛏 = B.*(2 .*collect(1:Ξ) .- Ξ .- 1)./(Ξ-2)
	μ = μ₀ + previousanswer*wₕ
	σ = √σ²ᵢ
	probabilityvector(μ, σ, 𝛏)[i]
end

"""
	compare_exact_approximate_transition_matrices(model)

Maximum absolute difference between the exact and approximate evaluation of the transition matrix

Approximate computation of the transition matrix is based on DePasquale et al., (2022)

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_04_01_test/data.mat");
julia> maxabsdiff = FHMDDM.compare_exact_approximate_transition_matrices(model)
```
"""
function compare_exact_approximate_transition_matrices(model::Model)
	@unpack Δt, Ξ = model.options
	Aexact, Aapprox, maxabsdiff = zeros(Ξ,Ξ), zeros(Ξ,Ξ), zeros(Ξ,Ξ)
	Aexact[1,1] = Aexact[Ξ,Ξ] = 1.0
	P = FHMDDM.Probabilityvector(model.options.Δt, model.θnative, model.options.Ξ);
	for i in eachindex(model.trialsets)
		for m in eachindex(model.trialsets[i].trials)
			trial = model.trialsets[i].trials[m]
			adaptedclicks = adapt(trial.clicks, model.θnative.k[1], model.θnative.ϕ[1])
			for t = 2:trial.ntimesteps
				update_for_transition_probabilities!(P, adaptedclicks, trial.clicks, t)
				transitionmatrix!(Aexact, P)
				approximatetransition!(Aapprox, P.Δt, P.Δξ, model.θnative.λ[1], P.Δc[1], P.Ξ, P.σ²[1], P.𝛏)
				for ij in eachindex(maxabsdiff)
					maxabsdiff[ij] = max(maxabsdiff[ij], abs(Aapprox[ij] - Aexact[ij]))
				end
			end
		end
	end
	return maxabsdiff
end

"""
    approximatetransition!(Aᵃ, dt, dx, λ, μ, n, σ², xc)

Compute the approximate transition matrix ``𝑝(𝑎ₜ ∣ 𝑎ₜ₋₁, clicks(𝑡), 𝜃)`` for a single time bin and store it in `Aᵃ`.

The computation makes use of the `λ`, a scalar indexing leakiness or instability; `μ` and `σ²`, mean and variance of the Gaussian noise added, time bin size `dt`, size of bins of the accumulator variable `dx`, number of bins of the accumulator variables `n`, and bin centers `xc`

The implementation is based on DePasquale et al., (2022)
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











"""
    transitionmatrix!(A, 𝛍, σ, 𝛏)

In-place computation of the transition matrix for the discretized Fokker-Planck system for a single time step

MODIFIED ARGUMENT
-`A`: a square matrix describing the transitions of the accumulator variable at a single time step

UNMODIFIED ARGUMENT
-`𝛍`: mean of the Gaussian PDF of the accumulator variable conditioned on its value in the previous time step
-`σ`: standard deviation of the Weiner process at this time step
-`𝛏`: a vector specifying the equally-spaced values into which the accumulator variable is discretized

RETURN
-nothing
"""
function transitionmatrix!(A::Matrix{T},
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
    transitionmatrix!(A, cL, cR, trialinvariant, θnative)

In-place computation of a transition matrix for a single time-step

MODIFIED ARGUMENT
-`A`: the transition matrix. Expects the `A[2:end,1] .== 0` and `A[1:end-1,end] .== 0`

UNMODIFIED ARGUMENT
-`cL`: input from the left
-`cR`: input from the right
-`trialinvariant`: structure containing quantities used for computations for each trial
-`θnative`: model parameters in native space
"""
function transitionmatrix!(A::Matrix{<:Real},
                           cL::Real,
						   cR::Real,
						   trialinvariant::Trialinvariant,
						   θnative::Latentθ)
    @unpack Δt, 𝛏 = trialinvariant
	𝛍 = conditionedmean(cR-cL, Δt, θnative.λ[1], 𝛏)
	σ = √( (cL+cR)*θnative.σ²ₛ[1] + θnative.σ²ₐ[1]*Δt )
	transitionmatrix!(A, 𝛍, σ, 𝛏)
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
    transitionmatrix!(A, ∂μ, ∂σ², ∂B, 𝛍, σ, 𝛚, 𝛏)

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
function transitionmatrix!(	A::Matrix{T},
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
    transitionmatrix!(A, ∂μ, ∂σ², ∂B, cL, cR, trialinvariant, θnative)

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
function transitionmatrix!(	A::Matrix{<:Real},
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
	transitionmatrix!(A, ∂μ, ∂σ², ∂B, 𝛍, σ, Ω, 𝛏)
	return nothing
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
