"""
    comparegradients(model)

Compare the hand-coded and automatically computed gradients of the log-likelihood of the model

INPUT
-`model`: a structure containing the parameters, data, and hyperparameters of a factorial hidden Markov drift-diffusion

RETURN
-maximum absolute difference between the hand-coded and automatically computed gradients
-hand-coded gradient
-automatically computed gradient

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_03_25_checkgradient/data.mat")
julia> maxabsdiff, handcoded, automatic = comparegradients(model)
```
"""
function comparegradients(model::Model)
    shared = Shared(model)
    ∇hand = similar(shared.concatenatedθ)
    γ =	map(model.trialsets) do trialset
    			map(CartesianIndices((model.options.Ξ, model.options.K))) do index
    				zeros(trialset.ntimesteps)
    			end
    		end
    ∇negativeloglikelihood!(∇hand, γ, model, shared, shared.concatenatedθ)
    f(x) = -loglikelihood(x, shared.indexθ, model)
    ∇auto = ForwardDiff.gradient(f, shared.concatenatedθ)
    return maximum(abs.(∇hand .- ∇auto)), ∇hand, ∇auto
end

"""
	compareHessians(B, μ, σ², Ξ)

Compare the automatically computed and hand-coded Hessian matrices

Each Hessian matrix consists of the second order partial derivatives of each element of the probability vector of the accumulator variable with respect to the bound height, mean, and variance.

INPUT
-`B`: bound height
-`μ`: mean
-`σ²`: variance
-`Ξ`: number of values into which the accumulator is discretied

RETURN
-scalar representing maximum absolute difference between the two types of Hessian matrices across all elements of the probability vector
-vector of matrices whose i-th is the automatically compute Hessian of the i-th element of the probability vector
-vector of matrices whose i-th is the hand-coded Hessian of the i-th element of the probability vector

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> maxabsdiff, Hauto, Hhand = FHMDDM.compareHessians(10.0, 1.0, 4.0, 53);
julia> maxabsdiff
	3.2786273695961654e-16
julia> size(Hauto), size(Hhand)
	((53,), (53,))
julia> eltype(Hauto), eltype(Hhand)
	(Matrix{Float64}, Matrix{Float64})
```
"""
function compareHessians(B::Real, μ::Real, σ²::Real, Ξ::Integer)
	@assert Ξ>0
	𝛏 = B.*(2collect(1:Ξ).-Ξ.-1)./(Ξ-2)
	x₀ = [B, μ, σ²]
	automatic_Hessians = collect(zeros(3,3) for i=1:Ξ)
	handcoded_Hessians = Hessian(x₀[2], √x₀[3], 𝛏)
	for i = 1:Ξ
		f(x) = FHMDDM.accumulatorprobability(i, Ξ, x)
		automatic_Hessians[i] = ForwardDiff.hessian(f, x₀)
	end
	maxabsdiff = map((x,y)-> maximum(abs.(x.-y)), automatic_Hessians, handcoded_Hessians)
	return maximum(maxabsdiff), automatic_Hessians, handcoded_Hessians
end

"""
	comparegradients(clicks, k, ϕ)

Compare the automatically differentiated and hand-coded first-order partial derivatives of the adapted click magnitude with respect to k and ϕ

INPUT
-`clicks`: a structure containing the times, sources, and time steps of the clicks in one trial
-`k`: change rate of adaptation
-`ϕ`: strength of adaptation

RETURN
-`maxabsdiff`: maximum absolute difference between the automatically computed and hand-coded gradients of the adapated impacts
-`automatic_gradients`: a vector of vectors whose i-th element is the automatically computed gradient of the adapted strength of the i-th click
-`handcoded_gradients`: a vector of vectors whose i-th element is the hand-coded gradient of the adapted strength of the i-th click

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> clicks = FHMDDM.sampleclicks(0.01, 40, 0.01, 100, 30; rng=MersenneTwister(1234))
julia> maxabsdiff, automatic_gradients, handcoded_gradients = FHMDDM.comparegradients(clicks, 0.5, 0.8)
```
"""
function comparegradients(clicks::Clicks, k::Real, ϕ::Real)
	C, dCdk, dCdϕ = FHMDDM.∇adapt(clicks, k, ϕ)
	x₀ = [k,ϕ]
	nclicks = length(clicks.time)
	automatic_gradients, handcoded_gradients = collect(zeros(2) for i=1:nclicks), collect(zeros(2) for i=1:nclicks)
	for i = 1:nclicks
		f(x) = adapt(clicks, x[1], x[2])[i]
		ForwardDiff.gradient!(automatic_gradients[i], f, x₀)
		handcoded_gradients[i][1] = dCdk[i]
		handcoded_gradients[i][2] = dCdϕ[i]
	end
	maxabsdiff = maximum(map((x,y)->maximum(abs.(x.-y)), automatic_gradients, handcoded_gradients))
	return maxabsdiff, automatic_gradients, handcoded_gradients
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
"""
function compareHessians(B::Real,
						 clicks::Clicks, Δt::Real, j::Integer, k::Real, λ::Real, ϕ::Real, σ²ₐ::Real, σ²ₛ::Real, t::Integer;
						 Ξ::Integer=53)
	@assert Ξ>0
	C, dC_dk, dC_dϕ, d²C_dkdk, d²C_dkdϕ, d²C_dϕdϕ = ∇∇adapt(clicks,k,ϕ)
	automatic_Hessians, handcoded_Hessians = collect(zeros(6,6) for i=1:Ξ), collect(zeros(6,6) for i=1:Ξ)
	cL = sum(C[clicks.left[t]])
	cR = sum(C[clicks.right[t]])
	Δc = cR-cL
	Σc = cR+cL
	μ = conditionedmean(Δc, Δt, θnative.λ[1], 𝛏)[j]
	σ = √(σ²ₛ*Σc + σ²ₛ*Δt)
	𝛏 = B.*(2collect(1:Ξ).-Ξ.-1)./(Ξ-2)
	𝛑, d𝛑_dB, d𝛑_dμ, d𝛑_dσ², d²𝛑_dBdB, d²𝛑_dBdμ, d²𝛑_dBdσ², d²𝛑_dμdμ, d²𝛑_dμdσ², d²𝛑_dσ²dσ² = zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ),
	probabilityvector!(𝛑, d𝛑_dB, d𝛑_dμ, d𝛑_dσ², d²𝛑_dBdB, d²𝛑_dBdμ, d²𝛑_dBdσ², d²𝛑_dμdμ, d²𝛑_dμdσ², d²𝛑_dσ²dσ², μ, σ, 𝛏)
	λΔt = λ*Δt
	expλΔt = exp(λΔt)
	dcR_dk = sum(dC_dk[clicks.right[t]])
	dcL_dk = sum(dC_dk[clicks.left[t]])
	dcR_dϕ = sum(dC_dϕ[clicks.right[t]])
	dcL_dϕ = sum(dC_dϕ[clicks.left[t]])
	d²cR_dkdk = sum(d²C_dkdk[clicks.right[t]])
	d²cL_dkdk = sum(d²C_dkdk[clicks.left[t]])
	d²cR_dkdϕ = sum(d²C_dkdϕ[clicks.right[t]])
	d²cL_dkdϕ = sum(d²C_dkdϕ[clicks.left[t]])
	d²cR_dϕdϕ = sum(d²C_dϕdϕ[clicks.right[t]])
	d²cL_dϕdϕ = sum(d²C_dϕdϕ[clicks.left[t]])
	dΔc_dk = dcR_dk - dcL_dk
	dΣc_dk = dcR_dk + dcL_dk
	dΔc_dϕ = dcR_dϕ - dcL_dϕ
	dΣc_dϕ = dcR_dϕ + dcL_dϕ
	d²Δc_dkdk = d²cR_dkdk - d²cL_dkdk
	d²Σc_dkdk = d²cR_dkdk + d²cL_dkdk
	d²Δc_dkdϕ = d²cR_dkdϕ - d²cL_dkdϕ
	d²Σc_dkdϕ = d²cR_dkdϕ + d²cL_dkdϕ
	d²Δc_dϕdϕ = d²cR_dϕdϕ - d²cL_dϕdϕ
	d²Σc_dϕdϕ = d²cR_dϕdϕ + d²cL_dϕdϕ
	dμ_dk = dΔc_dk*dμ_dΔc
	dμ_dϕ = dΔc_dϕ*dμ_dΔc
	dσ²_dΣc = σ²ₛ
	dσ²_dk = dΣc_dk*dσ²_dΣc
	dσ²_dϕ = dΣc_dϕ*dσ²_dΣc
	if abs(λΔt) > 1e-10
		dμ_dΔc = (expλΔt - 1.0)/λΔt
	else
		dμ_dΔc = 1.0
	end
	if abs(λ) > 1e-3
		d²μ_dΔcdλ = (expλΔt - (expλΔt - 1.0)/λΔt)/λ
	else
		d²μ_dΔcdλ = Δt/2
	end
	if abs(λ) > 1e-2
		d³μ_dΔcdλdλ = (expλΔt*(λΔt^2 - 2λΔt + 2)-2)/λΔt/λ^2
	else
		d³μ_dΔcdλdλ = Δt^2/3
	end
	dμ_dλ = Δt*expλΔt*𝛏[j] + Δc*d²μ_dΔcdλ
	dξ_dB = (2j-Ξ-1)/(Ξ-2)
	d²μ_dBdλ = Δt*expλΔt*dξ_dB + Δc*d²μ_dΔcdλ
	d²μ_dλdλ = Δt^2*expλΔt*𝛏[j] + Δc*d³μ_dΔcdλdλ
	for i = 1:Ξ
		handcoded_Hessian[i][1,1] = d²𝛑_dBdB[i]
		handcoded_Hessian[i][1,2] = handcoded_Hessian[i][2,1] = dμ_dk*d²𝛑_dBdμ[i] + dσ²_dk*d²𝛑_dBdσ²[i] #d²πᵢ_dBdk
		handcoded_Hessian[i][1,3] = handcoded_Hessian[i][3,1] = d²μ_dBdλ*d𝛑_dμ[i] + dμ_dλ*d²𝛑_dBdμ[i] #d²πᵢ_dBdλ
		handcoded_Hessian[i][1,4] = handcoded_Hessian[i][4,1] = dμ_dϕ*d²𝛑_dBdμ[i] + dσ²_dϕ*d²𝛑_dBdσ²[i] #d²πᵢ_dBdϕ
		handcoded_Hessian[i][1,5] = handcoded_Hessian[i][5,1] = Δt*d²𝛑_dBdσ²[i] #d²πᵢ_dBdσ²ₐ
		handcoded_Hessian[i][1,6] = handcoded_Hessian[i][6,1] = Σc*d²𝛑_dBdσ²[i] #d²πᵢ_dBdσ²ₛ
		handcoded_Hessian[i][2,2] = dμ_dΔc*(d²Δc_dkdk*d𝛑_dμ[i] + dΔc_dk^2*dμ_dΔc*d²𝛑_dμdμ[i]) + dσ²_dΣc*(d²Σc_dkdk*d𝛑_dσ²[i] + dΣc_dk^2*dσ²_dΣc*d²𝛑_dσ²dσ²[i]) #d²πᵢ_dkdk
		d²πᵢ_dλdΔc = d²μ_dΔcdλ*d𝛑_dμ[i] + dμ_dΔc*dμ_dλ*d²𝛑_dμdμ[i]
		d²πᵢ_dλdΣc = dσ²_dΣc*dμ_dλ*d²𝛑_dμdσ²[i]
		handcoded_Hessian[i][2,3] = handcoded_Hessian[i][3,2] = dΔc_dk*d²πᵢ_dλdΔc + dΣc_dk*d²πᵢ_dλdΣc #d²πᵢ_dkdλ
		handcoded_Hessian[i][2,4] = handcoded_Hessian[i][4,2] = dμdΔc*(d²Δc_dkdϕ*d𝛑_dμ[i] + dΔc_dk*dΔc_dϕ*dμ_dΔc*d²𝛑_dμdμ[i]) + dσ²_dΣc*(d²Σc_dkdϕ*d𝛑_dσ²[i] + dσ²_dΣc*dΣc_dk*dΣc_dϕ*d²𝛑_dσ²dσ²) #d²πᵢ_dkdϕ
		d²πᵢ_dkdσ² = dμ_dk*d²𝛑_dμdσ²[i] + dσ²_dk*d²𝛑_dσ²dσ²[i]
		handcoded_Hessian[i][2,5] = handcoded_Hessian[i][5,2] = Δt*d²πᵢ_dkdσ² #d²πᵢ_dkdσ²ₐ
		handcoded_Hessian[i][2,6] = handcoded_Hessian[i][6,2] = Σc*d²πᵢ_dkdσ² + dΔc_dk*d𝛑_dσ²[i] #d²πᵢ_dkdσ²ₛ
		handcoded_Hessian[i][3,3] = d²μ_dλdλ*d𝛑_dμ[i] + dμ_dλ*d²𝛑_dμdμ[i] #d²πᵢ_dλdλ
		handcoded_Hessian[i][3,4] = handcoded_Hessian[i][4,3] = dΔc_dϕ*d²πᵢ_dλdΔc + dΣc_dϕ*d²πᵢ_dλdΣc #d²πᵢ_dλdϕ
		d²πᵢ_dλdσ² = dμ_dλ*d²𝛑_dμdσ²[i]
		handcoded_Hessian[i][3,5] = handcoded_Hessian[i][5,3] = Δt*d²πᵢ_dλdσ² #d²πᵢ_dλdσ²ₐ
		handcoded_Hessian[i][3,6] = handcoded_Hessian[i][6,3] = Σc*d²πᵢ_dλdσ² #d²πᵢ_dλdσ²ₛ
		handcoded_Hessian[i][4,4] = dμ_dΔc*(d²Δc_dϕdϕ*d𝛑_dμ[i] + dΔc_dϕ^2*dμ_dΔc*d²𝛑_dμdμ[i]) + dσ²_dΣc*(d²Σc_dϕdϕ*d𝛑_dσ²[i] + dΣc_dϕ^2*dσ²_dΣc*d²𝛑_dσ²dσ²[i]) #d²πᵢ_dϕdϕ
		d²πᵢ_dϕdσ² = dμ_dϕ*d²𝛑_dμdσ²[i] + dσ²_dϕ*d²𝛑_dσ²dσ²[i]
		handcoded_Hessian[i][4,5] = handcoded_Hessian[i][5,4] = Δt*d²πᵢ_dϕdσ² #d²πᵢ_dϕdσ²ₐ
		handcoded_Hessian[i][4,6] = handcoded_Hessian[i][6,4] = Σc*d²πᵢ_dϕdσ² + dΔc_dϕ*d𝛑_dσ²[i] #d²πᵢ_dϕdσ²ₛ
		handcoded_Hessian[i][5,5] = Δt^2*d²𝛑_dσ²dσ²[i] #d²πᵢ_dσ²ₐdσ²ₐ
		handcoded_Hessian[i][5,6] = handcoded_Hessian[i][6,5] = Δt*Σc*d²𝛑_dσ²dσ²[i] #d²πᵢ_dσ²ₐdσ²ₛ
		handcoded_Hessian[i][6,6] = Σc^2*d²𝛑_dσ²dσ²[i] #d²πᵢ_dσ²ₛdσ²ₛ
	end
	x₀ = [B, k, λ, ϕ, σ²ₐ, σ²ₛ]
	for i = 1:Ξ
		f(x) = FHMDDM.accumulatorprobability(clicks, Δt, i, j, t, Ξ, x)
		automatic_Hessians[i] = ForwardDiff.hessian(f, x₀)
	end
	maxabsdiff = maximum(map((x,y)->maximum(abs.(x.-y)), automatic_Hessians, handcoded_Hessians))
	return maxabsdiff, automatic_Hessians, handcoded_Hessians
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
julia> p = FHMDDM.accumulatorprobability(clicks,0.01,4,10,20,53,x)
```
"""
function accumulatorprobability(clicks::Clicks,
								Δt::AbstractFloat=0.01,
                                i::Integer,
                                j::Integer,
                                t::Integer,
								Ξ::Integer=53,
                                x::Vector{<:Real})
	@assert t > 1
	@assert length(x)==6
	B = x[1]
    k = x[2]
    λ = x[3]
    ϕ = x[4]
    σ²ₐ = x[5]
    σ²ₛ = x[6]
    C = adapt(clicks, k, ϕ)
    cL = sum(C[clicks.left[t]])
    cR = sum(C[clicks.right[t]])
	𝛏 = B.*(2 .*collect(1:Ξ) .- Ξ .- 1)./(Ξ-2)
	μ = conditionedmean(cR-cL, Δt, λ, 𝛏)[j]
	σ = √( (cL+cR)*σ²ₛ + Δt*σ²ₐ )
	probabilityvector(μ, σ, 𝛏)[i]
end

"""
	accumulatorprobability(i, Ξ, x)

Probability of the accumulator being equal to its i-th discrete value

The probability is conditioned on the number of discrete values Ξ, bound height B, Gaussian mean μ, and Gaussian variance σ²

INPUT
-`i`: index of the discrete value
-`Ξ`: number of discrete values
-`x`: vector of the concatenated values of the bound height B=x[1], Gaussian mean μ=x[2], and Gaussian variance σ²=x[3]

RETURN
-a scalar representing probability of the accumulator being equal to its i-th discrete value

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> FHMDDM.accumulatorprobability(10, 53, [10.0, 1.0, 4.0])
	5.264468722481375e-5
```
"""
function accumulatorprobability(i::Integer, Ξ::Integer, x::Vector{<:Real})
	B = x[1]
	μ = x[2]
	σ = √x[3]
	𝛏 = B.*(2.0.*collect(1:Ξ) .- Ξ .- 1)./(Ξ-2)
	probabilityvector(μ, σ, 𝛏)[i]
end

"""
	Hessian(μ, σ², 𝛏)

Hessian of each element of a probability vector with respect to bound height B, mean μ, variance σ²

INPUT
-`μ`: mean of the Gaussian distribution
-`σ`: standard deviation of the Gaussian distribution
-`𝛏`: discrete values of the distribution

RETURN
-`𝗛`: a vector whose element 𝗛[i] is the 3x3 Hessian matrix of the i-th element of a the probability vector with respect to B, μ, and σ²

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> μ = 1.0; σ = 2.0; B = 10.0; Ξ = 53; 𝛏 = (2collect(1:Ξ) .- Ξ .- 1)./(Ξ-2);
julia> Hessian(μ, σ, 𝛏)
	53-element Vector{Matrix{Float64}}:
	⋮
```
"""
function Hessian(μ::AbstractFloat,
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
	𝗛 = collect(zeros(3,3) for x=1:Ξ)
	for i=1:Ξ
		if i == 1
			𝗛[i][1,1] 			  	= Δfωξ[1]/B²σ - 2∂B[1]/B 	#∂B∂B
			𝗛[i][1,2] = 𝗛[i][2,1] = -Δfξ[1]/BΔξσ - ∂μ[1]/B 	#∂B∂μ
			𝗛[i][1,3] = 𝗛[i][3,1] = -Δfωz[1]/Bσ²2 - ∂σ²[1]/B 	#∂B∂σ²
			𝗛[i][2,3] = 𝗛[i][3,2] = Δfz[1]/Δξσ²2 				#∂μ∂σ²
			𝗛[i][3,3] 			  	= Δζ[1]						#∂σ²∂σ²
		elseif i < Ξ
			𝗛[i][1,1] 			  	= (Δfωξ[i] - Δfωξ[i-1])/B²σ - 2∂B[i]/B	#∂B∂B
			𝗛[i][1,2] = 𝗛[i][2,1] = (Δfξ[i-1]-Δfξ[i])/BΔξσ - ∂μ[i]/B 		#∂B∂μ
			𝗛[i][1,3] = 𝗛[i][3,1] = (Δfωz[i-1]-Δfωz[i])/Bσ²2 - ∂σ²[i]/B 	#∂B∂σ²
			𝗛[i][2,3] = 𝗛[i][3,2] = (Δfz[i]-Δfz[i-1])/Δξσ²2				#∂μ∂σ²
			𝗛[i][3,3] 			  	= Δζ[i] - Δζ[i-1]						#∂σ²∂σ²
		else
			𝗛[i][1,1]				= -Δfωξ[Ξ_1]/B²σ - 2∂B[Ξ]/B #∂B∂B
			𝗛[i][1,2] = 𝗛[i][2,1]	= Δfξ[Ξ_1]/BΔξσ - ∂μ[Ξ]/B 	#∂B∂μ
			𝗛[i][1,3] = 𝗛[i][3,1] = Δfωz[Ξ_1]/Bσ²2 - ∂σ²[Ξ]/B #∂B∂σ²
			𝗛[i][2,3] = 𝗛[i][3,2] = -Δfz[Ξ_1]/Δξσ²2 			#∂μ∂σ²
			𝗛[i][3,3] 			  	= -Δζ[Ξ_1]					#∂σ²∂σ²
		end
		𝗛[i][2,2] = 2∂σ²[i] #∂μ∂μ
	end
	return 𝗛
end
