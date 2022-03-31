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
julia> maxabsdiff, automatic_Hessians, handcoded_Hessians = FHMDDM.compareHessians(10.0, 1.0, 4.0, 53);
julia> maxabsdiff
	3.2786273695961654e-16
julia> automatic_Hessians[27]
	3×3 Matrix{Float64}:
	 -9.86457e-6   -0.00168064  -0.000634302
	 -0.00168064   -0.0128575    0.00584635
	 -0.000634302   0.00584635   0.00166927
julia> handcoded_Hessians[27]
	3×3 Matrix{Float64}:
	 -9.86457e-6   -0.00168064  -0.000634302
	 -0.00168064   -0.0128575    0.00584635
	 -0.000634302   0.00584635   0.00166927
```
"""
function compareHessians(B::Real, μ::Real, σ²::Real, Ξ::Integer)
	@assert Ξ>0
	𝛏 = B.*(2collect(1:Ξ).-Ξ.-1)./(Ξ-2)
	x₀ = [B, μ, σ²]
	automatic_Hessians = collect(zeros(3,3) for i=1:Ξ)
	handcoded_Hessians = Hessian(B, μ, σ², Ξ)
	for i = 1:Ξ
		f(x) = FHMDDM.accumulatorprobability(i, Ξ, x)
		automatic_Hessians[i] = ForwardDiff.hessian(f, x₀)
	end
	maxabsdiff = map((x,y)-> maximum(abs.(x.-y)), automatic_Hessians, handcoded_Hessians)
	return maximum(maxabsdiff), automatic_Hessians, handcoded_Hessians
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
	compareHessians(B,clicks,Δt,j,k,λ,ϕ,σ²ₐ,σ²ₛ,t,Ξ)

Compare the automatically differentiated and hand-coded second-order partial derivatives with respect to the parameters governing transition dynamics

ARGUMENT
-`B`: bound height
-`clicks`: a structure containing the timing, source, and time step of the auditory clicks in a trial
-`Δt`: width of each time step
-`j`: index of the discrete value of the accumulator in the previous time step that is being conditioned on. The second-order partial derivatives are computed for each i-th element of p(aₜ = ξᵢ ∣ aₜ₋₁ = ξⱼ)
-`k`: change rate of the adaptation
-`λ`: feedback of the accumulator onto itself
-`ϕ`: strength of adaptation
-`σ²ₐ`: variance of diffusion noise
-`σ²ₛ`: variance of per-click noise
-`t`: index of the time step
-`Ξ`: Number of discrete values into which the accumulator is discretized

RETURN
-`maxabsdiff`: a matrix representing the maximum absolute difference between the automatically computed and hand-coded Hessians for each partial derivative
-`automatic_Hessians`: a vector of matrices whose i-th element is the automatically computed Hessian matrix of p(aₜ = ξᵢ ∣ aₜ₋₁ = ξⱼ)
-`handcoded_Hessians`: a vector of matrices whose i-th element is the hand-coded Hessian matrix of p(aₜ = ξᵢ ∣ aₜ₋₁ = ξⱼ)

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> clicks = FHMDDM.sampleclicks(0.01, 40, 0.01, 20, 1; rng=MersenneTwister(1234))
julia> maxabsdiff, automatic_Hessians, handcoded_Hessians = FHMDDM.compareHessians(10.0, clicks, 0.01, 20, 0.5, -0.5, 0.8, 2.0, 0.5, 6, 53)
julia> maxabsdiff
```
"""
function compareHessians(B::Real,
						 clicks::Clicks,
						 Δt::Real,
						 j::Integer,
						 k::Real,
						 λ::Real,
						 ϕ::Real,
						 σ²ₐ::Real,
						 σ²ₛ::Real,
						 t::Integer,
						 Ξ::Integer)
	@assert t>1
	@assert t<=length(clicks.inputindex)
	@assert Ξ>0
	@assert	j<=Ξ
	C, dC_dk, dC_dϕ, d²C_dkdk, d²C_dkdϕ, d²C_dϕdϕ = ∇∇adapt(clicks,k,ϕ)
	automatic_Hessians, handcoded_Hessians = collect(zeros(6,6) for i=1:Ξ), collect(zeros(6,6) for i=1:Ξ)
	cL = sum(C[clicks.left[t]])
	cR = sum(C[clicks.right[t]])
	Δc = cR-cL
	Σc = cR+cL
	∂𝛏_∂B = (2collect(1:Ξ).-Ξ.-1)./(Ξ-2)
	𝛏 = B.*∂𝛏_∂B
	λΔt = λ*Δt
	expλΔt = exp(λΔt)
	𝛈 = ∂𝛏_∂B .- expλΔt.*∂𝛏_∂B[j]
	𝛚 = 𝛈.*(Ξ-2)/2
	dμ_dΔc = differentiate_μ_wrt_Δc(Δt, λ)
	μ = expλΔt*𝛏[j] + Δc*dμ_dΔc
	σ² = Σc*σ²ₛ + Δt*σ²ₐ
	σ = √σ²

	𝛑, d𝛑_dB, d𝛑_dμ, d𝛑_dσ², d²𝛑_dBdB, d²𝛑_dBdμ, d²𝛑_dBdσ², d²𝛑_dμdμ, d²𝛑_dμdσ², d²𝛑_dσ²dσ² = zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ)
	CΦ, Δf, ΔΦ, 𝐟, Φ, 𝐳 = probabilityvector!(𝛑, d𝛑_dμ, d𝛑_dσ², d𝛑_dB, μ, 𝛚, σ, 𝛏)
	Ξ_1 = Ξ-1
	Δξ=𝛏[2]-𝛏[1]
	fη = 𝐟.*𝛈
	Δfη = diff(fη)
	Δfω = diff(𝐟.*𝛚)
	Δfωz = diff(𝐟.*𝛚.*𝐳)
	Δfz = diff(𝐟.*𝐳)
	Δζ = diff(𝐟.*(𝐳.^2 .- 1.0)./4.0./σ.^3.0./Δξ)
	Δξσ²2 = Δξ*σ^2*2
	for i=1:Ξ
		if i == 1
			d²𝛑_dBdB[i] 	= ((fη[1] + 𝛚[2]*Δfη[1])/σ - 2d𝛑_dB[1])/B
			d²𝛑_dBdμ[i] 	= (-Δfω[1]/σ - d𝛑_dμ[1])/B
			d²𝛑_dBdσ²[i] 	= (-Δfωz[1]/2/σ² - d𝛑_dσ²[1])/B
			d²𝛑_dμdσ²[i] 	= Δfz[1]/Δξσ²2
			d²𝛑_dσ²dσ²[i]	= Δζ[1]
		elseif i < Ξ
			d²𝛑_dBdB[i] 	= ((𝛚[i+1]*Δfη[i] - 𝛚[i-1]*Δfη[i-1])/σ - 2d𝛑_dB[i])/B
			d²𝛑_dBdμ[i] 	= ((Δfω[i-1]-Δfω[i])/σ - d𝛑_dμ[i])/B
			d²𝛑_dBdσ²[i] 	= ((Δfωz[i-1]-Δfωz[i])/2/σ² - d𝛑_dσ²[i])/B
			d²𝛑_dμdσ²[i] 	= (Δfz[i]-Δfz[i-1])/Δξσ²2
			d²𝛑_dσ²dσ²[i] 	= Δζ[i] - Δζ[i-1]
		else
			d²𝛑_dBdB[i]	= -((fη[Ξ] + 𝛚[Ξ_1]*Δfη[Ξ_1])/σ + 2d𝛑_dB[Ξ])/B
			d²𝛑_dBdμ[i]	= (Δfω[Ξ_1]/σ - d𝛑_dμ[Ξ])/B
			d²𝛑_dBdσ²[i] 	= (Δfωz[Ξ_1]/2/σ² - d𝛑_dσ²[Ξ])/B
			d²𝛑_dμdσ²[i] 	= -Δfz[Ξ_1]/Δξσ²2
			d²𝛑_dσ²dσ²[i] 	= -Δζ[Ξ_1]
		end
		d²𝛑_dμdμ[i] = 2d𝛑_dσ²[i]
	end

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
	d²μ_dkdk = d²Δc_dkdk*dμ_dΔc
	d²μ_dkdϕ = d²Δc_dkdϕ*dμ_dΔc
	d²μ_dϕdϕ = d²Δc_dϕdϕ*dμ_dΔc
	dσ²_dΣc = σ²ₛ
	dσ²_dk = dΣc_dk*dσ²_dΣc
	dσ²_dϕ = dΣc_dϕ*dσ²_dΣc
	d²σ²_dkdk = d²Σc_dkdk*dσ²_dΣc
	d²σ²_dkdϕ = d²Σc_dkdϕ*dσ²_dΣc
	d²σ²_dϕdϕ = d²Σc_dϕdϕ*dσ²_dΣc
	expλΔt = exp(λΔt)
	d²μ_dΔcdλ = differentiate_μ_wrt_Δcλ(Δt, λ)
	dμ_dλ = Δt*expλΔt*𝛏[j] + Δc*d²μ_dΔcdλ
	dξ_dB = (2j-Ξ-1)/(Ξ-2)
	d²μ_dBdλ = Δt*expλΔt*dξ_dB
	d³μ_dΔcdλdλ = differentiate_μ_wrt_Δcλλ(Δt, λ)
	d²μ_dλdλ = Δt^2*expλΔt*𝛏[j] + Δc*d³μ_dΔcdλdλ
	for i = 1:Ξ
		handcoded_Hessians[i][1,1] = d²𝛑_dBdB[i]
		handcoded_Hessians[i][1,2] = handcoded_Hessians[i][2,1] = dμ_dk*d²𝛑_dBdμ[i] + dσ²_dk*d²𝛑_dBdσ²[i] #d²πᵢ_dBdk
		handcoded_Hessians[i][1,3] = handcoded_Hessians[i][3,1] = d²μ_dBdλ*d𝛑_dμ[i] + dμ_dλ*d²𝛑_dBdμ[i] #d²πᵢ_dBdλ
		handcoded_Hessians[i][1,4] = handcoded_Hessians[i][4,1] = dμ_dϕ*d²𝛑_dBdμ[i] + dσ²_dϕ*d²𝛑_dBdσ²[i] #d²πᵢ_dBdϕ
		handcoded_Hessians[i][1,5] = handcoded_Hessians[i][5,1] = Δt*d²𝛑_dBdσ²[i] #d²πᵢ_dBdσ²ₐ
		handcoded_Hessians[i][1,6] = handcoded_Hessians[i][6,1] = Σc*d²𝛑_dBdσ²[i] #d²πᵢ_dBdσ²ₛ
		handcoded_Hessians[i][2,2] = d²μ_dkdk*d𝛑_dμ[i] + d²σ²_dkdk*d𝛑_dσ²[i] + dμ_dk^2*d²𝛑_dμdμ[i] + 2dμ_dk*dσ²_dk*d²𝛑_dμdσ²[i] + dσ²_dk^2*d²𝛑_dσ²dσ²[i] #d²πᵢ_dkdk
		d²πᵢ_dλdΔc = d²μ_dΔcdλ*d𝛑_dμ[i] + dμ_dΔc*dμ_dλ*d²𝛑_dμdμ[i]
		d²πᵢ_dλdΣc = dσ²_dΣc*dμ_dλ*d²𝛑_dμdσ²[i]
		handcoded_Hessians[i][2,3] = handcoded_Hessians[i][3,2] = dΔc_dk*d²πᵢ_dλdΔc + dΣc_dk*d²πᵢ_dλdΣc #d²πᵢ_dkdλ
		handcoded_Hessians[i][2,4] = handcoded_Hessians[i][4,2] = d²μ_dkdϕ*d𝛑_dμ[i] + d²σ²_dkdϕ*d𝛑_dσ²[i] + dμ_dk*dμ_dϕ*d²𝛑_dμdμ[i] + d²𝛑_dμdσ²[i]*(dμ_dϕ*dσ²_dk + dσ²_dϕ*dμ_dk) + dσ²_dk*dσ²_dϕ*d²𝛑_dσ²dσ²[i]  #d²πᵢ_dkdϕ
		d²πᵢ_dkdσ² = dμ_dk*d²𝛑_dμdσ²[i] + dσ²_dk*d²𝛑_dσ²dσ²[i]
		handcoded_Hessians[i][2,5] = handcoded_Hessians[i][5,2] = Δt*d²πᵢ_dkdσ² #d²πᵢ_dkdσ²ₐ
		handcoded_Hessians[i][2,6] = handcoded_Hessians[i][6,2] = Σc*d²πᵢ_dkdσ² + dΔc_dk*d𝛑_dσ²[i] #d²πᵢ_dkdσ²ₛ
		handcoded_Hessians[i][3,3] = d²μ_dλdλ*d𝛑_dμ[i] + (dμ_dλ)^2*d²𝛑_dμdμ[i] #d²πᵢ_dλdλ
		handcoded_Hessians[i][3,4] = handcoded_Hessians[i][4,3] = dΔc_dϕ*d²πᵢ_dλdΔc + dΣc_dϕ*d²πᵢ_dλdΣc #d²πᵢ_dλdϕ
		d²πᵢ_dλdσ² = dμ_dλ*d²𝛑_dμdσ²[i]
		handcoded_Hessians[i][3,5] = handcoded_Hessians[i][5,3] = Δt*d²πᵢ_dλdσ² #d²πᵢ_dλdσ²ₐ
		handcoded_Hessians[i][3,6] = handcoded_Hessians[i][6,3] = Σc*d²πᵢ_dλdσ² #d²πᵢ_dλdσ²ₛ
		handcoded_Hessians[i][4,4] = d²μ_dϕdϕ*d𝛑_dμ[i] + d²σ²_dϕdϕ*d𝛑_dσ²[i] + dμ_dϕ^2*d²𝛑_dμdμ[i] + 2dμ_dϕ*dσ²_dϕ*d²𝛑_dμdσ²[i] + dσ²_dϕ^2*d²𝛑_dσ²dσ²[i] #d²πᵢ_dϕdϕ
		d²πᵢ_dϕdσ² = dμ_dϕ*d²𝛑_dμdσ²[i] + dσ²_dϕ*d²𝛑_dσ²dσ²[i]
		handcoded_Hessians[i][4,5] = handcoded_Hessians[i][5,4] = Δt*d²πᵢ_dϕdσ² #d²πᵢ_dϕdσ²ₐ
		handcoded_Hessians[i][4,6] = handcoded_Hessians[i][6,4] = Σc*d²πᵢ_dϕdσ² + dΔc_dϕ*d𝛑_dσ²[i] #d²πᵢ_dϕdσ²ₛ
		handcoded_Hessians[i][5,5] = Δt^2*d²𝛑_dσ²dσ²[i] #d²πᵢ_dσ²ₐdσ²ₐ
		handcoded_Hessians[i][5,6] = handcoded_Hessians[i][6,5] = Δt*Σc*d²𝛑_dσ²dσ²[i] #d²πᵢ_dσ²ₐdσ²ₛ
		handcoded_Hessians[i][6,6] = Σc^2*d²𝛑_dσ²dσ²[i] #d²πᵢ_dσ²ₛdσ²ₛ
	end
	x₀ = [B, k, λ, ϕ, σ²ₐ, σ²ₛ]
	for i = 1:Ξ
		f(x) = FHMDDM.accumulatorprobability(clicks, Δt, i, j, t, Ξ, x)
		automatic_Hessians[i] = ForwardDiff.hessian(f, x₀)
	end
	maxabsdiff = zeros(6,6)
	for i = 1:length(automatic_Hessians)
	for j in eachindex(maxabsdiff)
	    maxabsdiff[j] = max(maxabsdiff[j], abs(automatic_Hessians[i][j] - handcoded_Hessians[i][j]))
	end
	end
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
    C = adapt(clicks, k, ϕ)
    cL = sum(C[clicks.left[t]])
    cR = sum(C[clicks.right[t]])
	𝛏 = B.*(2 .*collect(1:Ξ) .- Ξ .- 1)./(Ξ-2)
	μ = conditionedmean(cR-cL, Δt, λ, 𝛏[j])
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
-`B`: bound height
-`μ`: mean of the Gaussian distribution
-`σ`: standard deviation of the Gaussian distribution
-`Ξ`: number of values into which the accumulator is discretized

RETURN
-`𝗛`: a vector whose element 𝗛[i] is the 3x3 Hessian matrix of the i-th element of a the probability vector with respect to B, μ, and σ²

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> 𝗛 = Hessian(10, 1, 4, 53);
julia> 𝗛[27]
	3×3 Matrix{Float64}:
	 -9.86457e-6   -0.00168064  -0.000634302
	 -0.00168064   -0.0128575    0.00584635
	 -0.000634302   0.00584635   0.00166927
```
"""
function Hessian(B::Real, μ::Real, σ²::Real, Ξ::Integer)
    @assert Ξ>0
	𝛏 = B.*(2collect(1:Ξ).-Ξ.-1)./(Ξ-2)
	𝛑, d𝛑_dB, d𝛑_dμ, d𝛑_dσ², d²𝛑_dBdB, d²𝛑_dBdμ, d²𝛑_dBdσ², d²𝛑_dμdμ, d²𝛑_dμdσ², d²𝛑_dσ²dσ² = zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ), zeros(Ξ)
	probabilityvector!(𝛑, d𝛑_dB, d𝛑_dμ, d𝛑_dσ², d²𝛑_dBdB, d²𝛑_dBdμ, d²𝛑_dBdσ², d²𝛑_dμdμ, d²𝛑_dμdσ², d²𝛑_dσ²dσ², μ, √σ², 𝛏)
	𝗛 = collect(zeros(3,3) for i=1:Ξ)
	for i=1:Ξ
		𝗛[i][1,1] = d²𝛑_dBdB[i]
		𝗛[i][1,2] = 𝗛[i][2,1] = d²𝛑_dBdμ[i]
		𝗛[i][1,3] = 𝗛[i][3,1] = d²𝛑_dBdσ²[i]
		𝗛[i][2,2] = d²𝛑_dμdμ[i]
		𝗛[i][2,3] = 𝗛[i][3,2] = d²𝛑_dμdσ²[i]
		𝗛[i][3,3] = d²𝛑_dσ²dσ²[i]
	end
	return 𝗛
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
