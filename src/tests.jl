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
