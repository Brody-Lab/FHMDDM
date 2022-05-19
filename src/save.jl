"""
    save

Save the model parameters and the expectation of the emissions

ARGUMENT
-model: a structure containing the parameters, hyperparameters, and data

RETURN
-nothing

EXAMPLE
```julia-repl
julia> using FHMDDM, LineSearches, Optim
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_01_test/data.mat"
julia> model = Model(datapath)
julia> initializeparameters!(model)
julia> losses, gradientnorms = FHMDDM.maximizeposterior!(model, 0.1, LBFGS(linesearch = LineSearches.BackTracking()))
julia> fbz = FHMDDM.posterior_first_state(model)
julia> λΔt, pchoice = FHMDDM.expectedemissions(model)
julia> save(model, fbz, gradientnorms, losses, λΔt, pchoice)
```
"""
function save(model::Model,
              fbz::Vector{<:Vector{<:Vector{<:AbstractFloat}}},
              gradientnorms::Vector{<:AbstractFloat},
              losses::Vector{<:AbstractFloat},
              λΔt::Vector{<:Vector{<:Vector{<:AbstractFloat}}},
              pchoice::Vector{<:Vector{<:AbstractFloat}})
    dict = Dict("theta_native"=> dictionary(model.θnative),
                "theta_real"=> dictionary(model.θreal),
                "theta0_native" => dictionary(model.θ₀native),
                "thetaglm"=>map(trialset->map(mpGLM->dictionary(mpGLM.θ), trialset.mpGLMs), model.trialsets),
                "Phi"=>model.trialsets[1].mpGLMs[1].Φ,
                "fbz"=>fbz,
                "gradientnorms"=>gradientnorms,
                "losses"=>losses,
                "pchoice" => pchoice,
                "lambdaDeltat" => λΔt)
    matwrite(model.options.resultspath, dict)
    return nothing
end

"""
"""
function save(model::Model)
    dict = Dict("theta_native"=> dictionary(model.θnative),
                "theta_real"=> dictionary(model.θreal),
                "theta0_native" => dictionary(model.θ₀native),
                "thetaglm"=>map(trialset->map(mpGLM->dictionary(mpGLM.θ), trialset.mpGLMs), model.trialsets),
                "Phi"=>model.trialsets[1].mpGLMs[1].Φ)
    matwrite(model.options.resultspath, dict)
    return nothing
end

"""
    save

Save the results of crossvalidation

ARGUMENT
-`cvresults`: an instance of `CVResults`, a drift-diffusion linear model
"""
function save(cvresults::CVResults, options::Options)
    path = dirname(options.resultspath)*"/cvresults.mat"
    matwrite(path, dictionary(cvresults))
    return nothing
end
