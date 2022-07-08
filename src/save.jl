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
                "shrinkagecoefficients"=>model.gaussianprior.𝛂,
                "smoothingcoefficients"=>model.gaussianprior.𝐬,
                "fbz"=>fbz,
                "gradientnorms"=>gradientnorms,
                "losses"=>losses,
                "pchoice" => pchoice,
                "lambdaDeltat" => λΔt)
    matwrite(model.options.resultspath, dict)
    return nothing
end

"""
    save(model)

Save the results of a model into file compatible with both MATLAB and Julia
"""
function save(model::Model)
    dict = Dict("theta_native"=> dictionary(model.θnative),
                "theta_real"=> dictionary(model.θreal),
                "theta0_native" => dictionary(model.θ₀native),
                "thetaglm"=>map(trialset->map(mpGLM->dictionary(mpGLM.θ), trialset.mpGLMs), model.trialsets),
                "Phi"=>model.trialsets[1].mpGLMs[1].Φ,
                "shrinkagecoefficients"=>model.gaussianprior.𝛂,
                "smoothingcoefficients"=>model.gaussianprior.𝐬)
    matwrite(model.options.resultspath, dict)
    return nothing
end

"""
    save(model, fbz, λΔt, pchoice)

Save the model parameters and the expectation of the emissions

ARGUMENT
-model: a structure containing the parameters, hyperparameters, and data

RETURN
-nothing

EXAMPLE

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_01_test/T176_2018_05_03/data.mat")
julia> maximizeevidence!(model)
julia> λΔt, pchoice = expectedemissions(model;nsamples=10)
julia> fbz = posterior_first_state(model)
julia> save(model, fbz, λΔt, pchoice)
```
"""
function save(model::Model,
              fbz::Vector{<:Vector{<:Vector{<:AbstractFloat}}},
              λΔt::Vector{<:Vector{<:Vector{<:AbstractFloat}}},
              pchoice::Vector{<:Vector{<:AbstractFloat}})
    dict = Dict("theta_native"=> dictionary(model.θnative),
                "theta_real"=> dictionary(model.θreal),
                "theta0_native" => dictionary(model.θ₀native),
                "thetaglm"=>map(trialset->map(mpGLM->dictionary(mpGLM.θ), trialset.mpGLMs), model.trialsets),
                "Phi"=>model.trialsets[1].mpGLMs[1].Φ,
                "shrinkagecoefficients"=>model.gaussianprior.𝛂,
                "smoothingcoefficients"=>model.gaussianprior.𝐬,
                "fbz"=>fbz,
                "pchoice" => pchoice,
                "lambdaDeltat" => λΔt)
    matwrite(model.options.resultspath, dict)
    return nothing
end

"""
    save(model, λΔt, pchoice)

Save the model parameters and the expectation of the emissions
"""
function save(model::Model,
              λΔt::Vector{<:Vector{<:Vector{<:AbstractFloat}}},
              pchoice::Vector{<:Vector{<:AbstractFloat}})
    dict = Dict("theta_native"=> dictionary(model.θnative),
                "theta_real"=> dictionary(model.θreal),
                "theta0_native" => dictionary(model.θ₀native),
                "thetaglm"=>map(trialset->map(mpGLM->dictionary(mpGLM.θ), trialset.mpGLMs), model.trialsets),
                "Phi"=>model.trialsets[1].mpGLMs[1].Φ,
                "shrinkagecoefficients"=>model.gaussianprior.𝛂,
                "smoothingcoefficients"=>model.gaussianprior.𝐬,
                "pchoice" => pchoice,
                "lambdaDeltat" => λΔt)
    matwrite(model.options.resultspath, dict)
    return nothing
end

"""
    save(cvresults,options)

Save the results of crossvalidation

ARGUMENT
-`cvresults`: an instance of `CVResults`, a drift-diffusion linear model
"""
function save(cvresults::CVResults, options::Options)
    path = dirname(options.resultspath)*"/cvresults.mat"
    matwrite(path, dictionary(cvresults))
    return nothing
end

"""
    save(options, trialsets)
"""
function save(options::Dict, trialsets::Vector{<:Trialset})
    dict = Dict("options"=> options,
                "trialsets"=> map(trialset->dictionary(trialset), trialsets))
    matwrite(options["datapath"], dict)
    return nothing
end
