"""
    save

Save the model parameters and the expectation of the emissions

ARGUMENT
-model: an instance of `DDLM`, a drift-diffusion linear model

RETURN
-nothing
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
