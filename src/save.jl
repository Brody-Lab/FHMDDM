"""
    save

Save the model parameters and the expectation of the emissions

ARGUMENT
-model: an instance of `DDLM`, a drift-diffusion linear model

OPTIONAL ARGUMENT
-`overwritedata`: whether to write over the file of the data (not the results, which is always overwritten), if it exist

RETURN
-nothing
"""
function save(model::Model,
              λΔt::Vector{<:Vector{<:Vector{<:AbstractFloat}}},
              pchoice::Vector{<:Vector{<:AbstractFloat}})
    dict = Dict("theta_native"=> dictionary(model.θnative),
                "theta_real"=> dictionary(model.θreal),
                "theta0_native" => dictionary(model.θ₀native),
                "thetaglm"=>map(trialset->map(mpGLM->dictionary(mpGLM.θ), trialset.mpGLMs), model.trialsets),
                "Phi"=>model.trialsets[1].mpGLMs[1].Φ,
                "pchoice" => pchoice,
                "lambdaDeltat" => λΔt)
    matwrite(model.options.resultspath, dict)
    return nothing
end
