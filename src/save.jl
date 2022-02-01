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
function save(model::FHMDDM,
              λΔt::Vector{<:Vector{<:Vector{<:AbstractFloat}}},
              pchoice::Vector{<:Vector{<:AbstractFloat}})
    dict = Dict("theta_native"=> dictionary(model.θnative),
                "theta_real"=> dictionary(model.θreal),
                "theta0_native" => dictionary(model.θ₀native),
                "u"=>map(trialset->map(mpGLM->mpGLM.𝐮, trialset.mpGLMs), model.trialsets),
                "l"=>map(trialset->map(mpGLM->mpGLM.𝐥, trialset.mpGLMs), model.trialsets),
                "r"=>map(trialset->map(mpGLM->mpGLM.𝐫, trialset.mpGLMs), model.trialsets),
                "Phi"=>model.trialsets[1].mpGLMs[1].Φ,
                "pchoice" => pchoice,
                "lambdaDeltat" => λΔt)
    matwrite(model.options.resultspath, dict)
    return nothing
end

"""
    savedata(model)

Save the data

INPUT
-`model`: a FHMDDM

OPTIONAL INPUT
-`overwritedata`: whether to overwrite previous data

OUTPUT
-nothing
"""
function savedata(model::FHMDDM; overwritedata::Bool=false)
    if !isfile(model.options.datapath) || overwritedata
        dict = Dict("data" => map(trialset->Dict(trialset), model.trialsets),
                    "options" => Dict(model.options))
        matwrite(model.options.datapath, dict)
    end
    return nothing
end
