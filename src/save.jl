"""
    save(model)

Save the results of a model into file compatible with both MATLAB and Julia
"""
function save(model::Model; filename="results.mat")
    dict = dictionary(model)
    path = joinpath(dirname(model.options.resultspath), filename)
    matwrite(path, dict)
    return nothing
end

"""
    save(model, predictions)

Save the model parameters and the expectation of the emissions
"""
function save(model::Model, predictions::Predictions; filename="results.mat")
    modeldict = dictionary(model)
    predictionsdict = dictionary(predictions)
    dict = merge(modeldict, predictionsdict)
    path = joinpath(dirname(model.options.resultspath), filename)
    matwrite(path, dict)
    return nothing
end

"""
    save(hessian_loglikelihood, model, predictions)

Save the model parameters, predictions, and hessian of the log-likelihood
"""
function save(hessian_loglikelihood::Matrix{<:AbstractFloat}, model::Model, predictions::Predictions; filename="results.mat")
    modeldict = dictionary(model)
    predictionsdict = dictionary(predictions)
    hessiandict = Dict("hessian_loglikelihood"=>hessian_loglikelihood)
    dict = merge(modeldict, predictionsdict, hessiandict)
    path = joinpath(dirname(model.options.resultspath), filename)
    matwrite(path, dict)
    return nothing
end

"""
    save(model, λΔt, pchoice)

Save the model parameters and the expectation of the emissions
"""
function save(λΔt::Vector{<:Vector{<:Vector{<:AbstractFloat}}}, model::Model, pchoice::Vector{<:Vector{<:AbstractFloat}}; filename="results.mat")
    modeldict = dictionary(model)
    dict = Dict("pchoice" => pchoice,
                "lambdaDeltat" => λΔt)
    path = joinpath(dirname(model.options.resultspath), filename)
    matwrite(path, dict)
    return nothing
end

"""
    save(hessian_loglikelihood, λΔt, model, pchoice)

Save the model parameters and the expectation of the emissions and a hessian
"""
function save(hessian_loglikelihood::Matrix{<:AbstractFloat}, model::Model; filename="preinitialization.mat")
    dict = Dict("theta_native"=> dictionary(model.θnative),
                "theta_real"=> dictionary(model.θreal),
                "theta0_native" => dictionary(model.θ₀native),
                "thetaglm"=>map(trialset->map(mpGLM->dictionary(mpGLM.θ), trialset.mpGLMs), model.trialsets),
                "Phiaccumulator"=>model.trialsets[1].mpGLMs[1].Φₐ,
                "Phihistory"=>model.trialsets[1].mpGLMs[1].Φₕ,
                "Phipremovement"=>model.trialsets[1].mpGLMs[1].Φₘ,
                "Phitime"=>model.trialsets[1].mpGLMs[1].Φₜ,
                "penaltycoefficients"=>model.gaussianprior.𝛂,
                "penaltymatrices"=>model.gaussianprior.𝐀,
                "penaltymatrixindices"=>model.gaussianprior.index𝐀,
                "precisionmatrix"=>model.gaussianprior.𝚲,
                "hessian_loglikelihood"=>hessian_loglikelihood)
    path = joinpath(dirname(model.options.resultspath), filename)
    matwrite(path, dict)
    return nothing
end

"""
    save(hessian_loglikelihood, λΔt, model, pchoice)

Save the model parameters and the expectation of the emissions and a hessian
"""
function save(hessian_loglikelihood::Matrix{<:AbstractFloat},
              λΔt::Vector{<:Vector{<:Vector{<:AbstractFloat}}},
              model::Model,
              pchoice::Vector{<:Vector{<:AbstractFloat}}; filename="results.mat")
    dict = Dict("theta_native"=> dictionary(model.θnative),
                "theta_real"=> dictionary(model.θreal),
                "theta0_native" => dictionary(model.θ₀native),
                "thetaglm"=>map(trialset->map(mpGLM->dictionary(mpGLM.θ), trialset.mpGLMs), model.trialsets),
                "Phiaccumulator"=>model.trialsets[1].mpGLMs[1].Φₐ,
                "Phihistory"=>model.trialsets[1].mpGLMs[1].Φₕ,
                "Phipremovement"=>model.trialsets[1].mpGLMs[1].Φₘ,
                "Phitime"=>model.trialsets[1].mpGLMs[1].Φₜ,
                "penaltycoefficients"=>model.gaussianprior.𝛂,
                "penaltymatrices"=>model.gaussianprior.𝐀,
                "penaltymatrixindices"=>model.gaussianprior.index𝐀,
                "precisionmatrix"=>model.gaussianprior.𝚲,
                "hessian_loglikelihood"=>hessian_loglikelihood,
                "pchoice" => pchoice,
                "lambdaDeltat" => λΔt)
    path = joinpath(dirname(model.options.resultspath), filename)
    matwrite(path, dict)
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
