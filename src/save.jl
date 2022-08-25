"""
    analyzeandsave(model)

Perform analyses on the model and save the analyses and the model

ARGUMENT
-`model`: a structure containing the data, parameters, and settings

OPTIONAL ARGUMENT
-`prefix`: prefix to the name of the files to be saved
"""
function analyzeandsave(model::Model; prefix="results")
    save(model::Model, prefix)
    folderpath = dirname(model.options.resultspath)
    save(Predictions(model), folderpath, prefix)
    save(∇∇loglikelihood(model)[3], folderpath, prefix)
end

"""
    save(model)

Save the results of a model into file compatible with both MATLAB and Julia

Saved as `model.options.resultspath/<prefix>.mat`

ARGUMENT
-`model`: a structure containing the data, parameters, and settings
-`prefix`: name of the file to be saved
"""
function save(model::Model, prefix::String)
    dict = dictionary(model)
    path = joinpath(dirname(model.options.resultspath), prefix*".mat")
    matwrite(path, dict)
    return nothing
end

"""
    save(predictions, folderpath, prefix)

Save predictions of the model:
-accumulator distribution: `folderpath/<prefix>_pa.mat`
-choice-conditioned accumulator distribution: `folderpath/<prefix>_pa_d.mat`
-choice- and spikes-conditioned accumulator distribution: `folderpath/<prefix>_pa_Yd.mat`
-choice- and spikes-conditioned coupling distribution: `folderpath/<prefix>_pc_Yd.mat`
-probability choosing right: `folderpath/<prefix>_pd.mat`
-spike counts: `folderpath/<prefix>_lambdaDeltat.mat`
-choice-conditioned spike counts: `folderpath/<prefix>_lambdaDeltat_d.mat`

ARGUMENT
-`model`: a structure containing the data, parameters, and settings
-'folderpath': full path of the folder
-`prefix`: name of the file to be saved
"""
function save(predictions::Predictions, folderpath::String, prefix::String)
    matwrite(joinpath(folderpath, prefix*"_pa"*".mat"), Dict("pa" => predictions.p𝐚))
    matwrite(joinpath(folderpath, prefix*"_pa_d"*".mat"), Dict("pa_d" => predictions.p𝐚_𝑑))
    matwrite(joinpath(folderpath, prefix*"_pa_Yd"*".mat"), Dict("pa_Yd" => predictions.p𝐚_𝐘𝑑))
    matwrite(joinpath(folderpath, prefix*"_pc_Yd"*".mat"), Dict("pc_Yd" => predictions.p𝐜_𝐘𝑑))
    matwrite(joinpath(folderpath, prefix*"_pd"*".mat"), Dict("pd" => predictions.p𝑑))
    matwrite(joinpath(folderpath, prefix*"_lambdaDeltat"*".mat"), Dict("lambdaDeltat" => predictions.λΔt))
    matwrite(joinpath(folderpath, prefix*"_lambdaDeltat_d"*".mat"), Dict("lambdaDeltat_d" => predictions.λΔt_𝑑))
    return nothing
end

"""
    save(hessian_loglikelihood, folderpath, prefix)

Save the hessian of the log-likelihood as `folderpath/<prefix>_hessian_loglikelihood.mat`

ARGUMENT
-`model`: a structure containing the data, parameters, and settings
-'folderpath': full path of the folder
-`prefix`: name of the file to be saved
"""
function save(hessian_loglikelihood::Matrix{<:AbstractFloat}, folderpath::String, prefix::String)
    matwrite(joinpath(folderpath, prefix*"_hessian_loglikelihood"*".mat"), Dict("hessian_loglikelihood"=>hessian_loglikelihood))
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
