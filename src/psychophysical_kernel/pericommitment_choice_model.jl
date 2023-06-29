"""
    PericommitmentKernel(trials, commitmenttimes)

Infer the psychophysical kernel aligned to the decision commitment

ARGUMENT
-`trial`: A vector whose elements are of the type `Trial`

ARGUMENT
-`commitment_timesteps`: the time step on each trial when commitment is assumed to have occured
-`nfunctions`: number of temporal basis functions
-`trials`: a vector whose each element contains the data for one trial

OPTIONAL ARGUMENT
-`α`: maximum lapse rate
-`Δt`: duration, in seconds, of each time step
-`kfold`: number of cross-validation folds
-`latency`: number of time steps the clicks are shifted
-`T`: number of time steps
"""
function PericommitmentKernel(commitment_timesteps::Vector{<:Integer}, nfunctions::Integer, trials::Vector{<:Trial}; α::AbstractFloat=0.2, Δt::AbstractFloat=0.01, Tpre::Integer=20, Tpost::Integer=10)
    trialsanalyzed =map(commitment_timesteps, trials) do tcommit, trial
                        t_lastinput = isempty(trial.clicks.inputtimesteps) ? 0 : maximum(trial.clicks.inputtimesteps)
                        (tcommit > Tpre) && (t_lastinput >= tcommit+Tpost-1)
                    end
    commitment_timesteps = commitment_timesteps[trialsanalyzed]
    trials = trials[trialsanalyzed]
    𝐝 = BitArray(collect(trial.choice for trial in trials))
    𝐂 = map(commitment_timesteps, trials) do tcommit, trial
            leftpulses = collect(length(left) for left in trial.clicks.left)
            rightpulses = collect(length(right) for right in trial.clicks.right)
            𝐜 = rightpulses .- leftpulses
            timesteps = tcommit-Tpre:tcommit+Tpost-1
            𝐜[timesteps]
        end
    𝐂 = vcat(collect(𝐜' for 𝐜 in 𝐂)...)
    𝛌Δt = collect(gamma2lambda(trial.γ)*Δt for trial in trials)
    𝐄 = 𝐂 .- 𝛌Δt
    T = Tpre+Tpost
    𝐗 = hcat(ones(length(trials)), 𝛌Δt.*T)
    if nfunctions > 0
        if nfunctions > 1
            begins0=false
            ends0=false
            stretch=1e-6
            scalefactor = 1.0
            Φ = temporal_basis_functions(begins0, ends0, nfunctions, T, scalefactor, stretch; orthogonal_to_ones=false)
        else nfunctions == 1
            Φ = ones(T,1)./T
        end
        𝐗 = hcat(𝐗, 𝐄*Φ)
    else
        Φ = zeros(T,0)
    end
    times_s = Δt.*collect(-Tpre:Tpost-1)
    PericommitmentKernel(α=α,
                        𝐂=𝐂,
                        commitment_timesteps=commitment_timesteps,
                        𝐝=𝐝,
                        Δt=Δt,
                        𝐄=𝐄,
                        𝛌Δt=𝛌Δt,
                        Φ=Φ,
                        times_s=times_s,
                        Tpost=Tpost,
                        Tpre=Tpre,
                        𝐗=𝐗)
end

"""
    PCPKCrossValidation(kfold, model)

Cross-validated models for inferring the psychophysical kernel aligned to the decision commitment

ARGUMENT
-`kfold`: number of cross-validation folds
-`model`: an object containing the data and hyperparameters of a per
"""
function PCPKCrossValidation(kfold::Integer, model::PericommitmentKernel)
    ntrials = length(model.𝐝)
    testingtrials, trainingtrials = cvpartition(kfold, ntrials)
    testingmodels = map(testingtrials) do testingtrials
                        PericommitmentKernel(α=model.α,
                                            𝐂 = model.𝐂[testingtrials,:],
                                            commitment_timesteps=model.commitment_timesteps[testingtrials],
                                            𝐝 = model.𝐝[testingtrials],
                                            𝐄=model.𝐄[testingtrials,:],
                                            Δt = model.Δt,
                                            𝛌Δt=model.𝛌Δt[testingtrials],
                                            Φ=model.Φ,
                                            times_s=model.times_s,
                                            Tpost=model.Tpost,
                                            Tpre=model.Tpre,
                                            𝐗 = model.𝐗[testingtrials,:])
                    end
    trainingmodels = map(trainingtrials) do trainingtrials
                        PericommitmentKernel(α=model.α,
                                            𝐂 = model.𝐂[trainingtrials,:],
                                            commitment_timesteps=model.commitment_timesteps[trainingtrials],
                                            𝐝 = model.𝐝[trainingtrials],
                                            𝐄=model.𝐄[trainingtrials,:],
                                            Δt = model.Δt,
                                            𝛌Δt=model.𝛌Δt[trainingtrials],
                                            Φ=model.Φ,
                                            times_s=model.times_s,
                                            Tpost=model.Tpost,
                                            Tpre=model.Tpre,
                                            𝐗 = model.𝐗[trainingtrials,:])
                    end
    PCPKCrossValidation(model=model,
                        testingmodels=testingmodels,
                        trainingmodels=trainingmodels,
                        testingtrials=testingtrials,
                        trainingtrials=trainingtrials)
end

"""
    crossvalidate!(cv)

Make out-of-sample predictions

OPTIONAL ARGUMENT
-`α`: hyperparameter for L2-regularization
"""
function crossvalidate!(cv::PCPKCrossValidation; α::AbstractFloat=0.0)
    for trainingmodel in cv.trainingmodels
        maximizeposterior!(trainingmodel; α=α)
    end
    for k = 1:cv.kfold
        cv.testingmodels[k].𝛃 .= cv.trainingmodels[k].𝛃
    end
    predict!(cv)
    return nothing
end

"""
    maximizeposterior!(model)

Find the maximum-a-posteriori parameters of an psychophysical kernel model.

The optimization algorithm is done using the quasi-Newton algorithm L-BFGS. The gradient of the log-likelihood is computed using automatic differentiation.

OPTIONAL ARGUMENT
-`α`: hyperparameter for L2-regularization
-`nrepeat`: number of repeats
"""
function maximizeposterior!(model::PericommitmentKernel; α::AbstractFloat=0.0, nrepeat::Integer=5, show_trace::Bool=false)
    f(𝛃) = -loglikelihood(𝛃, model) + α*dot(𝛃,𝛃)
    minloss = Inf
    𝛃ML = copy(model.𝛃)
    for i = 1:nrepeat
        𝛃₀ = 1.0 .- 2.0.*rand(length(model.𝛃))
        od = OnceDifferentiable(f, 𝛃₀; autodiff = :forward);
        optimizer = LBFGS(linesearch = LineSearches.BackTracking())
        options = Optim.Options(show_trace=show_trace)
        optimizationresults = optimize(od, 𝛃₀, optimizer, options)
        show_trace && println(optimizationresults)
        thisloss = Optim.minimum(optimizationresults)
        if thisloss < minloss
            𝛃ML = Optim.minimizer(optimizationresults)
            minloss = thisloss
        end
    end
    model.𝛃 .= 𝛃ML
    model.hessian .= hessian(model; α=α)
    return nothing
end

"""
    loglikelihood(𝛃, model)

Log-likelihood of the parameters of a psychophysical kernel model given the data
"""
function loglikelihood(𝛃::Vector{<:Real}, model::PericommitmentKernel)
    @unpack α, 𝐝, 𝐗 = model
    𝐋 = 𝐗*𝛃[2:end]
    ℓ = 0
    ψ = α*logistic(𝛃[1])
    pright_lapse = logistic(𝛃[2])
    for i = 1:length(𝐝)
        pright = ψ*pright_lapse + (1-ψ)*logistic(𝐋[i])
        ℓ += 𝐝[i] ? log(pright) : log(1-pright)
    end
    return ℓ
end

"""
    predict!(cv)

Make out-of-sample predictions
"""
function predict!(cv::PCPKCrossValidation)
    𝐩right =   map(cv.testingmodels) do testingmodel
                    @unpack α, 𝛃, 𝐝, 𝐗 = testingmodel
                    𝐋 = 𝐗*𝛃[2:end]
                    ψ = α*logistic(𝛃[1])
                    pright_lapse = logistic(𝛃[2])
                    (ψ*pright_lapse) .+ (1-ψ).*logistic.(𝐋)
                end
    𝐩right = vcat(𝐩right...)
    sortindices = sortperm(vcat(cv.testingtrials...))
    cv.𝐩right .= 𝐩right[sortindices]
    for i = 1:length(cv.ℓ)
        p = cv.𝐩right[i]
        cv.ℓ[i] = cv.model.𝐝[i] ? log(p) : log(1-p)
    end
    ℓbernoulli = map(cv.testingmodels, cv.trainingmodels) do testingmodel, trainingmodel
                    p = mean(trainingmodel.𝐝)
                    collect(d ? log(p) : log(1-p) for d in testingmodel.𝐝)
                end
    ℓbernoulli = vcat(ℓbernoulli...)
    cv.ℓbernoulli .= ℓbernoulli[sortindices]
    if size(cv.testingmodels[1].Φ,2) > 0
        for k = 1:cv.kfold
            @unpack Φ, 𝛃 = cv.testingmodels[k]
            cv.Φ𝐰[k] .= Φ*𝛃[4:end]
            cv.psychophysicalkernel[k] .= 𝛃[3] .+ cv.Φ𝐰[k]
        end
    end
    return nothing
end

"""
    hessian(model)

Hessian matrix of the log-posterior
"""
function hessian(model::PericommitmentKernel; α::AbstractFloat)
    f(𝛃) = -loglikelihood(𝛃, model) + α*dot(𝛃,𝛃)
    ForwardDiff.hessian(f, model.𝛃)
end

"""
    save(cv, filepath)

Save the cross-validation results of an psychophysical kernel model
"""
function save(cv::PCPKCrossValidation, filepath::String)
    dict = Dict("betas"=>collect(trainingmodel.𝛃 for trainingmodel in cv.trainingmodels),
                "choice"=>cv.model.𝐝,
                "clickdifference_hz"=>vec(mean(cv.model.𝐂, dims=2))./cv.model.Δt,
                "pright"=>cv.𝐩right,
                "loglikelihood"=>cv.ℓ,
                "loglikelihood_bernoulli"=>cv.ℓbernoulli,
                "psychophysicalkernel"=>cv.psychophysicalkernel,
                "fluctuationsweight"=>cv.Φ𝐰,
                "temporal_basis_functions"=>cv.testingmodels[1].Φ,
                "times_s"=>cv.model.times_s,
                "hessians"=>collect(trainingmodel.hessian for trainingmodel in cv.trainingmodels))
    matwrite(filepath, dict)
end
