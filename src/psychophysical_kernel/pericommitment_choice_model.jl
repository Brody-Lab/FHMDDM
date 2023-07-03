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
-`Tpre`: minimum number of time steps with stimulus input before commitment
-`Tpost`: minimum number of time steps with stimulus input after commitment
-`η`: a positive scalar indicating the magnitude to which basis functions evaluated near the `startingvalue` is compressed and functions evaluated far from the `startingvalue` is stretched
"""
function PericommitmentKernel(commitment_timesteps::Vector{<:Integer}, nfunctions::Integer, trials::Vector{<:Trial}; α::AbstractFloat=0.2, Δt::AbstractFloat=0.01, Tpre::Integer=20, Tpost::Integer=20, η::Real=0.0, omit_t0::Bool=true)
    trialsanalyzed =map(commitment_timesteps, trials) do tcommit, trial
                        t_lastinput = isempty(trial.clicks.inputtimesteps) ? 0 : maximum(trial.clicks.inputtimesteps)
                        (tcommit > Tpre) && (t_lastinput >= tcommit+Tpost)
                    end
    commitment_timesteps = commitment_timesteps[trialsanalyzed]
    trials = trials[trialsanalyzed]
    𝐝 = BitArray(collect(trial.choice for trial in trials))
    𝐂 = map(commitment_timesteps, trials) do tcommit, trial
            leftpulses = collect(length(left) for left in trial.clicks.left)
            rightpulses = collect(length(right) for right in trial.clicks.right)
            𝐜 = rightpulses .- leftpulses
            if omit_t0
                timesteps = vcat(tcommit-Tpre:tcommit-1, tcommit+1:tcommit+Tpost)
            else
                timesteps = vcat(tcommit-Tpre:tcommit+Tpost)
            end
            𝐜[timesteps]
        end
    𝐂 = vcat(collect(𝐜' for 𝐜 in 𝐂)...)
    if omit_t0
        times_s = Δt.*vcat((-Tpre:-1)..., (1:Tpost)...)
    else
        times_s = Δt.*collect(-Tpre:Tpost)
    end
    T = length(times_s)
    𝛌Δt = collect(gamma2lambda(trial.γ)*Δt for trial in trials)
    # leftevidence = 𝛌Δt .< 0
    # rightevidence = 𝛌Δt .> 0
    # 𝐯 = mean(𝐂[rightevidence,:], dims=1) .- mean(𝐂[leftevidence,:], dims=1)
    # 𝐯 = vec(𝐯)
    # 𝐯 ./= norm(𝐯)
    # 𝐚 = 𝐂*𝐯
    # 𝐄 = 𝐂 .- 𝐚*𝐯'
    # 𝐗 = hcat(ones(length(trials)), 𝐚)
    𝐄 = 𝐂 .- 𝛌Δt
    𝐗 = hcat(ones(length(trials)), 𝛌Δt.*T)
    if nfunctions > 0
        if nfunctions == T
            Φ₀ = Φ = Matrix(1.0I, T, T)
        elseif nfunctions > 1
            if omit_t0
                npre = ceil(Int, nfunctions/2)
                npost = floor(Int, nfunctions/2)
                Φ₀pre = FHMDDM.basisfunctions(Tpre, npre; η=η, startingvalue=Tpre, orthogonalize=false)
                Φ₀post = FHMDDM.basisfunctions(Tpost, npost; η=η, startingvalue=1, orthogonalize=false)
                Φpre = FHMDDM.orthonormalbasis(Φ₀pre)
                Φpost = FHMDDM.orthonormalbasis(Φ₀post)
                Φ₀ = vcat(hcat(Φ₀pre, zeros(Tpre, npost)), hcat(zeros(Tpost, npre), Φ₀post))
                Φ = vcat(hcat(Φpre, zeros(Tpre, npost)), hcat(zeros(Tpost, npre), Φpost))
            else
                Φ₀ = basisfunctions(T, nfunctions; η=η, startingvalue=Tpre+1, orthogonalize=false)
                Φ = orthonormalbasis(Φ₀)
            end
        else nfunctions == 1
            Φ₀ = Φ = ones(T,1)./T
        end
        𝐗 = hcat(𝐗, 𝐄*Φ)
    else
        Φ₀ = Φ = zeros(T,0)
    end
    PericommitmentKernel(α=α,
                        𝐂=𝐂,
                        commitment_timesteps=commitment_timesteps,
                        𝐝=𝐝,
                        Δt=Δt,
                        𝐄=𝐄,
                        η=η,
                        𝛌Δt=𝛌Δt,
                        Φ₀=Φ₀,
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
                                            η=model.η,
                                            Δt = model.Δt,
                                            𝛌Δt=model.𝛌Δt[testingtrials],
                                            Φ₀=model.Φ₀,
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
                                            η=model.η,
                                            Δt = model.Δt,
                                            𝛌Δt=model.𝛌Δt[trainingtrials],
                                            Φ₀=model.Φ₀,
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
function maximizeposterior!(model::PericommitmentKernel; α::AbstractFloat=0.0, nrepeat::Integer=1, show_trace::Bool=false)
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
                "eta"=>cv.model.η,
                "C"=>cv.model.𝐂,
                "choice"=>cv.model.𝐝,
                "E"=>cv.model.𝐄,
                "lambdaDeltat"=>cv.model.𝛌Δt,
                "pright"=>cv.𝐩right,
                "loglikelihood"=>cv.ℓ,
                "loglikelihood_bernoulli"=>cv.ℓbernoulli,
                "psychophysicalkernel"=>cv.psychophysicalkernel,
                "fluctuationsweight"=>cv.Φ𝐰,
                "temporal_basis_functions"=>cv.testingmodels[1].Φ,
                "temporal_basis_functions_not_orthogonalized"=>cv.testingmodels[1].Φ₀,
                "times_s"=>cv.model.times_s,
                "hessians"=>collect(trainingmodel.hessian for trainingmodel in cv.trainingmodels))
    matwrite(filepath, dict)
end

"""
    basisfunctions(nvalues, nfunctions)

Nonlinear basis functions of an one-dimensional integer input

ARGUMENT
-`nvalues`: number of values of the input tiled by the basis functions
-`D`: number of basis functions

OPTIONAL ARGUMENT
-`startingvalue`: an integer indicating the shift in the input values
-`η`: a positive scalar indicating the magnitude to which basis functions evaluated near the `startingvalue` is compressed and functions evaluated far from the `startingvalue` is stretched
-`begins0`: whether the value of each basis function at the first input value is 0
-`ends0`: whether the value of each basis function at the last input value is 0
-`orthogonal_to_ones:` whether the basis functions should be orthogonal to a constant vector

RETURN
-`Φ`: Matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis function at the i-th input
"""
function basisfunctions(nvalues::Integer, D::Integer; begins0::Bool=false, ends0::Bool=false, η::Real=0.0, startingvalue::Integer=1, orthogonal_to_ones::Bool=false, orthogonalize::Bool=true)
    if D == 1
        Φ = ones(nvalues,1)./nvalues
    else
        𝐱 = collect(1:nvalues) .- startingvalue
        if η > eps()
            𝐱 = asinh.(η.*𝐱)
        end
        Φ = raisedcosines(begins0, D, ends0, 𝐱)
        if orthogonal_to_ones
            Φ = orthogonalize_to_ones(Φ)
        end
        if orthogonalize
            Φ = orthonormalbasis(Φ)
        end
    end
    return Φ
end

"""
    raisedcosines(begins0, D, ends0, 𝐱)

Equally spaced raised cosine basis functions

ARGUMENT
-`begins0`: whether the first temporal basis function begins at the trough or at the peak
-`ends0`: whether the last temporal basis function begins at the trough or at the peak
-`D`: number of bases
-`𝐱`: input to the raised cosine function

RETURN
-`Φ`: Matrix whose element Φ[i,j] corresponds to the value of the j-th temporal basis function at the i-th input
"""
function raisedcosines(begins0::Bool, D::Integer, ends0::Bool, 𝐱::Vector{<:Real})
    Δx = 𝐱[end]-𝐱[1]
    if begins0
        if ends0
            Δcenter = Δx / (D+3)
        else
            Δcenter = Δx / (D+1)
        end
        centers = 𝐱[1] .+ 2Δcenter .+ collect(0:max(1,D-1)).*Δcenter
    else
        if ends0
            Δcenter = Δx / (D+1)
        else
            Δcenter = Δx / (D-1)
        end
        centers = 𝐱[1] .+ collect(0:max(1,D-1)).*Δcenter
    end
    ω = π/2/Δcenter
    Φ = raisedcosines(centers, ω, 𝐱)
    if !begins0 && !ends0 # allow temporal basis functions to parametrize a constant function
        lefttail = raisedcosines([centers[1]-Δcenter], ω, 𝐱)
        righttail = raisedcosines([centers[end]+Δcenter], ω, 𝐱)
        Φ[:,1] += lefttail
        Φ[:,end] += righttail
        indices = 𝐱 .<= centers[1] + 2/Δcenter
        deviations = 2.0 .- sum(Φ,dims=2) # introduced by time compression
        Φ[indices,1] .+= deviations[indices]
    end
    return Φ
end
