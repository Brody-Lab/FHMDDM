"""
    drift_design_matrix(options, stereoclick_times_s, trialdurations, 𝐲)

The design matrix for regressing a neuron's firing rates against time across trials in a session

The time in a session is parametrized by using basis functions, and the number of basis functions is selected by modelling the firing rate on each trial using a Gaussian linear regression model and finding the model with the greatest out-of-sample log-likelihood

ARGUMENT
-`options`: fixed hyperparameters of the model
-`𝐲`: spike trains of a neuron, concatenated across trials
-`stereoclick_times_s`: the time of the stereoclick on each trial in a session, in seconds
-`ntimesteps_each_trial`: number of time steps on each trial

RETURN
-`Φ`: a matrix with a number of rows equal to the number of trials and a number of columns equal to the optimal number of basis functions
-`𝐔`: a matrix with a number of rows equal to the total number of time steps across trials (i.e., `sum(trialdurations)`) and a number of columns equal to the optimal number of basis functions

OPTIONAL ARGUMENT
-`kfold`: number of cross-validation fold used in selecting the optimal number of basis functions
-`max_number_basis_functions`: maximum number of basis functions
-`time_step_duration_s`: the time resolution for modelling the drift
"""
function drift_design_matrix(options::Options, stereoclick_times_s::Vector{<:AbstractFloat}, trialdurations::Vector{<:Integer}, 𝐲::Vector{<:Integer}; kfold::Integer=10, time_step_duration_s::Real=1.0)
    𝐫 = spikerate(options.Δt, trialdurations, 𝐲)
    stereoclick_timesteps = ceil.(Int, stereoclick_times_s./time_step_duration_s)
	stereoclick_timesteps = stereoclick_timesteps .- minimum(stereoclick_timesteps) .+ 1
	ntrials = length(stereoclick_timesteps)
    mse = zeros(options.tbf_gain_maxfunctions)
    for i = 1:options.tbf_gain_maxfunctions
        Φ = drift_design_matrix(i, options, stereoclick_timesteps, trialdurations)
        testindices, trainindices = cvpartition(kfold, ntrials)
        mse[i] += gaussianmse(testindices, trainindices, Φ, 𝐫)
    end
    Φ = drift_design_matrix(argmin(mse), options, stereoclick_timesteps, trialdurations)
	𝐰 = (Φ'*Φ) \ (Φ'*𝐫)
	Φ = Φ*𝐰
	Φ = reshape(Φ,length(Φ),1)
	𝐔 = vcat((repeat(Φ[m,:]', trialdurations[m]) for m in eachindex(trialdurations))...)
    return Φ, 𝐔
end

"""
    spikerate(Δt, trialdurations, 𝐲)

Compute the spikes per second of a neuron on each trial

ARGUMENT
-`Δt`: duration, in seconds, of the time step of the neuron's spike train
-`trialdurations`: the number of time steps on each trial
-`𝐲`: the spike train of a neuron, concatenated across trials

RETURN
-`𝐫`: a vector whose each element is the spikes per second on a trial
"""
function spikerate(Δt::Real, trialdurations::Vector{<:Integer}, 𝐲::Vector{<:Integer})
    𝐫 = zeros(length(trialdurations))
    ∑durations = 0
    for m in eachindex(trialdurations)
        indices = ∑durations .+ (1:trialdurations[m])
        𝐫[m] = sum(𝐲[indices])/trialdurations[m]/Δt
        ∑durations = indices[end]
    end
    return 𝐫
end

"""
	drift_design_matrix(nfunctions, options, stereoclick_timesteps, trialdurations)

The design matrix for regressing a neuron's firing rates against time across trials in a session

The scaling of the temporal basis vectors is adjusted to be based on the maximum number of time steps on each trial across trials, so that scaling of the regressors are similar to the scaling of other regressors in the Poisson mixture GLM

ARGUMENT
-`nfunctions`: number of basis functions
-`options`: fixed hyperparameters
-`s`: scale factor coefficient
-`stereoclick_timesteps`: the time of the stereoclick on each trial
-`trialdurations`: the number of time steps on each trial

RETURN
-`𝐗`: a  matrix with a number of rows equal to the number of trials (i.e., `length(stereoclick_timesteps)` or `length(trialdurations)`) and a number of columns equal to `nfunctions`
"""
function drift_design_matrix(nfunctions::Integer, options::Options, stereoclick_timesteps::Vector{<:Integer}, trialdurations::Vector{<:Integer})
	scalefactor = options.sf_tbf[1]*options.tbf_gain_scalefactor
	if nfunctions == 0
		error("nfunctions must be > 0")
	elseif nfunctions == 1
		scalingadjustment = scalefactor/sqrt(maximum(trialdurations))
		return fill(scalingadjustment, length(trialdurations), 1)
	else
	    begins0=false
		ends0=false
	    ntimesteps = maximum(stereoclick_timesteps)
		stretch=1e-6
	    Φ = temporal_basis_functions(begins0, ends0, nfunctions, ntimesteps, scalefactor, stretch; orthogonal_to_ones=false)
		scalingadjustment = sqrt(maximum(stereoclick_timesteps)/maximum(trialdurations))
	    return Φ[stereoclick_timesteps,:].*scalingadjustment
	end
end

"""
    gaussianmse(testindices, trainindices, 𝐗, 𝐲)

Out-of-sample mean squared error, which proportional to the log-likelihood, of a gaussian linear model

ARGUMENT
-`testindices`: nested vector whose each k-th element is a vector of integers indicating the indices of the samples used for testing in the k-th cross-validation fold
-`trainindices`: nested vector whose each k-th element is a vector of integers indicating the indices of the samples used for trainining in the k-th cross-validation fold
-`𝐗`: design matrix; i.e., the predictor. Each row corresponds to a sample, and each column a regressor
-`𝐲`: independent variable; i.e., the response variable.

RETURN
-a scalar value representing the out-of-sample mean-squared error
"""
function gaussianmse(testindices::Vector{<:Vector{<:Integer}}, trainindices::Vector{<:Vector{<:Integer}}, 𝐗::Matrix{<:Real}, 𝐲::Vector{<:Real})
    mse = 0
    for (testindices, trainindices) in zip(testindices, trainindices)
        𝐗train = 𝐗[trainindices,:]
        𝐲train = 𝐲[trainindices]
        𝐰 = (𝐗train'*𝐗train) \ (𝐗train'*𝐲train)
        𝐗test = 𝐗[testindices,:]
        𝐲test = 𝐲[testindices]
        𝛆 = (𝐲test - 𝐗test*𝐰)
        mse += (𝛆'*𝛆)*length(testindices)
    end
    mse/sum(length.(testindices))
end

"""
    cvpartition(kfold, n)

Training and test indices for k-fold cross-valiation

ARGMENT
-`kfold`: number of folds
-`n`: total number of samples

RETURN
-`testindices`: a nest array whose each element is a vector of indices for the test set of a fold
-`trainindices`: a nest array whose each element is a vector of indices for the training set of a fold
"""
function cvpartition(kfold::integertype, n::integertype) where {integertype<:Integer}
    @assert kfold > 1
    @assert n >= kfold
    trainindices = map(k->integertype[], 1:kfold)
    testindices = map(k->integertype[], 1:kfold)
    ntestmax = cld(n, kfold)
    partitioned = collect(Base.Iterators.partition(shuffle(1:n), ntestmax))
    for k=1:kfold
        testindices[k] = sort(partitioned[k])
        trainindices[k] = sort(vcat(partitioned[vcat(1:k-1, k+1:kfold)]...))
    end
    return testindices, trainindices
end
