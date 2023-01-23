
"""
	poststereoclick_time_histogram_sets(𝐄, model)

Post-stereoclick time histogram for each task condition for each neuron in each trialset

ARGUMENT
"""
function poststereoclick_time_histogram_sets(𝐄::Vector{<:Vector{<:ExpectedEmissions}},
											model::Model;
											confidencelevel::AbstractFloat=0.95,
											σ_s::AbstractFloat=0.1)
	cimethod = Bootstrap.ConfIntMethod=BCaConfInt(confidencelevel)
	σ = ceil(σ_s*model.options.Δt)
	map(𝐄, model.trialsets) do 𝐄, trialset
		T = maximum((trial.ntimesteps for trial in trialset.trials))
		linearfilter = causalgaussian(σ, T)
		time_s = collect(1:T).*model.options.Δt
		map(1:length(trialset.mpGLMs)) do n
			peths = map(fieldnames(PETHSet)) do condition
						PerieventTimeHistogram(cimethod,condition,𝐄,linearfilter,n,"stereoclick",time_s,trialset)
					end
			PETHSet(peths...)
		end
	end
end

"""
	PerieventTimeHistogram(cimethod,condition,𝐄,linearfilter,n,referenceevent,)
"""
function PerieventTimeHistogram(cimethod::Bootstrap.ConfIntMethod,
								condition::Symbol,
								𝐄::Vector{<:ExpectedEmissions},
								linearfilter::SpikeTrainLinearFilter,
								n::Integer,
								referenceevent::String,
								time_s::Vector{<:AbstractFloat},
								trialset::Trialset)
	trialsubset = filter(trial->selecttrial(condition, trial), trialset.trials)
	𝐲_eachtrial = align_spike_trains(referenceevent, trialsubset, trialset.mpGLMs[n].𝐲)
	𝐱_eachtrial = collect(smooth(linearfilter, 𝐲) for 𝐲 in 𝐲_eachtrial)
	observed, lowerconfidencelimit, upperconfidencelimit = trialaverage(cimethod, 𝐱_eachtrial)
	𝐲̄_expected = trialaverage(condition, 𝐄, n, trials)
	predicted = smooth(linearfilter, 𝐲̄_expected)
	PerieventTimeHistogram(confidence_interval_method=cimethod,
							linearfilter=linearfilter,
						   	lowerconfidencelimit=lowerconfidencelimit,
						   	observed=observed,
						   	predicted=predicted,
						   	referenceevent=referenceevent,
						   	time_s=time_s,
						   	upperconfidencelimit=upperconfidencelimit)
end

"""
	selecttrial(condition, trial)

Does the trial fall under the given task condition?

ARGUMENT
-`condition`: a symbol naming the condition
-`trial`: a composite containing the data of a given trial

RETURN
-a Bool
"""
function selecttrial(condition::Symbol, trial::Trial)
	if condition==:unconditioned
		true
	elseif condition==:leftchoice
		!trial.choice
	elseif condition==:rightchoice
		trial.choice
	elseif condition==:leftevidence
		selecttrial(condition, γ)
	elseif condition==:rightevidence
		selecttrial(condition, γ)
	elseif condition==:leftchoice_strong_leftevidence
		(!trial.choice) && selecttrial(condition, γ)
	elseif condition==:leftchoice_weak_leftevidence
		(!trial.choice) && selecttrial(condition, γ)
	elseif condition==:rightchoice_strong_rightevidence
		(trial.choice) && selecttrial(condition, γ)
	elseif condition==:rightchoice_weak_rightevidence
		(trial.choice) && selecttrial(condition, γ)
	else
		error("unrecognized condition")
	end
end

"""
	selecttrial(condition, γ)

Does the generative click rate fall under the given task condition?

ARGUMENT
-`condition`: a symbol naming the condition
-`γ`: log ratio of the generative right and left click rates

RETURN
-a Bool
"""
function selecttrial(condition::Symbol, γ::AbstractFloat)
	if condition==:leftevidence
		γ < 0
	elseif condition==:rightevidence
		γ > 0
	elseif condition==:leftchoice_strong_leftevidence
		γ < -2.25
	elseif condition==:leftchoice_weak_leftevidence
		(γ >= -2.25) && (γ < 0)
	elseif condition==:rightchoice_strong_rightevidence
		γ > 2.25
	elseif condition==:rightchoice_weak_rightevidence
		(γ <= 2.25) && (γ > 0)
	else
		error("unrecognized condition")
	end
end

"""
	align_spike_trains(referenceevent, trials, 𝐲)

ARGUMENT
-`referenceevent`: event in the trial to which spike trains are aligned
-`trials`: a vector whose each element is a composite containing the behavioral data of one trial
-`𝐲`: spike trains of one neuron concatenated across trials

RETURN
-`𝐲_eachtrial`: a nested array whose element `𝐲_eachtrial[m][t]` corresponds to the spike train on the m-th trial and t-th time step from the reference event
"""
function align_spike_trains(referenceevent::String, trials::Vector{<:Trial}, 𝐲::Vector{<:Integer})
	if referenceevent=="stereoclick"
		collect(𝐲[trial.τ₀ .+ (1:trial.ntimesteps)] for trial in trials)
	else
		error("unrecognized reference event")
	end
end

"""
	trialaverage(cimethod, 𝐱_eachtrial)

Average across trials the filtered spike train

ARGUMENT
-`cimethod`: method for estimating the confidence interval
-`𝐱_eachtrial`: filtered spike trains organized such that `𝐱_eachtrial[m][t]` corresponds the filtered response at the t-th time step on the m-th trial

RETURN
-`pointestimate`: the point estimate of the trial mean of the filtered response. The output is a vector whose length corresponds to the maximum time steps across trials.
-`lowerconfidencelimit`: the lower limit of the estimated confidence interval of the mean
-`upperconfidencelimit`: the upper limit of the estimated confidence interval of the mean
"""
function trialaverage(cimethod::Bootstrap.ConfIntMethod, 𝐱_eachtrial::Vector{<:Vector{<:AbstractFloat}})
	maxtimesteps = maximum(length.(𝐱_eachtrial))
	pointestimate, lowerconfidencelimit, upperconfidencelimit = zeros(maxtimesteps), zeros(maxtimesteps), zeros(maxtimesteps)
	emptyvector = empty(𝐱_eachtrial[1])
	for t=1:T
		𝐱ₜ_eachtrial = vcat(((length(𝐱) >= t ? 𝐱[t] : emptyvector) for 𝐱 in 𝐱_eachtrial)...)
		bootsamples = bootstrap(mean, 𝐱ₜ_eachtrial, BasicSampling(1000))
		pointestimate[t], lowerconfidencelimit[t], upperconfidencelimit[t] = confint(bootsamples, cimethod)[1]
	end
	return pointestimate, lowerconfidencelimit, upperconfidencelimit
end

"""
	trialaverage(condition, 𝐄, n, trials)

ARGUMENT
-`condition`: a symbol naming the condition
-`𝐄`: expected emissions across trials in a trialset
-`n`: integer of the neuron
-`trials`: a vector whose each element is a composite containing the behavioral data of one trial

RETURN
-trial average of a neuron across trials
"""
function trialaverage(condition::Symbol, 𝐄::Vector{<:ExpectedEmissions}, n::Integer, trials::Vector{<:Trials})
	𝐲̄, 𝐰 = zeros(maxtimesteps), zeros(maxtimesteps)
	for (E, trial) in zip(𝐄, trials)
		𝐲, w = expectedspiketrain(condition, E, trial.γ, n)
		for t in eachindex(𝐲)
			𝐲̄[t] += 𝐲[t]*w
			𝐰[t] += w
		end
	end
	for t in eachindex(∑𝐲)
		𝐲̄[t] /= 𝐰[t]
	end
	return 𝐲̄
end

"""
	expectedspiketrain

Spike train of a neuron expected on a trial, under a given task condition, and its weight in the conditional trial-average

ARGUMENT
-`condition`: task condition
-`E`: a composite containing the expected emissions on one trial
-`γ`: log ratio of the generative right and left click rates
-`n`: index of the spike train in this trial

RETURN
-`𝐲`: the expected spike train represented by a vector of floats
-`w`: weight of this spike train when averaging across all trials within the given task condition
"""
function expectedspiketrain(condition::Symbol, E::ExpectedEmissions, γ::AbstractFloat, n::Integer)
	if condition==:unconditioned
		𝐲 = marginalize_over_choice(E,n)
		w = 1.0
	elseif condition==:leftchoice
		𝐲 = E.spiketrain_leftchoice[n]
		w = 1.0-E.rightchoice
	elseif condition==:rightchoice
		𝐲 = E.spiketrain_rightchoice[n]
		w = E.rightchoice
	elseif condition==:leftevidence
		w = 1.0
		𝐲 = selecttrial(condition, γ) ? marginalize_over_choice(E,n) : empty(E.spiketrain_leftchoice[n])
	elseif condition==:rightevidence
		w = 1.0
		𝐲 = selecttrial(condition, γ) ? marginalize_over_choice(E,n) : empty(E.spiketrain_rightchoice[n])
	elseif condition==:leftchoice_strong_leftevidence
		w = 1.0-E.rightchoice
		𝐲 = selecttrial(condition, γ) ? E.spiketrain_leftchoice[n] : empty(E.spiketrain_leftchoice[n])
	elseif condition==:leftchoice_weak_leftevidence
		w = 1.0-E.rightchoice
		𝐲 = selecttrial(condition, γ) ? E.spiketrain_leftchoice[n] : empty(E.spiketrain_leftchoice[n])
	elseif condition==:rightchoice_strong_rightevidence
		w = E.rightchoice
		𝐲 = selecttrial(condition, γ) ? E.spiketrain_rightchoice[n] : empty(E.spiketrain_rightchoice[n])
	elseif condition==:righchoice_weak_rightevidence
		w = E.rightchoice
		𝐲 = selecttrial(condition, γ) ? E.spiketrain_rightchoice[n] : empty(E.spiketrain_rightchoice[n])
	else
		error("unrecognized condition")
	end
	return 𝐲, w
end

"""
	marginalize_over_choice(E,n)

Compute the expected spike train for a trial for a neuron

ARGUMENT
-`E`: A composite containing the expectation of the choice and the spike train for a trial
-`n`: index of the neuron

RETURN
-a vector of floats whose each element corresponds to expectation of the spike count at a time step
"""
function marginalize_over_choice(E::ExpectedEmissions, n::Integer)
	wₗ = 1-E.rightchoice
	wᵣ = E.rightchoice
	collect(wₗ*yₗ + wᵣ*yᵣ for (yₗ, yᵣ) in zip(E.spiketrain_leftchoice[n], E.spiketrain_rightchoice[n]))
end

"""
	smooth(linearfilter, 𝐲)

Smoothing a spike train by processing it with a linear filter.

The impulse response vector of the linear filter 𝐡 is convolved with the spike train 𝐲. The output, 𝐱, is then normalied  by the convolution of the same impulse response with a ones vector.

The equation is given by:
	𝐱[t] ≡ (∑_{τ=0}^{max(T,τ)} 𝐡[τ]*x[t-τ])/𝐇[t]
where
	𝐇[t] = ∑_{τ=0}^t 𝐡[τ]

ARGUMENT
-`linearfilter`: a composite containing the impulse response vector and also a weight vector
-`𝐲`: the spike train

OUTPUT
-a vector the same size as 𝐲, representing the smoothed version of the spike train
"""
function smooth(linearfilter::SpikeTrainLinearFilter, 𝐲::Vector{<:AbstractFloat})
	𝐱 = conv(𝐲, linearfilter.impulseresponse)
	collect(w*x for (w,x) in zip(linearfilter.weights,𝐱))
end

"""
"""
function causalgaussian(σ, maxtimesteps)
	normal = Normal(0,σ)
	SpikeTrainLinearFilter(impulseresponse=collect(pdf(normal,t) for t=0:maxtimesteps-1))
end
