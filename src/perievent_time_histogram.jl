"""
	poststereoclick_time_histogram_sets(𝐄, model)

Post-stereoclick time histogram for each task condition for each neuron in each trialset

ARGUMENT
-`𝐄`: expected emissions of each trial in each trialset
-`model`: structure containing the data, parameters, and hyperparameters

OPTIONAL ARGUMENT
-`confidencelevel`: fraction of the time when a random confidence interval contains the true parameter
-`σ_s`: standard deviation, in seconds, of the causal gaussian used to filter the spike train

RETURN
-`psthsets`: a nested array whose element `psthsets[i][n]` corresponds to n-th neuron in the i-th trialset and is a composite containing the post-stereoclick time histogram for each task condition.
"""
function poststereoclick_time_histogram_sets(𝐄::Vector{<:Vector{<:ExpectedEmissions}},
											model::Model;
											confidencelevel::AbstractFloat=0.95,
											σ_s::AbstractFloat=0.1)
	cimethod = BCaConfInt(confidencelevel)
	map(𝐄, model.trialsets) do 𝐄, trialset
		T = maximum((trial.ntimesteps for trial in trialset.trials))
		linearfilter = causalgaussian(model.options.Δt, σ_s, T)
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
	PerieventTimeHistogram(cimethod,condition,𝐄,linearfilter,n,referenceevent,time_s, trialset)

Peri-event time histogram of a neuron under a given task condition

ARGUMENT
-`cimethod`: method to estimate the confidence interval
-`condition`: a symbol indicating the task condition
-`𝐄`: expected emissions of each trial in a trialset
-`linearfilter`: used to filter the spike train
-`n`: index of the neuron
-`referenceevent`: event in the trial to which the

RETURN
-a composite containing the estimate of the peri-event time histogram of a neuron in a given task condition, as well as the estimate of the confidence interval and the prediction
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
	𝐲̄_expected = trialaverage(condition, 𝐄, n, trialset.trials)
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
		selecttrial(condition, trial.γ)
	elseif condition==:rightevidence
		selecttrial(condition, trial.γ)
	elseif condition==:leftchoice_strong_leftevidence
		(!trial.choice) && selecttrial(condition, trial.γ)
	elseif condition==:leftchoice_weak_leftevidence
		(!trial.choice) && selecttrial(condition, trial.γ)
	elseif condition==:rightchoice_strong_rightevidence
		(trial.choice) && selecttrial(condition, trial.γ)
	elseif condition==:rightchoice_weak_rightevidence
		(trial.choice) && selecttrial(condition, trial.γ)
	else
		error("unrecognized condition: "*String(condition))
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
		error("unrecognized condition: "*String(condition))
	end
end

"""
	selecttrial(condition, Erightchoice, trial)

Does expected spike trains of a given trial fall under the given task condition?

ARGUMENT
-`condition`: a symbol naming the condition
-`Erightchoice`: expectation a random simulation of this trial results in a right choice
-`trial`: a composite containing the data of a given trial

RETURN
-a Bool
"""
function selecttrial(condition::Symbol, Erightchoice::AbstractFloat, trial::Trial)
	if condition==:unconditioned
		true
	elseif condition==:leftchoice
		Erightchoice < 1.0
	elseif condition==:rightchoice
		Erightchoice > 0.0
	elseif condition==:leftevidence
		selecttrial(condition, trial.γ)
	elseif condition==:rightevidence
		selecttrial(condition, trial.γ)
	elseif condition==:leftchoice_strong_leftevidence
		(Erightchoice < 1.0) && selecttrial(condition, trial.γ)
	elseif condition==:leftchoice_weak_leftevidence
		(Erightchoice < 1.0) && selecttrial(condition, trial.γ)
	elseif condition==:rightchoice_strong_rightevidence
		(Erightchoice > 0.0) && selecttrial(condition, trial.γ)
	elseif condition==:rightchoice_weak_rightevidence
		(Erightchoice > 0.0) && selecttrial(condition, trial.γ)
	else
		error("unrecognized condition: "*String(condition))
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
		error("unrecognized reference event: "*referenceevent)
	end
end

"""
	trialaverage(cimethod, 𝐱_eachtrial)

Average across trials the filtered spike train

ARGUMENT
-`cimethod`: method for estimating the confidence interval
-`𝐱_eachtrial`: filtered spike trains organized such that `𝐱_eachtrial[m][t]` corresponds the filtered response at the t-th time step on the m-th trial

OPTIONAL ARGUMENT
-`minrange`: if the range of the filter responses at a given time step across all trials a positive value below which the confidence interval method may throw an error

RETURN
-`pointestimate`: the point estimate of the trial mean of the filtered response. The output is a vector whose length corresponds to the maximum time steps across trials.
-`lowerconfidencelimit`: the lower limit of the estimated confidence interval of the mean
-`upperconfidencelimit`: the upper limit of the estimated confidence interval of the mean
"""
function trialaverage(cimethod::Bootstrap.ConfIntMethod, 𝐱_eachtrial::Vector{<:Vector{<:AbstractFloat}}; minrange::AbstractFloat=1e-8)
	if length(𝐱_eachtrial) == 0
		pointestimate, lowerconfidencelimit, upperconfidencelimit = zeros(0), zeros(0), zeros(0)
	elseif length(𝐱_eachtrial) == 1
		pointestimate, lowerconfidencelimit, upperconfidencelimit = 𝐱_eachtrial[1], 𝐱_eachtrial[1], 𝐱_eachtrial[1]
	else
		maxtimesteps = maximum(length.(𝐱_eachtrial))
		pointestimate, lowerconfidencelimit, upperconfidencelimit = zeros(maxtimesteps), zeros(maxtimesteps), zeros(maxtimesteps)
		emptyvector = empty(𝐱_eachtrial[1])
		for t=1:maxtimesteps
			𝐱ₜ_eachtrial = collect(𝐱[t] for 𝐱 in filter(x->length(x)>=t, 𝐱_eachtrial))
			minval, maxval = extrema(𝐱ₜ_eachtrial)
			if (maxval-minval)<minrange
				avg = mean(𝐱ₜ_eachtrial)
				pointestimate[t], lowerconfidencelimit[t], upperconfidencelimit[t] = avg, avg, avg
			else
				bootsamples = bootstrap(mean, 𝐱ₜ_eachtrial, BasicSampling(1000))
				pointestimate[t], lowerconfidencelimit[t], upperconfidencelimit[t] = confint(bootsamples, cimethod)[1]
			end
		end
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
function trialaverage(condition::Symbol, 𝐄::Vector{<:ExpectedEmissions}, n::Integer, trials::Vector{<:Trial})
	trialindices = collect(selecttrial(condition, E.rightchoice, trial) for (E,trial) in zip(𝐄, trials))
	if sum(trialindices) > 0
		trialsubset = trials[trialindices]
		𝐄subset = 𝐄[trialindices]
		maxtimesteps = maximum((trial.ntimesteps for trial in trialsubset))
		𝐲̄, 𝐰 = zeros(maxtimesteps), zeros(maxtimesteps)
		for (E, trial) in zip(𝐄subset, trialsubset)
			𝐲, w = expectedspiketrain(condition, E, trial.γ, n)
			for t in eachindex(𝐲)
				𝐲̄[t] += 𝐲[t]*w
				𝐰[t] += w
			end
		end
		for t in eachindex(𝐲̄)
			𝐲̄[t] /= 𝐰[t]
		end
	else
		𝐲̄ = zeros(0)
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
	if (condition==:unconditioned) || (condition==:leftevidence) || (condition==:rightevidence)
		𝐲 = marginalize_over_choice(E,n)
		w = 1.0
	elseif (condition==:leftchoice) || (condition==:leftchoice_strong_leftevidence) || (condition==:leftchoice_weak_leftevidence)
		𝐲 = E.spiketrain_leftchoice[n]
		w = 1.0-E.rightchoice
	elseif (condition==:rightchoice) || (condition==:rightchoice_strong_rightevidence) || (condition==:rightchoice_weak_rightevidence)
		𝐲 = E.spiketrain_rightchoice[n]
		w = E.rightchoice
	else
		error("unrecognized condition: "*String(condition))
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
function smooth(linearfilter::SpikeTrainLinearFilter, 𝐲::Vector{<:Real})
	if isempty(𝐲)
		similar(linearfilter.weights,0)
	else
		𝐱 = conv(𝐲, linearfilter.impulseresponse)
		collect(linearfilter.weights[t]*𝐱[t] for t in eachindex(𝐲))
	end
end

"""
	causalgaussian(σ, maxtimesteps)

Return a composite containing a causal gaussian filter
"""
function causalgaussian(Δt::AbstractFloat, σ_s::AbstractFloat, maxtimesteps::Integer)
	normal = Normal(0,σ_s)
	h = collect(pdf(normal,t*Δt) for t=0:maxtimesteps-1)
	w = 1.0./(conv(ones(maxtimesteps).*Δt, h)[1:maxtimesteps])
	SpikeTrainLinearFilter(impulseresponse=h, weights=w)
end

"""
	save(pethsets, folderpath)
"""
function save(pethsets::Vector{<:Vector{<:PETHSet}}, folderpath::String)
	if !isdir(folderpath)
		mkdir(folderpath)
		@assert isdir(folderpath)
	end
	referenceevent = pethsets[1][1].unconditioned.referenceevent
	filepath = joinpath(folderpath, "pethsets_"*referenceevent*".mat")
	dict = dictionary(pethsets)
	matwrite(filepath, dict)
end


"""
	dictionary(pethsets)
"""
function dictionary(pethsets::Vector{<:Vector{<:PETHSet}})
	representativepeth = pethsets[1][1].unconditioned
	cimethod = representativepeth.confidence_interval_method
	linearfilter= representativepeth.linearfilter
	dict = Dict("confidence_interval_method"=>String(Symbol(cimethod)),
				"confidencelevel"=>cimethod.level,
				"linearfilter"=>dictionary(representativepeth.linearfilter),
				"referenceevent"=>representativepeth.referenceevent,
				"time_s"=>representativepeth.time_s,
				"pethsets"=>map(pethset->map(dictionary, pethset), pethsets))
end

"""
	dictionary(PETHSet)
"""
dictionary(pethset::PETHSet) = Dict((String(fieldname)=>dictionary(getfield(pethset,fieldname)) for fieldname in fieldnames(PETHSet))...)

"""
	dictionary(peth)
"""
dictionary(peth::PerieventTimeHistogram) = Dict((String(fieldname)=>getfield(peth,fieldname) for fieldname in (:observed, :lowerconfidencelimit, :upperconfidencelimit, :predicted))...)
