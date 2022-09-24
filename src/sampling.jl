"""
	Predictions(model)

ARGUMENT
-`model`: a structure containing the data, parameters, and hyperparameters of the model

RETURN
-a structure containing the predictions of the model
"""
function Predictions(model::Model; nsamples::Integer=100)
    @unpack trialsets, options, θnative = model
	@unpack Ξ, K = options
    λΔt = map(trialsets) do trialset
			map(trialset.mpGLMs) do mpGLM
				zeros(trialset.ntimesteps)
			end
		  end
	λΔt_𝑑 = deepcopy(λΔt)
	p𝐚 = map(trialsets) do trialset
			map(trialset.trials) do trial
				collect(zeros(Ξ) for t=1:trial.ntimesteps)
			end
		  end
	p𝐜_𝐘𝑑 = map(trialsets) do trialset
			map(trialset.trials) do trial
				collect(zeros(K) for t=1:trial.ntimesteps)
			end
		  end
	p𝐚_𝑑, p𝐚_𝐘𝑑 = deepcopy(p𝐚), deepcopy(p𝐚)
	p𝑑 = collect(zeros(trialset.ntrials) for trialset in trialsets)
	memory = Memoryforgradient(model)
	P = FHMDDM.update!(memory, model, concatenateparameters(model)[1])
	@unpack Aᵃinput, Aᵃsilent, Aᶜ, p𝐚₁, πᶜ = memory
	f⨀b = memory.f
	p𝑑_𝐚 = ones(Ξ)
	maxtimesteps = length(f⨀b)
	a = zeros(Int, maxtimesteps)
	c = zeros(Int, maxtimesteps)
	𝐄𝐞_𝐡_𝛚 = map(trialsets) do trialset
			map(trialset.mpGLMs) do mpGLM
				@unpack Δt, Φₕ, 𝐗,  = mpGLM
				@unpack 𝐠, 𝐮, 𝐯 = mpGLM.θ
				𝛚 = transformaccumulator(mpGLM)
				n_spikehistory_parameters = size(Φₕ,2)
				𝐡 = Φₕ*𝐮[1:n_spikehistory_parameters]
				𝐞 = 𝐮[n_spikehistory_parameters+1:end]
				indices_time_move_in_𝐗 = 1+n_spikehistory_parameters .+ (1:length(𝐞))
				𝐄 = @view 𝐗[:,indices_time_move_in_𝐗]
				𝐄𝐞 = 𝐄*𝐞
				return 𝐄𝐞, 𝐡, 𝛚
			end
		end
    for trialset in trialsets
		for trial in trialset.trials
			i = trial.trialsetindex
			m = trial.index_in_trialset
			𝛕 = trial.τ₀ .+ (1:trial.ntimesteps)
			forward!(memory, P, θnative, trial)
			backward!(memory, P, trial)
			accumulatorprobability!(p𝐚[i][m], p𝐚₁, Aᵃinput, Aᵃsilent, trial)
			accumulator_probability_given_choice!(p𝐚_𝑑[i][m], p𝑑_𝐚, Aᵃinput, Aᵃsilent, p𝐚[i][m], θnative.ψ[1], trial)
			for t = 1:trial.ntimesteps
				p𝐚_𝐘𝑑[i][m][t] = dropdims(sum(f⨀b[t], dims=2), dims=2)
				p𝐜_𝐘𝑑[i][m][t] = dropdims(sum(f⨀b[t], dims=1), dims=1)
			end
			for s = 1:nsamples
				samplecoupling!(c, Aᶜ, trial.ntimesteps, πᶜ)
				sampleaccumulator!(a, Aᵃinput, Aᵃsilent, p𝐚₁, trial)
				p𝑑[i][m] += sample(a[trial.ntimesteps], θnative.ψ[1], Ξ)/nsamples
				for (𝐄𝐞_𝐡_𝛚, λΔt, mpGLM) in zip(𝐄𝐞_𝐡_𝛚[i], λΔt[i], trialset.mpGLMs)
					λΔt[𝛕] .+= sample(a, c, 𝐄𝐞_𝐡_𝛚[1], 𝐄𝐞_𝐡_𝛚[2], mpGLM, 𝐄𝐞_𝐡_𝛚[3], 𝛕)./nsamples
				end
				sample_accumulator_given_choice!(a, Aᵃinput, Aᵃsilent, p𝐚[i][m], p𝐚_𝑑[i][m][trial.ntimesteps], trial)
				for (𝐄𝐞_𝐡_𝛚, λΔt_𝑑, mpGLM) in zip(𝐄𝐞_𝐡_𝛚[i], λΔt_𝑑[i], trialset.mpGLMs)
					λΔt_𝑑[𝛕] .+= sample(a, c, 𝐄𝐞_𝐡_𝛚[1], 𝐄𝐞_𝐡_𝛚[2], mpGLM, 𝐄𝐞_𝐡_𝛚[3], 𝛕)./nsamples
				end
			end
		end
	end
    return Predictions(	p𝐚 = p𝐚,
						p𝐚_𝑑 = p𝐚_𝑑,
						p𝐚_𝐘𝑑 = p𝐚_𝐘𝑑,
						p𝐜_𝐘𝑑 = p𝐜_𝐘𝑑,
						p𝑑 = p𝑑,
						λΔt = λΔt,
						λΔt_𝑑 = λΔt_𝑑,
						nsamples = nsamples)
end

"""
	accumulatorprobability!(Aᵃinput, P, p𝐚, Aᵃsilent, θnative, trial)

Probability of the accumulator at each time step

MODIFIED ARGUMENT
-`Aᵃinput`: vector of matrices used as memory for computing the transition probability of the accumulator on time steps with stimulus input
-`p𝐚`: a vector whose element p𝐚[t][i] represents p(a[t] = ξ[i])

UNMODIFIED ARGUMENT
-`Aᵃsilent`: transition probability of the accumulator on timesteps without stimulus input
-`p𝐚₁`: prior distribution of the accumulator
-`trial`: structure containing information on a trial

"""
function accumulatorprobability!(p𝐚::Vector{<:Vector{<:AbstractFloat}},
								p𝐚₁::Vector{<:AbstractFloat},
 								Aᵃinput::Vector{<:Matrix{<:AbstractFloat}},
 								Aᵃsilent::Matrix{<:AbstractFloat},
								trial::Trial)
	p𝐚[1] .= p𝐚₁
	@inbounds for t=2:trial.ntimesteps
		if isempty(trial.clicks.inputindex[t])
			Aᵃ = Aᵃsilent
		else
			Aᵃ = Aᵃinput[trial.clicks.inputindex[t][1]]
		end
		p𝐚[t] = Aᵃ * p𝐚[t-1]
	end
	return nothing
end

"""
	accumulator_probability_given_choice!(p, choice, p𝐚_end, ψ)

Conditional distribution of the accumulator variable given the behavioral choice

MODIFIED ARGUMENT
-`p`: a vector serving as memory

UNMODIFIED ARGUMENT
-`Aᵃinput`: memory for computing the transition matrix during a timestep with stimulus input
-`Aᵃsilent`: transition matrix during a timestep without stimulus input
-`p𝐚`: distribution of the accumulator at the each time step of the trial
-`ψ`: lapse rate
-`trial`: a structure containing information on the trial being considered

"""
function accumulator_probability_given_choice!(p𝐚_𝑑::Vector{<:Vector{<:AbstractFloat}},
											p𝑑_𝐚::Vector{<:AbstractFloat},
											Aᵃinput::Vector{<:Matrix{<:AbstractFloat}},
											Aᵃsilent::Matrix{<:AbstractFloat},
											p𝐚::Vector{<:Vector{<:AbstractFloat}},
											ψ::AbstractFloat,
											trial::Trial)
	choicelikelihood!(p𝑑_𝐚, trial.choice, ψ) # `p𝐚_𝑑[ntimesteps]` now reprsents p(𝑑 ∣ a)
	p𝐚_𝑑[trial.ntimesteps] .= p𝑑_𝐚.*p𝐚[trial.ntimesteps] # `p𝐚_𝑑[ntimesteps]` now reprsents p(𝑑, a)
	D = sum(p𝐚_𝑑[trial.ntimesteps])
	p𝐚_𝑑[trial.ntimesteps] ./= D # `p𝐚_𝑑[ntimesteps]` now reprsents p(a ∣ 𝑑)
	b = ones(length(p𝑑_𝐚))
	for t = trial.ntimesteps-1:-1:1
		inputindex = trial.clicks.inputindex[t+1]
		if isempty(inputindex)
			Aᵃ = Aᵃsilent
		else
			Aᵃ = Aᵃinput[inputindex[1]]
		end
		if t+1 == trial.ntimesteps
			b = Aᵃ' * (p𝑑_𝐚.*b./D)
		else
			b = Aᵃ' * b
		end
		p𝐚_𝑑[t] = p𝐚[t] .* b
	end
	return nothing
end

"""
	sampleaccumulator!(a, Aᵃinput, Aᵃsilent, p𝐚₁, trial)

Sample the values of the accumulator variable in one trial

MODIFIED ARGUMENT
-`a`: a vector containing the sample value of the coupling variable in each time step

UNMODIFIED ARGUMENT
-`Aᵃinput`: memory for computing the transition matrix during a timestep with stimulus input
-`Aᵃsilent`: transition matrix during a timestep without stimulus input
-`p𝐚₁`: prior distribution of the accumulator
-`trial`: a structure containing information on the trial being considered
"""
function sampleaccumulator!(a::Vector{<:Integer}, Aᵃinput::Vector{<:Matrix{<:Real}}, Aᵃsilent::Matrix{<:Real}, p𝐚₁::Vector{<:AbstractFloat}, trial::Trial)
	a[1] = findfirst(rand() .< cumsum(p𝐚₁))
	for t = 2:trial.ntimesteps
		if isempty(trial.clicks.inputindex[t])
			Aᵃ = Aᵃsilent
		else
			Aᵃ = Aᵃinput[trial.clicks.inputindex[t][1]]
		end
		p𝐚ₜ_aₜ₋₁ = Aᵃ[:,a[t-1]]
		a[t] = findfirst(rand() .< cumsum(p𝐚ₜ_aₜ₋₁))
	end
	return nothing
end

"""
	sample_accumulator_given_choice!(a, Aᵃinput, Aᵃsilent, p𝐚_𝑑, trial)

A sample of the accumulator in one trial conditioned on the behavioral choice

MODIFIED ARGUMENT
-`a`: a vector representing the value of the accumulator at each time step of the trial

UNMODIFIED ARGUMENT
-`Aᵃinput`: vector of matrices used as memory for computing the transition probability of the accumulator on time steps with stimulus input
-`Aᵃsilent`: transition probability of the accumulator on timesteps without stimulus input
-`p𝐚`: probability of the accumulator at each time step of the trial
-`p𝐚_end_𝑑`: posterior probability of the accumulator, given the choice, at the last time step. The i-th element represents p(a=ξᵢ ∣ 𝑑)
-`trial`: structure containing information on a trial
"""
function sample_accumulator_given_choice!(a::Vector{<:Integer},
										Aᵃinput::Vector{<:Matrix{<:AbstractFloat}},
 										Aᵃsilent::Matrix{<:AbstractFloat},
										p𝐚::Vector{<:Vector{<:AbstractFloat}},
										p𝐚_end_𝑑::Vector{<:AbstractFloat},
										trial::Trial)
	a[trial.ntimesteps] = findfirst(rand() .< cumsum(p𝐚_end_𝑑))
	for t = trial.ntimesteps-1:-1:1
		inputindex = trial.clicks.inputindex[t+1]
		if isempty(inputindex)
			Aᵃ = Aᵃsilent
		else
			Aᵃ = Aᵃinput[inputindex[1]]
		end
		p_𝐚ₜ_aₜ₊₁ = Aᵃ[a[t+1],:] .* p𝐚[t] ./ p𝐚[t+1][a[t+1]]
		a[t] = findfirst(rand() .< cumsum(p_𝐚ₜ_aₜ₊₁))
	end
	return nothing
end

"""
	samplecoupling!(c, Aᶜ, ntimesteps, πᶜ)

Sample the values of the coupling variable in one trial

MODIFIED ARGUMENT
-`c`: a vector containing the sample value of the coupling variable in each time step

ARGUMENT
-`Aᶜ`: transition matrix of the coupling variable
-`ntimesteps`: number of time steps in the trial
-`πᶜ`: prior probability of the coupling variable
"""
function samplecoupling!(c::Vector{<:Integer}, Aᶜ::Matrix{<:Real}, ntimesteps::Integer, πᶜ::Vector{<:Real})
	if length(πᶜ) == 1
		c .= 1
	else
		cumulativep𝐜 = cumsum(πᶜ)
	    c[1] = findfirst(rand() .< cumulativep𝐜)
		cumulativeAᶜ = cumsum(Aᶜ, dims=1)
	    for t = 2:ntimesteps
	        cumulativep𝐜 = cumulativeAᶜ[:,c[t-1]]
	        c[t] = findfirst(rand() .< cumulativep𝐜)
	    end
	end
	return nothing
end

"""
	sample(a_end, ψ, Ξ)

Sample a choice on a trial

ARGUMENT
-`a_end`: state of the accumulator at the last time step of the trial
-`ψ`: lapse rate
-`Ξ`: number of states that the accumulator can take
"""
function sample(a_end::Integer, ψ::AbstractFloat, Ξ::Integer)
	zeroindex = cld(Ξ,2)
	if a_end < zeroindex
		p_right_choice = ψ/2
	elseif a_end > zeroindex
		p_right_choice = 1-ψ/2
	else a_end == zeroindex
		p_right_choice = 0.5
	end
	choice = rand() < p_right_choice
end

"""
	sample(a, c, 𝛕, mpGLM)

Generate a sample of spiking response on each time step of one trial

ARGUMENT
-`a`: a vector representing the state of the accumulator at each time step of a trial. Note that length(a) >= length(𝛕).
-`c`: a vector representing the state of the coupling variable at each time step. Note that length(c) >= length(𝛕).
-`𝐄𝐞`: input from events
-`𝐡`: weight of post-spikefilter at each time lag
-`mpGLM`: a structure containing information on the mixture of Poisson GLM of a neuron
-`𝛚`: transformed values of the accumulator
-`𝛕`: time steps in the trialset. The number of time steps in the trial corresponds to the length of 𝛕.

RETURN
-`𝐲̂`: a vector representing the sampled spiking response at each time step
"""
function sample(a::Vector{<:Integer}, c::Vector{<:Integer}, 𝐄𝐞::Vector{<:AbstractFloat}, 𝐡::Vector{<:AbstractFloat}, mpGLM::MixturePoissonGLM, 𝛚::Vector{<:AbstractFloat}, 𝛕::UnitRange{<:Integer}, )
	@unpack Δt, Φₕ, 𝐕, 𝐲 = mpGLM
	@unpack 𝐠, 𝐮, 𝐯 = mpGLM.θ
	max_spikehistory_lag = size(Φₕ,1)
	K𝐠 = length(𝐠)
	K𝐯 = length(𝐯)
	max_spikes_per_step = floor(1000Δt)
    𝐲̂ = zeros(Int, length(𝛕))
    for t = 1:length(𝛕)
        τ = 𝛕[t]
        j = a[t]
        k = c[t]
		gₖ = 𝐠[min(k, K𝐠)]
		𝐯ₖ = 𝐯[min(k, K𝐯)]
		L = gₖ + 𝐄𝐞[τ]
		for i in eachindex(𝐯ₖ)
			L+= 𝛚[j]*𝐕[τ,i]*𝐯ₖ[i]
		end
		for lag = 1:min(max_spikehistory_lag, t-1)
			if 𝐲̂[t-lag] > 0
				L += 𝐡[lag]*𝐲̂[t-lag]
			end
		end
        λ = softplus(L)
        𝐲̂[t] = min(rand(Poisson(λ*Δt)), max_spikes_per_step)
    end
	return 𝐲̂
end

"""
    sample(model)

Generate latent and emission variables for all trials of all trialsets

ARGUMENT
-`model`: an instance of the factorial-hidden Markov drift-diffusion model

OUTPUT
-a structure with data sampled from the parameters of the model
"""
function sample(model::Model; folderpath::String = dirname(model.options.datapath))
	predictions = Predictions(model; nsamples=1)
	newtrialsets = 	map(model.trialsets, predictions.p𝑑, predictions.λΔt) do trialset, p𝑑, λΔt
						newtrials =	map(trialset.trials, p𝑑) do oldtrial, p𝑑
										Trial(clicks=oldtrial.clicks,
						                      choice=Bool(p𝑑),
											  movementtime_s=oldtrial.movementtime_s,
						                      ntimesteps=oldtrial.ntimesteps,
						                      previousanswer=oldtrial.previousanswer,
											  index_in_trialset=oldtrial.index_in_trialset,
											  τ₀=oldtrial.τ₀,
											  trialsetindex=oldtrial.trialsetindex)
									end
						new_mpGLMs = map(trialset.mpGLMs, λΔt) do old_mpGLM, λΔt
										MixturePoissonGLM(Δt=old_mpGLM.Δt,
						  								d𝛏_dB=old_mpGLM.d𝛏_dB,
														Φₐ=old_mpGLM.Φₐ,
														Φₕ=old_mpGLM.Φₕ,
														Φₘ=old_mpGLM.Φₘ,
														Φₜ=old_mpGLM.Φₜ,
														θ=FHMDDM.copy(old_mpGLM.θ),
														𝐕=old_mpGLM.𝐕,
														𝐗=old_mpGLM.𝐗,
														𝐲=convert.(Int,λΔt))
									end
						Trialset(trials=newtrials, mpGLMs=new_mpGLMs)
					end
		options = dictionary(model.options)
		options["datapath"] = joinpath(folderpath,"data.mat")
		options["resultspath"] = joinpath(folderpath,"results.mat")
		Model(options=Options(model.options.nunits, options),
			gaussianprior=GaussianPrior(model.options, newtrialsets),
			θnative=FHMDDM.copy(model.θnative),
			θreal=FHMDDM.copy(model.θreal),
			θ₀native=FHMDDM.copy(model.θ₀native),
			trialsets=newtrialsets)
end

"""
	samples(model, nsamples)

Generate and save samples of the data

ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters of a factorial hidden-Markov drift-diffusion model
-`nsamples`: number of samples to make
"""
function samples(model::Model, nsamples::Integer)
	@assert nsamples > 0
	pad = ceil(Int, log10(nsamples))
	open(joinpath(dirname(model.options.datapath), "samplepaths.txt"), "w") do io
	    for i=1:nsamples
	        folderpath = joinpath(dirname(model.options.datapath),"sample"*string(i;pad=pad))
	        !isdir(folderpath) && mkdir(folderpath)
	        filepath = joinpath(folderpath, "data.mat")
	        println(io, filepath)
	        sampledmodel = sample(model; folderpath=folderpath)
	        savedata(sampledmodel)
	    end
	end
	return nothing
end

"""
    expectedemissions(model; nsamples=100)

Compute the probability of a right choice and the expected spike rate

The emissions consist of the behavioral choice in each trial and the spike train response of each neuron at each time step of each trial. The behavioral choice is modelled as a Bernoulli random variable, and the spike train response is modelled as a Poisson random variable. Here we compute the expectated value of these random variablees by averaging across samples of the model.

ARGUMENT
- a structure containing information for a factorial hidden Markov drift-diffusion model

OPTION ARGUMENT
-`nsamples`: number of samples over which the expectation is taken

RETURN
-`λΔt`: expectd value of the number of spikes per `Δt` at each time step. Element λΔt[i][n][t] corresponds to the t-timestep and the n-th neuron in the i-th trialset
-`pchoice`: estimate of the mean of the probability of a right. Element pchoice[i][m] corresponds to the m-th trial in the i-th trialset

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_07_29a_test/T176_2018_05_03/data.mat"
julia> model = Model(datapath)
julia> λΔt, pchoice = expectedemissions(model; nsamples=2)
julia> save(λΔt, model, pchoice; filename="postinitialization.mat")
julia>
```
"""
function expectedemissions(model::Model; nsamples::Integer=100)
    @unpack trialsets, options, θnative = model
    pchoice = map(trialset->zeros(trialset.ntrials), trialsets)
    λΔt = map(trialsets) do trialset
			map(trialset.mpGLMs) do mpGLM
				zeros(trialset.ntimesteps)
			end
		  end
	memory = Memoryforgradient(model)
	P = update!(memory, model, concatenateparameters(model)[1])
    for i in eachindex(trialsets)
        for s = 1:nsamples
			sampledtrials =	collect(sample!(memory, P, θnative, trial) for trial in trialsets[i].trials)
			for m in eachindex(sampledtrials)
				pchoice[i][m] += sampledtrials[m].choice
			end
            for n in eachindex(trialsets[i].mpGLMs)
				λΔt[i][n] .+= sampleemissions(trialsets[i].mpGLMs[n], sampledtrials)
            end
        end
    	pchoice[i] ./= nsamples
        for n in eachindex(λΔt[i])
            λΔt[i][n] ./= nsamples
        end
    end
    return λΔt, pchoice
end

"""
    sample!(memory, P, θnative, trial)

Sample the latent variables and behavioral choice for one trial

ARGUMENT
-`θnative`: model parameters in their native space
-`trial`: stucture containing the stimulus information of one trial being used for sampling

RETURN
-an instance of `Trial` containing the generated behavioral choice as well as the sequence of latent variables
"""
function sample!(memory::Memoryforgradient, P::Probabilityvector, θnative::Latentθ, trial::Trial)
    @unpack Aᵃinput, Aᵃsilent, Aᶜ, πᶜ, Ξ = memory
	c = samplecoupling(Aᶜ, trial.ntimesteps, πᶜ)
    a = sampleaccumulator(Aᵃinput, Aᵃsilent, P, θnative, trial)
	zeroindex = cld(Ξ,2)
	if a[end] < zeroindex
		p_right_choice = θnative.ψ[1]/2
	elseif a[end] == zeroindex
		p_right_choice = 0.5
	else
		p_right_choice = 1-θnative.ψ[1]/2
	end
	choice = rand() < p_right_choice
    Trial(	clicks=trial.clicks,
          	choice=choice,
			movementtime_s=trial.movementtime_s,
		  	ntimesteps=trial.ntimesteps,
			previousanswer=trial.previousanswer,
			a=a,
			c=c)
end

"""
	sampleaccumulator(Aᵃinput, Aᵃsilent, P, θnative, trial)

Sample the values of the accumulator variable in one trial

ARGUMENT
-`Aᵃinput`: memory for computing the transition matrix during a timestep with stimulus input
-`Aᵃsilent`: transition matrix during a timestep without stimulus input
-`P`: memory for computing the prior probability or transition matrix
-`θnative`: parameters controlling the latent variables in native space
-`trial`: a structure containing information on the trial being considered

RETURN
-`a`: a vector containing the sample value of the coupling variable in each time step
"""
function sampleaccumulator(Aᵃinput::Vector{<:Matrix{<:Real}}, Aᵃsilent::Matrix{<:Real}, P::Probabilityvector, θnative::Latentθ, trial::Trial)
	@unpack clicks, ntimesteps, previousanswer,
	a = zeros(Int, ntimesteps)
	priorprobability!(P, previousanswer)
	a[1] = findfirst(rand() .< cumsum(P.𝛑))
	if length(clicks.time) > 0
		adaptedclicks = adapt(clicks, θnative.k[1], θnative.ϕ[1])
	end
	for t = 2:ntimesteps
		if isempty(clicks.inputindex[t])
			Aᵃ = Aᵃsilent
		else
			Aᵃ = Aᵃinput[clicks.inputindex[t][1]]
			update_for_transition_probabilities!(P, adaptedclicks, clicks, t)
			transitionmatrix!(Aᵃ, P)
		end
		p𝐚 = Aᵃ[:,a[t-1]]
		a[t] = findfirst(rand() .< cumsum(p𝐚))
	end
	return a
end

"""
	samplecoupling(Aᶜ, ntimesteps, πᶜ)

Sample the values of the coupling variable in one trial

ARGUMENT
-`Aᶜ`: transition matrix of the coupling variable
-`ntimesteps`: number of time steps in the trial
-`πᶜ`: prior probability of the coupling variable

RETURN
-`c`: a vector containing the sample value of the coupling variable in each time step
"""
function samplecoupling(Aᶜ::Matrix{<:Real}, ntimesteps::Integer, πᶜ::Vector{<:Real})
	if length(πᶜ) == 1
		return ones(Int, ntimesteps)
	else
		c = zeros(Int, ntimesteps)
		cumulativep𝐜 = cumsum(πᶜ)
	    c[1] = findfirst(rand() .< cumulativep𝐜)
		cumulativeAᶜ = cumsum(Aᶜ, dims=1)
	    for t = 2:ntimesteps
	        cumulativep𝐜 = cumulativeAᶜ[:,c[t-1]]
	        c[t] = findfirst(rand() .< cumulativep𝐜)
	    end
		return c
	end
end

"""
	sampleemissions(mpGLM, trials)

Generate one sample from the mixture of Poisson generalized linear model (GLM) of a neuron

ARGUMENT
-`mpGLM`: the fitted mixture of Poisson GLM of a neuron
-`trials`: a vector of structures, one of which contains the generated states of the accumulator and coupling variable of one trial

RETURN
-`𝐲̂`: a sample of the spike train response for each timestep
"""
function sampleemissions(mpGLM::MixturePoissonGLM, trials::Vector{<:Trial})
	@unpack Δt, Φₕ, 𝐗, 𝐕, 𝐲 = mpGLM
	@unpack 𝐠, 𝐮, 𝐯 = mpGLM.θ
	𝛚 = transformaccumulator(mpGLM)
	max_spikehistory_lag, n_spikehistory_parameters = size(Φₕ)
	𝐡 = Φₕ*𝐮[1:n_spikehistory_parameters]
	𝐞 = 𝐮[n_spikehistory_parameters+1:end]
	indices_time_move_in_𝐗 = 1+n_spikehistory_parameters .+ (1:length(𝐞))
	𝐄 = @view 𝐗[:,indices_time_move_in_𝐗]
	𝐄𝐞 = 𝐄*𝐞
	K𝐠 = length(𝐠)
	K𝐯 = length(𝐯)
	Ξ = length(𝛚)
	max_spikes_per_step = floor(1000Δt)
    𝐲̂ = similar(𝐲)
    τ = 0
    for m in eachindex(trials)
        for t = 1:trials[m].ntimesteps
            τ += 1
            j = trials[m].a[t]
            k = trials[m].c[t]
			gₖ = 𝐠[min(k, K𝐠)]
			𝐯ₖ = 𝐯[min(k, K𝐯)]
			L = gₖ + 𝐄𝐞[τ]
			for i in eachindex(𝐯ₖ)
				L+= 𝛚[j]*𝐕[τ,i]*𝐯ₖ[i]
			end
			for lag = 1:min(max_spikehistory_lag, t-1)
				if 𝐲̂[τ-lag] > 0
					L += 𝐡[lag]*𝐲̂[τ-lag]
				end
			end
            λ = softplus(L)
            𝐲̂[τ] = min(rand(Poisson(λ*Δt)), max_spikes_per_step)
        end
    end
	return 𝐲̂
end

"""
	sample_and_save(model)

Generate samples of a model and save

ARGUMENT
-`model`: an instance of the factorial-hidden Markov drift-diffusion model

OPTIONAL ARGUMENT
-`datafilename`: name of the file containing the generated data
-`resultsfilename`: name of the file to which fitted results will be saved
-`nsamples`: number of samples to generate

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_07_06a_test/T176_2018_05_03_b5K1K1/data.mat"
julia> model = Model(datapath)
julia> FHMDDM.sample_and_save(model;nsamples=2)
julia> newmodel = Model(dirname(datapath)*"/sample1.mat")
julia>

julia> using FHMDDM
julia> datapath = "/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_07_08e_pr/T274_2020_12_14/sample10.mat"
julia> newmodel = Model(datapath)
julia>
```
"""
function sample_and_save(model::Model; datafilename::String="sample", nsamples::Integer=10, resultsfilename::String="results")
	folderpath = dirname(model.options.datapath)
	optionsdict = dictionary(model.options)
	pad = length(string(nsamples))
	samplepaths, resultpaths = fill("", nsamples), fill("", nsamples)
	samplepaths = open(folderpath*"/samplepaths.txt", "w")
	resultpaths = open(folderpath*"/resultpaths.txt", "w")
	for i = 1:nsamples
		sampleindex = string(i, pad=pad)
	    optionsdict["datapath"] = folderpath*"/"*datafilename*sampleindex*".mat"
	    optionsdict["resultspath"] = folderpath*"/"*datafilename*sampleindex*"_"*resultsfilename*".mat"
		if i > 1
			write(samplepaths, "\n")
			write(resultpaths, "\n")
		end
		write(samplepaths, optionsdict["datapath"])
		write(resultpaths, optionsdict["resultspath"])
		trialsets = sample(model)
		save(optionsdict, trialsets)
	end
	close(samplepaths)
	close(resultpaths)
end

"""
    sample(mpGLM, sampledtrials)

Generate one sample from the mixture of Poisson generalized linear model (GLM) of a neuron

ARGUMENT
-`mpGLM`: the fitted mixture of Poisson GLM of a neuron
-`sampledtrials`: a vector of structures, one of which contains the generated states of the accumulator and coupling variable of one trial

RETURN
-`mpGLM`: a sample of the mixture of Poisson GLM
"""
function sample(mpGLM::MixturePoissonGLM, sampledtrials::Vector{<:Trial})
    𝐲̂ = sampleemissions(mpGLM, sampledtrials)
	θ = GLMθ(b = copy(mpGLM.θ.b),
			b_scalefactor = mpGLM.θ.b_scalefactor,
			𝐠 = copy(mpGLM.θ.𝐠),
			𝐮 = copy(mpGLM.θ.𝐮),
			𝐯 = map(𝐯ₖ->copy(𝐯ₖ), mpGLM.θ.𝐯))
    MixturePoissonGLM(Δt=mpGLM.Δt,
                      d𝛏_dB=mpGLM.d𝛏_dB,
					  Φₐ=mpGLM.Φₐ,
					  Φₕ=mpGLM.Φₕ,
					  Φₘ=mpGLM.Φₘ,
					  Φₜ=mpGLM.Φₜ,
					  θ=θ,
                      𝐕=mpGLM.𝐕,
                      𝐗=mpGLM.𝐗,
                      𝐲=𝐲̂)
end

"""
	sampleclicks(a_latency_s, clickrate_Hz, Δt, ntimesteps, right2left)

Create a structure containing information on a sequence of simulated clicks

INPUT
-`a_latency_s`: latency, in second, of the response of the accumulator to the clicks
-`clickrate_Hz`: number of left and right clicks, combined, per second
-`Δt`: size of the time step
-`ntimesteps`: number of time steps in the trial
-`right2left`: ratio of the right to left click rate

OPTIONAL INPUT
-`rng`: random number generator

RETURN
-a structure containing the times and time step indices of simulated clicks

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> FHMDDM.sampleclicks(0.01, 40, 0.01, 100, 30; rng=MersenneTwister(1234))
	Clicks{Vector{Float64}, BitVector, Vector{Int64}, Vector{Vector{Int64}}}
	  time: Array{Float64}((46,)) [0.0479481935162798, 0.06307130174886962, 0.0804820073564533, 0.11317136052678396, 0.18273464895638575, 0.2000809403010865, 0.2086987723064543, 0.21917011781456938, 0.23527419909502842, 0.25225718711259393  …  0.8251247971779945, 0.8461572549605891, 0.847170493491451, 0.8519321105940183, 0.8555972472927873, 0.8670437145405672, 0.93879550239758, 0.9419273975453288, 0.9484616835697396, 0.9755875605263443]
	  inputtimesteps: Array{Int64}((35,)) [6, 8, 10, 13, 20, 22, 23, 25, 27, 30  …  77, 79, 83, 84, 86, 87, 88, 95, 96, 99]
	  inputindex: Array{Vector{Int64}}((100,))
	  source: BitVector
	  left: Array{Vector{Int64}}((100,))
	  right: Array{Vector{Int64}}((100,))
```
"""
function sampleclicks(a_latency_s::Real,
					  clickrate_Hz::Real,
					  Δt::Real,
					  ntimesteps::Integer,
					  right2left::Real;
					  rng::AbstractRNG=MersenneTwister())
	leftrate = clickrate_Hz/(1+right2left)
	rightrate = clickrate_Hz - leftrate
	duration_s = ntimesteps*Δt
	leftclicktimes = samplePoissonprocess(leftrate, duration_s; rng=rng)
	rightclicktimes = samplePoissonprocess(rightrate, duration_s; rng=rng)
	Clicks(a_latency_s, Δt, leftclicktimes, ntimesteps, rightclicktimes)
end

"""
	samplePoissonprocess(λ, T)

Return the event times from sampling a Poisson process with rate `λ` for duration `T`

INPUT
-`λ`: expected number of events per unit time
-`T`: duration in time to simulate the process

OPTIONAL INPUT
-`rng`: random number generator

RETURN
-a vector of event times

EXAMPLE
```julia-repl
julia> using FHMDDM, Random
julia> FHMDDM.samplePoissonprocess(10, 1.0; rng=MersenneTwister(1234))
	6-element Vector{Float64}:
	 0.24835053723904896
	 0.40002089777669625
	 0.4604645464869504
	 0.5300512053508091
	 0.6607031685057758
	 0.9387319245195712
```
"""
function samplePoissonprocess(λ::Real,
							  T::Real;
							  rng::AbstractRNG=MersenneTwister())
	@assert λ > 0
	@assert T > 0
	times = zeros(1)
	while times[end] < T
		times = vcat(times, times[end]+randexp(rng)/λ)
	end
	return times[2:end-1]
end
