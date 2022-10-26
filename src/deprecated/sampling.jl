
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
