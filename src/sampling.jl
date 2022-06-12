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
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_05_05_test/data.mat");
julia> λΔt, pchoice = expectedemissions(model; nsamples =2)
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
function sample!(memory::Memoryforgradient,
				P::Probabilityvector,
				θnative::Latentθ,
                trial::Trial)
    @unpack Aᵃinput, Aᵃsilent, Aᶜ, Δt, πᶜ, Ξ = memory
    @unpack clicks = trial
	@unpack inputtimesteps, inputindex = clicks
    a = zeros(Int, trial.ntimesteps)
    c = zeros(Int, trial.ntimesteps)
	priorprobability!(P, trial.previousanswer)
	p𝐚 = P.𝛑
	p𝐜 = πᶜ
    a[1] = findfirst(rand() .< cumsum(p𝐚))
    c[1] = findfirst(rand() .< cumsum(p𝐜))
	if length(clicks.time) > 0
		adaptedclicks = adapt(clicks, θnative.k[1], θnative.ϕ[1])
	end
    for t = 2:trial.ntimesteps
        if isempty(clicks.inputindex[t])
			Aᵃ = Aᵃsilent
		else
			Aᵃ = Aᵃinput[clicks.inputindex[t][1]]
			update_for_transition_probabilities!(P, adaptedclicks, clicks, t)
			transitionmatrix!(Aᵃ, P)
		end
		p𝐚 = Aᵃ[:,a[t-1]]
        p𝐜 = Aᶜ[:,c[t-1]]
        a[t] = findfirst(rand() .< cumsum(p𝐚))
        c[t] = findfirst(rand() .< cumsum(p𝐜))
    end
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
		  	ntimesteps=trial.ntimesteps,
			previousanswer=trial.previousanswer,
			a=a,
			c=c)
end

"""
	sampleemissions(mpGLM, spikehistorylags, trials)

Generate one sample from the mixture of Poisson generalized linear model (GLM) of a neuron

ARGUMENT
-`mpGLM`: the fitted mixture of Poisson GLM of a neuron
-`trials`: a vector of structures, one of which contains the generated states of the accumulator and coupling variable of one trial

RETURN
-`𝐲̂`: a sample of the spike train response for each timestep
"""
function sampleemissions(mpGLM::MixturePoissonGLM, trials::Vector{<:Trial})
	@unpack Δt, d𝛏_dB, max_spikehistory_lag, 𝐗, 𝐕, 𝐲 = mpGLM
	@unpack 𝐠, 𝐮, 𝐯 = mpGLM.θ
	𝐡 = 𝐮[1:max_spikehistory_lag]
	𝐞 = 𝐮[max_spikehistory_lag+1:end]
	indices𝐄 = length(𝐠[1]) .+ max_spikehistory_lag .+ (1:length(𝐞))
	𝐄 = @view 𝐗[:,indices𝐄]
	𝐄𝐞 = 𝐄*𝐞
	K𝐠 = length(𝐠)
	K𝐯 = length(𝐯)
	Ξ = length(d𝛏_dB)
	max_spikes_per_step = floor(1000Δt)
    𝐲̂ = similar(𝐲)
    τ = 0
    for m in eachindex(trials)
        for t = 1:trials[m].ntimesteps
            τ += 1
            j = trials[m].a[t]
            k = trials[m].c[t]
			gₖ = 𝐠[min(k, K𝐠)][1]
			𝐯ₖ = 𝐯[min(k, K𝐯)]
			L = gₖ + 𝐄𝐞[τ]
			for i in eachindex(𝐯ₖ)
				L+= d𝛏_dB[j]*𝐕[τ,i]*𝐯ₖ[i]
			end
			for lag = 1:min(max_spikehistory_lag, t-1)
				L += 𝐡[lag]*𝐲̂[τ-lag]
			end
            λ = softplus(L)
            𝐲̂[τ] = min(rand(Poisson(λ*Δt)), max_spikes_per_step)
        end
    end
	return 𝐲̂
end

"""
    sample(model)

Generate latent and emission variables for all trials of all trialsets

ARGUMENT
-`model`: an instance of the factorial-hidden Markov drift-diffusion model

OUTPUT
-`model`: an instance of FHM-DDM with generated variables
"""
function sample(model::Model;
                datafilename::String="sample.mat",
                resultsfilename::String="resultsofsample.mat")
    optionsdict = dictionary(model.options)
    optionsdict["datapath"] = dirname(optionsdict["datapath"])*"/"*datafilename
    optionsdict["resultspath"] = dirname(optionsdict["resultspath"])*"/"*resultsfilename
	options = Options(optionsdict)
	memory = Memoryforgradient(model)
	P = update!(memory, model, concateateparameters(model)[1])
    sampledtrialsets =
		map(model.trialsets) do trialset
			sampledtrials =	map(trialset.trials) do trial
								sample!(memory, P, model.θnative, trial)
							end
			sampledmpGLMs =map(trialset.mpGLMs) do mpGLM
						sample(mpGLM, sampledtrials)
					end
			return Trialset(mpGLMs=sampledmpGLMs, trials=sampledtrials)
		end
	Model(options, trialsets)
end

"""
    sample(mpGLM, spikehistorylags, trials)

Generate one sample from the mixture of Poisson generalized linear model (GLM) of a neuron

ARGUMENT
-`mpGLM`: the fitted mixture of Poisson GLM of a neuron
-`sampledtrials`: a vector of structures, one of which contains the generated states of the accumulator and coupling variable of one trial

RETURN
-`mpGLM`: a sample of the mixture of Poisson GLM
"""
function sample(mpGLM::MixturePoissonGLM,
                sampledtrials::Vector{<:Trial})
    𝐲̂ = sampleemissions(mpGLM, sampledtrials)
	θ = GLMθ(𝐠 = map(𝐠ₖ->copy(𝐠ₖ), mpGLM.θ.𝐠),
			𝐮 = copy(mpGLM.θ.𝐮),
			𝐯 = map(𝐯ₖ->copy(𝐯ₖ), mpGLM.θ.𝐯))
    MixturePoissonGLM(Δt=mpGLM.Δt,
                      d𝛏_dB=mpGLM.d𝛏_dB,
					  max_spikehistory_lag=mpGLM.max_spikehistory_lag,
					  Φ=mpGLM.Φ,
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
