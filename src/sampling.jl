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
"""
function expectedemissions(model::Model; nsamples::Integer=100)
    @unpack trialsets, options, θnative = model
	@unpack spikehistorylags = options
    pchoice = map(trialset->zeros(trialset.ntrials), trialsets)
    λΔt = map(trialsets) do trialset
			map(trialset.mpGLMs) do mpGLM
				zeros(trialset.ntimesteps)
			end
		  end
	trialinvariant = Trialinvariant(model; purpose="gradient")
    for i in eachindex(trialsets)
        for s = 1:nsamples
            choices, 𝐘 = sampleemissions(spikehistorylags, θnative, trialinvariant, trialsets[i])
			for m in eachindex(choices)
				pchoice[i][m] += choices[m]
			end
            for n in eachindex(𝐘)
                λΔt[i][n] .+= 𝐘[n]
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
	sampleemissions(spikehistorylags, θnative, trialinvariant, trialset)

Generate emission variables for one trialset

ARGUMENT
-`spikehistorylags`: a vector indicating the lags of the autorregressive terms
-`θnative`: a structure containing parameters of the model in native space
-`trialinvariant`: quantities used across trials
-`trialset`: a structure containing the data of one trialset

OUTPUT
-`choices`: a sample of the choice in each trial
-`𝐘`: a sample of the spike train response of each neuron in each timestep
"""
function sampleemissions(spikehistorylags::Vector{<:Integer},
		                θnative::Latentθ,
						trialinvariant::Trialinvariant,
		                trialset::Trialset)
    trials =pmap(trialset.trials) do trial
				sample(θnative, trial, trialinvariant)
			end
	𝐘 =map(trialset.mpGLMs) do mpGLM
			sampleemissions(mpGLM, spikehistorylags, trials)
		end
	choices = map(trial->trial.choice, trials)
	return choices, 𝐘
end

"""
	sampleemissions(mpGLM, spikehistorylags, trials)

Generate one sample from the mixture of Poisson generalized linear model (GLM) of a neuron

ARGUMENT
-`mpGLM`: the fitted mixture of Poisson GLM of a neuron
-`spikehistorylags`: a vector indicating the lags of the autorregressive terms
-`trials`: a vector of structures, one of which contains the generated states of the accumulator and coupling variable of one trial

RETURN
-`𝐲̂`: a sample of the spike train response for each timestep
"""
function sampleemissions(mpGLM::MixturePoissonGLM,
                		spikehistorylags::Vector{integertype},
                		trials::Vector{<:Trial}) where {integertype<:Integer}
	@unpack Δt, K, 𝚽, 𝛏, 𝐲 = mpGLM
	@unpack 𝐮, 𝐯 = mpGLM.θ
	max_spikes_per_step = floor(Δt/1e-3)
    nspikehistorylags = length(spikehistorylags)
    if nspikehistorylags>0
        𝑙ₘᵢₙ= spikehistorylags[1]
        𝑙ₘₐₓ = spikehistorylags[end]
    end
    𝐔 = copy(mpGLM.𝐔)
    𝐔[:, 1:nspikehistorylags] .= 0.
	fa = rectifya(mpGLM.θ.a[1])
    𝚽𝐯 = 𝚽*𝐯
    𝐲̂ = similar(𝐲)
	f𝛏 = map(ξ->transformaccumulator(mpGLM.θ.b[1], ξ), 𝛏)
	zeroindex = (length(𝛏)+1)/2
    t = 0
    for m in eachindex(trials)
        for tₘ in 1:trials[m].ntimesteps
            t += 1
            if nspikehistorylags>0 && tₘ > 𝑙ₘᵢₙ
                𝑙 = min(𝑙ₘₐₓ, tₘ-1)
                𝐔[t,1:𝑙-𝑙ₘᵢₙ+1] = 𝐲̂[t-𝑙ₘᵢₙ:-1:t-𝑙]
            end
            j = trials[m].a[tₘ]
            k = trials[m].c[tₘ]
            if k == 1
                if j < zeroindex
                    𝐰ᵀ𝐱 = 𝐔[t,:]⋅𝐮 + f𝛏[j]*𝚽𝐯[t]
                elseif j > zeroindex
                    𝐰ᵀ𝐱 = 𝐔[t,:]⋅𝐮 + fa*f𝛏[j]*𝚽𝐯[t]
                else
                    𝐰ᵀ𝐱 = 𝐔[t,:]⋅𝐮
                end
            else
                𝐰ᵀ𝐱 = 𝐔[t,:]⋅𝐮
            end
            λ = softplus(𝐰ᵀ𝐱)
            𝐲̂[t] = min(rand(Poisson(λ*Δt)), max_spikes_per_step)
        end
    end
	return 𝐲̂
end

"""
    sample(spikehistorylags, θnative, trialinvariant, trialset)

Generate latent and emission variables for one trialset

ARGUMENT
-`spikehistorylags`: a vector indicating the lags of the autorregressive terms
-`θnative`: a structure containing parameters of the model in native space
-`trialinvariant`: quantities used across trials
-`trialset`: a structure containing the data of one trialset

OUTPUT
-an instance of `Trialset`
"""
function sample(spikehistorylags::Vector{<:Integer},
                θnative::Latentθ,
				trialinvariant::Trialinvariant,
                trialset::Trialset)
    trials =pmap(trialset.trials) do trial
				sample(θnative, trial, trialinvariant)
			end
	mpGLMs =map(trialset.mpGLMs) do mpGLM
				sample(mpGLM, spikehistorylags, trials)
			end
	Trialset(mpGLMs=mpGLMs, trials=trials)
end

"""
    sample(θnative, trial, trialinvariant)

Sample the latent variables and behavioral choice for one trial

ARGUMENT
-`θnative`: model parameters in their native space
-`trial`: stucture containing the stimulus information of one trial being used for sampling
-`trialinvariant`: quantities used across trials

RETURN
-an instance of `Trial` containing the generated behavioral choice as well as the sequence of latent variables
"""
function sample(θnative::Latentθ,
                trial::Trial,
                trialinvariant::Trialinvariant)
    @unpack Aᵃsilent, Aᶜ, Δt, πᶜᵀ, Ξ, 𝛏 = trialinvariant
    @unpack clicks = trial
    a = zeros(Int, trial.ntimesteps)
    c = zeros(Int, trial.ntimesteps)
    μ₁ = θnative.μ₀[1] + trial.previousanswer*θnative.wₕ[1]
    p𝐚 = probabilityvector(μ₁, √θnative.σ²ᵢ[1], 𝛏)
    p𝐜 = vec(πᶜᵀ)
    a[1] = findfirst(rand() .< cumsum(p𝐚))
    c[1] = findfirst(rand() .< cumsum(p𝐜))
    C = adapt(clicks, θnative.k[1], θnative.ϕ[1])
    Aᵃ = zeros(Ξ, Ξ)
    for t = 2:trial.ntimesteps
        if isempty(clicks.inputindex[t])
			p𝐚 = @view Aᵃsilent[:,a[t-1]]
		else
			cL = sum(C[clicks.left[t]])
			cR = sum(C[clicks.right[t]])
			stochasticmatrix!(Aᵃ, cL, cR, trialinvariant, θnative)
			p𝐚 = @view Aᵃ[:,a[t-1]]
		end
        p𝐜 = Aᶜ*p𝐜
        a[t] = findfirst(rand() .< cumsum(p𝐚))
        c[t] = findfirst(rand() .< cumsum(p𝐜))
    end
	zeroindex = cld(Ξ,2)
    pright = a[end]==zeroindex ? 0.5 : a[end]>zeroindex ? 1-θnative.ψ[1]/2 : θnative.ψ[1]/2
    Trial(	clicks=trial.clicks,
          	choice=rand() < pright,
		  	ntimesteps=trial.ntimesteps,
			previousanswer=trial.previousanswer,
			a=a,
			c=c)
end

"""
    sample(mpGLM, spikehistorylags, trials)

Generate one sample from the mixture of Poisson generalized linear model (GLM) of a neuron

ARGUMENT
-`mpGLM`: the fitted mixture of Poisson GLM of a neuron
-`spikehistorylags`: a vector indicating the lags of the autorregressive terms
-`trials`: a vector of structures, one of which contains the generated states of the accumulator and coupling variable of one trial

RETURN
-`mpGLM`: a sample of the mixture of Poisson GLM
"""
function sample(mpGLM::MixturePoissonGLM,
                spikehistorylags::Vector{<:Integer},
                trials::Vector{<:Trial})
    𝐲̂ = sampleemissions(mpGLM, spikehistorylags, trials)
	θ = GLMθ(𝐮 = copy(mpGLM.θ.𝐮),
			𝐯 = copy(mpGLM.θ.𝐯),
			a = copy(mpGLM.θ.a),
			b = copy(mpGLM.θ.b))
    MixturePoissonGLM(Δt=mpGLM.Δt,
                      K=mpGLM.K,
					  𝐗=mpGLM.𝐗,
                      𝐔=mpGLM.𝐔,
                      𝚽=mpGLM.𝚽,
                      Φ=mpGLM.Φ,
					  θ=θ,
                      𝛏=mpGLM.𝛏,
                      𝐲=𝐲̂)
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
	trialinvariant = Trialinvariant(model; purpose="gradient")
    trialsets = map(trialset->sample(options.spikehistorylags, model.θnative, trialinvariant, trialset), model.trialsets)
	Model(options, trialsets)
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

RETURN
-a structure containing the times and time step indices of simulated clicks

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> FHMDDM.sampleclicks(0.01, 40, 0.01, 100, 30)
Clicks{Vector{Float64}, BitVector, Vector{Int64}, Vector{Vector{Int64}}}
  time: Array{Float64}((35,)) [0.03535593874845616, 0.11620327763153825, 0.20070443480234917, 0.20267578816244386, 0.24651121564481937, 0.25271348149289374, 0.28212598515585835, 0.2923893454249192, 0.38639914440058487, 0.39104694133712664  …  0.7358489636299059, 0.7496389490980643, 0.7775989000479905, 0.7882768608067173, 0.7901513878713159, 0.7957808498880533, 0.813886931295781, 0.8293245350122752, 0.8709161080727402, 0.9751747027643045]
  inputtimesteps: Array{Int64}((29,)) [5, 13, 22, 26, 27, 30, 31, 40, 41, 45  …  72, 75, 76, 79, 80, 81, 83, 84, 89, 99]
  inputindex: Array{Vector{Int64}}((100,))
  source: BitVector
  left: Array{Vector{Int64}}((100,))
  right: Array{Vector{Int64}}((100,))
```
"""
function sampleclicks(a_latency_s::Number, clickrate_Hz::Number, Δt::Number, ntimesteps::Integer, right2left::Number)
	leftrate = 1/(1+right2left)
	rightrate = clickrate_Hz - leftrate
	duration_s = ntimesteps*Δt
	leftclicktimes = samplePoissonprocess(leftrate, duration_s)
	rightclicktimes = samplePoissonprocess(rightrate, duration_s)
	Clicks(a_latency_s, Δt, leftclicktimes, ntimesteps, rightclicktimes)
end

"""
	samplePoissonprocess(λ, T)

Return the event times from sampling a Poisson process with rate `λ` for duration `T`

INPUT
-`λ`: expected number of events per unit time
-`T`: duration in time to simulate the process

RETURN
-a vector of event times

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> FHMDDM.samplePoissonprocess(10, 1.0)
9-element Vector{Float64}:
 0.07905414067672215
 0.34836141788431263
 0.4290121941428464
 0.4760418862082705
 0.48733013889669546
 0.4900673382176315
 0.6172597288986647
 0.988594823030946
 0.990668151277051
```
"""
function samplePoissonprocess(λ::Number, T::Number)
	@assert λ > 0
	@assert T > 0
	times = zeros(1)
	while times[end] < T
		times = vcat(times, times[end]+randexp()/λ)
	end
	return times[2:end-1]
end
