"""
    expectedemissions(model; nsamples=100)

Compute the probability of a right choice and the expected spike rate

The emissions consist of the behavioral choice in each trial and the spike train response of each neuron at each time step of each trial. The behavioral choice is modelled as a Bernoulli random variable, and the spike train response is modelled as a Poisson random variable. Here we compute the expectated value of these random variablees by averaging across samples of the model.

ARGUMENT
-`model`: An instance of the factorial hidden Markov drift-diffusion model (FHMDDM)

OPTION ARGUMENT
-`nsamples`: number of samples over which the expectation is taken

RETURN
-`λΔt`: expectd value of the number of spikes per `Δt` at each time step. Element λΔt[i][n][t] corresponds to the t-timestep and the n-th neuron in the i-th trialset
-`pchoice`: estimate of the mean of the probability of a right. Element pchoice[i][m] corresponds to the m-th trial in the i-th trialset
"""
function expectedemissions(model::FHMDDM; nsamples::Integer=100)
    @unpack trialsets, options, θnative = model
	@unpack spikehistorylags = options
    pchoice = map(trialset->zeros(trialset.ntrials), trialsets)
    λΔt = map(trialsets) do trialset
			map(trialset.mpGLMs) do mpGLM
				zeros(trialset.ntimesteps)
			end
		  end
	trialinvariant = Trialinvariant(options, θnative; purpose="gradient")
    for i in eachindex(trialsets)
        for s = 1:nsamples
            sampledtrialset = sample(spikehistorylags, θnative, trialinvariant, trialsets[i])
			for m in eachindex(sampledtrialset.trials)
				pchoice[i][m] += sampledtrialset.trials[m].choice
			end
            for n in eachindex(sampledtrialset.mpGLMs)
                λΔt[i][n] .+= sampledtrialset.mpGLMs[n].𝐲
            end
        end
    	pchoice[i] ./= nsamples
        for n in eachindex(λΔt)
            λΔt[i][n] ./= nsamples
        end
    end
    return λΔt, pchoice
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
    mpGLMs =pmap(trialset.mpGLMs) do mpGLM
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
    @unpack Δt, K, 𝐮, 𝐥, 𝐫, 𝚽, 𝛏, 𝐲 = mpGLM
    nspikehistorylags = length(spikehistorylags)
    if nspikehistorylags>0
        𝑙ₘᵢₙ= spikehistorylags[1]
        𝑙ₘₐₓ = spikehistorylags[end]
    end
    𝐔 = copy(mpGLM.𝐔)
    𝐔[:, 1:nspikehistorylags] .= 0.
    𝚽𝐥 = 𝚽*𝐥
    𝚽𝐫 = 𝚽*𝐫
    𝐲̂ = similar(𝐲)
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
            if k == 1 || K == 1
                if 𝛏[j] < 0
                    𝐰ᵀ𝐱 = 𝐔[t,:]⋅𝐮 + 𝛏[j]*𝚽𝐥[t]
                elseif 𝛏[j] > 0
                    𝐰ᵀ𝐱 = 𝐔[t,:]⋅𝐮 + 𝛏[j]*𝚽𝐫[t]
                else
                    𝐰ᵀ𝐱 = 𝐔[t,:]⋅𝐮
                end
            else
                𝐰ᵀ𝐱 = 𝐔[t,:]⋅𝐮
            end
            λ = softplus(𝐰ᵀ𝐱)
            𝐲̂[t] = rand(Poisson(λ*Δt))
        end
    end
    MixturePoissonGLM(Δt=Δt,
                      K=mpGLM.K,
					  𝐗=mpGLM.𝐗,
                      𝐔=𝐔,
                      𝐮=𝐮,
                      𝐥=𝐥,
                      𝐫=𝐫,
                      𝚽=𝚽,
                      Φ=mpGLM.Φ,
                      𝛏=𝛏,
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
function sample(model::FHMDDM;
                datafilename::String="sample.mat",
                resultsfilename::String="resultsofsample.mat")
    optionsdict = dictionary(model.options)
    optionsdict["datapath"] = dirname(optionsdict["datapath"])*"/"*datafilename
    optionsdict["resultspath"] = dirname(optionsdict["resultspath"])*"/"*resultsfilename
	options = FHMDDMoptions(optionsdict)
	trialinvariant = Trialinvariant(options, θnative; purpose="gradient")
    trialsets = map(trialset->sample(options.spikehistorylags, model.θnative, trialinvariant, trialset), model.trialsets)
	FHMDDM(options, trialsets)
end
