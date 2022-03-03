"""
    crosssvalidate(model, kfold, 𝛌)

Assess how well the factorial hidden Markov drift-diffusion model generalizes to independent datasets

ARGUMENT
-`model`: a structure containing the settings, data, and parameters of a factorial hidden-Markov drift-diffusion model
-`kfold`: number of cross-validation folds
-`𝐬`: a vector of floating point numbers representing the L2 regularization weight on each parameter

OUTPUT
-an instance of `CVResults`

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_02_18_test/data.mat")
julia> concatenatedθ, indexθ = concatenateparameters(model)
julia> 𝐬 = 0.1.*ones(length(concatenatedθ))
julia> cvresults = crossvalidate(model, 5, 𝐬)
"""
function crossvalidate(model::Model,
                       kfold::Integer,
                       𝐬::Vector{<:AbstractFloat};
					   iterations= 1000)
    cvindices = CVIndices(model, kfold)
    θ₀native = map(k->Latentθ(), 1:kfold)
    θnative = map(k->Latentθ(), 1:kfold)
	losses = map(k->fill(NaN, iterations), 1:kfold)
	gradientnorms = map(k->fill(NaN, iterations), 1:kfold)
    glmθ = map(1:kfold) do k
				map(model.trialsets) do trialset
					map(trialset.mpGLMs) do mpGLM
						GLMθ(𝐮=copy(mpGLM.θ.𝐮),
							 𝐯=copy(mpGLM.θ.𝐯),
							 a=copy(mpGLM.θ.a),
							 b=copy(mpGLM.θ.b))
					end
				end
			end
    for k=1:kfold
        θ₀native[k] = initializeparameters(model.options)
        trainingmodel = Model(trialsets = trainingset(cvindices[k], model.trialsets),
                              options = model.options,
                              θ₀native = θ₀native[k],
                              θnative = Latentθ(([getfield(θ₀native[k], f)...] for f in fieldnames(Latentθ))...),
                              θreal = native2real(model.options, θ₀native[k]))
        maximizechoiceLL!(trainingmodel)
        initializeparameters!(trainingmodel)
        losses[k], gradientnorms[k] = maximizeposterior!(trainingmodel, 𝐬, Optim.LBFGS(linesearch = LineSearches.BackTracking()); iterations=iterations)
        θnative[k] = trainingmodel.θnative
        for i in eachindex(model.trialsets)
            for n in eachindex(model.trialsets[i].mpGLMs)
                glmθ[k][i][n] = model.trialsets[i].mpGLMs[n].θ
            end
        end
    end
    rll_choice, rll_spikes = relative_loglikelihood(cvindices, glmθ, model.options, θnative, model.trialsets)
    CVResults(cvindices = cvindices,
              θ₀native = θ₀native,
              θnative = θnative,
              glmθ = glmθ,
              losses = losses,
              gradientnorms = gradientnorms,
              rll_choice = rll_choice,
              rll_spikes = rll_spikes)
end

"""
	relative_loglikelihood(cvindices, glmθ, options, θnative, trialsets)

Relative log-likelihood of the choices and of the spike train responses

ARGUMENT
-`cvindices`: indices of the trials and timesteps used for training and testing in each fold
-`glmθ`: optimized parameters of the Poisson mixture GLM of each neuron in each fold
-`options`: model settings
-`θnative`: optimized parameters specifying the latent variables in each fold
-`trialsets`: the data

OUTPUT
-`rll_choice`: A vector of floating point numbers whose element `rll_choice[i]` represents the trial-averaged log-likelihood of the choices in the i-th trialset. Subtracted from this value is the baseline trial-averaged log-likelihood of the choices, computed under a Bernoulli distribution parametrized by the fraction of right responses.
-`rll_spikes`: A nested vector of floating point numbers element `rll_spikes[i][n]` represents the time-average log-likelihood of the n-th neuron's spike train in the i-th trialset. Subtracted from this value this is baseline time-average log-likelihood computed under a Poisson distribution parametrized by the average spike train response across time steps. The difference is further divided by the number of spikes observed for each neuron.
"""
function relative_loglikelihood(cvindices::Vector{<:CVIndices},
								glmθ::Vector{<:Vector{<:Vector{<:GLMθ}}},
                                options::Options,
                                θnative::Vector{<:Latentθ},
                                trialsets::Vector{<:Trialset})
    rll_choice = zeros(length(trialsets))
    rll_spikes = map(trialset->zeros(length(trialset.mpGLMs)), trialsets)
    kfold = length(cvindices)
    for k=1:kfold
        testingmodel = Model(trialsets = testingset(cvindices[k], trialsets),
                             options = options,
                             θ₀native = θnative[k],
                             θnative = θnative[k],
                             θreal = native2real(options, θnative[k]))
        for i in eachindex(testingmodel.trialsets)
            for n in eachindex(testingmodel.trialsets[i].mpGLMs)
                testingmodel.trialsets[i].mpGLMs[n].θ.𝐮 .= glmθ[k][i][n].𝐮
                testingmodel.trialsets[i].mpGLMs[n].θ.𝐯 .= glmθ[k][i][n].𝐯
                testingmodel.trialsets[i].mpGLMs[n].θ.a .= glmθ[k][i][n].a
                testingmodel.trialsets[i].mpGLMs[n].θ.b .= glmθ[k][i][n].b
            end
        end
		𝛌₀Δt = map(trialsets, cvindices[k].trainingtimesteps) do trialset, trainingtimesteps
					map(trialset.mpGLMs) do mpGLM
						mean(mpGLM.𝐲[trainingtimesteps])
					end
				end
		fractionright = map(trialsets, cvindices[k].trainingtrials) do trialset, trainingtrials
							mean(map(trialset.trials[trainingtrials]) do trial
									trial.choice
								 end)
						end
		ℓ𝑑, ℓ𝑦 = relative_loglikelihood(testingmodel, 𝛌₀Δt, fractionright)
		for i in eachindex(testingmodel.trialsets)
			rll_choice[i] += ℓ𝑑[i]/kfold
            for n in eachindex(testingmodel.trialsets[i].mpGLMs)
				rll_spikes[i][n] += ℓ𝑦[i][n]/kfold
        	end
		end
    end
	return rll_choice, rll_spikes
end

"""
	relative_loglikelihood(model, 𝛌₀, fractionright)

Compute the relative log-likelihood of the choices and spike train responses

ARGUMENT
-`model`: a structure containing the settings, data, and parameters of a factorial hidden-Markov drift-diffusion model
-`𝛌₀Δt`: the mean number of spikes per timestep of each neuron
-`fractionright`: the fraction of responding right

OUTPUT
-`ℓ𝑑`: ℓ𝑑[i] corresponds to the log-likelihood of the choices per trial in the i-th trialset, relative to the log-likelihood under a Bernoulli parametrized by `fractioncorrect`
-`ℓ𝑦`: ℓ𝑦[i][n] corresponds to the log-likelihood per spike of the n-th spike train in the i-th trialset, relative to the log-likelihood under a Poisson parametrized by `𝛌₀`
"""
function relative_loglikelihood(model::Model,
								𝛌₀Δt::Vector{<:Vector{<:AbstractFloat}},
								fractionright::Vector{<:AbstractFloat})
	@unpack options, θnative, trialsets = model
	@unpack Aᶜ₁₁, Aᶜ₂₂, πᶜ₁ = θnative
	@unpack Δt, K, Ξ = options
	Aᶜᵀ = [Aᶜ₁₁[1] 1-Aᶜ₁₁[1]; 1-Aᶜ₂₂[1] Aᶜ₂₂[1]]
	πᶜᵀ = [πᶜ₁[1] 1-πᶜ₁[1]]
	𝛏 = θnative.B[1]*(2collect(1:Ξ) .- Ξ .- 1)/(Ξ-2)
	𝛍 = conditionedmean(0.0, Δt, θnative.λ[1], 𝛏)
	Aᵃ, Aᵃsilent = zeros(Ξ,Ξ), zeros(Ξ,Ξ)
	stochasticmatrix!(Aᵃsilent, 𝛍, √(θnative.σ²ₐ[1]*Δt), 𝛏)
	σᵢ = √θnative.σ²ᵢ[1]
	ℓ𝑑 = zeros(length(model.trialsets))
	ℓ𝑦 = map(trialset->zeros(length(trialset.mpGLMs)), model.trialsets)
	p𝑦 = zeros(Ξ,K)
	homogeneousPoissons = map(λΔt->map(λΔt->Poisson(λΔt),λΔt),𝛌₀Δt)
	for i in eachindex(model.trialsets)
		τ = 0
		for m in eachindex(model.trialsets[i].trials)
			@unpack choice, clicks, ntimesteps, previousanswer = trialsets[i].trials[m]
			C = adapt(clicks, θnative.k[1], θnative.ϕ[1])
			μ = θnative.μ₀[1] + previousanswer*θnative.wₕ[1]
			p𝐚 = probabilityvector(μ, σᵢ, 𝛏)
			p𝐜ᵀ = copy(πᶜᵀ)
			for t=1:ntimesteps
				τ+=1
				if t > 1
					if isempty(clicks.inputindex[t])
						p𝐚 = Aᵃsilent*p𝐚
					else
						cL = sum(C[clicks.left[t]])
						cR = sum(C[clicks.right[t]])
						𝛍 = conditionedmean(cR-cL, Δt, θnative.λ[1], 𝛏)
						σ = √( (cL+cR)*θnative.σ²ₛ[1] + θnative.σ²ₐ[1]*Δt )
						stochasticmatrix!(Aᵃ, 𝛍, σ, 𝛏)
						p𝐚 = Aᵃ*p𝐚
					end
					p𝐜ᵀ = p𝐜ᵀ*Aᶜᵀ
				end
				for n in eachindex(trialsets[i].mpGLMs)
					likelihood!(p𝑦, trialsets[i].mpGLMs[n], τ)
					ℓ𝑦[i][n] += log(sum(p𝑦.*(p𝐚*p𝐜ᵀ))) - Distributions.logpdf(homogeneousPoissons[i][n], trialsets[i].mpGLMs[n].𝐲[τ])
				end
			end
			p𝑑 = conditionallikelihood(choice, θnative.ψ[1], Ξ)
			ℓ𝑑[i] += log(sum(p𝑑.*p𝐚)) - log(choice ? fractionright[i] : 1-fractionright[i])
		end
  		ℓ𝑑[i] /= length(model.trialsets[i].trials)
		for n in eachindex(trialsets[i].mpGLMs)
			ℓ𝑦[i][n] /= sum(trialsets[i].mpGLMs[n].𝐲)
		end
	end
	return ℓ𝑑, ℓ𝑦
end

"""
    likelihood(choice, ψ, Ξ)

Conditional probability of a right choice given the state of the accumulator

MODIFIED ARGUMENT
-`p𝐘ₜ𝑑`: A matrix whose element p𝐘ₜ𝑑[j,k] ≡ p(𝐘ₜ, 𝑑 ∣ aₜ = ξⱼ, zₜ = k) for time bin t that is the at the end of the trial

UNMODIFIED ARGUMENT
-`choice`: the observed choice, either right (`choice`=true) or left.
-`ψ`: the prior probability of a lapse state

OPTIONAL ARGUMENT
- `zeroindex`: the index of the bin for which the accumulator variable equals zero
"""
function conditionallikelihood(choice::Bool,
					             ψ::type,
								 Ξ::Integer) where {type<:Real}
	p = ones(type, Ξ)
	zeroindex = cld(Ξ,2)
    p[zeroindex] = 0.5
    if choice
        p[1:zeroindex-1] .= ψ/2
        p[zeroindex+1:end] .= 1-ψ/2
    else
        p[1:zeroindex-1]   .= 1-ψ/2
        p[zeroindex+1:end] .= ψ/2
    end
    return p
end

"""
	likelihood!(p, mpGLM, t)

Conditional likelihood the spike train response at a single timestep

MODIFIED ARGUMENT
-`p`: a matrix whose element `p[i,j]` represents the likelihood conditioned on the accumulator in the i-th state and the coupling in the j-th state

UNMODIFIED ARGUMENT
-`mpGLM`:a structure with information on the mixture of Poisson GLM of a neuron
-`t`: timestep
"""
function likelihood!(p::Matrix{<:Real}, mpGLM::MixturePoissonGLM, t::Integer)
	@unpack Δt, 𝐔, 𝚽, 𝛏, θ, 𝐲, 𝐲! = mpGLM
	𝐔ₜ𝐮 = 𝐔[t,:] ⋅ θ.𝐮
	𝚽ₜ𝐯 = 𝚽[t,:] ⋅ θ.𝐯
	Ξ = size(p,1)
	K = size(p,2)
	if K > 1
		λΔt = softplus(𝐔ₜ𝐮)*Δt
		p[:,2] .= likelihood(λΔt, 𝐲[t],  𝐲![t])
	end
	fa = rectifya(θ.a[1])
	for i = 1:Ξ
		fξ = transformaccumulator(θ.b[1], 𝛏[i])
		λΔt = softplus(𝐔ₜ𝐮 + fa*fξ*𝚽ₜ𝐯)*Δt
		p[i,1] = likelihood(λΔt, 𝐲[t],  𝐲![t])
	end
	return nothing
end

"""
    CVIndices(model, kfold)

Create indices for cross-validation

ARGUMENT
-`model`: a structure containing the settings, data, and parameters of a factorial hidden-Markov drift-diffusion model
-`kfold`: number of cross-validation folds

OUTPUT
-a vector of instances of `CVIndices`

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_02_18_test/data.mat")
julia> cvindices = CVIndices(model, 5)
```
"""
function CVIndices(model::Model,
                   kfold::integertype) where {integertype<:Integer}
    testingtrials = map(k->map(trialset->integertype[], model.trialsets), 1:kfold)
    trainingtrials = map(k->map(trialset->integertype[], model.trialsets), 1:kfold)
    testingtimesteps = map(k->map(trialset->integertype[], model.trialsets), 1:kfold)
    trainingtimesteps = map(k->map(trialset->integertype[], model.trialsets), 1:kfold)
    for i in eachindex(model.trialsets)
        ntrials = length(model.trialsets[i].trials)
        ntestingtrials = cld(ntrials, kfold)
        partitionedtrials = collect(Base.Iterators.partition(shuffle(1:ntrials), ntestingtrials))
        timesteps = map(trial->collect(1:trial.ntimesteps), model.trialsets[i].trials)
        count = 0
        for m in eachindex( model.trialsets[i].trials)
            timesteps[m] .+= count
            count +=  model.trialsets[i].trials[m].ntimesteps
        end
        for k=1:kfold
            testingtrials[k][i] = sort(partitionedtrials[k])
            trainingtrials[k][i] = sort(vcat(partitionedtrials[vcat(1:k-1, k+1:kfold)]...))
            testingtimesteps[k][i] =  vcat(timesteps[testingtrials[k][i]]...)
            trainingtimesteps[k][i] =  vcat(timesteps[trainingtrials[k][i]]...)
        end
    end
    map(1:kfold,
        trainingtrials,
        testingtrials,
        trainingtimesteps,
        testingtimesteps) do    k,
                                trainingtrials,
                                testingtrials,
                                trainingtimesteps,
                                testingtimesteps
        CVIndices(trainingtrials = trainingtrials,
                  testingtrials = testingtrials,
                  trainingtimesteps = trainingtimesteps,
                  testingtimesteps = testingtimesteps)
    end
end

"""
    trainingset(cvindices, trialsets)

Subsample the data for training

ARGUMENT
-`cvindices`: an instance of `CVIndices`
-`trialsets`: a vector of instances of `Trialset`

OUTPUT
- a vector of instances of `Trialset`

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_02_18_test/data.mat")
julia> cvindices = CVIndices(model, 5)
julia> first_training_set = trainingset(cvindices[1], model.trialsets)
```
"""
function trainingset(cvindices::CVIndices, trialsets::Vector{<:Trialset})
    map(trialsets, cvindices.trainingtrials, cvindices.trainingtimesteps) do trialset, trainingtrials, trainingtimesteps
        subsample(trialset, trainingtrials, trainingtimesteps)
    end
end

"""
    testingset(cvindices, trialsets)

Subsample the data for testing

ARGUMENT
-`cvindices`: an instance of `CVIndices`
-`trialsets`: a vector of instances of `Trialset`

OUTPUT
- a vector of instances of `Trialset`

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_02_18_test/data.mat")
julia> cvindices = CVIndices(model, 5)
julia> first_testing_set = testingset(cvindices[1], model.trialsets)
```
"""
function testingset(cvindices::CVIndices, trialsets::Vector{<:Trialset})
    map(trialsets, cvindices.testingtrials, cvindices.testingtimesteps) do trialset, testingtrials, testingtimesteps
        subsample(trialset, testingtrials, testingtimesteps)
    end
end

"""
    subsample(trialset, trialindices, timesteps)

Create an instance of `Trialset` by subsampling

ARGUMENT
-`trialset`: a structure corresponding to a collection of trials in which the same neurons werre recorded
-`trialindices`: a vector of integers indexing the trials to include
-`timesteps`: a vector of integers indexing the timesteps of spiketrains to include

OUTPUT
- an instance of `Trialset`
"""
function subsample(trialset::Trialset,
                   trialindices::Vector{<:Integer},
                   timesteps::Vector{<:Integer})
    Trialset(trials = trialset.trials[trialindices],
             mpGLMs = map(mpGLM->subsample(mpGLM, timesteps), trialset.mpGLMs))
end

"""
    subsample(mpGLM, timesteps)

Create a mixture of Poisson GLM by subsampling the spike train of a neuron

ARGUMENT
-`mpGLM`: a structure with information on the mixture of Poisson GLM of a neuron
-`timesteps`: a vector of integers indexing the timesteps to include

OUTPUT
-an instance of `MixturePoissonGLM`

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_02_18_test/data.mat")
julia> ntimesteps = length(model.trialsets[1].mpGLMs[1].𝐲)
julia> firsthalf = collect(1:cld(ntimesteps,2))
julia> mpGLM = subsample(model.trialsets[1].mpGLMs[1], firsthalf)
```
"""
function subsample(mpGLM::MixturePoissonGLM, timesteps::Vector{<:Integer})
	θ = GLMθ(𝐮=copy(mpGLM.θ.𝐮),
			𝐯=copy(mpGLM.θ.𝐯),
			a=copy(mpGLM.θ.a),
			b=copy(mpGLM.θ.b))
    MixturePoissonGLM(Δt = mpGLM.Δt,
                        K = mpGLM.K,
                        𝐔 = mpGLM.𝐔[timesteps, :],
                        𝚽 = mpGLM.𝚽[timesteps, :],
                        Φ = mpGLM.Φ,
                        θ = θ,
                        𝐗 = mpGLM.𝐗[timesteps, :],
                        𝛏 = mpGLM.𝛏,
                        𝐲 =mpGLM.𝐲[timesteps])
end
