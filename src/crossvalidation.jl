"""
    crosssvalidate(model)

Assess how well the factorial hidden Markov drift-diffusion model generalizes to independent datasets

ARGUMENT
-`kfold`: number of cross-validation folds
-`model`: a structure containing the settings, data, and parameters of a factorial hidden-Markov drift-diffusion model

OUTPUT
-an instance of `CVResults`

EXAMPLE
```julia-repl
julia> using FHMDDM
julia> model = Model("/mnt/cup/labs/brody/tzluo/analysis_data/analysis_2022_06_07a_test/T176_2018_05_03/data.mat")
julia> cvresults = crossvalidate(2, model)
julia> save(cvresults, model.options)
```
"""
function crossvalidate(kfold::Integer, model::Model)
    cvindices = CVIndices(model, kfold)
	trainingmodels = pmap(cvindices->train(cvindices, model), cvindices)
    λΔt, pchoice, rll_choice, rll_spikes = test(cvindices, model, trainingmodels)
    CVResults(cvindices = cvindices,
			θ₀native = collect(trainingmodel.θ₀native for trainingmodel in trainingmodels),
			θnative = collect(trainingmodel.θnative for trainingmodel in trainingmodels),
			glmθ = collect(collect(collect(mpGLM.θ for mpGLM in trialset.mpGLMs) for trialset in trainingmodel.trialsets) for trainingmodel in trainingmodels),
			λΔt = λΔt,
			pchoice = pchoice,
			rll_choice = rll_choice,
			rll_spikes = rll_spikes)
end

"""
	train(cvindices, model)

Fit training models

ARGUMENT
-`cvindices`: indices of the trials and timesteps used for training and testing in each fold
-`model`: structure containing the full dataset, parameters, and hyperparameters

RETURN
-`trainingmodel`: structure containing the data in the training trials, parameters optimized for the data in the trainings, and hyperparameters
"""
function train(cvindices::CVIndices, model::Model)
	θ₀native = initializeparameters(model.options)
	trainingmodel = Model(trialsets = trainingset(cvindices, model.trialsets),
						  options = model.options,
						  precisionmatrix = copy(model.precisionmatrix),
						  θ₀native = θ₀native,
						  θnative = Latentθ(([getfield(θ₀native, f)...] for f in fieldnames(Latentθ))...),
						  θreal = native2real(model.options, θ₀native))
	learnparameters!(trainingmodel)
	return trainingmodel
end

"""
	test(cvindices, model, trainingmodels)

Relative log-likelihood of the choices and of the spike train responses

ARGUMENT
-`cvindices`: indices of the trials and timesteps used for training and testing in each fold
-`model`: structure containing both the test and training data
-`trainingmodel`: structure containing only the training data and the parameters learned for those data

OUTPUT
-`rll_choice`: Relative log-likelihood of each choice. The element `rll_choice[i][m]` represents the log-likelihood(base 2) of the choice in the m-th trial in the i-th trialset. Subtracted from this value is the log-likelihood(base 2) of the choice predicted by a Bernoulli distribution parametrized by the fraction of right responses in the training data.
-`rll_spikes`: Relative log-likelihood of the spike train response of each neuron, divided by the total number of spikes. The element `rll_spikes[i][n]` represents the time-average log-likelihood(base 2) of the n-th neuron's spike train in the i-th trialset. Subtracted from this value this is baseline time-average log-likelihood computed under a Poisson distribution parametrized by the average spike train response across time steps in the training data. This difference is divided by the total number of spikes.
"""
function test(cvindices::Vector{<:CVIndices}, model::Model, trainingmodels::Vector{<:Model})
	λΔt = map(model.trialsets) do trialset
			map(trialset.mpGLMs) do mpGLM
				zeros(trialset.ntimesteps)
			end
		  end
	pchoice = map(trialset->zeros(trialset.ntrials), model.trialsets)
	rll_choice = map(trialset->fill(NaN, length(trialset.trials)), model.trialsets)
	rll_spikes = map(trialset->fill(NaN, length(trialset.mpGLMs)), model.trialsets)
	for (cvindices, trainingmodel) in zip(cvindices, trainingmodels)
		test!(λΔt, pchoice, rll_choice, rll_spikes, cvindices, model, trainingmodel)
	end
	for i in eachindex(rll_spikes)
		for n in eachindex(rll_spikes[i])
			rll_spikes[i][n] /= sum(model.trialsets[i].mpGLMs[n].𝐲)
		end
	end
	return λΔt, pchoice, rll_choice, rll_spikes
end

"""
	test!(cvindices, model, trainingmodel)

Out-of-sample relative log-likelihood of choices and spiking

MODIFIED INPUT
-`rll_choice`: Relative log-likelihood of each choice. The element `rll_choice[i][m]` represents the log-likelihood(base 2) of the choice in the m-th trial in the i-th trialset. Subtracted from this value is the log-likelihood(base 2) of the choice predicted by a Bernoulli distribution parametrized by the fraction of right responses in the training data.
-`rll_spikes`: Relative log-likelihood of the spike train response of each neuron, divided by the total number of spikes. The element `rll_spikes[i][n]` represents the time-average log-likelihood(base 2) of the n-th neuron's spike train in the i-th trialset. Subtracted from this value this is baseline time-average log-likelihood computed under a Poisson distribution parametrized by the average spike train response across time steps in the training data. This differece has not yet been divided by the total number of spikes.

UNMODIFIED INPUT
-`cvindices`: indices of the trials and timesteps used for training and testing in each fold
-`model`: structure containing the full dataset, parameters, and hyperparameters
"""
function test!(λΔt::Vector{<:Vector{<:Vector{<:AbstractFloat}}},
				pchoice::Vector{<:Vector{<:AbstractFloat}},
				rll_choice::Vector{<:Vector{<:AbstractFloat}},
				rll_spikes::Vector{<:Vector{<:AbstractFloat}},
				cvindices::CVIndices,
				model::Model,
				trainingmodel::Model)
	testmodel = test(cvindices, model, trainingmodel)
	bernoullis = map(trainingmodel.trialsets) do trialset
					p = mean(map(trial->trial.choice, trialset.trials))
					Bernoulli(p)
				end
	poissons = map(trainingmodel.trialsets) do trialset
				map(trialset.mpGLMs) do mpGLM
					Poisson(mean(mpGLM.𝐲))
				end
			end
	ℓ𝑑, ℓ𝑦 = test(testmodel, bernoullis, poissons)
	for i in eachindex(model.trialsets)
		rll_choice[i][cvindices.testingtrials[i]] .= ℓ𝑑[i]
		for n in eachindex(model.trialsets[i].mpGLMs)
			rll_spikes[i][n] = ℓ𝑦[i][n]
		end
	end
	λΔt_test, pchoice_test = expectedemissions(testmodel)
	for i in eachindex(model.trialsets)
		pchoice[i][cvindices.testingtrials[i]] .= pchoice_test[i]
		for n in eachindex(model.trialsets[i].mpGLMs)
			λΔt[i][n][cvindices.testingtimesteps[i]] .= λΔt_test[i][n]
		end
	end
	return nothing
end

"""
	test(cvindices, model, trainingmodel)

Construct a model with only the test data and the parameters learned from the training data

ARGUMENT
-`cvindices`: indices of the trials and timesteps used for training and testing in each fold
-`model`: structure containing both the test and training data
-`trainingmodel`: structure containing only the training data and the parameters learned for those data

OUTPUT
-`testmodel`: a structure containing only the test data and the parameters learned from the training data
"""
function test(cvindices::CVIndices, model::Model, trainingmodel::Model)
	testmodel = Model(trialsets = testingset(cvindices, model.trialsets),
					options = model.options,
					precisionmatrix = trainingmodel.precisionmatrix,
					θ₀native = trainingmodel.θ₀native,
					θnative = trainingmodel.θnative,
					θreal = native2real(model.options, trainingmodel.θnative))
	for (test_trialset, training_trialset) in zip(testmodel.trialsets, trainingmodel.trialsets)
		for (test_mpGLM, training_mpGLM) in zip(test_trialset.mpGLMs, training_trialset.mpGLMs)
			test_mpGLM.θ.𝐮 .= training_mpGLM.θ.𝐮
			for (test_𝐯, training_𝐯) in zip(test_mpGLM.θ.𝐯, training_mpGLM.θ.𝐯)
				test_𝐯 .= training_𝐯
			end
		end
	end
	return testmodel
end

"""
	test(testmodel, bernoullis, poissons)

Compute the relative log-likelihood of the choices and spike train responses

ARGUMENT
-`testmodel`: a structure containing only the test data and the parameters learned from the training data
-`bernoullis`: Bernoulli models of choice probability. The i-th element corresponds a Bernoulli model of the probability of a right choice in the i-th trialset, with the parameter of the Bernoulli model inferred as the mean fraction of a right choice in the training trials of the i-th trialset.
-`poissons`: Homogeneous Poisson model of spike response. The element `poissons[i][n]` corresponds a homogeneous Poisson model of the spiking response of the n-th neuron in the i-th trialset. The intensity of the Poisson with the parameter of the Bernoulli model inferred as the mean fraction of a right choice in the training trials of the i-th trialset.

OUTPUT
-`ℓ𝑑`: ℓ𝑑[i][m] corresponds to the log-likelihood(base 2) of the choice in the m-th trial in the i-th trialset, relative to the log-likelihood predicted by a Bernoulli model whose parameter is inferred from the fraction of right choices in the training data
-`ℓ𝑦`: ℓ𝑦[i][n] corresponds to the log-likelihood(base 2) of the spiking of the n-th neuron in the i-th trialset, relative to the log-likelihood predicted by a homogenous Poisson whose intensity is inferred from the mean response of the training data of the same neuron.
"""
function test(testmodel::Model, bernoullis::Vector{<:Bernoulli}, poissons::Vector{<:Vector{<:Poisson}})
	@unpack options, θnative, trialsets = testmodel
	@unpack Aᶜ₁₁, Aᶜ₂₂, πᶜ₁ = θnative
	@unpack Δt, K, minpa, Ξ = options
	Aᶜᵀ = [Aᶜ₁₁[1] 1-Aᶜ₁₁[1]; 1-Aᶜ₂₂[1] Aᶜ₂₂[1]]
	πᶜᵀ = [πᶜ₁[1] 1-πᶜ₁[1]]
	Aᵃinput, Aᵃsilent = zeros(Ξ,Ξ), zeros(Ξ,Ξ)
	expλΔt = exp(θnative.λ[1]*Δt)
	dμ_dΔc = differentiate_μ_wrt_Δc(Δt, θnative.λ[1])
	d𝛏_dB = (2 .*collect(1:Ξ) .- Ξ .- 1)./(Ξ-2)
	𝛏 = θnative.B[1].*d𝛏_dB
	transitionmatrix!(Aᵃsilent, minpa, expλΔt.*𝛏, √(Δt*θnative.σ²ₐ[1]), 𝛏)
	σᵢ = √θnative.σ²ᵢ[1]
	ℓ𝑑 = map(trialset->fill(NaN, length(trialset.trials)), testmodel.trialsets)
	ℓ𝑦 = map(trialset->zeros(length(trialset.mpGLMs)), testmodel.trialsets)
	p𝑦 = zeros(Ξ,K)
	log2e = log2(exp(1))
	for i in eachindex(testmodel.trialsets)
		τ = 0
		ntrials = length(testmodel.trialsets[i].trials)
		for m in eachindex(testmodel.trialsets[i].trials)
			@unpack choice, clicks, ntimesteps, previousanswer = trialsets[i].trials[m]
			μ = θnative.μ₀[1] + previousanswer*θnative.wₕ[1]
			p𝐚 = probabilityvector(minpa, μ, σᵢ, 𝛏)
			p𝐜ᵀ = πᶜᵀ
			if length(clicks.time) > 0
				adaptedclicks = adapt(clicks, θnative.k[1], θnative.ϕ[1])
			end
			for t=1:ntimesteps
				τ+=1
				if t > 1
					if t ∈ clicks.inputtimesteps
						cL = sum(adaptedclicks.C[clicks.left[t]])
						cR = sum(adaptedclicks.C[clicks.right[t]])
						𝛍 = expλΔt.*𝛏 .+ (cR-cL).*dμ_dΔc
						σ = √((cR+cL)*θnative.σ²ₛ[1] + Δt*θnative.σ²ₐ[1])
						transitionmatrix!(Aᵃinput, minpa, 𝛍, σ, 𝛏)
						Aᵃ = Aᵃinput
					else
						Aᵃ = Aᵃsilent
					end
					p𝐚 = Aᵃ*p𝐚
					p𝐜ᵀ = p𝐜ᵀ*Aᶜᵀ
				end
				for n in eachindex(trialsets[i].mpGLMs)
					conditionallikelihood!(p𝑦, trialsets[i].mpGLMs[n], τ)
					ℓ𝑦[i][n] += log2(sum(p𝑦.*p𝐚.*p𝐜ᵀ)) - log2e*Distributions.logpdf(poissons[i][n], trialsets[i].mpGLMs[n].𝐲[τ])
				end
			end
			p𝑑 = conditionallikelihood(choice, θnative.ψ[1], Ξ)
			ℓ𝑑[i][m] = log2(sum(p𝑑.*p𝐚)) - log2e*Distributions.logpdf(bernoullis[i], choice)
		end
	end
	return ℓ𝑑, ℓ𝑦
end

"""
    conditionallikelihood(choice, ψ, Ξ)

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
	conditionallikelihood!(p, mpGLM, t)

Conditional likelihood the spike train response at a single timestep

MODIFIED ARGUMENT
-`p`: a matrix whose element `p[i,j]` represents the likelihood conditioned on the accumulator in the i-th state and the coupling in the j-th state

UNMODIFIED ARGUMENT
-`mpGLM`:a structure with information on the mixture of Poisson GLM of a neuron
-`t`: timestep
"""
function conditionallikelihood!(p::Matrix{<:Real}, mpGLM::MixturePoissonGLM, t::Integer)
	@unpack Δt, d𝛏_dB, θ, 𝐕, 𝐗, 𝐲 = mpGLM
	@unpack 𝐮, 𝐯 = θ
	𝐔ₜ𝐮 = 0
	for i in eachindex(𝐮)
		𝐔ₜ𝐮 += 𝐗[t,i]*𝐮[i]
	end
	Ξ, K = size(p)
	for k=1:K
		𝐕ₜ𝐯 = 0
		for i in eachindex(𝐯[k])
			𝐕ₜ𝐯 += 𝐕[t,i]*𝐯[k][i]
		end
		for j=1:Ξ
			L = 𝐔ₜ𝐮 + d𝛏_dB[j]*𝐕ₜ𝐯
			p[j,k] = poissonlikelihood(Δt, L, 𝐲[t])
		end
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
"""
function subsample(mpGLM::MixturePoissonGLM, timesteps::Vector{<:Integer})
    MixturePoissonGLM(Δt = mpGLM.Δt,
                        d𝛏_dB = mpGLM.d𝛏_dB,
						max_spikehistory_lag = mpGLM.max_spikehistory_lag,
						Φ = mpGLM.Φ,
						θ = GLMθ(mpGLM.θ, eltype(mpGLM.θ.𝐮)),
                        𝐕 = mpGLM.𝐕[timesteps, :],
                        𝐗 = mpGLM.𝐗[timesteps, :],
                        𝐲 =mpGLM.𝐲[timesteps])
end
