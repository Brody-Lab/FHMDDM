"""
	dictionary(sample)
"""
dictionary(sample::Sample) = Dict("trialsets"=>map(dictionary, sample.trialsets))

"""
	dictionary(trialsetsample)
"""
dictionary(trialsetsample::TrialsetSample) = Dict("trials"=>map(dictionary, trialsetsample.trials))

"""
	dictionary(trialsample)
"""
dictionary(trialsample::TrialSample) = Dict("choice"=>trialsample.choice, "spiketrains"=>trialsample.spiketrains)

"""
	drawsamples(model, nsamples)

Simulate the behavioral choice and neuronal spike trains

The model is run forward in time on each trial using the actual auditory clicks.

ARGUMENT
-`model`: a composite containing the data, parameters, and hyperparameters of a factorial hidden-Markov drift-diffusion model
-`nsamples`: number of samples to draw

RETURN
-`samples`: A vector of composites of the data type `Sample`
"""
function drawsamples(model::Model, nsamples::Integer)
	memory = Memoryforgradient(model)
	P = update_for_latent_dynamics!(memory, model.options, model.θnative)
	a = zeros(Int, memory.maxtimesteps)
	c = zeros(Int, memory.maxtimesteps)
<<<<<<< Updated upstream
=======
	𝛏 = model.trialsets[1].mpGLMs[1].d𝛏_dB.*model.θnative.B[1]
>>>>>>> Stashed changes
	trialsamples =
		map(model.trialsets) do trialset
			𝐄𝐞 = map(mpGLM->externalinput(mpGLM), trialset.mpGLMs)
			𝐡 = map(mpGLM->postspikefilter(mpGLM), trialset.mpGLMs)
			𝛚 = map(mpGLM->transformaccumulator(mpGLM), trialset.mpGLMs)
			map(trialset.trials) do trial
				accumulator_prior_transitions!(memory.Aᵃinput, P, memory.p𝐚₁, trial)
<<<<<<< Updated upstream
				collect(sampletrial!(a, c, 𝐄𝐞, 𝐡, memory, 𝛚, model.θnative.ψ[1], trial, trialset) for s=1:nsamples)
=======
				collect(sampletrial!(a, c, 𝐄𝐞, 𝐡, memory, 𝛚, model.θnative.ψ[1], trial, trialset, 𝛏) for s=1:nsamples)
>>>>>>> Stashed changes
			end
		end
	map(1:nsamples) do s
		Sample(trialsets =
				map(eachindex(model.trialsets)) do i
					TrialsetSample(trials =
									map(eachindex(model.trialsets[i].trials)) do m
										trialsamples[i][m][s]
									end)
				end)
	end
end

"""
<<<<<<< Updated upstream
	sampletrial!(a, c, 𝐄𝐞, 𝐡, memory 𝛚, trial, trialset)
=======
	sampletrial!(a, c, 𝐄𝐞, 𝐡, memory 𝛚, trial, trialset, 𝛏)
>>>>>>> Stashed changes

Simulate the choice and spike trains on a trial.

The model is run forward in time, and the value of the latent variables are simulated using the actual timing of the auditory clicks. Then, the choice and spike trains generated based on the simulated values of the latent variable.

MODIFIED ARGUMENT
-`a`: a vector used for the simulation of the accumulator state
-`c`: a vector used for the simulation of the coupling state

UNMODIFIED ARGUMENT
-`𝐄𝐞`: external input. Element `𝐄𝐞[n][τ]` corresponds to the n-th neuron at the τ-th time step among timesteps concatenated across trials in a trialset
-`𝐡`: postspike filter. Element `𝐡[n][q]` corresponds to the n-th neuron at the q-th time step after the spike
-`memory`: a composite used for in-place computations
-`𝛚`: transformed accumulated evidence. Element `𝛚[n][i]` corresponds to the n-th neuron at the i-th accumulator state
-`ψ`: behavioral lapse rate
-`trial`: a composite containing the behavioral choice and click timing in a trial
-`trialset`: a composite containing the behavioral, auditory, and neuronal data of a set of trials
<<<<<<< Updated upstream
=======
-`𝛏`: a vector of floats representing the value of the accumulator in each state
>>>>>>> Stashed changes

RETURN
-simulation of the choice and the spike trains in a trial
"""
function sampletrial!(a::Vector{<:Integer},
					c::Vector{<:Integer},
					𝐄𝐞::Vector{<:Vector{<:AbstractFloat}},
					𝐡::Vector{<:Vector{<:AbstractFloat}},
					memory::Memoryforgradient,
					𝛚::Vector{<:Vector{<:AbstractFloat}},
					ψ::AbstractFloat,
					trial::Trial,
<<<<<<< Updated upstream
					trialset::Trialset)
=======
					trialset::Trialset,
					𝛏::Vector{<:AbstractFloat})
>>>>>>> Stashed changes
	sampleaccumulator!(a, memory.Aᵃinput, memory.Aᵃsilent, memory.p𝐚₁, trial)
	samplecoupling!(c, memory.Aᶜ, trial.ntimesteps, memory.πᶜ)
	choice = samplechoice(a[trial.ntimesteps], ψ, memory.Ξ)
	timesteps = trial.τ₀ .+ (1:trial.ntimesteps)
<<<<<<< Updated upstream
	spiketrains=map(𝐄𝐞, 𝐡, trialset.mpGLMs, 𝛚) do 𝐄𝐞, 𝐡, mpGLM, 𝛚
					samplespiketrain(a, c, 𝐄𝐞, 𝐡, mpGLM, 𝛚, timesteps)
				end
	TrialSample(choice=choice, spiketrains=spiketrains)
=======
	outputs = map(𝐄𝐞, 𝐡, trialset.mpGLMs, 𝛚) do 𝐄𝐞, 𝐡, mpGLM, 𝛚
					samplespiketrain(a, c, 𝐄𝐞, 𝐡, mpGLM, 𝛚, timesteps)
				end
	λ = collect(output[1] for output in outputs)
	spiketrains = collect(output[2] for output in outputs)
	accumulator = collect(𝛏[a[t]] for t=1:trial.ntimesteps)
	TrialSample(accumulator=accumulator,
				choice=choice,
				λ=λ,
				spiketrains=spiketrains)
>>>>>>> Stashed changes
end

"""
    Model(model, sample)

Package into a composite simulated choices and spike trains with the auditory clicks, parameters, and hyperparameters used for simulation

ARGUMENT
-`model`: a composite containing the parameters and hyperparameters of a factorial-hidden Markov drift-diffusion model, and the data used to fit the model

OPTIONAL ARGUMENT
-`folderpath`: the absolute path of the folder in which the data, summary, and simulations of the model would be saved

OUTPUT
-a composite containing the parameters, hyperparameters, auditory clicks, and simulations
"""
function Model(model::Model, sample::Sample; folderpath::String = dirname(model.options.datapath))
	newtrialsets = 	map(model.trialsets, sample.trialsets) do trialset, trialsetsample
						newtrials =	map(trialset.trials, trialsetsample.trials) do trial, trialsample
										Trial(((fieldname == :choice) ? trialsample.choice : getfield(trial, fieldname) for fieldname in fieldnames(Trial))...)
									end
						new_mpGLMs = map(trialset.mpGLMs, eachindex(trialset.mpGLMs)) do old_mpGLM, n
										values = map(fieldnames(MixturePoissonGLM)) do fieldname
													if fieldname == :θ
														FHMDDM.copy(old_mpGLM.θ)
													elseif fieldname == :𝐲
														vcat((trialsample.spiketrains[n] for trialsample in trialsetsample.trials)...)
													else
														getfield(old_mpGLM, fieldname)
													end
												end
										MixturePoissonGLM(values...)
									end
						Trialset(trials=newtrials, mpGLMs=new_mpGLMs)
					end
		options = dictionary(model.options)
		options["datapath"] = joinpath(folderpath,"data.mat")
		Model(options=Options(model.options.nunits, options),
			gaussianprior=GaussianPrior(model.options, newtrialsets),
			θnative=FHMDDM.copy(model.θnative),
			θreal=FHMDDM.copy(model.θreal),
			θ₀native=FHMDDM.copy(model.θ₀native),
			trialsets=newtrialsets)
<<<<<<< Updated upstream
=======
end

"""
    save_accumulator_λ(folderpath, sample)

Save

ARGUMENT
-`folderpath`: the absolute path of folder in which the simulated moment-to-moment values of the accumulated evidence and firing rates are saved
-`sample`: an object containing a sample of simulated moment-to-moment values of the latent variables and the firing rates as well as the emissions generated by the latent variables
"""
function save_accumulator_λ(folderpath::String, sample::Sample)
	accumulator = map(sample.trialsets) do trialset
					map(trialset.trials) do trial
						trial.accumulator
					end
				end
	matwrite(joinpath(folderpath, "accumulator.mat"), Dict("accumulator"=>accumulator))
	lambda_spikes_per_s = map(sample.trialsets) do trialset
							map(trialset.trials) do trial
								trial.λ
							end
						end
	matwrite(joinpath(folderpath, "lambda_spikes_per_s.mat"), Dict("lambda_spikes_per_s"=>lambda_spikes_per_s))
>>>>>>> Stashed changes
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
function sampleaccumulator!(a::Vector{<:Integer},
							Aᵃinput::Vector{<:Matrix{<:Real}},
							Aᵃsilent::Matrix{<:Real},
							p𝐚₁::Vector{<:AbstractFloat},
							trial::Trial)
	a[1] = findfirst(rand() .< cumsum(p𝐚₁))
	@inbounds for t = 2:trial.ntimesteps
		Aᵃ = transitionmatrix(trial.clicks, Aᵃinput, Aᵃsilent, t)
		p𝐚ₜ_aₜ₋₁ = Aᵃ[:,a[t-1]]
		a[t] = findfirst(rand() .< cumsum(p𝐚ₜ_aₜ₋₁))
	end
	return nothing
end

"""
	samplechoice(a_end, ψ, Ξ)

Sample a choice on a trial

ARGUMENT
-`a_end`: state of the accumulator at the last time step of the trial
-`ψ`: lapse rate
-`Ξ`: number of states that the accumulator can take
"""
function samplechoice(a_end::Integer, ψ::AbstractFloat, Ξ::Integer)
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

"""
	simulate(model)

Simulate choices and spike trains and package them into a composite containing the parameters and hyperparameters used to generate them
"""
simulate(model::Model; folderpath::String = dirname(model.options.datapath)) = Model(model, drawsamples(model, 1)[1]; folderpath=folderpath)

"""
	simulateandsave(model, nsamples)

Generate and save samples drawn from the model

ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters of a factorial hidden-Markov drift-diffusion model
-`nsamples`: number of samples to make

<<<<<<< Updated upstream
RETURN
-`samplepaths`: a vector of String indicating the path to the data of each sample
"""
function simulateandsave(model::Model, nsamples::Integer)
	@assert nsamples > 0
	pad = ceil(Int, log10(nsamples))
	open(joinpath(dirname(model.options.datapath), "samplepaths.txt"), "w") do io
		samplepaths = Vector{String}(undef, nsamples)
	    for i=1:nsamples
	        folderpath = joinpath(dirname(model.options.datapath), "sample"*string(i;pad=pad))
	        !isdir(folderpath) && mkdir(folderpath)
	        samplepaths[i] = joinpath(folderpath, "data.mat")
	        println(io, samplepaths[i])
	        simulation = simulate(model; folderpath=folderpath)
	        savedata(simulation)
=======
OPTIONAL ARGUMENT
-`offset`: offset in the naming

RETURN
-`samplepaths`: a vector of String indicating the path to the data of each sample
"""
function simulateandsave(model::Model, nsamples::Integer; offset::Integer=0)
	@assert nsamples > 0
	pad = ceil(Int, log10(nsamples+offset))
	open(joinpath(dirname(model.options.datapath), "samplepaths.txt"), "w") do io
		samplepaths = Vector{String}(undef, nsamples)
	    for i=1:nsamples
	        folderpath = joinpath(dirname(model.options.datapath), "sample"*string(i+offset;pad=pad))
	        !isdir(folderpath) && mkdir(folderpath)
	        samplepaths[i] = joinpath(folderpath, "data.mat")
	        println(io, samplepaths[i])
			sample = drawsamples(model, 1)[1]
			simulation = Model(model, sample; folderpath=folderpath)
	        savedata(simulation)
			save_accumulator_λ(folderpath,sample)
>>>>>>> Stashed changes
	    end
		return samplepaths
	end
end
<<<<<<< Updated upstream
=======

"""
    savedata(model)

Save the data used to fit a model.

This function is typically used when the data is sampled

ARGUMENT
-`model`: structure containing the data, parameters, and hyperparameters used to fit a factorial hidden Markov drift-diffusion mdoel

OPTIONAL ARGUMENT
-`filename`: the data will be saved at the path `joinpath(dirname(model.options.datapath), filename)`
"""
function savedata(model::Model; filename::String=basename(model.options.datapath))
    data =  map(model.trialsets, 1:length(model.trialsets)) do trialset, index
                Dict("trials"=>map(trial->packagedata(trial, model.options.a_latency_s), trialset.trials),
                     "units"=>map(mpGLM->packagedata(mpGLM), trialset.mpGLMs),
                     "index"=>index)
            end
    dict = Dict("data"=>data, "options"=>dictionary(model.options))
    path = joinpath(dirname(model.options.datapath), filename)
    matwrite(path, dict)
end
>>>>>>> Stashed changes
