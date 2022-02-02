"""
    likelihood(mpGLM, j, k)

Conditional likelihood of the spike train, given the index of the state of the accumulator `j` and the state of the coupling `k`, and also by the prior likelihood of the regression weights

UNMODIFIED ARGUMENT
-`mpGLM`: the mixture of Poisson generalized linear model for one neuron
-`j`: state of accumulator variable
-`k`: state of the coupling variable

RETURN
-`𝐩`: a vector by which the conditional likelihood of the spike train and the prior likelihood of the regression weights are multiplied against
"""
function likelihood(mpGLM::MixturePoissonGLM, j::Integer, k::Integer)
    @unpack Δt, 𝐲, 𝐲! = mpGLM
    𝛌 = lambda(mpGLM, j, k)
    𝐩 = 𝛌 # reuse memory
    for i in eachindex(𝛌)
        𝐩[i] = (𝛌[i]*Δt)^𝐲[i] / exp(𝛌[i]*Δt) / 𝐲![i]
    end
    return 𝐩
end

"""
    likelihood!(𝐩, mpGLM, j, k)

In-place multiplication of `𝐩` by the conditional likelihood of the spike train, given the index of the state of the accumulator `j` and the state of the coupling `k`, and also by the prior likelihood of the regression weights

MODIFIED ARGUMENT
-`𝐩`: a vector by which the conditional likelihood of the spike train and the prior likelihood of the regression weights are multiplied against

UNMODIFIED ARGUMENT
-`mpGLM`: the mixture of Poisson generalized linear model for one neuron
-`j`: state of accumulator variable
-`k`: state of the coupling variable

RETURN
-`nothing`
"""
function likelihood!(𝐩, mpGLM::MixturePoissonGLM, j::Integer, k::Integer)
    @unpack Δt, 𝐲, 𝐲! = mpGLM
    𝛌 = lambda(mpGLM, j, k)
    for i in eachindex(𝛌)
        𝐩[i] *= (𝛌[i]*Δt)^𝐲[i] / exp(𝛌[i]*Δt) / 𝐲![i]
    end
    return nothing
end

"""
    loglikelihood(mpGLM, ξ, k)

Compute the conditional log-likelihood of the spike train and the input weights given the state of the accumulator variable and of the coupling variable

ARGUMENT
-`mpGLM`: the mixture of Poisson generalized linear model of one neuron
-`j`: state of the accumulator variable
-`k`: state of the coupling variable

RETURN
-a vector whose 𝑡-th element is the log-likelihood 𝑙𝑜𝑔 𝑝(𝐲[t] ∣ 𝐮, 𝐥, 𝐫, 𝐔, 𝚽)
"""
function loglikelihood(mpGLM::MixturePoissonGLM, j::Integer, k::Integer)
    @unpack Δt, 𝐲, log_𝐲! = mpGLM
    𝛌 = lambda(mpGLM, j, k)
    𝐲.*log.(𝛌.*Δt) .- 𝛌.*Δt .- log_𝐲!
end

"""
    lambda(mpGLM, j, k)

Compute λ of the mixture of Poisson GLM given the j-th accumulator state and k-th coupling state

ARGUMENT
-`mpGLM`: the mixture of Poisson generalized linear model of one neuron
-`j`: state of the accumulator variable
-`k`: state of the coupling variable

RETURN
-`𝛌`: a vector whose element 𝛌[t] corresponds to the t-th time bin in the trialset
"""
function lambda(mpGLM::MixturePoissonGLM, j::Integer, k::Integer)
    @unpack 𝐔, 𝐮, 𝐗, 𝛏, 𝐥, 𝐫 = mpGLM
    if k == 1 && 𝛏[j] != 0.0
        if 𝛏[j] < 0
            𝐰 = vcat(𝐮, 𝛏[j].*𝐥)
        else
            𝐰 = vcat(𝐮, 𝛏[j].*𝐫)
        end
        softplus.(𝐗*𝐰)
    else
        softplus.(𝐔*𝐮)
    end
end

"""
    estimatefilters!(trialsets, γ)

Update the filters in the mixture of Poisson generalized linear models

MODIFIED ARGUMENT
-`trialsets`: vector of data for each group of trials

UNMODIFIED ARGUMENT
-`γ`: joint posterior likelihood of each accumulator state and each coupling state at each time bin. `γ[i][ξ,k][t]` corresponds to the joint posterior of accumulator state ξ and coupling state k at the t-th time bin concatenated across trials in the i-th trialset

RETURN
-nothing
"""
function estimatefilters!(trialsets::Vector{<:Trialset},
                          γ::Vector{<:Matrix{<:Vector{<:AbstractFloat}}};
                          show_trace::Bool=true)
    concatentatedθ = map(trialsets, γ) do trialset, γ
                        pmap(trialset.mpGLMs) do mpGLM
                            estimatefilters(γ, mpGLM,; show_trace=show_trace)
                        end
                    end
    Pᵤ = length(trialsets[1].mpGLMs[1].𝐮)
    Pₗ = length(trialsets[1].mpGLMs[1].𝐥)
    for i in eachindex(concatentatedθ)
        for n in eachindex(concatentatedθ[i])
            trialsets[i].mpGLMs[n].𝐮 .= concatentatedθ[i][n][1:Pᵤ]
            trialsets[i].mpGLMs[n].𝐥 .= concatentatedθ[i][n][Pᵤ+1:Pᵤ+Pₗ]
            trialsets[i].mpGLMs[n].𝐫 .= concatentatedθ[i][n][Pᵤ+Pₗ+1:end]
        end
    end
    return nothing
end

"""
    estimatefilters(γ, mpGLM; show_trace)

Estimate the filters of the Poisson mixture GLM of one neuron

ARGUMENT
-`γ`: posterior probabilities of the latent
-`mpGLM`: the Poisson mixture GLM of one neuron

OPTIONAL ARGUMENT
-`show_trace`: whether to show information about each step of the optimization

RETURN
-weights concatenated into a single vector
"""
function estimatefilters(γ::Matrix{<:Vector{<:AbstractFloat}},
                         mpGLM::MixturePoissonGLM;
                         show_trace::Bool=true)
    @unpack 𝐮, 𝐥, 𝐫 = mpGLM
    x₀ = vcat(𝐮, 𝐥, 𝐫)
    f(x) = negativeexpectation(γ, mpGLM, x)
    g!(∇, x) = ∇negativeexpectation!(∇, γ, mpGLM, x)
    h!(𝐇, x) = 𝐇negativeexpectation!(𝐇, γ, mpGLM, x)
    results = Optim.optimize(f, g!, h!, x₀, NewtonTrustRegion(), Optim.Options(show_trace=show_trace))
    show_trace && println("The model converged: ", Optim.converged(results))
    return Optim.minimizer(results)
end

"""
    negativeexpectation(γ, mpGLM, x)

Negative of the expectation of the log-likelihood of the mixture of Poisson generalized model of one neuron

ARGUMENT
-`γ`: posterior probability of the latent variable
-`mpGLM`: the GLM of one neuron
-`x`: weights of the linear filters of the GLM concatenated as a vector of floating-point numbers

RETURN
-expectation of the log-likelihood of the spike train of one neuron
"""
function negativeexpectation(γ::Matrix{<:Vector{<:AbstractFloat}},
                             mpGLM::MixturePoissonGLM,
                             x::Vector{<:AbstractFloat})
    @unpack Δt, 𝐔, 𝚽, 𝛏, 𝐗, 𝐲 = mpGLM
    Pᵤ = size(𝐔,2)
    Pₗ = size(𝚽,2)
    𝐮 = x[1:Pᵤ]
    𝐥 = x[Pᵤ+1:Pᵤ+Pₗ]
    𝐫 = x[Pᵤ+Pₗ+1:end]
    𝐔𝐮 = 𝐔*𝐮
    Ξ = size(γ,1)
    zeroindex = cld(Ξ,2)
    if size(γ,2) > 1 # i.e, the coupling variable has more than one state
        ∑γdecoupled = γ[zeroindex,1] .+ sum(γ[:,2])
    else
        ∑γdecoupled = γ[zeroindex,1]
    end
    f𝐗𝐰 = softplus.(𝐔𝐮)
    neg𝒬 = ∑γdecoupled ⋅ (f𝐗𝐰.*Δt .- 𝐲.*log.(f𝐗𝐰))
    for i=1:Ξ
        if i != zeroindex
            if i < zeroindex
                f𝐗𝐰 = softplus.(𝐗*vcat(𝐮, 𝛏[i].*𝐥))
            else
                f𝐗𝐰 = softplus.(𝐗*vcat(𝐮, 𝛏[i].*𝐫))
            end
            neg𝒬 += γ[i,1] ⋅ (f𝐗𝐰.*Δt .- 𝐲.*log.(f𝐗𝐰))
        end
    end
    return neg𝒬
end

"""
    ∇negativeexpectation!(∇, γ, mpGLM, x)

Gradient of the negative of the expectation of the log-likelihood of the mixture of Poisson generalized model of one neuron

MODIFIED ARGUMENT
-`∇`: The gradient

UNMODIFIED ARGUMENT
-`γ`: Joint posterior probability of the accumulator and coupling variable. γ[i,k][t] corresponds to the i-th accumulator state and the k-th coupling state in the t-th time bin in the trialset.
-`mpGLM`: structure containing information for the mixture of Poisson GLM for one neuron
-`x`: vector of parameters for the mixture of Poisson GLM

RETURN
-nothing
"""
function ∇negativeexpectation!(∇::Vector{<:AbstractFloat},
                               γ::Matrix{<:Vector{<:AbstractFloat}},
                               mpGLM::MixturePoissonGLM,
                               x::Vector{<:AbstractFloat})
    Pᵤ = size(mpGLM.𝐔,2)
    Pₗ = size(mpGLM.𝚽,2)
    mpGLM.𝐮 .= x[1:Pᵤ]
    mpGLM.𝐥 .= x[Pᵤ+1:Pᵤ+Pₗ]
    mpGLM.𝐫 .= x[Pᵤ+Pₗ+1:end]
    ∇ .= ∇negativeexpectation(γ, mpGLM)
end

"""
    ∇negativeexpectation(γ, mpGLM)

Gradient of the negative of the expectation of the log-likelihood of the mixture of Poisson generalized model of one neuron

ARGUMENT
-`γ`: Joint posterior probability of the accumulator and coupling variable. γ[i,k][t] corresponds to the i-th accumulator state and the k-th coupling state in the t-th time bin in the trialset.
-`mpGLM`: structure containing information for the mixture of Poisson GLM for one neuron

RETURN
-∇: the gradient
"""
function ∇negativeexpectation(γ::Matrix{<:Vector{<:AbstractFloat}},
                              mpGLM::MixturePoissonGLM)
    @unpack Δt, 𝐔, 𝚽, 𝐗, 𝛏, 𝐲, 𝐮, 𝐥, 𝐫 = mpGLM
    Ξ = size(γ,1)
    zeroindex = cld(Ξ,2)
    if size(γ,2) > 1 # i.e, the coupling variable has more than one state
        ∑γdecoupled = γ[zeroindex,1] .+ sum(γ[:,2])
    else
        ∑γdecoupled = γ[zeroindex,1]
    end
    𝐔𝐮 = 𝐔*𝐮
    tmp𝐮 = ∑γdecoupled .* logistic.(𝐔𝐮) .* (Δt .- 𝐲 ./ softplus.(𝐔𝐮))
    tmp𝐥 = zeros(size(𝐲))
    tmp𝐫 = zeros(size(𝐲))
    for i=1:zeroindex-1
        𝐗𝐰 = 𝐗*vcat(𝐮, 𝛏[i].*𝐥)
        tmp = γ[i,1] .* logistic.(𝐗𝐰) .* (Δt .- 𝐲 ./ softplus.(𝐗𝐰))
        tmp𝐥 .+= 𝛏[i].*tmp
        tmp𝐮 .+= tmp
    end
    for i=zeroindex+1:Ξ
        𝐗𝐰 = 𝐗*vcat(𝐮, 𝛏[i].*𝐫)
        tmp = γ[i,1] .* logistic.(𝐗𝐰) .* (Δt .- 𝐲 ./ softplus.(𝐗𝐰))
        tmp𝐫 .+= 𝛏[i].*tmp
        tmp𝐮 .+= tmp
    end
    Pᵤ = length(𝐮)
    Pₗ = length(𝐥)
    𝚽ᵀ = transpose(𝚽)
    ∇ = zeros(Pᵤ+2Pₗ)
    ∇[1:Pᵤ] = transpose(𝐔)*tmp𝐮
    ∇[Pᵤ+1:Pᵤ+Pₗ] = 𝚽ᵀ*tmp𝐥
    ∇[Pᵤ+Pₗ+1:end] = 𝚽ᵀ*tmp𝐫
    return ∇
end

"""
    𝐇negativeexpection(𝐇, γ, mpGLM, x)

Compute the Hessian of the negative of the terms in the expectation that depend on the GLM filters

MODIFIED ARGUMENT
-`𝐇`: Hessian matrix

UNMODIFIED ARGUMENT
-`γ`: posterior probabilities of the latents
-`mpGLM`: the Poisson mixture GLM of one neuron
-`x`: filters of the Poisson mixture GLM

RETURN
-nothing
"""
function 𝐇negativeexpectation!(𝐇::Matrix{<:AbstractFloat},
                               γ::Matrix{<:Vector{<:AbstractFloat}},
                               mpGLM::MixturePoissonGLM,
                               x::Vector{<:AbstractFloat})
    @unpack Δt, 𝐔, 𝚽, 𝛏, 𝐗, 𝐲 = mpGLM
    Pᵤ = size(𝐔,2)
    Pₗ = size(𝚽,2)
    indices𝐮 = 1:Pᵤ
    indices𝐥 = Pᵤ+1:Pᵤ+Pₗ
    indices𝐫 = Pᵤ+Pₗ+1:Pᵤ+2Pₗ
    𝐮 = x[indices𝐮]
    𝐥 = x[indices𝐥]
    𝐫 = x[indices𝐫]
    𝐔𝐮 = 𝐔*𝐮
    𝐔ᵀ = transpose(𝐔)
    𝚽ᵀ = transpose(𝚽)
    T = length(𝐲)
    Ξ = size(γ,1)
    zeroindex = cld(Ξ,2)
    if size(γ,2) > 1 # i.e, the coupling variable has more than one state
        ∑γdecoupled = γ[zeroindex,1] .+ sum(γ[:,2])
    else
        ∑γdecoupled = γ[zeroindex,1]
    end
    f₀ = softplus.(𝐔𝐮)
    f₁ = logistic.(𝐔𝐮) # first derivative
    f₂ = f₁ .* (1 .- f₁) # second derivative
    tmp𝐮𝐮 = ∑γdecoupled .* (f₂.*Δt .- 𝐲.*(f₀.*f₂ .- f₁.^2)./f₀.^2)
    tmp𝐥𝐮 = zeros(T)
    tmp𝐫𝐮 = zeros(T)
    tmp𝐥𝐥 = zeros(T)
    tmp𝐫𝐫 = zeros(T)
    for i = 1:Ξ
        if i < zeroindex
            𝐗𝐰 = 𝐗*vcat(𝐮, 𝛏[i].*𝐥)
        elseif i > zeroindex
            𝐗𝐰 = 𝐗*vcat(𝐮, 𝛏[i].*𝐫)
        else
            continue
        end
        f₀ = softplus.(𝐗𝐰)
        f₁ = logistic.(𝐗𝐰) # first derivative
        f₂ = f₁ .* (1 .- f₁) # second derivative
        tmp = γ[i,1].*(f₂.*Δt .- 𝐲.*(f₀.*f₂ .- f₁.^2)./f₀.^2)
        if i < zeroindex
            tmp𝐥𝐥 .+= 𝛏[i]^2 .* tmp
            tmp𝐥𝐮 .+= 𝛏[i].*tmp
        elseif i > zeroindex
            tmp𝐫𝐫 .+= 𝛏[i]^2 .* tmp
            tmp𝐫𝐮 .+= 𝛏[i].*tmp
        end
        tmp𝐮𝐮 .+= tmp
    end
    𝐔ᵀ_tmp𝐥𝐮_𝚽 = 𝐔ᵀ*(tmp𝐥𝐮.*𝚽)
    𝐔ᵀ_tmp𝐫𝐮_𝚽 = 𝐔ᵀ*(tmp𝐫𝐮.*𝚽)
    𝐇 .= 0
    𝐇[indices𝐮, indices𝐮] = 𝐔ᵀ*(tmp𝐮𝐮.*𝐔)
    𝐇[indices𝐥, indices𝐥] = 𝚽ᵀ*(tmp𝐥𝐥.*𝚽)
    𝐇[indices𝐫, indices𝐫] = 𝚽ᵀ*(tmp𝐫𝐫.*𝚽)
    𝐇[indices𝐮, indices𝐥] = 𝐔ᵀ_tmp𝐥𝐮_𝚽
    𝐇[indices𝐮, indices𝐫] = 𝐔ᵀ_tmp𝐫𝐮_𝚽
    𝐇[indices𝐥, indices𝐮] = transpose(𝐔ᵀ_tmp𝐥𝐮_𝚽)
    𝐇[indices𝐫, indices𝐮] = transpose(𝐔ᵀ_tmp𝐫𝐮_𝚽)
    return nothing
end
