function initializedecoupling(elapsedΔclicks::Vector{<:Real}, mpGLM::MixturePoissonGLM; kfold::Integer=5)
    @unpack Δt, 𝐲 = mpGLM
    𝐗 = [mpGLM.𝐗 elapsedΔclicks]
    testindices, trainindices = cvpartition(size(𝐗,1), kfold)
    ℓ = 0
    for k = 1:kfold
        𝐰 = maximizeloglikelihood(Δt, 𝐗[trainindices[k],:], 𝐲[trainindices[k]])
        𝐋 = (@view 𝐗[testindices[k],:])*𝐰
        for i in eachindex(testindices[k])
            timestep = testindices[k][i]
            ℓ += poissonloglikelihood(Δt, 𝐋[timestep], 𝐲[timestep])
        end
    end
end
