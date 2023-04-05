"""
    functions

-GLMDerivatives(mpGLM)
-differentiate_twice_loglikelihood!(D,L,y)
-differentiate_loglikelihood!(D,L,y)
-differentiate_twice_overdispersion!(D,a)
-differentiate_overdispersion!(D,a)
"""

"""
    GLMDerivatives(mpGLM)

Create an object of the composite type `GLMDerivatives`
"""
GLMDerivatives(mpGLM::MixturePoissonGLM) = GLMDerivatives(Δt=mpGLM.Δt, fit_overdispersion=mpGLM.θ.fit_overdispersion)

"""
    differentiate_twice_loglikelihood!(D,L,y)

Compute in-place second-order partial derivatives by modifying fields of an object of the composite type `GLMDerivatives`

MODIFIED ARGUMENT
-`D`: An object containing quantities for computing the derivatives

UNMODIFIED ARGUMENT
-`L`: linear predictor
-`y`: spike count response

EXAMPLE
```julia-repl
julia> using FHMDDM, ForwardDiff, Random
julia> a, L, y = rand(), rand(), rand(0:2)
julia> D = FHMDDM.GLMDerivatives(Δt=0.01, fit_overdispersion=true)
julia> FHMDDM.differentiate_twice_overdispersion!(D, a)
julia> FHMDDM.differentiate_twice_loglikelihood!(D,L,y)
julia> x0 = [a, L]
julia> f(x) = FHMDDM.negbinloglikelihood(FHMDDM.inverselink(x[1]), D.Δt, FHMDDM.inverselink(x[2]), y)
julia> g = ForwardDiff.gradient(f, x0)
julia> H = ForwardDiff.hessian(f, x0)
julia> maxabsdiff = max(abs(D.dℓ_da[1]-g[1]), abs(D.dℓ_dL[1]-g[2]), abs(D.d²ℓ_da²[1]-H[1,1]), abs(D.d²ℓ_dadL[1]-H[2,1]), abs(D.d²ℓ_dL²[1]-H[2,2]))
julia> println("The maximum absolute difference between the automatically and hand-computed derivatives is ϵ=", maxabsdiff)
```
"""
function differentiate_twice_loglikelihood!(D::GLMDerivatives, L::AbstractFloat, y::Integer)
    if D.α[1] > 0.0
        μ = inverselink(L)
        D.ℓ[1] = negbinloglikelihood(D.α[1], D.Δt, μ, y)
        dμ_dL = differentiate_inverselink(L)
        d²μ_dL² = differentiate_twice_inverselink(L)
        differentiate_loglikelihood_wrt_overdispersion_mean!(D.g, D.H, D.α[1], D.Δt, μ, y)
        dℓ_dα = D.g[1]
        d²ℓ_dα² = D.H[1,1]
        D.dℓ_da[1] = dℓ_dα*D.dα_da[1]
        D.d²ℓ_da²[1] = d²ℓ_dα²*D.dα_da[1]^2 + dℓ_dα*D.d²α_da²[1]
        dℓ_dμ = D.g[2]
        D.dℓ_dL[1] = dℓ_dμ*dμ_dL
        d²ℓ_dμ² = D.H[2,2]
        D.d²ℓ_dL²[1] = d²ℓ_dμ²*dμ_dL^2 + dℓ_dμ*d²μ_dL²
        d²ℓ_dαdμ = D.H[2,1]
        D.d²ℓ_dadL[1] = d²ℓ_dαdμ*D.dα_da[1]*dμ_dL
    else
        D.d²ℓ_dL²[1], D.dℓ_dL[1], D.ℓ[1] = differentiate_twice_loglikelihood_wrt_linearpredictor(D.Δt, L, y)
    end
    return nothing
end

"""
    differentiate_loglikelihood!(D, L, y)

Compute in-place first-order partial derivatives by modifying fields of an object of the composite type `GLMDerivatives`

See `differentiate_twice_loglikelihood!(D,L,y)`
"""
function differentiate_loglikelihood!(D::GLMDerivatives, L::AbstractFloat, y::Integer)
    if D.α[1] > 0.0
        μ = inverselink(L)
        D.ℓ[1] = negbinloglikelihood(D.α[1], D.Δt, μ, y)
        dμ_dL = differentiate_inverselink(L)
        differentiate_loglikelihood_wrt_overdispersion_mean!(D.g, D.α[1], D.Δt, μ, y)
        dℓ_dα = D.g[1]
        dℓ_dμ = D.g[2]
        D.dℓ_da[1] = dℓ_dα*D.dα_da[1]
        D.dℓ_dL[1] = dℓ_dμ*dμ_dL
    else
        λ = inverselink(L)
        D.dℓ_dL[1] = differentiate_loglikelihood_wrt_linearpredictor(D.Δt, L, λ, y)
        D.ℓ[1] = poissonloglikelihood(λ*D.Δt, y)
    end
end

"""
    differentiate_twice_overdispersion!(D,a)

Compute second-order partial derivative of the non-negative-valued parameter of overdispersion with respect to the real-valued parameter

Both the real-valued and non-negative-valued parameters of overdispersion are updated

MODIFIED ARGUMENT
-`D`: object of composite type `GLMDerivatives`

UNMODIFIED ARGUMENT
-`a`: real-valued parameter of overdispersion
"""
function differentiate_twice_overdispersion!(D::GLMDerivatives, a::AbstractFloat)
    D.α[1] = inverselink(a)
    D.dα_da[1] = differentiate_inverselink(a)
    D.d²α_da²[1] = differentiate_twice_inverselink(a)
    return nothing
end

"""
    differentiate_overdispersion!(D,a)

Compute first-order partial derivative of the non-negative-valued parameter of overdispersion with respect to the real-valued parameter

Both the real-valued and non-negative-valued parameters of overdispersion are updated
"""
function differentiate_overdispersion!(D::GLMDerivatives, a::AbstractFloat)
    D.α[1] = inverselink(a)
    D.dα_da[1] = differentiate_inverselink(a)
    return nothing
end