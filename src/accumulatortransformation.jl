"""
    transformaccumulator

Nonlinearly transform the normalized values of the accumulator

ARGUMENT
-`ξ`: value of the accumulator: expected to be between -1 and 1
-`b`: parameter specifying the transformation

RETURN
-transformed value of the accumulator
"""
function transformaccumulator(b::Real, ξ::Real)
    if b == 0.0
        ξ
    else
        if ξ < 0
            if b > 709.0 # 709 is close to which exp returns Inf for a 64-bit floating point number
                ξ == -1.0 ? -1.0 : 0.0
            else
                -expm1(-b*ξ)/expm1(b)
            end
        elseif ξ > 0
            if b > 709.0
                ξ == 1.0 ? 1.0 : 0.0
            else
                expm1(b*ξ)/expm1(b)
            end
        else
            0.0
        end
    end
end

"""
    dtransformaccumulator

Derivative of the nonlinear transformation of the normalized values of the accumulator with respect to b

ARGUMENT
-`ξ`: value of the accumulator: expected to be between -1 and 1
-`b`: parameter specifying the transformation

RETURN
-derivative of the transformed value of the accumulator
"""
function dtransformaccumulator(b::Real, ξ::Real)
    if ξ == -1.0 || ξ == 0.0 || ξ == 1.0 || b > 709.0 # 709 is close to which exp returns Inf for a 64-bit floating point number
        0.0
    elseif abs(b) < 1e-6
        ξ < 0 ? (-ξ^2-ξ)/2 : (ξ^2-ξ)/2
    elseif ξ < 0
        eᵇ = exp(b)
        eᵇm1 = expm1(b)
        e⁻ᵇˣ = exp(-b*ξ)
        e⁻ᵇˣm1 = expm1(-b*ξ)
        if b < 1
            (ξ*e⁻ᵇˣ*eᵇm1 + e⁻ᵇˣm1*eᵇ)/eᵇm1^2
        else
            ξ*e⁻ᵇˣ/eᵇm1 + e⁻ᵇˣm1/(eᵇ-2+exp(-b))
        end
    elseif ξ > 0
        eᵇ = exp(b)
        eᵇm1 = expm1(b)
        eᵇˣ = exp(b*ξ)
        eᵇˣm1 = expm1(b*ξ)
        if b < 1
            ξ*eᵇˣ/eᵇm1 - eᵇˣm1*eᵇ/eᵇm1^2
        else
            ξ*eᵇˣ/eᵇm1 - eᵇˣm1/(eᵇ-2+exp(-b))
        end
    end
end

"""
    test_dtransformaccumulator()

OPTIONAL ARGUMENT
-`Ξ`: number of values into which accumulated evidence is discretized

RETURN
-maximum absolute difference between automatically and analytically computed derivatives
-absolute differences across nonlinearity parameters and accumulator values
-`𝐛`: vector of nonlinearity parameters
-`𝛏`: normalized accumulator value

TEST
```julia-repl
julia> using FHMDDM
julia> maxD, D, 𝐛, 𝛏 = FHMDDM.test_dtransformaccumulator()
julia>
```
"""
function test_dtransformaccumulator(; Ξ::Int=53)
    𝛏 = (2collect(1:Ξ) .- Ξ .- 1)./(Ξ-2)
    𝐛 = 10.0 .^collect(-8:4)
    𝐛 = vcat(-reverse(𝐛), 0.0, 𝐛)
    D = zeros(length(𝐛), Ξ)
    for j in eachindex(𝛏)
        f(b) = transformaccumulator(b, 𝛏[j])
        for i in eachindex(𝐛)
            autoderivative = ForwardDiff.derivative(f, 𝐛[i])
            analderivative = dtransformaccumulator(𝐛[i], 𝛏[j])
            D[i,j] = abs(autoderivative-analderivative)
        end
    end
    return maximum(D), D, 𝐛, 𝛏
end

"""
    d²transformaccumulator

Second derivative of the nonlinear transformation of the normalized values of the accumulator with respect to b

ARGUMENT
-`ξ`: value of the accumulator: expected to be between -1 and 1
-`b`: parameter specifying the transformation

RETURN
-second derivative of the transformed value of the accumulator
"""
function d²transformaccumulator(b::type, ξ::type) where {type<:Real}
    if abs(b) > 230.0
        zero(type)
    elseif abs(b) > 1e-4
        if ξ < 0.0
            eᵇ = exp(b)
            eᵇm1 = expm1(b)
            e⁻ᵇˣ = exp(-b*ξ)
            e⁻ᵇˣm1 = expm1(-b*ξ)
            (-ξ*e⁻ᵇˣ*eᵇm1*(ξ*eᵇm1+2eᵇ) - eᵇ*(eᵇ+1)*e⁻ᵇˣm1)/eᵇm1^3
        else
            (exp(b*(ξ+2))*(ξ-1)^2 + exp(b*(ξ+1))*(-2ξ^2 + 2ξ + 1) + exp(b*ξ)*ξ^2 - exp(2b) - exp(b))/expm1(b)^3
        end
    else
        d²transformaccumulator_limit_b_0(ξ)
    end
end

"""
    d²transformaccumulator_limit_b_0

Limit of the second derivative as 'b' approaches zero

ARGUMENT
-`ξ`: value of the accumulator: expected to be between -1 and 1

RETURN
-second derivative of the transformed value of the accumulator
"""
function d²transformaccumulator_limit_b_0(ξ::Real)
    if ξ < 0
        (2ξ^3 + 3ξ^2 + ξ)/6
    else
        (2ξ^3 - 3ξ^2 + ξ)/6
    end
end

"""
    test_d²transformaccumulator()

OPTIONAL ARGUMENT
-`Ξ`: number of values into which accumulated evidence is discretized

RETURN
-maximum absolute difference between automatically and analytically computed derivatives
-absolute differences across nonlinearity parameters and accumulator values
-`𝐛`: vector of nonlinearity parameters
-`𝛏`: normalized accumulator value

TEST
```julia-repl
julia> using FHMDDM
julia> maxD, D, 𝐛, 𝛏 = FHMDDM.test_d²transformaccumulator()
julia>
```
"""
function test_d²transformaccumulator(; Ξ::Int=7)
    𝛏 = (2collect(1:Ξ) .- Ξ .- 1)./(Ξ-2)
    𝐛 = 10.0 .^collect(-8:1:4)
    𝐛 = vcat(-reverse(𝐛), 0.0, 𝐛)
    D = zeros(length(𝐛), Ξ)
    for j in eachindex(𝛏)
        f(b) = transformaccumulator(b, 𝛏[j])
        g(b) = ForwardDiff.derivative(f, b)
        for i in eachindex(𝐛)
            autoderivative = ForwardDiff.derivative(g, 𝐛[i])
            analderivative = d²transformaccumulator(𝐛[i], 𝛏[j])
            D[i,j] = abs(autoderivative-analderivative)
        end
    end
    return maximum(D), D, 𝐛, 𝛏
end
