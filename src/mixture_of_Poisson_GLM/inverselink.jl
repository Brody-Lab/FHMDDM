inverselink(x::Real) = softplus(x)
differentiate_inverselink(x::Real) = logistic(x)
differentiate_twice_inverselink(x::Real) = logistic(x)*(1-logistic(x))