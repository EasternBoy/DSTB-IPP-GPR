mutable struct polyBound
    s_max::Float64 # constant of bound constraint
    x_min::Float64 
    x_max::Float64 # constant of bound constraint
    y_min::Float64 
    y_max::Float64 # constant of bound constraint

    function polyBound(s_max::Float64, x_min::Float64, x_max::Float64, y_min::Float64, y_max::Float64)
        return new(s_max, x_min, x_max, y_min, y_max)
    end
end

mutable struct robot
    index::Int64
    T::Float64 # Sampling time
    H::Integer # Horizon length
    R::Float64
    r::Float64
    σ::Float64  # Gaussian Noise on the measurement
    pBnd::polyBound # Interval Bounds


    data::Matrix{Vector{Float64}}
    posn::Vector{Float64}  # Current states: x, y
    meas::Float64 # measurement

    loca::Matrix{Float64}  # All locations up to current time
    obsr::Vector{Float64}  # All observation up to current time

    # odeprob # ODEProblem for solving the ODE
    opti
    mGP::GPBase    

    β::Float64
    σκ2::Float64
    ϕℓ2::Vector{Float64}
    σω2::Float64
    iΦ2::Matrix{Float64}
    iCθ::Matrix{Float64}



    function robot(index::Int64, T::Float64, H::Integer, R::Float64, r::Float64, σ::Float64, 
                                                            pBnd::polyBound, x0::AbstractVector)

        obj         = new(index,T, H, R, r, σ, pBnd)
        obj.posn    = x0  # Copy x0
        
        obj.data    = [-ones(3) for i in 1:M, j in 1:L+1] # -1 means no data
        obj.loca    = Matrix{Float64}(undef, length(x0), 0)
        obj.obsr    = Vector{Float64}(undef, 0)

        obj.β       = 1.
        obj.ϕℓ2     = 1e2*ones(2)
        obj.σκ2     = 1.
        obj.σω2     = 1e-2

        obj.opti = JuMP.Model(Ipopt.Optimizer)
        set_silent(obj.opti)
        return obj
    end
end