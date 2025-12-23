using LinearAlgebra
using Printf

# Exact solution: u(x,t) = sin(x - t) for u_t = u_x with a=1
function exact_solution(x, t)
    return sin.(x .- t)
end

function scheme_1(mu, uinit, dt, Tfinal)
    #=
    Forward in time, forward in space (FTFS)
    u^{n+1}_j = u^n_j - a*dt/dx * (u^n_{j+1} - u^n_j)
    For a=1: u^{n+1}_j = u^n_j - mu * (u^n_{j+1} - u^n_j)
    =#
    t = 0.0
    u = copy(uinit)
    n = length(u)
    unew = similar(u)
    
    while t < Tfinal
        # Adjust dt to hit Tfinal exactly
        dt_current = min(dt, Tfinal - t)
        mu_current = dt_current / (2*pi/n)
        
        for j in 1:n
            jp1 = mod(j, n) + 1
            unew[j] = u[j] - mu_current * (u[jp1] - u[j])
        end
        u .= unew
        t += dt_current
    end
    return u
end

function scheme_2(mu, uinit, dt, Tfinal)
    #=
    Forward in time, backward in space (FTBS)
    u^{n+1}_j = u^n_j - a*dt/dx * (u^n_j - u^n_{j-1})
    For a=1: u^{n+1}_j = u^n_j - mu * (u^n_j - u^n_{j-1})
    =#
    t = 0.0
    u = copy(uinit)
    n = length(u)
    unew = similar(u)
    
    while t < Tfinal
        dt_current = min(dt, Tfinal - t)
        mu_current = dt_current / (2*pi/n)
        
        for j in 1:n
            jm1 = mod(j-2, n) + 1
            unew[j] = u[j] - mu_current * (u[j] - u[jm1])
        end
        u .= unew
        t += dt_current
    end
    return u
end

function scheme_3(mu, uinit, dt, Tfinal)
    #=
    Lax-Wendroff scheme (a = 1)
    u^{n+1}_j = u^n_j - (a*dt)/(2*dx) * (u^n_{j+1} - u^n_{j-1}) 
                + (a^2*dt^2)/(2*dx^2) * (u^n_{j+1} - 2*u^n_j + u^n_{j-1})
    =#
    t = 0.0
    u = copy(uinit)
    n = length(u)
    unew = similar(u)
    
    while t < Tfinal
        dt_current = min(dt, Tfinal - t)
        mu_current = dt_current / (2*pi/n)
        
        for j in 1:n
            jp1 = mod(j, n) + 1
            jm1 = mod(j-2, n) + 1
            unew[j] = u[j] - (mu_current/2) * (u[jp1] - u[jm1]) + 
                      (mu_current^2/2) * (u[jp1] - 2*u[j] + u[jm1])
        end
        u .= unew
        t += dt_current
    end
    return u
end

# Compute L2 error
function compute_error(numerical, exact)
    return sqrt(sum((numerical .- exact).^2) / length(numerical))
end

# Main computation
mu = 0.5
grid_sizes = [20, 40, 80, 160, 320]
Tfinal = 1.0

schemes = [
    (scheme_1, "Scheme 1 (FTFS)"),
    (scheme_2, "Scheme 2 (FTBS)"),
    (scheme_3, "Scheme 3 (Lax-Wendroff)")
]

for (scheme_func, scheme_name) in schemes
    errors = []
    println(scheme_name)
    println(@sprintf("%10s %12s %12s %12s", "N", "dx", "L2 Error", "Order"))
    
    for N in grid_sizes
        # Setup grid
        x = LinRange(0, 2π, N+1)[1:end-1]  # Periodic: exclude last point
        dx = 2π / N
        dt = mu * dx
        
        # Initial condition
        uinit = sin.(x)
        
        # Run scheme
        u_numerical = scheme_func(mu, uinit, dt, Tfinal)
        
        # Exact solution at t = Tfinal
        u_exact = exact_solution(x, Tfinal)
        
        # Compute error
        error = compute_error(u_numerical, u_exact)
        push!(errors, error)
        
        if length(errors) > 1
            order = log2(errors[end-1] / errors[end])
            println(@sprintf("%10d %12.6f %12.6e %12.2f", N, dx, error, order))
        else
            println(@sprintf("%10d %12.6f %12.6e %12s", N, dx, error, "-"))
        end
    end
end