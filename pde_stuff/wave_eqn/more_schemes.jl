using LinearAlgebra
using Printf

# Exact solution: u(x,t) = sin(x - t) for u_t = u_x with a=1
function exact_solution(x, t)
    return sin.(x .+ t)
end

function scheme_1(uinit, dt, Tfinal)
    #=
    u^{n+1}_j = u^n_j + dt/2*dx * (u^n_(j+1) - u^n_{j-1})
    =#
    t = 0.0
    u = copy(uinit)
    n = length(u)
    dx = (2*pi/n)
    unew = similar(u)
    
    while t < Tfinal
        dt_current = min(dt, Tfinal - t)
        mu_current = dt_current / 2 * dx
        
        for j in 1:n
            jm1 = mod(j-2, n) + 1
            jp1 = mod(j, n) + 1
            unew[j] = u[j] + mu_current * (u[jp1] - u[jm1])
        end
        u .= unew
        t += dt_current
    end
    return u
end

function scheme_2(uinit, dt, Tfinal)
    #=
    Forward in time, forward in space (FTFS)
    u^{n+1}_j = u^n_j  + dt/dx * (u^n_{j+1} - u^n_j)
    =#
    t = 0.0
    u = copy(uinit)
    n = length(u)
    dx = (2*pi/n)
    unew = similar(u)
    
    while t < Tfinal
        # Adjust dt to hit Tfinal exactly
        dt_current = min(dt, Tfinal - t)
        mu_current = dt_current / dx
        
        for j in 1:n
            jp1 = mod(j, n) + 1
            unew[j] = u[j] + mu_current * (u[jp1] - u[j])
        end
        u .= unew
        t += dt_current
    end
    return u
end

function scheme_3(uinit, dt, Tfinal)
    #=
    Leap Frog scheme
    u^{n+1}_j = u^(n-1)_j + 2dt/2dx * (u^n_(j+1) - u^n_{j-1})
    =#
    t = 0.0
    u = copy(uinit)
    uold = copy(uinit)
    n = length(u)
    dx = (2*pi/n)
    unew = similar(u)

    for j in 1:n
        jm1 = mod(j-2, n) + 1
        jp1 = mod(j, n) + 1
        u[j] = uold[j] + (dt/(2*dx)) * (uold[jp1] - uold[jm1])
    end
    t += dt
    
    while t < Tfinal
        dt_current = min(dt, Tfinal - t)
        mu_current = dt_current / dx
        
        for j in 1:n
            jm1 = mod(j-2, n) + 1
            jp1 = mod(j, n) + 1
            unew[j] = uold[j] + mu_current * (u[jp1] - u[jm1])
        end
        uold .= u
        u .= unew
        t += dt_current
    end
    return u
end

function scheme_4(uinit, dt, Tfinal)
    #=
    Lax-Wendroff scheme (a = 1)
    u^{n+1}_j = u^n_j + (dt)/(2*dx) * (u^n_{j+1} - u^n_{j-1}) 
                + (dt^2)/(2*dx^2) * (u^n_{j+1} - 2*u^n_j + u^n_{j-1})
    =#
    t = 0.0
    u = copy(uinit)
    n = length(u)
    dx = (2*pi/n)
    unew = similar(u)
    
    while t < Tfinal
        dt_current = min(dt, Tfinal - t)
        mu_current = dt_current / dx
        
        for j in 1:n
            jp1 = mod(j, n) + 1
            jm1 = mod(j-2, n) + 1
            unew[j] = u[j] + (mu_current/2) * (u[jp1] - u[jm1]) + 
                      (mu_current^2/2) * (u[jp1] - 2*u[j] + u[jm1])
        end
        u .= unew
        t += dt_current
    end
    return u
end

function scheme_5(uinit, dt, Tfinal)
    #=
    Implicit Scheme
    u^{n+1}_j = u^n_j + (dt/dx)*(u^{n+1}_{j+1} - u^{n+1}_j)
    can be written as:
    u^{n+1}_j(1 + (dt/dx)) - (dt/dx)*(u^{n+1}_{j+1}) = u^n_j
    =#
    t = 0.0
    u = copy(uinit)
    n = length(u)
    dx = 2*pi/n
    
    while t < Tfinal
        dt_current = min(dt, Tfinal - t)
        mu_current = dt_current/dx
        
        # Build tridiagonal system
        A = zeros(n, n)
        for j in 1:n
            A[j, j] = 1 + mu_current
            jp1 = mod(j, n) + 1
            A[j, jp1] = -mu_current
        end
        
        u = A \ u
        t += dt_current
    end
    return u
end

function scheme_6(uinit, dt, Tfinal)
    #=
    Crank-Nicolson Scheme
    (u^{n+1}_j - u^n_j)/dt = (1/2) * [(u^{n+1}_{j+1} - u^{n+1}_{j-1})/(2dx) 
                                     + (u^n_{j+1} - u^n_{j-1})/(2dx)]
    =#
    t = 0.0
    u = copy(uinit)
    n = length(u)
    dx = 2*pi/n
    
    while t < Tfinal
        dt_current = min(dt, Tfinal - t)
        mu_current = dt_current/(4*dx)

        A = zeros(n, n)
        for j in 1:n
            jm1 = mod(j-2, n) + 1
            jp1 = mod(j, n) + 1
            A[j, jm1] = -1
            A[j, jp1] = 1
        end
        
        LHS = I - mu_current*A
        RHS = (I + mu_current*A) * u
        
        u = LHS \ RHS
        t += dt_current
    end
    return u
end

# Compute L2 error
function compute_error(numerical, exact)
    return sqrt(sum((numerical .- exact).^2) / length(numerical))
end

# Main computation
n = 100
x = range(0, 2*pi, length=n+1)[1:end-1]
dx = 2*pi/n
dt = 0.01
Tfinal = 1.0
mu = dt/dx

uinit = sin.(x)
uexact = exact_solution(x, Tfinal)

schemes = [
    ("Scheme 1 (Central/Forward)", scheme_1),
    ("Scheme 2 (FTFS)", scheme_2),
    ("Scheme 3 (Leap Frog)", scheme_3),
    ("Scheme 4 (Lax-Wendroff)", scheme_4),
    ("Scheme 5 (Implicit)", scheme_5),
    ("Scheme 6 (Crank-Nicolson)", scheme_6)
]

for (name, scheme) in schemes
    try
        u_numerical = scheme(uinit, dt, Tfinal)
        error = compute_error(u_numerical, uexact)
        println("$name: L2 error = $(error)")
    catch e
        println("$name failed: $e")
    end
end

#=
Solution:
Scheme 1 (Central/Forward): L2 error = 0.6755606073536279
Scheme 2 (FTFS): L2 error = 0.01843502808223345
Scheme 3 (Leap Frog): L2 error = 0.0004380609069721324
Scheme 4 (Lax-Wendroff): L2 error = 0.0004533810601902246
Scheme 5 (Implicit): L2 error = 0.025279165583409355
Scheme 6 (Crank-Nicolson): L2 error = 0.0004710466219773588
=#