function [e, x] = simulate_system(sys,x0, d)
    nx = size(x0,1); ne = sys.ne; 
    N = length(d);
    e = zeros(ne, N);
    x = zeros(nx, N+1);
    x(:,1) = x0; % initial condition
    for k = 1:N
        d_k = d(:,k); x_k = x(:,k);
        z_k = sys.C2d*x_k + sys.D21d*d_k; % needs to be changed if D22 is not zero
        w_k = sys.nl(z_k);
        e(:,k) = sys.Cd*x_k + sys.Dd*d_k + sys.D12d*w_k;
        x(:,k+1) = sys.Ad*x_k + sys.Bd*d_k + sys.B2d*w_k;
    end
end