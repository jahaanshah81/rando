%% =========================================================================
%% build_transition_library.m
%%
%% OFFLINE script — run ONCE to precompute all feasible pairwise transitions
%% between trim primitives and store them in transition_library.mat.
%%
%% Output
%%   transition_library  : struct array (N x N) where
%%       .feasible       : true if OCP converged, false otherwise
%%       .X              : (N_nodes x 13) optimal state trajectory
%%       .U              : (N_nodes x 4)  optimal control trajectory
%%       .T              : (N_nodes x 1)  time vector (starts at 0)
%%       .Tf             : scalar total transition time used
%%       .cost           : IPOPT objective value
%%       .solver_status  : IPOPT return string
%%
%% Feasibility pre-filter (avoids wasting OCP solves on impossible pairs):
%%   dV   <= V_THRESH   (airspeed mismatch)
%%   dAlpha <= A_THRESH  (angle-of-attack mismatch)
%%   dBeta  <= B_THRESH  (sideslip mismatch)
%%   dPhi   <= P_THRESH  (bank angle mismatch)
%%   same primitive excluded (i == j)
%%
%% Tf is chosen adaptively per pair — larger state mismatches get more time.
%% =========================================================================

clc; clear;
import casadi.*

%% ---- Configuration -------------------------------------------------------
V_THRESH  = 80;            % max airspeed difference  [m/s]
A_THRESH  = deg2rad(15);   % max alpha difference     [rad]
B_THRESH  = deg2rad(10);   % max sideslip difference  [rad]
P_THRESH  = deg2rad(60);   % max bank-angle difference [rad]

Tf_BASE   = 5;            % minimum transition time  [s]
Tf_MAX    = 25;            % maximum transition time  [s]

SAVE_FILE = 'transition_library.mat';

%% ---- Load Prerequisites --------------------------------------------------
fprintf('Loading maneuver_library.mat ...\n');
load('maneuver_library.mat', 'maneuver_library');
aircraft = get_f18_harv_parameters();
N = length(maneuver_library);
fprintf('  Found %d primitives in library.\n\n', N);

%% ---- Pre-extract Physical Parameters from Every Trim State ---------------
% This avoids recomputing per pair inside the nested loop.
fprintf('Extracting aerodynamic parameters from trim states...\n');
params = extract_trim_params(maneuver_library, N);

%% ---- Feasibility Check: Build Candidate Pair List ------------------------
fprintf('Running feasibility pre-filter...\n');
candidates = [];   % Mx2 list of [i, j] index pairs

for i = 1:N
    for j = 1:N
        if i == j, continue; end

        dV     = abs(params.V(i)   - params.V(j));
        dAlpha = abs(params.alpha(i) - params.alpha(j));
        dBeta  = abs(params.beta(i)  - params.beta(j));
        dPhi   = abs(angdiff(params.phi(i), params.phi(j)));  % wrap-safe

        if dV     <= V_THRESH  && ...
           dAlpha <= A_THRESH  && ...
           dBeta  <= B_THRESH  && ...
           dPhi   <= P_THRESH
            candidates(end+1, :) = [i, j];
        end
    end
end

n_cand = size(candidates, 1);
fprintf('  %d / %d pairs pass feasibility filter (%.1f%%).\n\n', ...
    n_cand, N*(N-1), 100*n_cand/(N*(N-1)));

%% ---- Initialise Output Structure -----------------------------------------
empty_entry.feasible      = false;
empty_entry.X             = [];
empty_entry.U             = [];
empty_entry.T             = [];
empty_entry.Tf            = NaN;
empty_entry.cost          = NaN;
empty_entry.solver_status = 'not_attempted';

% Pre-fill entire NxN grid with empty entries
transition_library(N, N) = empty_entry;
for ii = 1:N
    for jj = 1:N
        transition_library(ii,jj) = empty_entry;
    end
end

%% ---- Main OCP Loop -------------------------------------------------------
fprintf('Starting OCP solves for %d candidate pairs...\n', n_cand);
fprintf('%-6s %-6s %-22s %-22s %-8s %-10s %-10s\n', ...
    'i','j','Type_i','Type_j','Tf(s)','Status','Cost');
fprintf('%s\n', repmat('-', 1, 90));

t_loop_start = tic;

for c = 1:n_cand
    i = candidates(c, 1);
    j = candidates(c, 2);

    man_i = maneuver_library{i};
    man_j = maneuver_library{j};

    x0    = man_i.trim_state;
    xref  = man_j.trim_state;

    u_trim1 = [man_i.trim_controls.thrust;
               man_i.trim_controls.delta_e;
               man_i.trim_controls.delta_a;
               man_i.trim_controls.delta_r];

    u_trim2 = [man_j.trim_controls.thrust;
               man_j.trim_controls.delta_e;
               man_j.trim_controls.delta_a;
               man_j.trim_controls.delta_r];

    % Adaptive Tf based on state distance
    Tf = compute_adaptive_Tf(params, i, j, Tf_BASE, Tf_MAX);

    % Solve OCP
    try
        [T_tr, X_tr, U_tr, cost_val, status] = ...
            solve_transition_ocp_6dof(x0, xref, u_trim1, u_trim2, Tf, aircraft);

        converged = contains(status, 'Solve_Succeeded') || ...
                    contains(status, 'Feasible_Point_Found');

        transition_library(i,j).feasible      = converged;
        transition_library(i,j).X             = X_tr;
        transition_library(i,j).U             = U_tr;
        transition_library(i,j).T             = T_tr;
        transition_library(i,j).Tf            = Tf;
        transition_library(i,j).cost          = cost_val;
        transition_library(i,j).solver_status = status;

        status_str = status;
        if length(status_str) > 22, status_str = status_str(1:22); end
    catch ME
        transition_library(i,j).solver_status = ME.message(1:min(end,30));
        status_str = 'MATLAB_ERROR';
        cost_val   = NaN;
        converged  = false;
    end

    % Progress report
    flag = '  ';
    if converged, flag = 'OK'; else, flag = '--'; end
    fprintf('[%s] %4d->%-4d  %-22s %-22s  Tf=%5.1f  %-28s  J=%.4g\n', ...
        flag, i, j, ...
        pad_str(man_i.maneuver_type, 22), ...
        pad_str(man_j.maneuver_type, 22), ...
        Tf, status_str, cost_val);

    % Periodic autosave every 50 pairs
    if mod(c, 50) == 0
        save(SAVE_FILE, 'transition_library');
        fprintf('  [AUTOSAVE] %d / %d pairs done (%.1f min elapsed).\n', ...
            c, n_cand, toc(t_loop_start)/60);
    end
end

%% ---- Final Save & Summary ------------------------------------------------
save(SAVE_FILE, 'transition_library');

n_ok = sum(arrayfun(@(i,j) transition_library(i,j).feasible, ...
    candidates(:,1), candidates(:,2)));

fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('DONE.  %d / %d transitions converged (%.1f%%).\n', ...
    n_ok, n_cand, 100*n_ok/max(n_cand,1));
fprintf('Elapsed: %.2f minutes.\n', toc(t_loop_start)/60);
fprintf('Saved to: %s\n', SAVE_FILE);
fprintf('%s\n\n', repmat('=', 1, 60));

%% =========================================================================
%% LOCAL HELPER FUNCTIONS
%% =========================================================================

%% ---- extract_trim_params ------------------------------------------------
function p = extract_trim_params(lib, N)
% Compute scalar aerodynamic parameters from each trim state for fast
% pairwise comparison.
    p.V     = zeros(1, N);
    p.alpha = zeros(1, N);
    p.beta  = zeros(1, N);
    p.phi   = zeros(1, N);
    p.gamma = zeros(1, N);
    p.type  = cell(1, N);

    for k = 1:N
        xs = lib{k}.trim_state;
        u  = xs(1); v = xs(2); w = xs(3);
        q0 = xs(7); q1 = xs(8); q2 = xs(9); q3 = xs(10);

        Vk = sqrt(u^2 + v^2 + w^2 + 1e-9);
        p.V(k)     = Vk;
        p.alpha(k) = atan2(w, u);
        p.beta(k)  = asin(v / Vk);

        % Bank angle from quaternion (roll about body x)
        p.phi(k) = atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1^2 + q2^2));

        % Flight-path angle: rotate body velocity to inertial, take vertical
        Rbi = quat2dcm([q0, q1, q2, q3]);   % built-in quaternion to DCM
        vel_i = Rbi * [u; v; w];
        p.gamma(k) = asin(-vel_i(3) / Vk);  % z_down convention

        p.type{k} = lib{k}.maneuver_type;
    end
end

%% ---- compute_adaptive_Tf ------------------------------------------------
function Tf = compute_adaptive_Tf(p, i, j, Tf_base, Tf_max)
% Scale transition time based on how different the two trim conditions are.
% Each axis is normalised by its threshold and contributes additively.
    V_THRESH  = 80;
    A_THRESH  = deg2rad(15);
    P_THRESH  = deg2rad(60);
    G_THRESH  = deg2rad(20);

    dV    = abs(p.V(i)     - p.V(j))     / V_THRESH;
    dA    = abs(p.alpha(i) - p.alpha(j)) / A_THRESH;
    dPhi  = abs(angdiff(p.phi(i), p.phi(j))) / P_THRESH;
    dGam  = abs(p.gamma(i) - p.gamma(j)) / G_THRESH;

    difficulty = dV + dA + dPhi + dGam;   % 0 = identical, ~4 = max mismatch
    Tf = Tf_base + (Tf_max - Tf_base) * min(difficulty / 2, 1.0);
    Tf = round(Tf)+2;   % integer seconds keeps N consistent
end

%% ---- pad_str ------------------------------------------------------------
function s = pad_str(s, n)
% Right-pad a string to length n for aligned console output.
    if length(s) > n
        s = s(1:n);
    else
        s = [s, repmat(' ', 1, n - length(s))];
    end
end

%% =========================================================================
%% OCP SOLVER (identical to master_trajectory.m version — kept local so
%% this script is fully self-contained)
%% =========================================================================

function [T_out, X_opt, U_opt, cost_val, status] = ...
        solve_transition_ocp_6dof(x0, xref, u0, uref, Tf, aircraft)

     import casadi.*

    N  = 80;          % control intervals (increase for longer Tf)
    dt = Tf / N;
    nx = 13;  nu = 4;

    %% Align quaternion signs (shortest path)
    if dot(x0(7:10), xref(7:10)) < 0, xref(7:10) = -xref(7:10); end

    %% Dynamics function
    xs = SX.sym('x', nx);
    us = SX.sym('u', nu);
    f  = Function('f', {xs, us}, {F18_6DOF_casadi(xs, us, aircraft)});

    %% Warm start — SLERP for quaternion, linear elsewhere
    X_ws = zeros(nx, N+1);
    for i = 1:nx
        X_ws(i,:) = linspace(x0(i), xref(i), N+1);
    end
    X_ws(2,:) = 0;                    % sideslip guess = 0
    qs = x0(7:10)/norm(x0(7:10));
    qe = xref(7:10)/norm(xref(7:10));
    if dot(qs,qe) < 0, qe = -qe; end
    th_q = acos(min(dot(qs,qe), 1.0));
    for k = 1:N+1
        tau = (k-1)/N;
        if sin(th_q) < 1e-8
            qi = (1-tau)*qs + tau*qe;
        else
            qi = (sin((1-tau)*th_q)*qs + sin(tau*th_q)*qe) / sin(th_q);
        end
        X_ws(7:10,k) = qi / norm(qi);
    end
    % Position: extrapolate from initial velocity
    X_ws(11,:) = x0(11) + linspace(0, x0(1)*Tf, N+1);
    X_ws(12,:) = x0(12) + linspace(0, x0(2)*Tf, N+1);
    X_ws(13,:) = x0(13) + linspace(0, x0(3)*Tf, N+1);

    U_ws = zeros(nu, N);
    for i = 1:nu
        U_ws(i,:) = linspace(u0(i), uref(i), N);
    end

    %% Weights
    % Control effort (normalised)
    R      = diag([1/142000^2, 1/(25*pi/180)^2, ...
                   1/(21*pi/180)^2, 1/(30*pi/180)^2]);
    R_rate = 1.5 * R;
    Q_beta = 5000;       % sideslip penalty
    % Running state error: velocity + angular rate states only
    q_run  = diag([1/50^2, 1/10^2, 1/50^2, ...
                   (5*pi/180)^(-2), (5*pi/180)^(-2), (5*pi/180)^(-2)]);
    % Terminal weights (strong)
    Q_term_vel  = 150 * q_run;
    Q_term_quat = 200 * eye(4);

    %% NLP build
    w   = {};  w0  = [];  lbw = [];  ubw = [];
    J   = 0;   g   = {};  lbg = [];  ubg = [];

    u_lb = [500;    -25*pi/180; -21*pi/180; -30*pi/180];  % FIX: thrust >= 500
    u_ub = [142000;  25*pi/180;  21*pi/180;  30*pi/180];

    % Initial node (fixed)
    Xk = SX.sym('X_0', nx);
    w  = [w, {Xk}];  lbw = [lbw; x0];  ubw = [ubw; x0];  w0 = [w0; X_ws(:,1)];

    U_prev = u0;
    for k = 0:N-1
        %% Control node
        Uk = SX.sym(['U_' num2str(k)], nu);
        w  = [w, {Uk}];
        if k == 0
            lbw = [lbw; u0];    ubw = [ubw; u0];      % lock entry control
        elseif k == N-1
            lbw = [lbw; uref];  ubw = [ubw; uref];    % lock exit control
        else
            lbw = [lbw; u_lb];  ubw = [ubw; u_ub];
        end
        w0 = [w0; U_ws(:,k+1)];

        %% RK4
        k1 = f(Xk, Uk);
        k2 = f(Xk + dt/2*k1, Uk);
        k3 = f(Xk + dt/2*k2, Uk);
        k4 = f(Xk + dt*k3,   Uk);
        Xk_phys = Xk + dt/6*(k1 + 2*k2 + 2*k3 + k4);

        %% Running cost
        J = J + (Uk' * R * Uk) * dt;
        % if k > 0
        %     J = J + ((Uk-U_prev)' * R_rate * (Uk-U_prev)) * dt;
        % end
        % U_prev = Uk;
        % dx_vel = Xk(1:6) - xref(1:6);
        % J = J + (dx_vel' * q_run * dx_vel) * dt;

        %% Next state node
        Xk_next = SX.sym(['X_' num2str(k+1)], nx);
        w   = [w, {Xk_next}];
        lbw = [lbw; -inf(nx,1)];  ubw = [ubw; inf(nx,1)];
        w0  = [w0; X_ws(:,k+2)];

        %% Defect constraint (dynamics)
        g   = [g, {Xk_next - Xk_phys}];
        lbg = [lbg; zeros(nx,1)];  ubg = [ubg; zeros(nx,1)];

        %% Quaternion unit-norm constraint
        q_n = Xk_next(7)^2 + Xk_next(8)^2 + Xk_next(9)^2 + Xk_next(10)^2;
        g   = [g, {q_n}];
        lbg = [lbg; 1.0];  ubg = [ubg; 1.0];

        %% Sideslip penalty + constraint  (±5 deg hard limit)
        v_n = Xk_next(2);  u_n = Xk_next(1);  w_n = Xk_next(3);
        V_n = sqrt(u_n^2 + v_n^2 + w_n^2 + 1e-6);
        beta_a = v_n / V_n;
        % J = J + Q_beta * beta_a^2 * dt;
        g   = [g, {beta_a}];
        lbg = [lbg; -10*pi/180];  ubg = [ubg; 10*pi/180];

        % %% Alpha path constraint  (-5 to +28 deg)
        % alpha_k = atan2(w_n, u_n);
        % g   = [g, {alpha_k}];
        % lbg = [lbg; -5*pi/180];  ubg = [ubg; 28*pi/180];
        % 
        % %% Load factor path constraint  (|Nz| <= 9g)
        % % FIX: added — was completely missing in original
        % qbar_k = 0.5 * 1.225 * V_n^2;
        % CL_k   = aircraft.aero.CL_alpha * alpha_k;
        % Nz_k   = qbar_k * aircraft.aero.S_ref * CL_k / (aircraft.mass * 9.81);
        % g   = [g, {Nz_k}];
        % lbg = [lbg; -3.0];   ubg = [ubg;  9.0];

        Xk = Xk_next;
    end

    %% Terminal cost
    % dx_v = Xk(1:6)   - xref(1:6);
    % dx_q = Xk(7:10)  - xref(7:10);
    % J = J + dx_v' * Q_term_vel  * dx_v;
    % J = J + dx_q' * Q_term_quat * dx_q;

    %% Terminal constraint — per-state physical tolerances
    % FIX: was uniform 0.04 for all states (wrong units)
    tol_term = [2.0;  1.0;  2.0;           % u, v, w  [m/s]
                0.05; 0.05; 0.05;          % p, q, r  [rad/s]
                0.02; 0.02; 0.02; 0.02];   % quaternion components
    g   = [g, {Xk(1:10) - xref(1:10)}];
    lbg = [lbg; -tol_term];
    ubg = [ubg;  tol_term];

    %% Solve
    prob = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}));
    opts = struct;
    opts.ipopt.max_iter       = 2000;
    opts.ipopt.tol            = 1e-5;
    opts.ipopt.acceptable_tol = 1e-4;
    opts.ipopt.acceptable_iter = 8;
    opts.ipopt.print_level    = 0;
    opts.ipopt.acceptable_obj_change_tol = 1e-4;

    solver = nlpsol('solver', 'ipopt', prob, opts);
    sol    = solver('x0', w0, 'lbx', lbw, 'ubx', ubw, 'lbg', lbg, 'ubg', ubg);

    % %% Convergence check  (FIX: was missing)
    % stats = solver.stats();
    % ok = strcmp(stats.return_status, 'Solve_Succeeded') || ...
    %      strcmp(stats.return_status, 'Solved_To_Acceptable_Level');
    % if ~ok
    %     warning('OCP (CasADi): solver returned "%s" — trajectory may be infeasible.', ...
    %             stats.return_status);
    % else
    %     fprintf('  [OK] %s  (iter=%d)\n', stats.return_status, stats.iter_count);
    % end

    %% Extract

    
    cost_val = full(sol.f);
    stats    = solver.stats();
    status   = stats.return_status;
    
    w_opt = full(sol.x);
    X_opt = zeros(N+1, nx);
    U_opt = zeros(N,   nu);
    idx = 1;
    for k = 1:N
        X_opt(k,:) = w_opt(idx:idx+nx-1)';  idx = idx+nx;
        U_opt(k,:) = w_opt(idx:idx+nu-1)';  idx = idx+nu;
    end
    X_opt(N+1,:) = w_opt(idx:idx+nx-1)';
    T_out = linspace(0, Tf, N+1)';
    U_opt = [U_opt; U_opt(end,:)];
end

%% ---- 6DOF Dynamics (CasADi) --------------------------------------------
function xdot = F18_6DOF_casadi(x, u, a)
    import casadi.*

    u_b=x(1); v=x(2); w=x(3);
    p=x(4);   q=x(5); r=x(6);
    q0=x(7); q1=x(8); q2=x(9); q3=x(10);
    T=u(1); de=u(2); da=u(3); dr=u(4);

    rho  = 1.225;
    V    = sqrt(u_b^2 + v^2 + w^2 + 1e-6);
    qbar = 0.5*rho*V^2;
    alp  = atan2(w, u_b);
    bet  = asin(v / V);

    % FIX: k_alpha2 drag term
    CL  = a.aero.CL_alpha * alp;
    CD  = a.aero.CD0 + a.aero.k*CL^2 + a.aero.k_alpha2*alp^2;
    % FIX: CY sideforce
    CY  = a.aero.CY_beta*bet + a.aero.CY_dr*dr;

    L    = qbar*a.aero.S_ref*CL;
    D    = qbar*a.aero.S_ref*CD;
    Yf   = qbar*a.aero.S_ref*CY;

    Fw  = [T-D; Yf; -L];
    Cwb = [cos(alp)*cos(bet),  sin(bet), sin(alp)*cos(bet);
          -cos(alp)*sin(bet),  cos(bet),-sin(alp)*sin(bet);
          -sin(alp),           0,        cos(alp)         ];
    Fb  = Cwb' * Fw;

    Rbi = [1-2*(q2^2+q3^2),   2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2);
            2*(q1*q2+q0*q3),  1-2*(q1^2+q3^2),   2*(q2*q3-q0*q1);
            2*(q1*q3-q0*q2),  2*(q2*q3+q0*q1),   1-2*(q1^2+q2^2)];

    Fg = a.mass * (Rbi' * [0;0;9.81]);
    FX = Fb(1)+Fg(1);  FY = Fb(2)+Fg(2);  FZ = Fb(3)+Fg(3);

    ud = FX/a.mass + r*v - q*w;
    vd = FY/a.mass + p*w - r*u_b;
    wd = FZ/a.mass + q*u_b - p*v;

    I3 = [a.Ixx, 0, -a.Ixz; 0, a.Iyy, 0; -a.Ixz, 0, a.Izz];
    om = [p;q;r];

    Cm = a.aero.Cm_alpha*alp + a.aero.Cm_q*(a.aero.c_bar/(2*V))*q + a.aero.Cm_delta_e*de;
    Cl = a.aero.Cl_beta*bet  + a.aero.Cl_p*(a.aero.b/(2*V))*p   + a.aero.Cl_r*(a.aero.b/(2*V))*r   + a.aero.Cl_delta_a*da + a.aero.Cl_delta_r*dr;
    Cn = a.aero.Cn_beta*bet  + a.aero.Cn_p*(a.aero.b/(2*V))*p   + a.aero.Cn_r*(a.aero.b/(2*V))*r   + a.aero.Cn_delta_a*da + a.aero.Cn_delta_r*dr;

    Lm = qbar*a.aero.S_ref*a.aero.b    *Cl;
    Mm = qbar*a.aero.S_ref*a.aero.c_bar*Cm;
    Nm = qbar*a.aero.S_ref*a.aero.b    *Cn;

    om_dot = I3 \ ([Lm;Mm;Nm] - cross(om, I3*om));

    Om  = [0,-p,-q,-r; p,0,r,-q; q,-r,0,p; r,q,-p,0];
    qd  = 0.5 * Om * [q0;q1;q2;q3];
    vi  = Rbi * [u_b;v;w];

    xdot = [ud; vd; wd; om_dot; qd; vi];
end

function a = get_f18_harv_parameters()
    a.mass = 15096.5;
    a.Ixx  =  31183.6;
    a.Iyy  = 205127.5;
    a.Izz  = 230432.1;
    a.Ixz  =   4028.0;

    a.aero.S_ref  = 37.16;
    a.aero.c_bar  =  3.511;
    a.aero.b      = 11.404;

    a.aero.CL_alpha = 5.0;

    a.aero.CD0      = 0.020;
    a.aero.k        = 0.080;
    a.aero.k_alpha2 = 0.300;   % FIX: was missing

    a.aero.Cm_alpha   = -0.45;   % FIX: was -0.02
    a.aero.Cm_q       = -4.50;   % FIX: was -8.00
    a.aero.Cm_delta_e = -0.50;   % FIX: was -0.015

    a.aero.Cl_beta    = -0.080;
    a.aero.Cl_p       = -0.300;
    a.aero.Cl_r       =  0.050;
    a.aero.Cl_delta_a = -0.150;
    a.aero.Cl_delta_r =  0.050;

    a.aero.Cn_beta    =  0.080;
    a.aero.Cn_p       = -0.030;
    a.aero.Cn_r       = -0.150;
    a.aero.Cn_delta_a = -0.010;
    a.aero.Cn_delta_r = -0.120;

    a.aero.CY_beta = -0.730;     % FIX: was missing (zero)
    a.aero.CY_dr   =  0.200;     % FIX: was missing (zero)

    a.ctrl.delta_e_min = deg2rad(-24.0);
    a.ctrl.delta_e_max = deg2rad( 10.5);
    a.ctrl.delta_a_min = deg2rad(-25.0);
    a.ctrl.delta_a_max = deg2rad( 45.0);
    a.ctrl.delta_r_min = deg2rad(-30.0);
    a.ctrl.delta_r_max = deg2rad( 30.0);
    a.ctrl.thrust_min  = 500;
    a.ctrl.thrust_max  = 143200;
end