%% =========================================================================
%% build_transition_library_parallel.m
%%
%% PARALLEL version — parfor over candidate pairs.
%%
%% CRITICAL FIX: No struct 'res' inside parfor. Instead, separate
%% pre-allocated arrays (chunk_feasible, chunk_X, ...) are used.
%% Each array is indexed only by the parfor loop variable c — this is
%% the only pattern MATLAB accepts for sliced output in parfor.
%%
%% BEFORE RUNNING:
%%   1. Set CASADI_PATH below to your actual CasADi folder.
%%      To find it, run in MATLAB:   which casadi.SX
%%      Look for the folder containing "+casadi" or "casadi.m"
%%   2. Make sure maneuver_library.mat is in your working directory.
%% =========================================================================

clc; clear;

%% =========================================================================
%% USER SETTINGS  — edit these lines
%% =========================================================================

CASADI_PATH = 'C:\Users\HP\Downloads\casadi-windows-matlabR2016a-v3.4.5';
%              ^^^ change this to your actual CasADi folder ^^^
%              Right-click folder in Windows Explorer -> "Copy as path"

NUM_WORKERS = 6;      % Inf = all cores, or e.g. 4

V_THRESH  = 80;         % max airspeed diff   [m/s]
A_THRESH  = deg2rad(15);% max alpha diff      [rad]
B_THRESH  = deg2rad(10);% max sideslip diff   [rad]
P_THRESH  = deg2rad(60);% max bank-angle diff [rad]

Tf_BASE   = 5;          % min transition time [s]
Tf_MAX    = 25;         % max transition time [s]

CHUNK_SIZE = 100;       % autosave every N pairs
SAVE_FILE  = 'transition_library.mat';

%% =========================================================================
%% SECTION 1 — VERIFY CASADI PATH
%% =========================================================================

if ~exist(CASADI_PATH, 'dir')
    error(['CasADi folder not found:\n  %s\n\n' ...
           'Edit CASADI_PATH at the top of this script.\n' ...
           'To find it, run:  which casadi.SX\n' ...
           'Or search for a folder containing "+casadi" or "casadi.m".'], ...
           CASADI_PATH);
end
addpath('C:\Users\HP\Downloads\casadi-windows-matlabR2016a-v3.4.5');
fprintf('CasADi path OK: %s\n', CASADI_PATH);

%% =========================================================================
%% SECTION 2 — LOAD DATA
%% =========================================================================

fprintf('Loading maneuver_library.mat ...\n');
load('maneuver_library.mat', 'maneuver_library');
aircraft = get_f18_harv_parameters();
N = length(maneuver_library);
fprintf('  Found %d primitives.\n\n', N);

fprintf('Extracting aerodynamic parameters...\n');
params = extract_trim_params(maneuver_library, N);

%% =========================================================================
%% SECTION 3 — FEASIBILITY PRE-FILTER
%% =========================================================================

fprintf('Running feasibility pre-filter...\n');
candidates = [];
for i = 1:N
    for j = 1:N
        if i == j, continue; end
        dV     = abs(params.V(i)     - params.V(j));
        dAlpha = abs(params.alpha(i) - params.alpha(j));
        dBeta  = abs(params.beta(i)  - params.beta(j));
        dPhi   = abs(angdiff(params.phi(i), params.phi(j)));
        if dV <= V_THRESH && dAlpha <= A_THRESH && ...
           dBeta <= B_THRESH && dPhi <= P_THRESH
            candidates(end+1, :) = [i, j]; %#ok<AGROW>
        end
    end
end
n_cand = size(candidates, 1);
fprintf('  %d / %d pairs pass filter (%.1f%%).\n\n', ...
    n_cand, N*(N-1), 100*n_cand/max(N*(N-1), 1));

%% =========================================================================
%% SECTION 4 — INITIALISE OUTPUT STRUCT ARRAY
%% =========================================================================

empty_entry.feasible      = false;
empty_entry.X             = [];
empty_entry.U             = [];
empty_entry.T             = [];
empty_entry.Tf            = NaN;
empty_entry.cost          = NaN;
empty_entry.solver_status = 'not_attempted';

transition_library(N, N)  = empty_entry;
for ii = 1:N
    for jj = 1:N
        transition_library(ii,jj) = empty_entry;
    end
end

%% =========================================================================
%% SECTION 5 — FLATTEN MANEUVER DATA FOR PARFOR
%%
%% parfor cannot efficiently broadcast a cell array of structs.
%% Extract everything into plain arrays beforehand.
%% =========================================================================

trim_states   = cell(N, 1);
trim_controls = zeros(N, 4);   % [thrust, delta_e, delta_a, delta_r]
for k = 1:N
    trim_states{k}      = maneuver_library{k}.trim_state;
    trim_controls(k, 1) = maneuver_library{k}.trim_controls.thrust;
    trim_controls(k, 2) = maneuver_library{k}.trim_controls.delta_e;
    trim_controls(k, 3) = maneuver_library{k}.trim_controls.delta_a;
    trim_controls(k, 4) = maneuver_library{k}.trim_controls.delta_r;
end

%% =========================================================================
%% SECTION 6 — OPEN PARALLEL POOL
%% =========================================================================

fprintf('Opening parallel pool...\n');
pool = gcp('nocreate');

% If a pool exists but has the wrong number of workers, shut it down
if ~isempty(pool)
    correct_size = isinf(NUM_WORKERS) || pool.NumWorkers == NUM_WORKERS;
    if ~correct_size
        fprintf('  Existing pool has %d workers (need %d) — restarting...\n', ...
                pool.NumWorkers, NUM_WORKERS);
        delete(pool);
        pool = [];
    end
end

% Open pool if not already running with correct size
if isempty(pool)
    if isinf(NUM_WORKERS)
        pool = parpool('local');
    else
        pool = parpool('local', NUM_WORKERS);
    end
end

fprintf('  Pool ready: %d workers.\n\n', pool.NumWorkers);

% Add CasADi on every worker once, before any parfor runs
parfevalOnAll(pool, @addpath, 0, CASADI_PATH);
fprintf('  CasADi path set on all workers.\n\n');

%% =========================================================================
%% SECTION 7 — PARALLEL OCP LOOP (chunked for autosave)
%%
%% THE FIX:
%%   WRONG  ->  res.feasible = true  (struct built field-by-field in parfor)
%%   RIGHT  ->  chunk_feasible(c) = true  (separate pre-allocated arrays)
%%
%% MATLAB parfor requires that output variables are indexed ONLY by the
%% loop variable c, and that every element is independent. Structs fail
%% this classification because MATLAB cannot prove which fields are written.
%% Separate arrays with chunk_X(c) or chunk_X{c} satisfy the classifier.
%% =========================================================================

fprintf('Starting OCP solves: %d pairs, chunk size %d...\n\n', n_cand, CHUNK_SIZE);

t_total           = tic;
n_converged_total = 0;
chunk_starts      = 1 : CHUNK_SIZE : n_cand;
n_chunks          = length(chunk_starts);

for ch = 1:n_chunks

    c_start   = chunk_starts(ch);
    c_end     = min(c_start + CHUNK_SIZE - 1, n_cand);
    chunk_len = c_end - c_start + 1;

    fprintf('[Chunk %d/%d]  pairs %d to %d ...\n', ch, n_chunks, c_start, c_end);
    t_chunk = tic;

    %% Slice candidate indices for this chunk (plain numeric vectors)
    chunk_i = candidates(c_start:c_end, 1);
    chunk_j = candidates(c_start:c_end, 2);

    %% ---------------------------------------------------------------
    %% Pre-allocate ONE array per output field (parfor-safe pattern).
    %% Each is indexed only by c — no structs, no nested indexing.
    %% ---------------------------------------------------------------
    chunk_feasible = false(chunk_len, 1);         % logical scalar
    chunk_X        = cell(chunk_len, 1);          % variable-size matrix
    chunk_U        = cell(chunk_len, 1);          % variable-size matrix
    chunk_T        = cell(chunk_len, 1);          % variable-size vector
    chunk_Tf       = nan(chunk_len, 1);           % double scalar
    chunk_cost     = nan(chunk_len, 1);           % double scalar
    chunk_status   = repmat({'not_attempted'}, chunk_len, 1); % cell of char

    %% ---------------------------------------------------------------
    %% PARFOR — one worker per (i,j) pair.
    %% Each iteration writes ONLY to chunk_*(c) — nothing shared.
    %% ---------------------------------------------------------------
    parfor c = 1:chunk_len

        % Every worker starts with a blank path — add CasADi locally.
        % addpath(CASADI_PATH);  
        import casadi.*

        i_idx = chunk_i(c);     
        j_idx = chunk_j(c);

        x0      = trim_states{i_idx};         %#ok<PFBNS>
        xref    = trim_states{j_idx};
        u_trim1 = trim_controls(i_idx, :)';   %#ok<PFBNS>
        u_trim2 = trim_controls(j_idx, :)';

        Tf_c = compute_adaptive_Tf(params, i_idx, j_idx, Tf_BASE, Tf_MAX);

        chunk_Tf(c) = Tf_c;   % written even if OCP throws

        try
            [T_tr, X_tr, U_tr, cost_val, sol_status] = ...
                solve_transition_ocp_6dof( ...
                    x0, xref, u_trim1, u_trim2, Tf_c, aircraft); %#ok<PFBNS>

            converged = contains(sol_status, 'Solve_Succeeded') || ...
                        contains(sol_status, 'Solved_To_Acceptable_Level') || ...
                        contains(sol_status, 'Feasible_Point_Found');

            chunk_feasible(c) = converged;
            chunk_X{c}        = X_tr;
            chunk_U{c}        = U_tr;
            chunk_T{c}        = T_tr;
            chunk_cost(c)     = cost_val;
            chunk_status{c}   = sol_status;

        catch ME
            chunk_status{c} = ME.message(1 : min(numel(ME.message), 80));
        end

    end   % ← end parfor

    %% ---------------------------------------------------------------
    %% Write chunk results into transition_library (serial, safe).
    %% ---------------------------------------------------------------
    % n_ok = 0;
    % for c = 1:chunk_len
    %     ii = chunk_i(c);
    %     jj = chunk_j(c);
    % 
    %     transition_library(ii,jj).feasible      = chunk_feasible(c);
    %     transition_library(ii,jj).X             = chunk_X{c};
    %     transition_library(ii,jj).U             = chunk_U{c};
    %     transition_library(ii,jj).T             = chunk_T{c};
    %     transition_library(ii,jj).Tf            = chunk_Tf(c);
    %     transition_library(ii,jj).cost          = chunk_cost(c);
    %     transition_library(ii,jj).solver_status = chunk_status{c};
    % 
    %     if chunk_feasible(c), n_ok = n_ok + 1; end
    % end
    % n_converged_total = n_converged_total + n_ok;


    %% Write chunk results into transition_library (serial, safe)
fprintf('\n  %-6s %-5s %-5s %-8s %-12s %s\n', ...
        'Flag','i','j','Tf(s)','Cost','Status');
fprintf('  %s\n', repmat('-',1,58));

n_ok = 0;
for c = 1:chunk_len
    ii = chunk_i(c);
    jj = chunk_j(c);
    st = chunk_status{c};

    % Classify status into a short flag
    if     contains(st, 'Solve_Succeeded')
        flag = '[OK] ';
    elseif contains(st, 'Solved_To_Acceptable_Level')
        flag = '[~OK]';   % feasible, looser tolerance — still saved as feasible
    elseif contains(st, 'Feasible_Point_Found')
        flag = '[FPF]';
    elseif contains(st, 'Infeasible_Problem_Detected')
        flag = '[INF]';
    elseif contains(st, 'Maximum_Iterations_Exceeded')
        flag = '[MAX]';
    elseif contains(st, 'not_attempted')
        flag = '[---]';
    else
        flag = '[ERR]';
    end

    fprintf('  %s  %3d->%-3d  %5.1f    %-12.4g  %s\n', ...
        flag, ii, jj, chunk_Tf(c), chunk_cost(c), st);

    % Write into struct
    transition_library(ii,jj).feasible      = chunk_feasible(c);
    transition_library(ii,jj).X             = chunk_X{c};
    transition_library(ii,jj).U             = chunk_U{c};
    transition_library(ii,jj).T             = chunk_T{c};
    transition_library(ii,jj).Tf            = chunk_Tf(c);
    transition_library(ii,jj).cost          = chunk_cost(c);
    transition_library(ii,jj).solver_status = chunk_status{c};

    if chunk_feasible(c), n_ok = n_ok + 1; end
end
n_converged_total = n_converged_total + n_ok;

% Chunk summary breakdown
n_ok_acc = sum(contains(chunk_status, 'Solved_To_Acceptable_Level'));
n_inf    = sum(contains(chunk_status, 'Infeasible_Problem_Detected'));
n_max    = sum(contains(chunk_status, 'Maximum_Iterations_Exceeded'));

% fprintf('\n  Summary: [OK]=%d  [~OK]=%d  [INF]=%d  [MAX]=%d  total=%d\n', ...
%     n_ok - n_ok_acc, n_ok_acc, n_inf, n_max, chunk_len);


n_fpf = sum(contains(chunk_status, 'Feasible_Point_Found'));
fprintf('\n  Summary: [OK]=%d  [~OK]=%d  [FPF]=%d  [INF]=%d  [MAX]=%d  total=%d\n', ...
    n_ok-n_ok_acc-n_fpf, n_ok_acc, n_fpf, n_inf, n_max, chunk_len);

    fprintf('  Converged: %d / %d  (%.1f s)\n', n_ok, chunk_len, toc(t_chunk));
    save(SAVE_FILE, 'transition_library');
    fprintf('  [AUTOSAVE] %d / %d total done  (%.1f min elapsed)\n\n', ...
            c_end, n_cand, toc(t_total)/60);

end   % ← end chunk loop

%% =========================================================================
%% SECTION 8 — FINAL SAVE & SUMMARY
%% =========================================================================

save(SAVE_FILE, 'transition_library');

fprintf('\n%s\n', repmat('=',1,60));
fprintf('DONE.\n');
fprintf('  Converged : %d / %d  (%.1f%%)\n', ...
    n_converged_total, n_cand, 100*n_converged_total/max(n_cand,1));
fprintf('  Elapsed   : %.2f minutes\n', toc(t_total)/60);
fprintf('  Saved to  : %s\n', SAVE_FILE);
fprintf('%s\n\n', repmat('=',1,60));


%% =========================================================================
%% HELPER FUNCTIONS
%% =========================================================================

function p = extract_trim_params(lib, N)
    p.V     = zeros(1,N);  p.alpha = zeros(1,N);
    p.beta  = zeros(1,N);  p.phi   = zeros(1,N);
    p.gamma = zeros(1,N);  p.type  = cell(1,N);
    for k = 1:N
        xs = lib{k}.trim_state;
        u  = xs(1); v = xs(2); w = xs(3);
        q0 = xs(7); q1 = xs(8); q2 = xs(9); q3 = xs(10);
        Vk = sqrt(u^2 + v^2 + w^2 + 1e-9);
        p.V(k)     = Vk;
        p.alpha(k) = atan2(w, u);
        p.beta(k)  = asin(v / Vk);
        p.phi(k)   = atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1^2 + q2^2));
        Rbi        = quat2dcm([q0, q1, q2, q3]);
        vel_i      = Rbi * [u; v; w];
        p.gamma(k) = asin(-vel_i(3) / Vk);
        p.type{k}  = lib{k}.maneuver_type;
    end
end

function Tf = compute_adaptive_Tf(p, i, j, Tf_base, Tf_max)
    V_THR = 80;  A_THR = deg2rad(15);
    P_THR = deg2rad(60);  G_THR = deg2rad(20);
    dV   = abs(p.V(i)     - p.V(j))            / V_THR;
    dA   = abs(p.alpha(i) - p.alpha(j))         / A_THR;
    dPhi = abs(angdiff(p.phi(i), p.phi(j)))     / P_THR;
    dGam = abs(p.gamma(i) - p.gamma(j))         / G_THR;
    difficulty = dV + dA + dPhi + dGam;
    Tf = Tf_base + (Tf_max - Tf_base) * min(difficulty/2, 1.0);
    Tf = min(round(Tf) + 2, Tf_max);
end

%% =========================================================================
%% OCP SOLVER
%% =========================================================================

function [T_out, X_opt, U_opt, cost_val, status] = ...
        solve_transition_ocp_6dof(x0, xref, u0, uref, Tf, aircraft)

    import casadi.*

    N_ctrl = 80;
    dt     = Tf / N_ctrl;
    nx = 13; nu = 4;

    if dot(x0(7:10), xref(7:10)) < 0
        xref(7:10) = -xref(7:10);
    end

    xs = SX.sym('x', nx);
    us = SX.sym('u', nu);
    f  = Function('f', {xs, us}, {F18_6DOF_casadi(xs, us, aircraft)});

    %% Warm start
    X_ws = zeros(nx, N_ctrl+1);
    for i = 1:nx
        X_ws(i,:) = linspace(x0(i), xref(i), N_ctrl+1);
    end
    X_ws(2,:) = 0;
    qs = x0(7:10)   / norm(x0(7:10));
    qe = xref(7:10) / norm(xref(7:10));
    if dot(qs,qe) < 0, qe = -qe; end
    th_q = acos(min(dot(qs,qe), 1.0));
    for k = 1:N_ctrl+1
        tau = (k-1)/N_ctrl;
        if sin(th_q) < 1e-8
            qi = (1-tau)*qs + tau*qe;
        else
            qi = (sin((1-tau)*th_q)*qs + sin(tau*th_q)*qe) / sin(th_q);
        end
        X_ws(7:10,k) = qi / norm(qi);
    end
    X_ws(11,:) = x0(11) + linspace(0, x0(1)*Tf, N_ctrl+1);
    X_ws(12,:) = x0(12) + linspace(0, x0(2)*Tf, N_ctrl+1);
    X_ws(13,:) = x0(13) + linspace(0, x0(3)*Tf, N_ctrl+1);
    U_ws = zeros(nu, N_ctrl);
    for i = 1:nu
        U_ws(i,:) = linspace(u0(i), uref(i), N_ctrl);
    end

    %% Weights
    R = diag([1/142000^2, 1/(25*pi/180)^2, ...
              1/(21*pi/180)^2, 1/(30*pi/180)^2]);

    %% NLP build
    w   = {}; w0 = []; lbw = []; ubw = [];
    J   = 0;  g  = []; lbg = []; ubg = [];

    u_lb = [500;    -25*pi/180; -21*pi/180; -30*pi/180];
    u_ub = [142000;  25*pi/180;  21*pi/180;  30*pi/180];

    Xk  = SX.sym('X_0', nx);
    w   = [w, {Xk}];
    lbw = [lbw; x0]; ubw = [ubw; x0]; w0 = [w0; X_ws(:,1)];

    for k = 0:N_ctrl-1
        Uk = SX.sym(['U_' num2str(k)], nu);
        w  = [w, {Uk}];
        if k == 0
            lbw = [lbw; u0];    ubw = [ubw; u0];
        elseif k == N_ctrl-1
            lbw = [lbw; uref];  ubw = [ubw; uref];
        else
            lbw = [lbw; u_lb];  ubw = [ubw; u_ub];
        end
        w0 = [w0; U_ws(:,k+1)];

        k1 = f(Xk, Uk);
        k2 = f(Xk + dt/2*k1, Uk);
        k3 = f(Xk + dt/2*k2, Uk);
        k4 = f(Xk + dt*k3,   Uk);
        Xk_phys = Xk + dt/6*(k1 + 2*k2 + 2*k3 + k4);

        J = J + (Uk' * R * Uk) * dt;

        Xk_next = SX.sym(['X_' num2str(k+1)], nx);
        w   = [w,   {Xk_next}];
        lbw = [lbw; -inf(nx,1)]; ubw = [ubw; inf(nx,1)];
        w0  = [w0;  X_ws(:,k+2)];

        g   = [g;   Xk_next - Xk_phys];
        lbg = [lbg; zeros(nx,1)];
        ubg = [ubg; zeros(nx,1)];

        q_n = Xk_next(7)^2 + Xk_next(8)^2 + Xk_next(9)^2 + Xk_next(10)^2;
        g   = [g;   q_n];
        % lbg = [lbg; 1.0]; ubg = [ubg; 1.0];
        lbg = [lbg; 0.9998]; ubg = [ubg; 1.0002];

        u_n = Xk_next(1); v_n = Xk_next(2); w_n = Xk_next(3);
        V_n = sqrt(u_n^2 + v_n^2 + w_n^2 + 1e-6);
        beta_a = v_n / V_n;
        g   = [g;   beta_a];
        lbg = [lbg; -10*pi/180]; ubg = [ubg; 10*pi/180];

        Xk = Xk_next;
    end

    tol_term = [2.0; 1.0; 2.0; 0.05; 0.05; 0.05; 0.02; 0.02; 0.02; 0.02];
    g   = [g;   Xk(1:10) - xref(1:10)];
    lbg = [lbg; -tol_term];
    ubg = [ubg;  tol_term];

    prob = struct('f', J, 'x', vertcat(w{:}), 'g', g);
    opts = struct;
    opts.ipopt.max_iter               = 2000;
    opts.ipopt.tol                    = 1e-5;
    opts.ipopt.acceptable_tol         = 1e-4;
    opts.ipopt.acceptable_iter        = 8;
    opts.ipopt.print_level            = 0;
    opts.ipopt.acceptable_obj_change_tol = 1e-4;

    solver   = nlpsol('solver', 'ipopt', prob, opts);
    sol      = solver('x0', w0, 'lbx', lbw, 'ubx', ubw, 'lbg', lbg, 'ubg', ubg);
    cost_val = full(sol.f);
    stats    = solver.stats();
    status   = stats.return_status;

    w_opt = full(sol.x);
    X_opt = zeros(N_ctrl+1, nx);
    U_opt = zeros(N_ctrl,   nu);
    idx = 1;
    for k = 1:N_ctrl
        X_opt(k,:) = w_opt(idx:idx+nx-1)'; idx = idx+nx;
        U_opt(k,:) = w_opt(idx:idx+nu-1)'; idx = idx+nu;
    end
    X_opt(N_ctrl+1,:) = w_opt(idx:idx+nx-1)';
    T_out = linspace(0, Tf, N_ctrl+1)';
    U_opt = [U_opt; U_opt(end,:)];
end

%% =========================================================================
%% 6DOF DYNAMICS
%% =========================================================================

function xdot = F18_6DOF_casadi(x, u_c, a)
%F18_6DOF_CASADI  6-DOF equations of motion for F-18 HARV (CasADi symbolic).
%
%  State  x  = [u v w  p q r  q0 q1 q2 q3  px py pz]   (13x1)
%  Input  u_c = [T  de  da  dr]                           (4x1)
%  Output xdot = dx/dt                                    (13x1)

import casadi.*

%% --- 1. Unpack state & controls -----------------------------------------
u_b = x(1);  v = x(2);  w = x(3);      % body-axis velocity components [m/s]
p   = x(4);  q = x(5);  r = x(6);      % body-axis angular rates [rad/s]
q0  = x(7);  q1 = x(8); q2 = x(9); q3 = x(10);  % quaternion (scalar-first)
% x(11:13) = inertial position [m] — kinematics only, not needed for forces

T  = u_c(1);   % thrust [N]
de = u_c(2);   % elevator deflection [rad]
da = u_c(3);   % aileron deflection  [rad]
dr = u_c(4);   % rudder deflection   [rad]

%% --- 2. Airspeed & aerodynamic angles ------------------------------------
V     = sqrt(u_b^2 + v^2 + w^2 + 1e-6);   % total airspeed [m/s], eps avoids /0
qbar  = 0.5 * 1.225 * V^2;                 % dynamic pressure [Pa], rho = 1.225 kg/m³
alpha = atan2(w, u_b);                      % angle of attack  [rad]
beta  = asin(v / V);                        % sideslip angle   [rad]

%% --- 3. Aerodynamic coefficients -----------------------------------------
CL = a.aero.CL_alpha * alpha;
CD = a.aero.CD0 + a.aero.k * CL^2 + a.aero.k_alpha2 * alpha^2;
CY = a.aero.CY_beta * beta + a.aero.CY_dr * dr;

%% --- 4. Aerodynamic forces (wind frame → body frame) --------------------
%  Wind-frame forces: [-D, Y, -L]
%  Cwb transforms wind → body; Cwb' (= Cbw) transforms body → wind,
%  so Cwb' * F_wind gives F_body.
ca = cos(alpha);  sa = sin(alpha);
cb = cos(beta);   sb = sin(beta);

Cwb = [ ca*cb,   sb,  sa*cb ;
       -ca*sb,   cb, -sa*sb ;
       -sa,      0,   ca   ];

S   = a.aero.S_ref;
L_f = qbar * S * CL;
D_f = qbar * S * CD;
Y_f = qbar * S * CY;

Fb_aero = Cwb' * [-D_f; Y_f; -L_f];   % aerodynamic force in body frame [N]

%% --- 5. Thrust force (body x-axis only) ----------------------------------
Fb_thrust = [T; 0; 0];

%% --- 6. Gravity in body frame --------------------------------------------
%  Rbi: body → inertial DCM (from quaternion, scalar-first convention)
Rbi = [1-2*(q2^2+q3^2),   2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2);
         2*(q1*q2+q0*q3), 1-2*(q1^2+q3^2),   2*(q2*q3-q0*q1);
         2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1), 1-2*(q1^2+q2^2)];

%  Inertial gravity vector is [0;0;g]; rotate to body frame via Rbi'
Fb_grav = a.mass * 9.80665 * (Rbi' * [0; 0; 1]);

%% --- 7. Translational acceleration (Newton, body frame) -----------------
Ftot = Fb_aero + Fb_thrust + Fb_grav;

ud = Ftot(1)/a.mass + r*v - q*w;
vd = Ftot(2)/a.mass + p*w - r*u_b;
wd = Ftot(3)/a.mass + q*u_b - p*v;

%% --- 8. Aerodynamic moment coefficients ----------------------------------
b    = a.aero.b;
cbar = a.aero.c_bar;
phat = b    / (2*V);   % non-dimensional roll-rate factor
qhat = cbar / (2*V);   % non-dimensional pitch-rate factor
rhat = b    / (2*V);   % non-dimensional yaw-rate factor

Cl = a.aero.Cl_beta*beta  + a.aero.Cl_p*phat*p  + a.aero.Cl_r*rhat*r ...
   + a.aero.Cl_delta_a*da + a.aero.Cl_delta_r*dr;

Cm = a.aero.Cm_alpha*alpha + a.aero.Cm_q*qhat*q  + a.aero.Cm_delta_e*de;

Cn = a.aero.Cn_beta*beta   + a.aero.Cn_p*phat*p  + a.aero.Cn_r*rhat*r ...
   + a.aero.Cn_delta_a*da  + a.aero.Cn_delta_r*dr;

%% --- 9. Moments & rotational dynamics (Euler equations) -----------------
Lm = qbar * S * b    * Cl;
Mm = qbar * S * cbar * Cm;
Nm = qbar * S * b    * Cn;

I      = [a.Ixx, 0,     -a.Ixz;
          0,     a.Iyy,  0    ;
         -a.Ixz, 0,      a.Izz];
omega  = [p; q; r];
om_dot = I \ ([Lm; Mm; Nm] - cross(omega, I*omega));

%% --- 10. Quaternion kinematics ------------------------------------------
%  qdot = 0.5 * Xi(q) * omega,  where Xi is the 4x3 left-product matrix
Xi = [-q1, -q2, -q3;
       q0, -q3,  q2;
       q3,  q0, -q1;
      -q2,  q1,  q0];
q_dot = 0.5 * Xi * omega;

%% --- 11. Inertial position kinematics -----------------------------------
pos_dot = Rbi * [u_b; v; w];   % body velocity → inertial frame

%% --- 12. Assemble state derivative --------------------------------------
xdot = [ud; vd; wd;          % 1-3  body velocity
        om_dot;               % 4-6  angular rates
        q_dot;                % 7-10 quaternion
        pos_dot];             % 11-13 inertial position
end
%% =========================================================================
%% AIRCRAFT PARAMETERS (F-18 HARV)
%% =========================================================================

function a = get_f18_harv_parameters()
% Mass & inertia (Chakraborty Table 2.1, slug·ft² → kg·m²)
a.mass = 15096.5;
a.Ixx  =  31183.6;
a.Iyy  = 205127.5;
a.Izz  = 230432.1;
a.Ixz  =   4028.0;   % positive magnitude; tensor uses -Ixz

% Reference geometry
a.aero.S_ref  = 37.16;
a.aero.c_bar  =  3.511;
a.aero.b      = 11.404;

% Lift (linear model, effective slope ~5 /rad)
a.aero.CL_alpha = 5.0;

% Drag
a.aero.CD0      = 0.020;
a.aero.k        = 0.080;
a.aero.k_alpha2 = 0.300;   % alpha² drag rise term [/rad²]

% Pitching moment
a.aero.Cm_alpha   = -0.45;
a.aero.Cm_q       = -4.50;
a.aero.Cm_delta_e = -0.50;

% Rolling moment
a.aero.Cl_beta    = -0.080;
a.aero.Cl_p       = -0.300;
a.aero.Cl_r       =  0.050;
a.aero.Cl_delta_a = -0.150;
a.aero.Cl_delta_r =  0.050;

% Yawing moment
a.aero.Cn_beta    =  0.080;
a.aero.Cn_p       = -0.030;
a.aero.Cn_r       = -0.150;
a.aero.Cn_delta_a = -0.010;
a.aero.Cn_delta_r = -0.120;

% Sideforce
a.aero.CY_beta = -0.730;
a.aero.CY_dr   =  0.200;

% Control limits (FA-18A, stored for OCP bound reference)
a.ctrl.delta_e_min = deg2rad(-24.0);
a.ctrl.delta_e_max = deg2rad( 10.5);
a.ctrl.delta_a_min = deg2rad(-25.0);
a.ctrl.delta_a_max = deg2rad( 45.0);
a.ctrl.delta_r_min = deg2rad(-30.0);
a.ctrl.delta_r_max = deg2rad( 30.0);
a.ctrl.thrust_min  = 500;
a.ctrl.thrust_max  = 143200;
end