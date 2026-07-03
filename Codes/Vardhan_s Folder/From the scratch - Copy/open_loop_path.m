clc; clear;

%% =========================================================================
%% master_trajectory.m  —  Open-Loop Simulation from Precomputed Library
%%
%% Workflow:
%%   1. Takes a path (sequence of primitive indices) as input
%%   2. Stitches trim controls + precomputed transition controls into one
%%      continuous [T_cmd, U_cmd] sequence  (no OCP at runtime)
%%   3. Runs a single forward RK4 open-loop simulation from x0
%%   4. Plots 3D trajectory, control inputs, and aerodynamic states
%% =========================================================================

%% ================= LOAD DATA =================
load maneuver_library.mat
load transition_library.mat
aircraft = get_f18_harv_parameters();


D1 = load('ref_trajectory.mat');
D2 = load('ref_time.mat');
% D3 = load('ref_controls.mat'); % Ensure this matches the filename you saved!

ref_X = D1.X_ref;
ref_T = D2.T_ref;

%% ================= USER INPUT =================
path = [6 22 7 21 6 23 9];



%% ================= VALIDATE PATH =================
fprintf('Validating path transitions against library...\n');

for i = 1:length(path)-1

    src = path(i);
    dst = path(i+1);

    tr = transition_library(src,dst);

    is_feasible = false;

    %% -----------------------------------------------------
    % Standard feasible flag
    %% -----------------------------------------------------

    if isfield(tr,'feasible')

        if tr.feasible == 1
            is_feasible = true;
        end

    end

    %% -----------------------------------------------------
    % Accept IPOPT acceptable solutions
    %% -----------------------------------------------------

    if isfield(tr,'solver_status')

        status = lower(string(tr.solver_status));

        if contains(status,'acceptable') || ...
           contains(status,'solve_succeeded') || ...
           contains(status,'optimal')

            is_feasible = true;

        end

        % Explicit rejection
        if contains(status,'infeasible') || ...
           contains(status,'fail') || ...
           contains(status,'diverged')

            is_feasible = false;

        end

    end

    %% -----------------------------------------------------
    % Final validation
    %% -----------------------------------------------------

    if ~is_feasible

        error(['No converged transition stored for pair (%d -> %d).\n' ...
               'Transition is infeasible or solver failed.'], ...
               src,dst);

    end

    fprintf('  %d -> %d : OK  (Tf = %.1f s,  cost = %.4g)\n', ...
        src, dst, tr.Tf, tr.cost);

end

fprintf('\n');

%% ================= BUILD CONTROL SEQUENCE =================
% Concatenate trim and transition controls into one time-stamped matrix.
% Trim segments use a fixed dt=0.01s (constant controls).
% Transition segments use the stored time vector from the OCP.
% First point of each new segment is dropped to avoid duplicate timestamps
% at stitch boundaries.

fprintf('Building concatenated control sequence...\n');

T_cmd        = [];   % (Ntotal x 1) time vector
U_cmd        = [];   % (Ntotal x 4) [thrust, delta_e, delta_a, delta_r]
segment_type = [];   % (Ntotal x 1)  1 = trim,  2 = transition

t_offset = 0;
dt_trim  = 0.01;     % [s] fixed step for trim segments

for i = 1:length(path)

    %% --- Trim Segment ---
    man    = maneuver_library{path(i)};
    Tf_man = man.duration;
    u_trim = [man.trim_controls.thrust;
              man.trim_controls.delta_e;
              man.trim_controls.delta_a;
              man.trim_controls.delta_r]';           % 1x4

    N_trim   = round(Tf_man / dt_trim) + 1;
    T_trim   = linspace(0, Tf_man, N_trim)' + t_offset;
    U_trim   = repmat(u_trim, N_trim, 1);            % constant over segment

    % Drop first point if this is not the very first segment (avoid duplicate)
    if isempty(T_cmd)
        T_cmd        = [T_cmd;        T_trim];
        U_cmd        = [U_cmd;        U_trim];
        segment_type = [segment_type; ones(N_trim, 1)];
    else
        T_cmd        = [T_cmd;        T_trim(2:end)];
        U_cmd        = [U_cmd;        U_trim(2:end,:)];
        segment_type = [segment_type; ones(N_trim-1, 1)];
    end

    t_offset = T_cmd(end);

    %% --- Transition Segment (library lookup) ---
    if i < length(path)
        src   = path(i);
        dst   = path(i+1);
        trans = transition_library(src, dst);

        % trans.T starts at 0 — offset to current time, drop duplicate first point
        T_tr  = trans.T(2:end) + t_offset;   % (N_tr-1) x 1
        U_tr  = trans.U(2:end, :);           % (N_tr-1) x 4

        T_cmd        = [T_cmd;        T_tr];
        U_cmd        = [U_cmd;        U_tr];
        segment_type = [segment_type; 2*ones(length(T_tr), 1)];

        t_offset = T_cmd(end);
    end
end

% Safety: remove any remaining duplicate timestamps
[T_cmd, ia] = unique(T_cmd, 'stable');
U_cmd        = U_cmd(ia, :);
segment_type = segment_type(ia);

N_sim = length(T_cmd);
fprintf('  Total simulation time : %.2f s\n',  T_cmd(end));
fprintf('  Total time steps      : %d\n\n', N_sim);

%% ================= INITIAL STATE =================
x0 = maneuver_library{path(1)}.trim_state;

%% ================= OPEN-LOOP SIMULATION =================
fprintf('Running open-loop RK4 simulation...\n');

X = zeros(N_sim, 13);
X(1,:) = x0';

for k = 1:N_sim-1
    dt = T_cmd(k+1) - T_cmd(k);
    tk = T_cmd(k);
    xk = X(k,:)';

    % Piecewise-constant controls
    c.thrust  = U_cmd(k,1);
    c.delta_e = U_cmd(k,2);
    c.delta_a = U_cmd(k,3);
    c.delta_r = U_cmd(k,4);

    % RK4
    k1 = F18_HARV_dynamics(tk,            xk,              c, aircraft);
    k2 = F18_HARV_dynamics(tk + 0.5*dt,   xk + 0.5*dt*k1, c, aircraft);
    k3 = F18_HARV_dynamics(tk + 0.5*dt,   xk + 0.5*dt*k2, c, aircraft);
    k4 = F18_HARV_dynamics(tk + dt,       xk + dt*k3,     c, aircraft);

    X(k+1,:) = (xk + dt/6*(k1 + 2*k2 + 2*k3 + k4))';

    % Normalise quaternion each step to prevent drift
    q_norm        = X(k+1, 7:10);
    X(k+1, 7:10)  = q_norm / norm(q_norm);
end

fprintf('Simulation complete.\n\n');
%% ================= POST-PROCESSING =================
% Convert quaternion trajectory back to Euler angles for plotting
[Phi_deg, Theta_deg, Psi_deg] = quat_to_euler(X(:, 7:10));



%% ================= PLOT: 3D TRAJECTORY =================
% Insert NaN breaks at segment-type boundaries so trim and transition
% appear as visually separate coloured lines.
X_plot = X;
for k = 2:length(segment_type)
    if segment_type(k) ~= segment_type(k-1)
        X_plot(k, 11:13) = NaN;
    end
end

idx_m = segment_type == 1;
idx_t = segment_type == 2;

figure('Color','w','Name','3D Position Trajectory');
hold on; grid on; axis equal;


% Plot trajectory
plot3(ref_X(:,11),ref_X(:,12),-ref_X(:,13),'--r', 'LineWidth', 1.5, 'DisplayName', 'Reference');

plot3(X_plot(idx_m,11), X_plot(idx_m,12), -X_plot(idx_m,13), ...
      'b', 'LineWidth', 2, 'DisplayName', 'Trim');
plot3(X_plot(idx_t,11), X_plot(idx_t,12), -X_plot(idx_t,13), ...
      'r', 'LineWidth', 2, 'DisplayName', 'Transition');
plot3(X(1,11),   X(1,12),   -X(1,13),   'go', ...
      'MarkerFaceColor','g', 'MarkerSize',10, 'DisplayName','Start');
plot3(X(end,11), X(end,12), -X(end,13), 'rs', ...
      'MarkerFaceColor','r', 'MarkerSize',10, 'DisplayName','End');

xlabel('East (m)'); ylabel('North (m)'); zlabel('Altitude (m)');
% axis([0 14000 -7000 7000 500 7500]);
title('F-18 HARV Open-Loop 3D Trajectory');
legend; view(45,30);

%% ================= PLOT: CONTROL INPUTS =================
figure('Color','w','Name','Control Inputs');

subplot(2,2,1);
plot(T_cmd, U_cmd(:,1), 'k', 'LineWidth',1.2); grid on;
xlabel('Time (s)'); ylabel('Thrust (N)'); title('Thrust');

subplot(2,2,2);
plot(T_cmd, rad2deg(U_cmd(:,2)), 'b', 'LineWidth',1.2); grid on;
xlabel('Time (s)'); ylabel('\delta_e (deg)'); title('Elevator');

subplot(2,2,3);
plot(T_cmd, rad2deg(U_cmd(:,3)), 'r', 'LineWidth',1.2); grid on;
xlabel('Time (s)'); ylabel('\delta_a (deg)'); title('Aileron');

subplot(2,2,4);
plot(T_cmd, rad2deg(U_cmd(:,4)), 'm', 'LineWidth',1.2); grid on;
xlabel('Time (s)'); ylabel('\delta_r (deg)'); title('Rudder');

sgtitle('Open-Loop Control Sequence');

%% ================= PLOT: AERODYNAMIC STATES =================
u_body  = X(:,1);  v_body = X(:,2);  w_body = X(:,3);
V_total   = sqrt(u_body.^2 + v_body.^2 + w_body.^2);
alpha_deg = atan2(w_body, u_body) * (180/pi);
beta_deg  = asin(v_body ./ V_total) * (180/pi);
q_deg_s   = X(:,5) * (180/pi);

figure('Color','w','Name','Aerodynamic States');

subplot(3,1,1);
plot(T_cmd, alpha_deg, 'b', 'LineWidth',1.5); grid on;
ylabel('\alpha (deg)'); title('Angle of Attack');

subplot(3,1,2);
plot(T_cmd, beta_deg, 'r', 'LineWidth',1.5); grid on;
ylabel('\beta (deg)'); title('Sideslip Angle');

subplot(3,1,3);
plot(T_cmd, q_deg_s, 'k', 'LineWidth',1.5); grid on;
ylabel('q (deg/s)'); xlabel('Time (s)'); title('Pitch Rate');

sgtitle('Aerodynamic States — Open-Loop Response');
%% -------- 3. Euler Angles Plot --------
figure('Color','w','Name','Euler Angles');
subplot(3,1,1);
plot(T_cmd, Phi_deg, 'r', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Roll (deg)'); grid on;
title('Roll Angle (\phi)');

subplot(3,1,2);
plot(T_cmd, Theta_deg, 'g', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Pitch (deg)'); grid on;
title('Pitch Angle (\theta)');

subplot(3,1,3);
plot(T_cmd, Psi_deg, 'b', 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Yaw (deg)'); grid on;
title('Yaw Angle (\psi)');
sgtitle('Aircraft Attitude (Converted from Quaternions)');

%% =========================================================================
%% FUNCTIONS
%% =========================================================================

function sd = F18_HARV_dynamics(~, state, ctrl, a)
u=state(1); v=state(2); w=state(3);
p=state(4); q=state(5); r=state(6);
q0=state(7);q1=state(8);q2=state(9);q3=state(10);

T=ctrl.thrust; de=ctrl.delta_e; da=ctrl.delta_a; dr=ctrl.delta_r;

rho  = 1.225;
V    = max(norm([u v w]), 1);
qbar = 0.5*rho*V^2;
alp  = atan2(w, u);
bet  = asin(v / V);

CL = a.aero.CL_alpha * alp;
CD = a.aero.CD0 + a.aero.k*CL^2 + a.aero.k_alpha2*alp^2;
CY = a.aero.CY_beta*bet + a.aero.CY_dr*dr;

Lift  = qbar*a.aero.S_ref*CL;
Drag  = qbar*a.aero.S_ref*CD;
Yside = qbar*a.aero.S_ref*CY;

Cwb = [cos(alp)*cos(bet),  sin(bet), sin(alp)*cos(bet);
      -cos(alp)*sin(bet),  cos(bet),-sin(alp)*sin(bet);
      -sin(alp),           0,        cos(alp)          ];
Fb  = Cwb' * [T-Drag; Yside; -Lift];

Rbi = [1-2*(q2^2+q3^2),   2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2);
        2*(q1*q2+q0*q3),  1-2*(q1^2+q3^2), 2*(q2*q3-q0*q1);
        2*(q1*q3-q0*q2),  2*(q2*q3+q0*q1), 1-2*(q1^2+q2^2)];

Fg = a.mass * (Rbi' * [0;0;9.81]);
X=Fb(1)+Fg(1); Y=Fb(2)+Fg(2); Z=Fb(3)+Fg(3);

u_d = X/a.mass + r*v - q*w;
v_d = Y/a.mass + p*w - r*u;
w_d = Z/a.mass + q*u - p*v;

I3  = [a.Ixx,0,-a.Ixz; 0,a.Iyy,0; -a.Ixz,0,a.Izz];
om  = [p;q;r];

Cm = a.aero.Cm_alpha*alp + a.aero.Cm_q*(a.aero.c_bar/(2*V))*q + a.aero.Cm_delta_e*de;
Cl = a.aero.Cl_beta*bet  + a.aero.Cl_p*(a.aero.b/(2*V))*p   + a.aero.Cl_r*(a.aero.b/(2*V))*r   + a.aero.Cl_delta_a*da + a.aero.Cl_delta_r*dr;
Cn = a.aero.Cn_beta*bet  + a.aero.Cn_p*(a.aero.b/(2*V))*p   + a.aero.Cn_r*(a.aero.b/(2*V))*r   + a.aero.Cn_delta_a*da + a.aero.Cn_delta_r*dr;

Lm = qbar*a.aero.S_ref*a.aero.b*Cl;
Mm = qbar*a.aero.S_ref*a.aero.c_bar*Cm;
Nm = qbar*a.aero.S_ref*a.aero.b*Cn;

om_d = I3 \ ([Lm;Mm;Nm] - cross(om, I3*om));

Om = [0,-p,-q,-r; p,0,r,-q; q,-r,0,p; r,q,-p,0];
qd = 0.5 * Om * [q0;q1;q2;q3];

vi = Rbi * [u;v;w];
sd = [u_d; v_d; w_d; om_d; qd; vi];
end

%% ===================================================================
%  AIRCRAFT PARAMETERS  (F-18A / HARV, corrected values)
% ====================================================================
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


function [phi, theta, psi] = quat_to_euler(Q)
    % Converts an Nx4 array of quaternions into Euler angles (in degrees)
    % Q = [q0, q1, q2, q3] where q0 is the scalar part
    q0 = Q(:,1); q1 = Q(:,2); q2 = Q(:,3); q3 = Q(:,4);
    
    % Roll (phi)
    phi = atan2(2.*(q0.*q1 + q2.*q3), 1 - 2.*(q1.^2 + q2.^2));
    
    % Pitch (theta) - Added min/max clamping to prevent complex numbers from roundoff errors
    sinp = 2.*(q0.*q2 - q3.*q1);
    sinp = max(min(sinp, 1), -1); 
    theta = asin(sinp);
    
    % Yaw (psi)
    psi = atan2(2.*(q0.*q3 + q1.*q2), 1 - 2.*(q2.^2 + q3.^2));
    
    % Convert to degrees
    phi = rad2deg(phi);
    theta = rad2deg(theta);
    psi = rad2deg(psi);
end