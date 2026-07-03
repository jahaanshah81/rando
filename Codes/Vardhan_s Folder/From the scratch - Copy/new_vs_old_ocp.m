% =========================================================================
%  compare_ocp_trajectories.m
%
%  Side-by-side comparison of OLD vs NEW OCP formulations on the same
%  maneuver path. Runs both solvers for every transition in 'path',
%  stitches complete trajectories, then produces comparison figures.
%
%  OLD OCP  — min(time) + control effort + control rate,  free Tf,
%             thrust mixed with drag in wind frame (original formulation)
%
%  NEW OCP  — min(control effort) + state-progress + angular rate energy
%             + terminal cost,  free Tf,  thrust correctly in body X axis
%
%  OUTPUT FIGURES
%    Fig 1  — 3D trajectory overlay  (blue=trim, red=old OCP, green=new OCP)
%    Fig 2  — Per-transition overlay  (alpha, beta, q  for both OCPs)
%    Fig 3  — Control inputs comparison  (T, de, da, dr  for both OCPs)
%    Fig 4  — Summary bar chart  (Tf and objective value per transition)
% =========================================================================

clc; clear; close all;
import casadi.*

% ------------------------------------------------------------------
%  LOAD DATA
% ------------------------------------------------------------------
load maneuver_library.mat
aircraft = get_f18_harv_parameters();

% ------------------------------------------------------------------
%  USER SETTINGS
% ------------------------------------------------------------------
path    = [8 9];      % maneuver indices from library
Tf_min  = 5;           % free-Tf lower bound [s]
Tf_max  = 25;          % free-Tf upper bound [s]

% NEW OCP weights — adjust here and re-run to tune
new_opt.w_ctrl      = 1.0;
new_opt.w_prog      = 1.0;   % increase for more aggressive transitions
new_opt.w_rate      = 0.5;
new_opt.w_terminal  = 80.0;

% ------------------------------------------------------------------
%  STORAGE  (separate for OLD and NEW)
% ------------------------------------------------------------------
% Each row: [state(13)] per time step
X_old = [];  U_old = [];  T_old = [];  seg_old = [];
X_new = [];  U_new = [];  T_new = [];  seg_new = [];

% Per-transition diagnostics
n_trans     = length(path) - 1;
Tf_old_list = zeros(1, n_trans);
Tf_new_list = zeros(1, n_trans);
J_old_list  = zeros(1, n_trans);
J_new_list  = zeros(1, n_trans);
stat_old    = cell(1, n_trans);
stat_new    = cell(1, n_trans);

t_off_old = 0;
t_off_new = 0;
x0_old    = maneuver_library{path(1)}.trim_state;
x0_new    = maneuver_library{path(1)}.trim_state;

% ------------------------------------------------------------------
%  MAIN LOOP
% ------------------------------------------------------------------
for i = 1:length(path)

    man   = maneuver_library{path(i)};
    u_vec = ctrl_to_vec(man.trim_controls);

    % ---- Trim maneuver (identical for both paths) -----------------
    [Tm_o, Xm_o, Um_o] = sim_trim(x0_old, u_vec, man.duration, aircraft);
    [Tm_n, Xm_n, Um_n] = sim_trim(x0_new, u_vec, man.duration, aircraft);

    X_old = [X_old; Xm_o];  U_old = [U_old; Um_o];
    T_old = [T_old; Tm_o + t_off_old];
    seg_old = [seg_old; ones(size(Xm_o,1),1)];

    X_new = [X_new; Xm_n];  U_new = [U_new; Um_n];
    T_new = [T_new; Tm_n + t_off_new];
    seg_new = [seg_new; ones(size(Xm_n,1),1)];

    t_off_old = T_old(end);
    t_off_new = T_new(end);
    x_end_old = Xm_o(end,:)';
    x_end_new = Xm_n(end,:)';

    % ---- Transition (different OCP for each path) -----------------
    if i < length(path)
        man_next  = maneuver_library{path(i+1)};
        xref      = man_next.trim_state;
        u0_c      = ctrl_to_vec(man.trim_controls);
        uref_c    = ctrl_to_vec(man_next.trim_controls);

        fprintf('\n=== Transition %d -> %d ===\n', path(i), path(i+1));

        % OLD OCP
        fprintf('  [OLD OCP] solving...\n');
        [To, Xo, Uo, Tf_o, Jo, so] = ocp_old( ...
            x_end_old, xref, u0_c, uref_c, Tf_min, Tf_max, aircraft);
        Tf_old_list(i) = Tf_o;
        J_old_list(i)  = Jo;
        stat_old{i}    = so;
        fprintf('  [OLD OCP] Tf=%.2f s  J=%.4g  status=%s\n', Tf_o, Jo, so);

        X_old = [X_old; Xo];  U_old = [U_old; Uo];
        T_old = [T_old; To + t_off_old];
        seg_old = [seg_old; 2*ones(size(Xo,1),1)];
        t_off_old = T_old(end);
        x0_old = Xo(end,:)';

        % NEW OCP
        fprintf('  [NEW OCP] solving...\n');
        [Tn, Xn, Un, Tf_n, Jn, sn] = ocp_new( ...
            x_end_new, xref, u0_c, uref_c, Tf_min, Tf_max, aircraft, new_opt);
        Tf_new_list(i) = Tf_n;
        J_new_list(i)  = Jn;
        stat_new{i}    = sn;
        fprintf('  [NEW OCP] Tf=%.2f s  J=%.4g  status=%s\n', Tf_n, Jn, sn);

        X_new = [X_new; Xn];  U_new = [U_new; Un];
        T_new = [T_new; Tn + t_off_new];
        seg_new = [seg_new; 2*ones(size(Xn,1),1)];
        t_off_new = T_new(end);
        x0_new = Xn(end,:)';
    end
end

% ------------------------------------------------------------------
%  SAVE
% ------------------------------------------------------------------
save('traj_old.mat', 'X_old','U_old','T_old','seg_old');
save('traj_new.mat', 'X_new','U_new','T_new','seg_new');
fprintf('\nSaved traj_old.mat and traj_new.mat\n');

% ------------------------------------------------------------------
%  FIG 1 — 3D TRAJECTORY OVERLAY
% ------------------------------------------------------------------
figure('Name','Fig 1 — 3D Trajectory Comparison','Color','w', ...
       'Position',[40 40 1200 600],'NumberTitle','off');

titles_3d = {'OLD OCP', 'NEW OCP'};
data_3d   = {X_old, seg_old; X_new, seg_new};

for col = 1:2
    subplot(1,2,col); hold on; grid on; axis equal; view(30,20);
    X_ = data_3d{col,1};  seg_ = data_3d{col,2};

    % NaN-break at segment boundaries for clean rendering
    X_plot = X_;
    for k = 2:length(seg_)
        if seg_(k) ~= seg_(k-1), X_plot(k,11:13) = NaN; end
    end

    idx_m = seg_ == 1;
    idx_t = seg_ == 2;
    plot3(X_plot(idx_m,11), X_plot(idx_m,12), -X_plot(idx_m,13), ...
          'b', 'LineWidth', 2, 'DisplayName','Trim maneuver');
    plot3(X_plot(idx_t,11), X_plot(idx_t,12), -X_plot(idx_t,13), ...
          'r', 'LineWidth', 2.5, 'DisplayName','Transition');

    % Mark start/end of each segment
    bounds = find(diff(seg_) ~= 0);
    for b = bounds'
        plot3(X_(b,11), X_(b,12), -X_(b,13), 'ko', ...
              'MarkerFaceColor','k','MarkerSize',7,'HandleVisibility','off');
    end
    xlabel('X North [m]'); ylabel('Y East [m]'); zlabel('Altitude [m]');
    title(titles_3d{col},'FontSize',11,'FontWeight','bold');
    legend('Location','best','FontSize',9);
end
sgtitle(sprintf('3D Trajectory: path [%s]', num2str(path)), ...
        'FontSize',12,'FontWeight','bold');

% ------------------------------------------------------------------
%  FIG 2 — AERODYNAMICS COMPARISON (alpha, beta, q)
% ------------------------------------------------------------------
figure('Name','Fig 2 — Aerodynamics Comparison','Color','w', ...
       'Position',[40 680 1200 520],'NumberTitle','off');

aero_vars = {
    @(X) atan2(X(:,3), X(:,1)) * 180/pi,    '\alpha [deg]',    [-5 30];
    @(X) asin(X(:,2)./max(vecnorm(X(:,1:3),2,2),1)) * 180/pi, '\beta [deg]', [-15 15];
    @(X) X(:,5) * 180/pi,                    'q [deg/s]',       [-30 30];
};

for row = 1:3
    subplot(3,2,2*row-1); hold on; grid on;
    val = aero_vars{row,1}(X_old);
    plot_colored(T_old, val, seg_old, 'OLD OCP');
    ylabel(aero_vars{row,2},'FontSize',10);
    if row == 1, title('OLD OCP','FontSize',10,'FontWeight','bold'); end
    if row == 3, xlabel('Time [s]'); end
    if ~isnan(aero_vars{row,3}(1))
        yline(aero_vars{row,3}(1),'k:','LineWidth',1,'HandleVisibility','off');
        yline(aero_vars{row,3}(2),'k:','LineWidth',1,'HandleVisibility','off');
    end

    subplot(3,2,2*row); hold on; grid on;
    val = aero_vars{row,1}(X_new);
    plot_colored(T_new, val, seg_new, 'NEW OCP');
    ylabel(aero_vars{row,2},'FontSize',10);
    if row == 1, title('NEW OCP','FontSize',10,'FontWeight','bold'); end
    if row == 3, xlabel('Time [s]'); end
    if ~isnan(aero_vars{row,3}(1))
        yline(aero_vars{row,3}(1),'k:','LineWidth',1,'HandleVisibility','off');
        yline(aero_vars{row,3}(2),'k:','LineWidth',1,'HandleVisibility','off');
    end
end
sgtitle('Aerodynamic States: OLD vs NEW OCP', ...
        'FontSize',12,'FontWeight','bold');

% ------------------------------------------------------------------
%  FIG 3 — CONTROL INPUTS COMPARISON
% ------------------------------------------------------------------
figure('Name','Fig 3 — Control Inputs Comparison','Color','w', ...
       'Position',[40 40 1200 650],'NumberTitle','off');

ctrl_labels = {'Thrust [N]', '\delta_e [deg]', '\delta_a [deg]', '\delta_r [deg]'};
ctrl_scale  = [1, 180/pi, 180/pi, 180/pi];
ctrl_lim    = {[500 143200], [-24 10.5], [-25 45], [-30 30]};

for row = 1:4
    subplot(4,2,2*row-1); hold on; grid on;
    val = U_old(:,row) * ctrl_scale(row);
    plot_colored(T_old, val, seg_old, '');
    ylabel(ctrl_labels{row},'FontSize',9);
    yline(ctrl_lim{row}(1),'k:','LineWidth',0.8,'HandleVisibility','off');
    yline(ctrl_lim{row}(2),'k:','LineWidth',0.8,'HandleVisibility','off');
    if row == 1, title('OLD OCP','FontSize',10,'FontWeight','bold'); end
    if row == 4, xlabel('Time [s]'); end

    subplot(4,2,2*row); hold on; grid on;
    val = U_new(:,row) * ctrl_scale(row);
    plot_colored(T_new, val, seg_new, '');
    ylabel(ctrl_labels{row},'FontSize',9);
    yline(ctrl_lim{row}(1),'k:','LineWidth',0.8,'HandleVisibility','off');
    yline(ctrl_lim{row}(2),'k:','LineWidth',0.8,'HandleVisibility','off');
    if row == 1, title('NEW OCP','FontSize',10,'FontWeight','bold'); end
    if row == 4, xlabel('Time [s]'); end
end
sgtitle('Control Inputs: OLD vs NEW OCP (dashed = limits)', ...
        'FontSize',12,'FontWeight','bold');

% ------------------------------------------------------------------
%  FIG 4 — TRANSITION SUMMARY BAR CHART
% ------------------------------------------------------------------
if n_trans > 0
    figure('Name','Fig 4 — Transition Summary','Color','w', ...
           'Position',[40 40 800 400],'NumberTitle','off');
    lbl = arrayfun(@(k) sprintf('%d→%d', path(k), path(k+1)), ...
                   1:n_trans, 'UniformOutput',false);
    x_pos = 1:n_trans;

    subplot(1,2,1);
    bar_data = [Tf_old_list; Tf_new_list]';
    b = bar(x_pos, bar_data, 0.6);
    b(1).FaceColor = [0.8 0.2 0.2];  % red = old
    b(2).FaceColor = [0.2 0.7 0.3];  % green = new
    set(gca,'XTick',x_pos,'XTickLabel',lbl);
    ylabel('Transition time [s]');
    title('Transition Time');
    legend('OLD OCP','NEW OCP','Location','best');
    grid on;

    subplot(1,2,2);
    % Normalise objectives for fair comparison (different units)
    J_norm_old = J_old_list / max(max(abs(J_old_list)),1);
    J_norm_new = J_new_list / max(max(abs(J_new_list)),1);
    bar_data2 = [J_norm_old; J_norm_new]';
    b2 = bar(x_pos, bar_data2, 0.6);
    b2(1).FaceColor = [0.8 0.2 0.2];
    b2(2).FaceColor = [0.2 0.7 0.3];
    set(gca,'XTick',x_pos,'XTickLabel',lbl);
    ylabel('Normalised objective value');
    title('Objective Value (normalised within each OCP)');
    legend('OLD OCP','NEW OCP','Location','best');
    grid on;

    sgtitle('Per-Transition Diagnostics','FontSize',12,'FontWeight','bold');

    % Print text summary
    fprintf('\n%s\n', repmat('=',1,60));
    fprintf('TRANSITION SUMMARY\n');
    fprintf('%-12s %10s %10s %10s %10s\n', ...
        'Pair','Tf_OLD','Tf_NEW','J_OLD','J_NEW');
    fprintf('%s\n', repmat('-',1,60));
    for k = 1:n_trans
        fprintf('%-12s %10.2f %10.2f %10.4g %10.4g\n', ...
            lbl{k}, Tf_old_list(k), Tf_new_list(k), J_old_list(k), J_new_list(k));
    end
    fprintf('%s\n', repmat('=',1,60));
end


% =========================================================================
%  OLD OCP  (original formulation from master_trajectory.m)
%  Objective: min time + control effort + control rate
%  Dynamics:  thrust mixed with drag in wind frame  (original bug kept)
%  Free Tf:   yes (Tf_sym as decision variable)
% =========================================================================
function [T_out, X_opt, U_opt, Tf_opt, cost_val, status] = ...
        ocp_old(x0, xref, u0, uref, Tf_min, Tf_max, aircraft)
import casadi.*

N  = 100;  nx = 13;  nu = 4;
if dot(x0(7:10), xref(7:10)) < 0, xref(7:10) = -xref(7:10); end

x_sym    = SX.sym('x', nx);
u_sym    = SX.sym('u', nu);
f        = Function('f', {x_sym,u_sym}, {F18_casadi_old(x_sym,u_sym,aircraft)});

% Warm start — forward simulation
Tf_g = (Tf_min+Tf_max)/2;
X_ws = zeros(nx,N+1); X_ws(:,1)=x0;
dt_w = Tf_g/N;
for k=1:N
    frac=( k-1)/(N-1); uc=u0+frac*(uref-u0); xk=X_ws(:,k);
    k1=F18_casadi_old(xk,uc,aircraft); k2=F18_casadi_old(xk+dt_w/2*k1,uc,aircraft);
    k3=F18_casadi_old(xk+dt_w/2*k2,uc,aircraft); k4=F18_casadi_old(xk+dt_w*k3,uc,aircraft);
    X_ws(:,k+1)=xk+dt_w/6*(k1+2*k2+2*k3+k4);
    X_ws(7:10,k+1)=X_ws(7:10,k+1)/norm(X_ws(7:10,k+1));
end
X_ws(11,:)=linspace(x0(11),x0(11)+x0(1)*Tf_g,N+1);
X_ws(12,:)=linspace(x0(12),x0(12)+x0(2)*Tf_g,N+1);
U_ws=zeros(nu,N);
for s=1:nu, U_ws(s,:)=linspace(u0(s),uref(s),N); end

w={};w0=[];lbw=[];ubw=[];J=0;g={};lbg=[];ubg=[];
R      = diag([1/143200^2; 1/deg2rad(24)^2; 1/deg2rad(25)^2; 1/deg2rad(30)^2]);
R_rate = 1.5*R;
w_time = 0.5;

Tf_sym=SX.sym('Tf'); w={w{:},Tf_sym};
lbw=[lbw;Tf_min]; ubw=[ubw;Tf_max]; w0=[w0;Tf_g];
dt=Tf_sym/N;

Xk=SX.sym('X_0',nx); w={w{:},Xk};
lbw=[lbw;x0]; ubw=[ubw;x0]; w0=[w0;X_ws(:,1)];
U_prev=u0;

for k=0:N-1
    Uk=SX.sym(['U_' num2str(k)],nu); w={w{:},Uk};
    if k==0,     lbw=[lbw;u0];    ubw=[ubw;u0];
    elseif k==N-1, lbw=[lbw;uref]; ubw=[ubw;uref];
    else
        lbw=[lbw;aircraft.ctrl.thrust_min;aircraft.ctrl.delta_e_min;
                  aircraft.ctrl.delta_a_min;aircraft.ctrl.delta_r_min];
        ubw=[ubw;aircraft.ctrl.thrust_max;aircraft.ctrl.delta_e_max;
                  aircraft.ctrl.delta_a_max;aircraft.ctrl.delta_r_max];
    end
    w0=[w0;U_ws(:,k+1)];

    k1=f(Xk,Uk); k2=f(Xk+dt/2*k1,Uk);
    k3=f(Xk+dt/2*k2,Uk); k4=f(Xk+dt*k3,Uk);
    Xk_p=Xk+dt/6*(k1+2*k2+2*k3+k4);

    J=J+(Uk'*R*Uk)*dt+w_time*dt;
    if k>0, dU=Uk-U_prev; J=J+(dU'*R_rate*dU)*dt; end
    U_prev=Uk;

    Xk_n=SX.sym(['X_' num2str(k+1)],nx); w={w{:},Xk_n};
    lbw=[lbw;-inf(nx,1)]; ubw=[ubw;inf(nx,1)]; w0=[w0;X_ws(:,k+2)];
    g={g{:},Xk_n-Xk_p}; lbg=[lbg;zeros(nx,1)]; ubg=[ubg;zeros(nx,1)];
    qn=Xk_n(7)^2+Xk_n(8)^2+Xk_n(9)^2+Xk_n(10)^2;
    g={g{:},qn}; lbg=[lbg;1]; ubg=[ubg;1];
    un=Xk_n(1);vn=Xk_n(2);wn=Xk_n(3);
    Vn=sqrt(un^2+vn^2+wn^2+1e-6);
    g={g{:},vn/Vn}; lbg=[lbg;deg2rad(-10)]; ubg=[ubg;deg2rad(10)];
    Xk=Xk_n;
end

tol_v=[1.0;1.0;1.0; 0.02;0.02;0.02; 0.01;0.01;0.01;0.01];
g={g{:},Xk(1:10)-xref(1:10)};
lbg=[lbg;-tol_v]; ubg=[ubg;tol_v];

prob=struct('f',J,'x',vertcat(w{:}),'g',vertcat(g{:}));
opts=struct; opts.ipopt.max_iter=2000; opts.ipopt.tol=1e-4;
opts.ipopt.print_level=0; opts.ipopt.acceptable_tol=1e-3;
opts.ipopt.acceptable_iter=8; opts.ipopt.acceptable_obj_change_tol=1e-4;
solver=nlpsol('solver','ipopt',prob,opts);
sol=solver('x0',w0,'lbx',lbw,'ubx',ubw,'lbg',lbg,'ubg',ubg);
stats=solver.stats(); status=stats.return_status;
cost_val=full(sol.f);
w_opt=full(sol.x); Tf_opt=w_opt(1);

X_opt=zeros(N+1,nx); U_opt=zeros(N,nu); idx=2;
for k=1:N
    X_opt(k,:)=w_opt(idx:idx+nx-1)'; idx=idx+nx;
    U_opt(k,:)=w_opt(idx:idx+nu-1)'; idx=idx+nu;
end
X_opt(N+1,:)=w_opt(idx:idx+nx-1)';
T_out=linspace(0,Tf_opt,N+1)'; U_opt=[U_opt;U_opt(end,:)];
end


% =========================================================================
%  NEW OCP  (updated formulation)
%  Objective: control effort + STATE PROGRESS + angular rate + terminal
%  Dynamics:  thrust correctly in body X axis
%  Free Tf:   yes (time-scaling with Tf_sym)
%  Path constraints: alpha, Nz, speed  (re-enabled)
% =========================================================================
function [T_out, X_opt, U_opt, Tf_opt, cost_val, status] = ...
        ocp_new(x0, xref, u0, uref, Tf_min, Tf_max, aircraft, opt)
import casadi.*

% N  = max(60, round(((Tf_min+Tf_max)/2)*12));
N = 100;
nx = 13;  nu = 4;
if dot(x0(7:10),xref(7:10))<0, xref(7:10)=-xref(7:10); end

w_ctrl     = opt.w_ctrl;
w_prog     = opt.w_prog;
w_rate     = opt.w_rate;
w_terminal = opt.w_terminal;
w_time     = 0.5;   % kept same as old for fair Tf comparison

x_sym=SX.sym('x',nx); u_sym=SX.sym('u',nu);
f=Function('f',{x_sym,u_sym},{F18_casadi_new(x_sym,u_sym,aircraft)});

% -- Warm start (SLERP quaternion + simulated forward) --
Tf_g=(Tf_min+Tf_max)/2;
X_ws=zeros(nx,N+1); X_ws(:,1)=x0;
qs=x0(7:10)/norm(x0(7:10)); qe=xref(7:10)/norm(xref(7:10));
if dot(qs,qe)<0,qe=-qe;end
th_q=acos(min(abs(dot(qs,qe)),1));
dt_w=Tf_g/N;
for k=1:N
    tau=(k-1)/N;
    if sin(th_q)<1e-8, qi=(1-tau)*qs+tau*qe;
    else, qi=(sin((1-tau)*th_q)*qs+sin(tau*th_q)*qe)/sin(th_q); end
    X_ws(7:10,k)=qi/norm(qi);
    for s=1:6, X_ws(s,k+1)=x0(s)+(xref(s)-x0(s))*(k/N); end
end
X_ws(7:10,N+1)=qe;
% Position warm start from attitude-corrected velocity
X_ws(11:13,1)=x0(11:13);
for k=1:N
    q0_=X_ws(7,k);q1_=X_ws(8,k);q2_=X_ws(9,k);q3_=X_ws(10,k);
    Rbi_=[1-2*(q2_^2+q3_^2),2*(q1_*q2_-q0_*q3_),2*(q1_*q3_+q0_*q2_);
           2*(q1_*q2_+q0_*q3_),1-2*(q1_^2+q3_^2),2*(q2_*q3_-q0_*q1_);
           2*(q1_*q3_-q0_*q2_),2*(q2_*q3_+q0_*q1_),1-2*(q1_^2+q2_^2)];
    X_ws(11:13,k+1)=X_ws(11:13,k)+Rbi_*X_ws(1:3,k);
end
p0=x0(11:13); pf=xref(11:13);
for s=1:3
    rng=X_ws(10+s,end)-X_ws(10+s,1);
    if abs(rng)>1e-3
        X_ws(10+s,:)=p0(s)+(X_ws(10+s,:)-X_ws(10+s,1))*(pf(s)-p0(s))/rng;
    else
        X_ws(10+s,:)=linspace(p0(s),pf(s),N+1);
    end
end
U_ws=zeros(nu,N);
for s=1:nu, U_ws(s,:)=linspace(u0(s),uref(s),N); end

% -- Weights --
T_max=aircraft.ctrl.thrust_max; de_max=abs(aircraft.ctrl.delta_e_min);
da_max=aircraft.ctrl.delta_a_max; dr_max=aircraft.ctrl.delta_r_max;
R_ctrl=diag([1/T_max^2,1/de_max^2,1/da_max^2,1/dr_max^2]);
Q_prog=diag([1/30^2,1/5^2,1/15^2, 1/deg2rad(10)^2,1/deg2rad(10)^2,1/deg2rad(10)^2]);
Q_tv =w_terminal*Q_prog; Q_tq=w_terminal*200*eye(4);

w={};w0=[];lbw=[];ubw=[];J=0;g={};lbg=[];ubg=[];

Tf_sym=SX.sym('Tf'); w={w{:},Tf_sym};
lbw=[lbw;Tf_min]; ubw=[ubw;Tf_max]; w0=[w0;Tf_g];
J=J+w_time*Tf_sym;   % time cost (same as old for fair comparison)
dt=Tf_sym/N;

Xk=SX.sym('X_0',nx); w={w{:},Xk};
lbw=[lbw;x0]; ubw=[ubw;x0]; w0=[w0;X_ws(:,1)];

for k=0:N-1
    Uk=SX.sym(['U_' num2str(k)],nu); w={w{:},Uk};
    if k==0,       lbw=[lbw;u0];    ubw=[ubw;u0];
    elseif k==N-1, lbw=[lbw;uref];  ubw=[ubw;uref];
    else
        lbw=[lbw;aircraft.ctrl.thrust_min;aircraft.ctrl.delta_e_min;
                  aircraft.ctrl.delta_a_min;aircraft.ctrl.delta_r_min];
        ubw=[ubw;aircraft.ctrl.thrust_max;aircraft.ctrl.delta_e_max;
                  aircraft.ctrl.delta_a_max;aircraft.ctrl.delta_r_max];
    end
    w0=[w0;U_ws(:,min(k+1,size(U_ws,2)))];

    k1=f(Xk,Uk); k2=f(Xk+dt/2*k1,Uk);
    k3=f(Xk+dt/2*k2,Uk); k4=f(Xk+dt*k3,Uk);
    Xk_p=Xk+dt/6*(k1+2*k2+2*k3+k4);

    % [J1] Control effort
    J=J+w_ctrl*(Uk'*R_ctrl*Uk)*dt;
    % [J2] State progress toward xref
    dx6=Xk(1:6)-xref(1:6);
    J=J+w_prog*(dx6'*Q_prog*dx6)*dt;
    % [J3] Angular rate energy
    J=J+w_rate*(Xk(4)^2+Xk(5)^2+Xk(6)^2)*dt;

    Xk_n=SX.sym(['X_' num2str(k+1)],nx); w={w{:},Xk_n};
    lbw=[lbw;-inf(nx,1)]; ubw=[ubw;inf(nx,1)];
    w0=[w0;X_ws(:,min(k+2,size(X_ws,2)))];

    % Defect
    g={g{:},Xk_n-Xk_p}; lbg=[lbg;zeros(nx,1)]; ubg=[ubg;zeros(nx,1)];
    % Quaternion norm
    qn=Xk_n(7)^2+Xk_n(8)^2+Xk_n(9)^2+Xk_n(10)^2;
    g={g{:},qn}; lbg=[lbg;1]; ubg=[ubg;1];
    % Sideslip ±8 deg
    un=Xk_n(1);vn=Xk_n(2);wn=Xk_n(3);
    Vn=sqrt(un^2+vn^2+wn^2+1e-6);
    g={g{:},vn/Vn}; lbg=[lbg;deg2rad(-8)]; ubg=[ubg;deg2rad(8)];
    % Alpha -5 to +28 deg
    alp_n=atan2(wn,un);
    g={g{:},alp_n}; lbg=[lbg;deg2rad(-5)]; ubg=[ubg;deg2rad(28)];
    % Load factor -3g to +9g
    qbar_n=0.5*1.225*(un^2+vn^2+wn^2+1e-6);
    CL_n=aircraft.aero.CL_alpha*alp_n;
    Nz_n=qbar_n*aircraft.aero.S_ref*CL_n/(aircraft.mass*9.81);
    g={g{:},Nz_n}; lbg=[lbg;-3]; ubg=[ubg;9];
    % Min speed
    g={g{:},Vn}; lbg=[lbg;70]; ubg=[ubg;400];

    Xk=Xk_n;
end

% [J4] Terminal cost
dx_v=Xk(1:6)-xref(1:6); dx_q=Xk(7:10)-xref(7:10);
J=J+dx_v'*Q_tv*dx_v+dx_q'*Q_tq*dx_q;

% Terminal hard constraint (tighter than old)
tol_v=[1.5;1.0;1.5; 0.04;0.04;0.04; 0.025;0.025;0.025;0.025];
g={g{:},Xk(1:10)-xref(1:10)};
lbg=[lbg;-tol_v]; ubg=[ubg;tol_v];

prob=struct('f',J,'x',vertcat(w{:}),'g',vertcat(g{:}));
opts=struct; opts.ipopt.max_iter=3000; opts.ipopt.tol=1e-4;
opts.ipopt.print_level=0; opts.ipopt.acceptable_tol=1e-3;
opts.ipopt.acceptable_iter=10; opts.ipopt.acceptable_obj_change_tol=1e-4;
opts.ipopt.mu_strategy='adaptive';
% opts.ipopt.hessian_approximation='limited-memory';
solver=nlpsol('solver','ipopt',prob,opts);
sol=solver('x0',w0,'lbx',lbw,'ubx',ubw,'lbg',lbg,'ubg',ubg);
stats=solver.stats(); status=stats.return_status;
cost_val=full(sol.f);
w_opt=full(sol.x); Tf_opt=w_opt(1);

X_opt=zeros(N+1,nx); U_opt=zeros(N,nu); idx=2;
for k=1:N
    X_opt(k,:)=w_opt(idx:idx+nx-1)'; idx=idx+nx;
    U_opt(k,:)=w_opt(idx:idx+nu-1)'; idx=idx+nu;
end
X_opt(N+1,:)=w_opt(idx:idx+nx-1)';
T_out=linspace(0,Tf_opt,N+1)'; U_opt=[U_opt;U_opt(end,:)];
end


% =========================================================================
%  DYNAMICS — OLD (thrust mixed with drag in wind frame — original)
% =========================================================================
function xdot = F18_casadi_old(x, u_c, a)
import casadi.*
u_b=x(1);v=x(2);w=x(3);p=x(4);q=x(5);r=x(6);
q0=x(7);q1=x(8);q2=x(9);q3=x(10);
T=u_c(1);de=u_c(2);da=u_c(3);dr=u_c(4);
rho=1.225; V=sqrt(u_b^2+v^2+w^2+1e-6); qbar=0.5*rho*V^2;
alpha=atan2(w,u_b); beta=asin(v/V);
CL=a.aero.CL_alpha*alpha;
CD=a.aero.CD0+a.aero.k*CL^2+a.aero.k_alpha2*alpha^2;
CY=a.aero.CY_beta*beta+a.aero.CY_dr*dr;
L=qbar*a.aero.S_ref*CL; D=qbar*a.aero.S_ref*CD; Ys=qbar*a.aero.S_ref*CY;
Fw=[T-D;Ys;-L];  % <-- original: thrust mixed in wind frame
Cwb=[cos(alpha)*cos(beta),sin(beta),sin(alpha)*cos(beta);
    -cos(alpha)*sin(beta),cos(beta),-sin(alpha)*sin(beta);
    -sin(alpha),0,cos(alpha)];
Fb=Cwb'*Fw;
Rbi=[1-2*(q2^2+q3^2),2*(q1*q2-q0*q3),2*(q1*q3+q0*q2);
      2*(q1*q2+q0*q3),1-2*(q1^2+q3^2),2*(q2*q3-q0*q1);
      2*(q1*q3-q0*q2),2*(q2*q3+q0*q1),1-2*(q1^2+q2^2)];
Fg=a.mass*9.80665*(Rbi'*[0;0;1]);
Ftot=Fb+Fg;
ud=Ftot(1)/a.mass+r*v-q*w; vd=Ftot(2)/a.mass+p*w-r*u_b; wd=Ftot(3)/a.mass+q*u_b-p*v;
Cm=a.aero.Cm_alpha*alpha+a.aero.Cm_q*(a.aero.c_bar/(2*V))*q+a.aero.Cm_delta_e*de;
Cl=a.aero.Cl_beta*beta+a.aero.Cl_p*(a.aero.b/(2*V))*p+a.aero.Cl_r*(a.aero.b/(2*V))*r+a.aero.Cl_delta_a*da+a.aero.Cl_delta_r*dr;
Cn=a.aero.Cn_beta*beta+a.aero.Cn_p*(a.aero.b/(2*V))*p+a.aero.Cn_r*(a.aero.b/(2*V))*r+a.aero.Cn_delta_a*da+a.aero.Cn_delta_r*dr;
Lm=qbar*a.aero.S_ref*a.aero.b*Cl; Mm=qbar*a.aero.S_ref*a.aero.c_bar*Cm; Nm=qbar*a.aero.S_ref*a.aero.b*Cn;
I=[a.Ixx,0,-a.Ixz;0,a.Iyy,0;-a.Ixz,0,a.Izz]; om=[p;q;r];
om_d=I\([Lm;Mm;Nm]-cross(om,I*om));
Om=[0,-p,-q,-r;p,0,r,-q;q,-r,0,p;r,q,-p,0];
qd=0.5*Om*[q0;q1;q2;q3]; vi=Rbi*[u_b;v;w];
xdot=[ud;vd;wd;om_d;qd;vi];
end


% =========================================================================
%  DYNAMICS — NEW  (thrust correctly in body X axis)
% =========================================================================
function xdot = F18_casadi_new(x, u_c, a)
import casadi.*
u_b=x(1);v=x(2);w=x(3);p=x(4);q=x(5);r=x(6);
q0=x(7);q1=x(8);q2=x(9);q3=x(10);
T=u_c(1);de=u_c(2);da=u_c(3);dr=u_c(4);
rho=1.225; V=sqrt(u_b^2+v^2+w^2+1e-6); qbar=0.5*rho*V^2;
alpha=atan2(w,u_b); beta=asin(v/V);
CL=a.aero.CL_alpha*alpha;
CD=a.aero.CD0+a.aero.k*CL^2+a.aero.k_alpha2*alpha^2;
CY=a.aero.CY_beta*beta+a.aero.CY_dr*dr;
L=qbar*a.aero.S_ref*CL; D=qbar*a.aero.S_ref*CD; Ys=qbar*a.aero.S_ref*CY;
Cwb=[cos(alpha)*cos(beta),sin(beta),sin(alpha)*cos(beta);
    -cos(alpha)*sin(beta),cos(beta),-sin(alpha)*sin(beta);
    -sin(alpha),0,cos(alpha)];
Fb_aero=Cwb'*[-D;Ys;-L];   % aero only — no thrust
Fb_thrust=[T;0;0];           % thrust in body X only
Rbi=[1-2*(q2^2+q3^2),2*(q1*q2-q0*q3),2*(q1*q3+q0*q2);
      2*(q1*q2+q0*q3),1-2*(q1^2+q3^2),2*(q2*q3-q0*q1);
      2*(q1*q3-q0*q2),2*(q2*q3+q0*q1),1-2*(q1^2+q2^2)];
Fg=a.mass*9.80665*(Rbi'*[0;0;1]);
Ftot=Fb_aero+Fb_thrust+Fg;
ud=Ftot(1)/a.mass+r*v-q*w; vd=Ftot(2)/a.mass+p*w-r*u_b; wd=Ftot(3)/a.mass+q*u_b-p*v;
Cm=a.aero.Cm_alpha*alpha+a.aero.Cm_q*(a.aero.c_bar/(2*V))*q+a.aero.Cm_delta_e*de;
Cl=a.aero.Cl_beta*beta+a.aero.Cl_p*(a.aero.b/(2*V))*p+a.aero.Cl_r*(a.aero.b/(2*V))*r+a.aero.Cl_delta_a*da+a.aero.Cl_delta_r*dr;
Cn=a.aero.Cn_beta*beta+a.aero.Cn_p*(a.aero.b/(2*V))*p+a.aero.Cn_r*(a.aero.b/(2*V))*r+a.aero.Cn_delta_a*da+a.aero.Cn_delta_r*dr;
Lm=qbar*a.aero.S_ref*a.aero.b*Cl; Mm=qbar*a.aero.S_ref*a.aero.c_bar*Cm; Nm=qbar*a.aero.S_ref*a.aero.b*Cn;
I=[a.Ixx,0,-a.Ixz;0,a.Iyy,0;-a.Ixz,0,a.Izz]; om=[p;q;r];
om_d=I\([Lm;Mm;Nm]-cross(om,I*om));
Om=[0,-p,-q,-r;p,0,r,-q;q,-r,0,p;r,q,-p,0];
qd=0.5*Om*[q0;q1;q2;q3]; vi=Rbi*[u_b;v;w];
xdot=[ud;vd;wd;om_d;qd;vi];
end


% =========================================================================
%  TRIM SIMULATOR
% =========================================================================
function [T, X, U] = sim_trim(x0, u_trim, Tf, aircraft)
dt=0.01; N=round(Tf/dt)+1;
X=zeros(N,13); U=zeros(N,4); T=linspace(0,Tf,N)';
X(1,:)=x0';
for k=1:N-1
    xk=X(k,:)';
    k1=F18_casadi_new(xk,u_trim,aircraft);
    k2=F18_casadi_new(xk+0.5*dt*k1,u_trim,aircraft);
    k3=F18_casadi_new(xk+0.5*dt*k2,u_trim,aircraft);
    k4=F18_casadi_new(xk+dt*k3,u_trim,aircraft);
    X(k+1,:)=(xk+dt/6*(k1+2*k2+2*k3+k4))';
    q=X(k+1,7:10); X(k+1,7:10)=q/norm(q);
    U(k,:)=u_trim';
end
U(end,:)=u_trim';
end


% =========================================================================
%  PLOT HELPER — colour-codes trim (blue) vs transition (red/green)
% =========================================================================
function plot_colored(T, val, seg, label)
idx_m = seg==1;  idx_t = seg==2;
val_m = val; val_m(~idx_m)=NaN;
val_t = val; val_t(~idx_t)=NaN;
plot(T, val_m, 'b',  'LineWidth',1.5,'DisplayName','Trim');
plot(T, val_t, 'r',  'LineWidth',2.0,'DisplayName','Transition');
end


% =========================================================================
%  CONTROL STRUCT → VECTOR
% =========================================================================
function u_vec = ctrl_to_vec(tc)
if isfield(tc,'delta_stab')
    u_vec=[tc.thrust;tc.delta_stab;tc.delta_ail;tc.delta_rud];
elseif isfield(tc,'delta_e')
    u_vec=[tc.thrust;tc.delta_e;tc.delta_a;tc.delta_r];
else
    error('Unknown control field names.'); end
end


% =========================================================================
%  AIRCRAFT PARAMETERS  (linear model)
% =========================================================================
function a = get_f18_harv_parameters()
a.mass=15096.5; a.Ixx=31183.6; a.Iyy=205127.5; a.Izz=230432.1; a.Ixz=4028.0;
a.aero.S_ref=37.16; a.aero.c_bar=3.511; a.aero.b=11.404;
a.aero.CL_alpha=5.0; a.aero.CD0=0.020; a.aero.k=0.080; a.aero.k_alpha2=0.300;
a.aero.Cm_alpha=-0.45; a.aero.Cm_q=-4.50; a.aero.Cm_delta_e=-0.50;
a.aero.Cl_beta=-0.080; a.aero.Cl_p=-0.300; a.aero.Cl_r=0.050;
a.aero.Cl_delta_a=-0.150; a.aero.Cl_delta_r=0.050;
a.aero.Cn_beta=0.080; a.aero.Cn_p=-0.030; a.aero.Cn_r=-0.150;
a.aero.Cn_delta_a=-0.010; a.aero.Cn_delta_r=-0.120;
a.aero.CY_beta=-0.730; a.aero.CY_dr=0.200;
a.ctrl.delta_e_min=deg2rad(-24.0); a.ctrl.delta_e_max=deg2rad(10.5);
a.ctrl.delta_a_min=deg2rad(-25.0); a.ctrl.delta_a_max=deg2rad(45.0);
a.ctrl.delta_r_min=deg2rad(-30.0); a.ctrl.delta_r_max=deg2rad(30.0);
a.ctrl.thrust_min=500; a.ctrl.thrust_max=143200;
end