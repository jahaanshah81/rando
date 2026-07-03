clc;
clear;
close all;

%% =========================================================
% BUILD_REFERENCE_TRAJECTORY.m
%
% Constructs a reference trajectory directly from:
%
%   maneuver_library
%   transition_library
%
% using a specified maneuver path.
%
% OUTPUT:
%
%   ref_trajectory.mat
%   ref_time.mat
%
% which can later be loaded into your master script.
%
%% =========================================================

%% ================= LOAD LIBRARIES =================

load('maneuver_library.mat','maneuver_library');
load('transition_library.mat','transition_library');

%% ================= USER PATH =================

path = [6 22 7 21 6 23 9];

%% ================= INITIALIZE =================

X_ref = [];
T_ref = [];
T_ref = zeros(0,1);

t_offset = 0;

fprintf('Building reference trajectory...\n');

%% =========================================================
% MAIN CONCATENATION LOOP
%% =========================================================

for i = 1:length(path)

    %% -----------------------------------------------------
    % MANEUVER SEGMENT
    %% -----------------------------------------------------

    man = maneuver_library{path(i)};

    Xman = man.trajectory_states;

    % Convert to [N x 13] if stored as [13 x N]
    if size(Xman,1) < size(Xman,2)

        Xman = Xman';

    end

    %% -----------------------------------------------------
    % BUILD MANEUVER TIME
    %% -----------------------------------------------------

    if isfield(man,'trajectory_time')

        Tman_local = man.trajectory_time;

    elseif isfield(man,'T')

        Tman_local = man.T;

    else

        % fallback estimate
        dt_trim = 0.01;

        Tman_local = (0:size(Xman,1)-1)' * dt_trim;

    end

    Tman = Tman_local + t_offset;

    %% -----------------------------------------------------
    % APPEND MANEUVER
    %% -----------------------------------------------------

    if isempty(X_ref)

        X_ref = [X_ref; Xman];
        T_ref = [T_ref; Tman];

    else

        % remove duplicate boundary point
       %% Force proper orientation

T_ref = T_ref(:);
Tman  = Tman(:);

%% Append
%% -------------------------------------------------
% GLOBAL POSITION STITCHING
%% -------------------------------------------------

if ~isempty(X_ref)

    % Previous endpoint
    prev_pos = X_ref(end,11:13);

    % Current segment start
    curr_pos = Xman(1,11:13);

    % Offset required
    delta_pos = prev_pos - curr_pos;

    % Shift entire maneuver
    Xman(:,11:13) = Xman(:,11:13) + delta_pos;

end
X_ref = [X_ref; Xman(2:end,:)];

T_ref = [T_ref;
         Tman(2:end)];

    end

    t_offset = T_ref(end);

    %% -----------------------------------------------------
    % TRANSITION SEGMENT
    %% -----------------------------------------------------

    if i < length(path)

        src = path(i);
        dst = path(i+1);

        tr = transition_library(src,dst);

        %% -------------------------------------------------
        % FEASIBILITY CHECK
        %% -------------------------------------------------

        is_feasible = false;

        if isfield(tr,'feasible')

            if tr.feasible == 1

                is_feasible = true;

            end

        end

        if isfield(tr,'solver_status')

            status = lower(string(tr.solver_status));

            if contains(status,'acceptable') || ...
               contains(status,'solve_succeeded') || ...
               contains(status,'optimal')

                is_feasible = true;

            end

            if contains(status,'infeasible') || ...
               contains(status,'fail')

                is_feasible = false;

            end

        end

        if ~is_feasible

            error('Transition (%d -> %d) is infeasible.', ...
                src,dst);

        end

        %% -------------------------------------------------
        % TRANSITION STATES
        %% -------------------------------------------------

        Xtr = tr.X;

        % Convert to [N x 13]
        if size(Xtr,1) < size(Xtr,2)

            Xtr = Xtr';

        end

        %% -------------------------------------------------
        % TRANSITION TIME
        %% -------------------------------------------------

        Ttr_local = tr.T;

        Ttr = Ttr_local + t_offset;

        %% -------------------------------------------------
        % APPEND TRANSITION
        %% -------------------------------------------------

        %% Force proper orientation

        T_ref = T_ref(:);
        Ttr   = Ttr(:);

        %% -------------------------------------------------
% GLOBAL POSITION STITCHING
%% -------------------------------------------------

if ~isempty(X_ref)

    % Previous endpoint
    prev_pos = X_ref(end,11:13);

    % Transition start
    curr_pos = Xtr(1,11:13);

    % Offset required
    delta_pos = prev_pos - curr_pos;

    % Shift transition
    Xtr(:,11:13) = Xtr(:,11:13) + delta_pos;

end
        
        %% Append

        
        X_ref = [X_ref; Xtr(2:end,:)];
        
        T_ref = [T_ref;
                 Ttr(2:end)];

        t_offset = T_ref(end);

    end

end

%% =========================================================
% SAVE
%% =========================================================

save('ref_trajectory.mat','X_ref');
save('ref_time.mat','T_ref');

fprintf('\n====================================\n');
fprintf('REFERENCE TRAJECTORY BUILT\n');
fprintf('====================================\n');

fprintf('Total trajectory points : %d\n',size(X_ref,1));
fprintf('Total duration          : %.2f sec\n',T_ref(end));

fprintf('\nSaved:\n');
fprintf('  ref_trajectory.mat\n');
fprintf('  ref_time.mat\n');

%% =========================================================
% QUICK PLOT
%% =========================================================

figure('Color','w');

plot3(X_ref(:,11), ...
      X_ref(:,12), ...
     -X_ref(:,13), ...
      'LineWidth',1.5);

grid on;
% axis equal;

xlabel('East (m)');
ylabel('North (m)');
zlabel('Altitude (m)');

title('Reference Trajectory');
view(45,30);