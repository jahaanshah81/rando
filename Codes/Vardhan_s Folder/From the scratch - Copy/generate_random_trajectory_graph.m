clc;
clear;
close all;

%% =========================================================
% LOAD LIBRARIES
% ==========================================================

load('maneuver_library.mat','maneuver_library');
load('transition_library.mat','transition_library');

N = length(maneuver_library);

%% =========================================================
% BUILD FEASIBLE ADJACENCY LIST
% ==========================================================

adjacency = cell(N,1);

for i = 1:N

    neighbors = [];

    for j = 1:N

        tr = transition_library(i,j);

        if isempty(tr)
            continue;
        end

        feasible = false;

        % Standard feasible flag
        if isfield(tr,'feasible')

            if tr.feasible == 1
                feasible = true;
            end

        end

        % Accept IPOPT acceptable solutions
        if isfield(tr,'solver_status')

            status = lower(string(tr.solver_status));

            if contains(status,'Acceptable') || ...
               contains(status,'solve_succeeded') || ...
               contains(status,'optimal')

                feasible = true;
            end

            if contains(status,'infeasible') || ...
               contains(status,'fail')

                feasible = false;
            end
        end

        if feasible

            neighbors(end+1) = j;

        end

    end

    adjacency{i} = neighbors;

end

%% =========================================================
% FIND VALID START NODE
% ==========================================================

valid_nodes = [];

for i = 1:N

    if ~isempty(adjacency{i})

        valid_nodes(end+1) = i;

    end

end

%% =========================================================
% RANDOM PATH GENERATION
% ==========================================================

num_segments = 10;

current_node = valid_nodes(randi(length(valid_nodes)));

path = current_node;

fprintf('\n====================================\n');
fprintf('RANDOM FEASIBLE MANEUVER PATH\n');
fprintf('====================================\n');

fprintf('\nStart Node : %d\n',current_node);
fprintf('Type       : %s\n', ...
    maneuver_library{current_node}.maneuver_type);

for k = 1:num_segments

    neighbors = adjacency{current_node};

    if isempty(neighbors)

        fprintf('\nNo feasible outgoing transitions.\n');
        break;

    end

    % Random next node
    next_node = neighbors(randi(length(neighbors)));

    % Store path
    path(end+1) = next_node;

    fprintf('\n%d --> %d',current_node,next_node);

    fprintf('   (%s)', ...
        maneuver_library{next_node}.maneuver_type);

    current_node = next_node;

end

%% =========================================================
% FINAL PATH
% ==========================================================

fprintf('\n\n====================================\n');
fprintf('FINAL NODE PATH\n');
fprintf('====================================\n');

disp(path);