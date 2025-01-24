% This script replicates the surface mapping results from experiments 
% described in our publication:
%   I. Cuiral-Zueco and G. López-Nicolás, "Time Consistent Surface Mapping for 
%   Deformable Object Shape Control," IEEE Transactions on Automation 
%   Science and Engineering, doi: 10.1109/TASE.2025.3529180.

% The code loads mesh data acquired from real experiments. 
% The mesh data structures (S1 and S2) were generated from the raw segmented point cloud data
% as illustrated in the `objectsurfacemeshing.m` script.

% Note: 
% - We refer the reader to the accompanying experiment videos in the "data" folder for 
%   visual context of the deformable object shape control experiments.

% Experiment IDs:
%     1: Mexican hat
%     2: T-shaped noodle
%     3: Pillow
%     4: Foam rectangle
%     5: Foam free-shape (foam cutout)

close all; clear all;
% Select the desired experiments (e.g., experiments = [1 3 4]).

experiments = 1:5;
NEXP = length(experiments);

% Global Settings
set(groot, 'defaultTextInterpreter', 'latex');
addpath(genpath('.'));

% Plotting Parameters
stepTicks = 100;
tamText = 20;
tamText3 = 40;
viewPoint = [90, 10];

% Loop over experiments
for inexp = 1:NEXP
    % Clear variables (except experiment-related)
    clearvars -except experiments inexp NEXP stepTicks tamText tamText3 viewPoint;
    close all;

    nexp = experiments(inexp);
    experimentName = ['experiment_data_', num2str(nexp)];
    experimentDir = ['data/', experimentName];
    load([experimentDir], 'iS1', 'iS2');

    % Initialize variables
    iMax = size(iS1, 2);
    timeCost = zeros(2, iMax);
    p2perror = zeros(2, iMax);
    numberofnodes = zeros(1, iMax);

    % Create figure handle
    loopfig = figure('Position',[100 100 600 900],'Name',['Experiment ', num2str(nexp)]);

    % Loop over time iterations
    for ii = 1:iMax
        % Load mesh data from real experiments (acquired using an
        % RGB-D Intel Realsense D435 camera and using the process in
        % 'objectsurfacemshing.m').
        S2 = cell2mat(iS2(1)); %target shape data (constant mesh)
        S1 = cell2mat(iS1(ii)); %deforming shape data (varying mesh)
        Vt = S2.VERT;
        V = S1.VERT;

        % Set parameters
        distanceType = 'euclidean';
        nhinitial = 4;  % Initial size of map C before refinement
        %I_update in the paper:
        nhupdate = 5;  % Re-used sub-matrix size of C in our time-consistent approach
        nh = 30;       % Eigenfunction truncation index
        Cstep = 4;     % Refinement step

        % Compute Laplacian Basis
        S1 = MESH.compute_LaplacianBasis(S1, nh);
        S2 = MESH.compute_LaplacianBasis(S2, nh);
        numberofnodes(ii) = S1.nv; % For mesh size information (see Fig. 9 in the paper)

        % Classic Zoom-out
        tic;
        useSlantedDiagonal = 0;
        [T12zo, C21zo, ~, ~] = classicZoomout(S1, S2, nhinitial, nh, distanceType, useSlantedDiagonal, Cstep);
        timeCost(1, ii) = toc; % For ZoomOut time cost (see Fig. 9 in the paper)

        % Time-consistent surface mapping (ours)
        useSlantedDiagonal = 1;
        tic;
        S1 = MESH.compute_LaplacianBasis(S1, nh);
        if ii > 1
            [T122, C212, S1, ~] = previousstatefmap(S1, S1ant, nhinitial, nh, distanceType, useSlantedDiagonal, Cstep);
            T0zo_rc_sd = knnsearch(S1.evecs, S0zo_rc_sd.evecs(:, 1:size(S1.evecs, 2)), 'Distance', distanceType, 'NSMethod', 'kdtree');
            S1cons.VERT = S1.VERT(T0zo_rc_sd, :);
            [T12zo_rc_sd, C21zo_rc_sd, S1, S2] = updatefmap(S1, S2, nhinitial, nhupdate, nh, C21ant, distanceType, useSlantedDiagonal, Cstep);
            S2cons.VERT = S2.VERT(T12zo_rc_sd(T0zo_rc_sd), :);
            C21ant = C21zo_rc_sd;
            S1ant = S1;
        else
            [T12zo_rc_sd, C21zo_rc_sd, S1, S2] = initialisefmap(S1, S2, nhinitial, nh, distanceType, useSlantedDiagonal, Cstep);
            C21ant = C21zo_rc_sd;
            S1ant = S1;
            S0zo_rc_sd = S1;
            S1init = S1;
            T12init = T12zo_rc_sd;
        end
        timeCost(2, ii) = toc; % For our method's time cost (see Fig. 9 in the paper,
        % note that this values can change depending on your computer's power)

        % Compute shape errors
        p2perror(1, ii) = norm(vecnorm(V - Vt(T12zo, :), 2, 2))^2;
        p2perror(2, ii) = norm(vecnorm(V - Vt(T12zo_rc_sd, :), 2, 2))^2;

        % Plotting
        figure(loopfig)
        clf(loopfig);

        subplot(3, 2, 1);
        visualize_map_on_target(S1, S2, T12zo_rc_sd);
        title(['$\mathcal{M}(k=', num2str(ii), '),\,\Pi_{}(k=', num2str(ii), ')$, ours'], 'Interpreter', 'latex', 'FontSize', tamText3 / 3);
        axis equal;
        view(viewPoint);

        subplot(3, 2, 2);
        visualize_map_on_source(S1, S2, T12zo_rc_sd);
        title(['$\mathcal{N}$ (target shape)'], 'Interpreter', 'latex', 'FontSize', tamText3 / 3);
        axis equal;
        view(viewPoint);

        subplot(3, 2, 3);
        visualize_map_on_target(S1, S2, T12zo);
        title(['$\mathcal{M}(k=', num2str(ii), '),\,\Pi_{\mathrm{NTC}}(k=', num2str(ii), ')$'], 'Interpreter', 'latex', 'FontSize', tamText3 / 3);
        axis equal;
        view(viewPoint);

        subplot(3, 2, 4);
        yyaxis right;
        plot(p2perror(1, 1:ii), 'color',[0.5 0.5 1], 'linestyle', '--');
        xlim([1 iMax]);
        current_ylim = ylim;
        ylim([0 current_ylim(2)]);
        ylabel('NTC shape error');
        ax = gca;
        xlabel('$k$');
        ax.YColor = 'b';

        yyaxis left;
        plot(p2perror(2, 1:ii), 'r','LineWidth',2);
        xlim([1 iMax]);
        current_ylim = ylim;
        ylim([0 current_ylim(2)]);
        ylabel('Our shape error');
        ax.YColor = 'r';

        subplot(3,2,5)
        title('Time cost')
        plot(timeCost(1,2:ii), '--b', 'LineWidth', 1.5); % ZoomOut: blue dashed
        hold on;
        plot(timeCost(2,2:ii), 'r', 'LineWidth', 1.5); % Ours: red
        xlim([1 iMax]);
        ylim([0 1]);
        hold off;
        xlabel('$k$');
        ylabel('Processing time cost [s]');
        legend('ZoomOut', 'Ours');

        subplot(3,2,6)
        title('Processed mesh size')
        plot(numberofnodes(1:ii),'black')
        xlabel('$k$');
        ylabel('No. mesh nodes');
        current_ylim = ylim;
        ylim([0 current_ylim(2)*1.1]);
        xlim([1 iMax]);


        % Print progress
        fprintf('Sequence completion [%3.2f%%]\r', 100 * ii / iMax);
    end
end
close all;