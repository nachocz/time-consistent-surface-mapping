function [T12,C212,S1,S1ant]=previousstatefmap(S1,S1ant,nhinitial,nh,distanceType,useSlantedDiagonal,Cstep)
%
% PREVIOUSSTATEFMAP Computes a surface mapping between two consecutive 
%   shapes assuming small movements.
%
%   [T12,C212,S1,S1ant] = PREVIOUSSTATEFMAP(S1,S1ant,nhinitial,nh,distanceType,useSlantedDiagonal,Cstep) 
%   computes a surface mapping between the current shape (S1) and the 
%   previous shape (S1ant) under the assumption of small movements. 
%
%   I. Cuiral-Zueco (IEEE T-ASE, 2025)
%
%   Inputs:
%       S1: Structure with current mesh.
%       S1ant: Structure with previous mesh.
%       nhinitial: Initial number of eigenvectors used for the mapping.
%       nh: Total number of eigenvectors to use (truncated basis size).
%       distanceType: Distance metric (e.g., 'euclidean').
%       useSlantedDiagonal: Flag to enable slanted diagonal in the mapping.
%       Cstep: Step size for refinement of the mapping.
%
%   Outputs:
%       T12: Mapping from current mesh vertices to previous mesh vertices.
%       C212: Mapping matrix.
%       S1: Current mesh with updated Laplacian basis.
%       S1ant: Previous mesh.
%
%   Note:
%       This function assumes small movements between the two deforming shapes. 
%       It initializes the mapping using nearest neighbor search in the 
%       embedding space and then refines the mapping using 

%% compute eigenbasis of the new shape
if  ~isfield(S1,'evecs')
    b=1;
    S1 = MESH.compute_LaplacianBasis(S1,nh);
end
%% set the initial map from p2p distance in the embedding
T12_ini = knnsearch(S1ant.surface.VERT-mean(S1ant.surface.VERT),...
    S1.surface.VERT-mean(S1.surface.VERT));
%% refine map
para.k_init = 15;
para.k_step = Cstep;
para.k_final = nh;
[T12, C212, ~, ~,~] = refinenonisometric(T12_ini, para,S1,S1ant,distanceType,useSlantedDiagonal);
%% Update evecs consistently
B1=S1.evecs;
S1.evecs(:,1:size(S1.evecs*C212,2))=S1.evecs*C212;
end