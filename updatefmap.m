function [T12,C21,S1,S2]=updatefmap(S1,S2,nhinitial,nhupdate,nh,C21,distanceType,useSlantedDiagonal,Cstep)
%
% UPDATEFMAP Updates a surface mapping between two meshes.
%
%   [T12,C21,S1,S2] = UPDATEFMAP(S1,S2,nhinitial,nhupdate,nh,C21,distanceType,useSlantedDiagonal,Cstep) 
%   updates a surface mapping between two meshes, S1 and S2, using the 
%   previous mapping (C21) to improve efficiency and time-consistency.
%
%   I. Cuiral-Zueco (IEEE T-ASE, 2025)
%
%   Inputs:
%       S1: Structure with source mesh.
%       S2: Structure with target mesh.
%       nhinitial: Initial number of eigenvectors used for the mapping.
%       nhupdate: Number of eigenvectors to use for the initial mapping update.
%       nh: Total number of eigenvectors to use.
%       C21: Previous mapping matrix.
%       distanceType: Distance metric (e.g., 'euclidean').
%       useSlantedDiagonal: Flag to enable slanted diagonal in the mapping.
%       Cstep: Step size for refinement of the mapping.
%
%   Outputs:
%       T12: Updated mapping from source mesh vertices to target mesh vertices.
%       C21: Updated mapping matrix.
%       S1: Source mesh with updated Laplacian basis (if necessary).
%       S2: Target mesh with updated Laplacian basis (if necessary).
%
%   Note:
%       This function utilizes the previous mapping (C21) to initialize 
%       the current mapping, improving efficiency and time-consistency. 
%       If the Laplacian basis of S1 is not already computed, it is 
%       computed within this function.

if  isempty(S1.evecs)
    S1 = MESH.compute_LaplacianBasis(S1,nh);
end
%% use previously computed map for initialisation
nhupdate=min(min(size(C21)),nhupdate); % in case slanted diagonal reduces size of C21 wrt nhupdate
C21_ini=C21(1:nhupdate,1:nhupdate);
B1 = S1.evecs(:,1:size(C21_ini,1));
B2 = S2.evecs(:,1:size(C21_ini,2));
T12_ini = knnsearch(B2*C21_ini', B1,'Distance',distanceType,'NSMethod','exhaustive');

%% update from there
para.k_init = nhupdate;
para.k_step = Cstep;
para.k_final = nh;
[T12, C21, ~, ~,~] = refinenonisometric(T12_ini, para,S1,S2,distanceType,useSlantedDiagonal);

end