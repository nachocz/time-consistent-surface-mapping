function     [T12,C21,S1,S2]=initialisefmap(S1,S2,nhinitial, nh,distanceType,useSlantedDiagonal,Cstep)
%
% INITIALISEFMAP Computes an initial surface mapping between two meshes.
%
%   [T12,C21,S1,S2] = INITIALISEFMAP(S1,S2,nhinitial, nh,distanceType,useSlantedDiagonal,Cstep) 
%   computes an initial surface mapping between two meshes, S1 and S2. 
%
%   I. Cuiral-Zueco (IEEE T-ASE, 2025)
%
%   Inputs:
%       S1: Structure with source mesh.
%       S2: Structure with target mesh.
%       nhinitial: Initial number of eigenvectors used for the mapping.
%       nh: Total number of eigenvectors to use.
%       distanceType: Distance metric (e.g., 'euclidean').
%       useSlantedDiagonal: Flag to enable slanted diagonal in the mapping.
%       Cstep: Step size for refinement of the mapping.
%
%   Outputs:
%       T12: Mapping from source mesh vertices to target mesh vertices.
%       C21: Functional map matrix.
%       S1: Source mesh with updated Laplacian basis.
%       S2: Target mesh with updated Laplacian basis.
%
%   Note:
%       This function provides a basic initialization. 
%       For more robust initialization, consider using iterative closest 
%       point (ICP) or other surface matching algorithms.

% compute laplacian basis
S1 = MESH.compute_LaplacianBasis(S1, nh);
S2 = MESH.compute_LaplacianBasis(S2, nh);
% obtain initial T12 and C21 by seeking embedding-consistent mapping
B1 = S1.evecs(:,1:nhinitial);
B2 = S2.evecs(:,1:nhinitial);

nInit=2000;
residualval=10e8;
for ii=1:nInit
    iiC21_ini=(diag(rand(1,nhinitial)-0.5));
    iiT12_ini = knnsearch(B2*iiC21_ini', B1,'Distance',distanceType,'NSMethod','exhaustive');
    extrinsicError=S1.VERT-mean(S1.VERT)-S2.VERT(iiT12_ini,:)+mean(S2.VERT(iiT12_ini,:));
    iiresidual=sum(vecnorm(extrinsicError,2,2));
    if iiresidual<residualval
        C21_ini=iiC21_ini;
        T12_ini=iiT12_ini;
        residualval=iiresidual;
    end
end
%% refine maps
para.k_init = nhinitial;
para.k_step = Cstep;
para.k_final = nh;
[T12, C21, ~, ~,~ ] = refinenonisometric(T12_ini, para,S1,S2,distanceType,useSlantedDiagonal);
end