function [T12, C21, all_T12, all_C21,slopeOfC] = refinenonisometric(T12, para,S1,S2,distanceType,useSlantedDiagonal)
%
%   [T12, C21, all_T12, all_C21,slopeOfC] = REFINENONISOMETRIC(T12, para,S1,S2,nIH,distanceType,useSlantedDiagonal) 
%   refines the functional map T12 between meshes S1 and S2 using a ZoomOut-style 
%   strategy. This implementation accounts for non-isometries by introducing 
%   slanted diagonals in the mapping.
%
%   I. Cuiral-Zueco (IEEE T-ASE, 2025)
%
%   Inputs:
%       T12: Initial mapping from source mesh vertices to target mesh vertices.
%       para: Structure containing refinement parameters:
%           para.k_init: Initial number of eigenvectors.
%           para.k_step: Step size for increasing the number of eigenvectors.
%           para.k_final: Final number of eigenvectors.
%       S1: Structure with source mesh.
%       S2: Structure with target mesh.
%       nIH: Number of initial harmonics.
%       distanceType: Distance metric (e.g., 'euclidean').
%       useSlantedDiagonal: Flag to enable slanted diagonal scaling.
%
%   Outputs:
%       T12: Refined mapping from source mesh vertices to target mesh vertices.
%       C21: Refined mapping matrix.
%       all_T12: Cell array containing the mapping at each refinement step (optional).
%       all_C21: Cell array containing the mapping matrix at each refinement step (optional).
%       slopeOfC: Scaling factor for non-isometric mapping.

if nargout > 2, all_T12 = {}; all_C21 = {}; end

evals1=S1.evals;
evals2=S2.evals;
B1_all=S1.evecs;
B2_all=S2.evecs;

Aratio=1;%regular method
if useSlantedDiagonal
    Aratio=sqrt(full(sum(S1.area)/sum(S2.area)));
    b=1;
end

slopeOfC=Aratio;
for k = [para.k_init : para.k_step : para.k_final para.k_final]

    for ii=1:3
        %     maxEvals2=max(evals2(1:k));
        if useSlantedDiagonal
            maxEvals1=max(evals1(1:k));
            maxEvals2=maxEvals1*Aratio;
            r=sum(evals2(:)<maxEvals2);
        else
            r=k;
        end
        B1 = [B1_all(:, 1:k)];%nm*km
        B2 = [B2_all(:, 1:r)];%nn*kn
        C21=(B1')*S1.A*B2(T12,:);
        T12 = knnsearch(B2*C21', B1,'Distance',distanceType,'NSMethod','kdtree');
        if nargout > 2, all_T12{end+1} = T12; all_C21{end+1} = C21;
        else
            break;
        end
    end

end

end
