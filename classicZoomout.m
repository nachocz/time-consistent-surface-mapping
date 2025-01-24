function     [T12,C21,S1,S2]=classicZoomout(S1,S2,nhinitial,nh,distanceType,useSlantedDiagonal,Cstep)
% Classic ZoomOut: Melzi, S., Ren, J., Rodola, E., Sharma, A., Wonka, P., & Ovsjanikov, M. (2019). 
% Zoomout: Spectral upsampling for efficient shape correspondence. arXiv preprint arXiv:1904.07865.
S1 = MESH.compute_LaplacianBasis(S1,nh);
S2 = MESH.compute_LaplacianBasis(S2,nh);

%%  initialise map
C21_ini=eye(nhinitial);
B1 = S1.evecs(:,1:nhinitial);
B2 = S2.evecs(:,1:nhinitial);
T12_ini = knnsearch(B2*C21_ini', B1,'Distance',distanceType,'NSMethod','exhaustive');

%% update from there
para.k_init = nhinitial;
para.k_step = Cstep;
para.k_final = nh;
[T12, C21, ~, ~,~] = refinenonisometric(T12_ini, para,S1,S2,distanceType,useSlantedDiagonal);

end