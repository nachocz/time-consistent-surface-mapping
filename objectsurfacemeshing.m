% This script demonstrates the process of generating mesh structures (S1, and S2) 
% from point cloud data segmented (in CieLAB space) during real
% experiments.

% 1. Load and preprocess point cloud data
% - Load point cloud data with XYZ coordinates and RGB color information.
ptCloud = pointCloud(xyz(:,1:3), 'Color', rgb);

% - Downsample the point cloud using grid averaging.
gridStep = 0.025; % Set grid step size
ptCloudOut = pcdownsample(ptCloud, 'gridAverage', gridStep);

% 2. Generate surface mesh from downsampled point cloud
% - Use the ball-pivoting algorithm to create a surface mesh from the 
%   downsampled point cloud.
initradii = gridStep * 1.5; % Set initial radius for ball-pivoting
[mesh,radii] = pc2surfacemesh(ptCloudOut,"ball-pivot",initradii);

% 3. Mesh repair and structure creation
% - Repair the generated mesh to ensure its validity.
[V,F] = meshcheckrepair(mesh.Vertices, mesh.Faces);

% - Create the mesh structure (S1) for subsequent processing.
S1.TRIV = F; 
S1.VERT = V;

% - Create a 'surface' substructure for compatibility with the 
%   functional map library.
S1.surface.TRIV = S1.TRIV;
S1.surface.VERT = S1.VERT;
S1.surface.X = S1.VERT(:,1);
S1.surface.Y = S1.VERT(:,2);
S1.surface.Z = S1.VERT(:,3);

% - Store the number of vertices in the mesh.
S1.nv = length(S1.VERT); 