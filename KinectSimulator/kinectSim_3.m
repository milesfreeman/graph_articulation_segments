function [im1,im2,im3] = kinectSim_3(fname)
    load(fname, 'vertices', 'normals', 'faces');
    fprintf('\nProgress: [');
    im1 = cameraAngles(vertices, faces, normals, 7*pi/4 + pi, 0, 'left');
    fprintf('---------------------')
    im2 = cameraAngles(vertices, faces, normals, pi, -pi/6, 'centre');
    fprintf('---------------------');
    im3 = cameraAngles(vertices, faces, normals, pi/4 + pi, 0, 'right');
    fprintf('---------------------]\n Complete')
    function DpthImg = cameraAngles(vertex, face, normal, theta, rho, name)
        folder_path = pwd;
        path_opcodemesh = [folder_path '/opcodemesh'];
        if ~contains(path,path_opcodemesh)
            disp('Adding opcodemesh path...\n')
            addpath(genpath(path_opcodemesh))
        end 
        
        vpcd = pointCloud(vertex);
        npcd = pointCloud(normal);
        
        rf_y = theta;
        transY = [cos(rf_y)  0 sin(rf_y) 0;
                       0        1 0       0;
                       -sin(rf_y) 0 cos(rf_y) 0;
                       0        0 0       1];
        rf_x = rho;
        transX = [1 0         0          0;
                  0 cos(rf_x) -sin(rf_x) 0;
                  0 sin(rf_x) cos(rf_x)  0;
                  0 0         0          1];
%         rf_z = pi;
%         transZ = [cos(rf_z) -sin(rf_z) 0 0;
%                   sin(rf_z) cos(rf_z)  0 0;
%                   0         0          1 0;
%                   0         0          0 1];
        transform_x = affine3d(transX);
        vpcd = pctransform(vpcd, transform_x); 
        npcd = pctransform(npcd, transform_x);
        transform_y = affine3d(transY);
        vpcd = pctransform(vpcd, transform_y); 
        npcd = pctransform(npcd, transform_y);
%         transform_z = affine3d(transZ);
%         vpcd = pctransform(vpcd, transform_z); 
%         npcd = pctransform(npcd, transform_z);
        
        vertex = vpcd.Location;
        
        vertex = transpose(100*vertex);
        vertex(3,:) = vertex(3,:) + 300;
        normal = npcd.Location;
        
        DpthImg = KinectSimulator_Depth(vertex,transpose(face),transpose(normal),...
           'none', 'none', 'none', [],'imgrng',[200 1000],'subray',[5 9]);
         save([fname name '.mat'], 'DpthImg');
    end
end