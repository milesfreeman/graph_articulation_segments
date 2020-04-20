function [im1,im2,im3] = camera_angles(fname)
    load(fname, 'vertices', 'normals', 'faces');
    rf = pi/2;
    rot = [1 0       0        0;
           0 cos(rf) -sin(rf) 0;
           0 sin(rf) cos(rf)  0;
           0 0       0        1];
    p_pcd = pointCloud(vertices);
    n_pcd = pointCloud(normals);
    rotate = affine3d(rot);
    mu_x = mean(vertices(1:length(vertices), 1));
    mu_y = mean(vertices(1:length(vertices), 2));
    mu_z = mean(vertices(1:length(vertices), 3));
    p_trans = repmat([-1*mu_x -1*mu_y -1*mu_z], length(vertices), 1);
    n_trans = repmat([-1*mu_x -1*mu_y -1*mu_z], length(normals), 1);
    p_pcd2 = pctransform(p_pcd, p_trans);
    n_pcd2 = pctransform(n_pcd, n_trans);
    p_pcd3 = pctransform(p_pcd2, rotate);
    n_pcd3 = pctransform(n_pcd2, rotate);
    vertices = p_pcd3.Location;
    normals = n_pcd3.Location;
    im1 = cameraAngles(vertices, faces, normals, 7*pi/4, 0, 'left');
    im2 = cameraAngles(vertices, faces, normals, 0, pi/6, 'centre');
    im3 = cameraAngles(vertices, faces, normals, pi/4, 0, 'right');
    function DpthImg = cameraAngles(vertex, face, normal, theta, rho, name)
        folder_path = pwd;
        path_opcodemesh = [folder_path '/opcodemesh'];
        if ~contains(path,path_opcodemesh)
            disp('Adding opcodemesh path...\n')
            addpath(genpath(path_opcodemesh))
        end 
        % sf = 1.0;
        % rf = -pi/4;
        % 
        % pcd = pointCloud(sf*vertex);
        % transMatrix = [cos(rf)  0 sin(rf) 0;
        %                0        1 0       0;
        %                -sin(rf) 0 cos(rf) 0;
        %                0        0 0       1];
        % 
        % transform = affine3d(transMatrix);
        % pcd = pctransform(pcd, transform);       



        % 
    %     lower = min([pcd.XLimits pcd.YLimits]);
    %     upper = max([pcd.XLimits pcd.YLimits]);
    %       
    %     xlimits = [lower upper];
    %     ylimits = [lower upper];
    %     zlimits = pcd.ZLimits;
    %     player = pcplayer(xlimits,ylimits,zlimits);
    %     view(player,pcd)
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
        

        transform_x = affine3d(transX);
        vpcd = pctransform(vpcd, transform_x); 
        npcd = pctransform(npcd, transform_x);
        transform_y = affine3d(transY);
        vpcd = pctransform(vpcd, transform_y); 
        npcd = pctransform(npcd, transform_y);
        vertex = vpcd.Location;
        vertex = transpose(100*vertex);
        vertex(3,:) = vertex(3,:) + 300;
        normal = npcd.Location;

        DpthImg = KinectSimulator_Depth(vertex,transpose(face),transpose(normal),...
           'none', 'none', 'none', [],'imgrng',[200 1000],'subray',[5 9]);
         save([fname name '.mat'], 'DpthImg');
    end
end