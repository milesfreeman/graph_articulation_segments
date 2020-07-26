function img = kinectSim(fname)

    load([fname '_Lf.mat'], 'vertices', 'normals', 'faces');
    fprintf('\nProgress: [');
    img = captures(vertices, faces, normals, '_Lf.F.mat');
    clearvars -except fname
    fprintf('----------1')
    
    load([fname '_Cu.mat'], 'vertices', 'normals', 'faces');
    img = captures(vertices, faces, normals, '_Cu.F.mat');
    clearvars -except fname
    fprintf('-----------2');
    
    load([fname '_Rf.mat'], 'vertices', 'normals', 'faces');
    img = captures(vertices, faces, normals, '_Rf.F.mat');
    clearvars -except fname
    fprintf('-----------3')

    load([fname '_Cd.mat'], 'vertices', 'normals', 'faces');
    img = captures(vertices, faces, normals, '_Cd.F.mat');
    clearvars -except fname
    fprintf('-----------4');

    load([fname '_Rb.mat'], 'vertices', 'normals', 'faces');
    img = captures(vertices, faces, normals, '_Rb.F.mat');
    clearvars -except fname
    fprintf('-----------5')

    load([fname '_Lb.mat'], 'vertices', 'normals', 'faces');
    img = captures(vertices, faces, normals, '_Lb.F.mat');
    fprintf('-----------6] \n Complete <3 thanks 4 waiting')


    
    function DpthImg = captures(vertex, face, normal, name)
        folder_path = pwd;
        path_opcodemesh = [folder_path '/opcodemesh'];
        if ~contains(path,path_opcodemesh)
            addpath(genpath(path_opcodemesh))
        end 
        
        vertex = transpose(100*vertex);
        vertex(3,:) = vertex(3,:) + 300;
        
        DpthImg = KinectSimulator_Depth(vertex,transpose(face),transpose(normal),...
           'none', 'none', 'none', [],'imgrng',[200 1000],'subray',[5 9]);
         save([fname name], 'DpthImg');
    end
end