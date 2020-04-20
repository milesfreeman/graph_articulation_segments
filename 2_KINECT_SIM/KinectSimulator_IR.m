%KINECTSIMULATOR_IR  Simulate Kinect IR images.
%   KINECTSIMULATOR_IR is a program developed to simulate high fidelity
%   Kinect IR images by closely following the Microsoft Kinect's mechanics. 
%   For a detailed description of how this simulator was developed, please 
%   refer to [1]. If this simulator is used for publication, please cite 
%   [1] in your references.
%
%   IRimg = KinectSimulator_IR(vertex,face,normalf) returns the simulated  
%   padded IR image 'IRimg' used for depth image estimation, given a   
%   correlation window size. The parameters 'vertex' and 'face' define the 
%   CAD model of the 3D scene used to generate an image.
%
%       vertex  - 3xn, n vertices of each 3D coordinate that defines the  
%                 CAD model.
%       face    - 3xm, m facets, each represented by 3 vertices that  
%                 defines the CAD model.
%       normalf - 3xm, m facets, representing the normal direction of each
%                 facet.
%
%   This program utilizes a Matlab wrapper written by Vipin Vijayan [2] of 
%   the ray casting program OPCODE. The Original OPCODE program was written 
%   by Pierre Terdiman [3].
%
%   [IRimg,IRimg_disp] = KinectSimulator_IR(vertex,face,normalf) returns  
%   the simulated IR image IRimg_disp that is the same size and resolution  
%   of the real outputted Kinect IR image (480 rows x 640 columns). Note,
%   all intersecting IR dots are displayed, regardless of the operational
%   range of depths.
%
%   [IRimg,IRimg_disp,IRimg_full] = KinectSimulator_IR(vertex,face,normalf) 
%   returns the simulated image of the fully projected IR dot pattern 
%   IRimg_full that falls outside the bounds of the padded IR image.
%
%   IRimg = KinectSimulator_IR(vertex,face,normalf,IR_intensity,IR_speckle,IR_detector) 
%   allows the user to specify a different IR intensity and noise models.
%
%       IR_intensity Options
%           'default' - IR intensity model determined empirically from data
%                       collected from IR images recorded by the Kinect
%                       positioned in front of a flat wall at various
%                       depths. The model follows an alpha*dot(-n,l)/r^2 
%                       intensity model, where r is the range between the
%                       sensor and the surface point, n is the surface
%                       normal, and l is the IR lighting direction. Alpha 
%                       accounts for a constant illumination, surface  
%                       properties, etc. The actual intensity model is
%                       
%                           I = Iu.*5.90x10^8*dot(-n,l)/r^2
%
%                       Iu is the fractional intensity contributed by a
%                       sub-ray of a transmitted dot. r is the distance
%                       between the center of the transmitter coordinate 
%                       system, and the 3D location of where the sub-ray 
%                       intersected the CAD model (range).
%
%           'simple'  - Additional IR intensity model determined 
%                       empirically from data collected from Kinect IR  
%                       images. 
%
%                           I = Iu.*5.96x10^8/r^2
%
%                       This model is a simplified version of the 'default'
%                       model, which excluded the surface normal and IR 
%                       lighting direction to fit to the data.
%
%           'none'    - No model is used to compute an IR intensity. This 
%                       is modeled by 
%
%                           I = Iu
%
%                       This option is used when generating reference 
%                       images for depth estimation.
%
%       @(Iu,r,n,l)fn - User defined IR intensity model given the
%                       fractional intensity, range of the transmitted
%                       sub-ray, surface normal, and lighting direction. 
%
%       IR_speckle Options
%           'default' - IR speckle noise model determined empirically from 
%                       data collected from IR images recorded by the 
%                       Kinect positioned in front of a flat wall at 
%                       various depths. The model has a multiplicative term 
%                       that fits a gamma distribution with shape value 
%                       4.54 and scale value 0.196. This noise is added to 
%                       each IR dot, separately.
%
%                           I = I*Gamma
%                             = I*gamrnd(4.54,0.196)
%
%           'none'    - No model is used to compute an IR noise. This is
%                       modeled by 
%
%                           I = Iu
%
%                       This option is used when generating reference 
%                       images for depth estimation. Note, this option must
%                       be set if the IR_intensity model is set to 'none'.
%
%           @(I)fn    - User defined IR noise model given the intensity of
%                       the sub-ray representing a part of the transmitted
%                       IR dot.
%
%       IR_detector Options
%           'default' - IR detector noise model determined empirically from 
%                       data collected from IR images recorded by the 
%                       Kinect positioned in front of a flat wall at 
%                       various depths. The model has an additive term 
%                       that fits a normal distribution with mean -0.126  
%                       and standard deviation 10.4, with units of 10-bit 
%                       intensity. This noise is added to each pixel,
%                       separately.
%
%                           Z = I + n
%                             = I - 0.126+10.4*randn()
%
%           'none'    - No model is used to compute an IR noise. This is
%                       modeled by 
%
%                           Z = I 
%
%                       This option is used when generating reference 
%                       images for depth estimation. Note, this option must
%                       be set if the IR_intensity model is set to 'none'.
%
%           @(I)fn    - User defined IR noise model given the intensity of
%                       the sub-ray representing a part of the transmitted
%                       IR dot.
%
%   IRimg = KinectSimulator_IR(vertex,face,normalf,IR_intensity,IR_speckle,IR_detector,wallDist) 
%   allows the user the option to add a wall to the CAD model of the
%   simulated 3D scene.
%
%       wallDist Options
%           []        - No wall is added to the scene. This is the default
%                       setting.
%
%           'max'     - A flat wall is added to the 3D scene at the maximum
%                       operational depth.
%
%           wallDist  - The user can input an single value in millimeters
%                       between the min and max operational depths.
%
%   IRimg = KinectSimulator_IR(vertex,face,normalf,IR_intensity,IR_speckle,IR_detector,wallDist,options)
%   gives the user the option to change default Kinect input parameters and
%   the default IR simulator parameters. The options are listed as follows:
%
%       IR Simulator Parameters
%           'window'  - Size of correlation window used to process IR image
%                       images for depth estimation. The default is set to
%                       9x9 rows and columns, i.e. [9 9] pixels. Note, 
%                       these values must be greater than zero, and must be 
%                       odd to allow the pixel being processed to be at the 
%                       center of the window. Also, given the limited size 
%                       of the idealized dot pattern used to simulate IR 
%                       images, the number of rows in the window cannot 
%                       exceed 15 pixels.
%
%           'subray'  - Size of sub-ray grid used to simulate the physical
%                       cross-sectional area of a transmitted IR dot. The
%                       default is set to 7x17 rows and cols, i.e. [7 17]. 
%                       Note, it is preferable to set each value to an odd
%                       number as to allow the center of the pixel to be 
%                       represented by a sub-ray. Also, since Kinect 
%                       performs an interpolation to achieve a sub-pixel 
%                       disparity accuracy of 1/8th of a pixel, there 
%                       should be at least 8 columns of sub-rays.
%
%           'pattern' - The dot pattern used to simulate the IR image. The
%                       default is adapted from the work done by Andreas 
%                       Reichinger, in his blog post entitled 'Kinect 
%                       Pattern Uncovered' [4]. Note, this idealized binary
%                       representation of Kinect's dot pattern does not
%                       include the pincushion distortion observed in real
%                       Kinect IR images.
%
%           'quant10' - The option to quantize IR intensity into a 10-bit
%                       value. The default is set to 'on', but can be 
%                       turned to 'off'. Note, keeping the IR image  
%                       quantization option set to 'on' will result in 
%                       introducing more noise to the output IR values on 
%                       the order of  10*log10(2^10) = 30.1 dB, which 
%                       impacts depth estimation in the depth image 
%                       simulator. Depending on the inputted IR intensity
%                       and noise models, this may introduce erroneous 
%                       depth error, so the user can opt to set this option
%                       to 'off' to avoid this, or at the very least remove
%                       the option to saturate values below 0 and above 
%                       2^10 on lines 690 and 692.
%
%           'display' - The option to display the output IR images. The
%                       default is set to 'on', but can be turned to 'off'.
%                       Note, the dotted line on the padded image IRimg
%                       represents the original IR image resolution from 
%                       IRimg_disp. If the full IR image is displayed, the
%                       dotted line represents the padded image IRimg, and
%                       the blue stars represent where the dot pattern had
%                       to be extended. Also, the yellow stars represent
%                       the bright center dot locations that represent the
%                       center of each repeated pattern in the 3x3 final
%                       grid.
%
%       Kinect Parameters
%           'imgfov'  - The field of view of Kinect's transmitter/receiver.
%                       The default is set to 45.6 x 58.5 degrees for the 
%                       verticle and horizontal FOVs, respectively, i.e. 
%                       [45.6 58.5].
%
%           'imgrng'  - The minimum and maximum operational depths of the
%                       Kinect sensor. Dots with depths that fall outside 
%                       of this range are filtered out from the IRimg
%                       image. This is important for the depth image
%                       simulator because reference images generated at the
%                       min and max depths with only be able to find
%                       matches in the simulated measured IR image between
%                       the set operational depth range. The default is set
%                       to 800 mm for the minimum depth, and 4000 mm for
%                       the maximum depth, i.e. [800 4000]. Also note, 
%                       depending on the input range, the horizontal FOV of
%                       the dot pattern may increase by adding a repeated 
%                       portion of the original pattern. This is done in 
%                       order to allow all possible IR dot disparities
%                       across the entire focal plane.
%
% References: 
%   [1] M. J. Landau, B. Y. Choo, P. A. Beling, “Simulating Kinect Infrared  
%       and Depth Images,” IEEE Transactions on Cybernetics. 2015.
%
%   [2] V. Vijayan, “Ray casting for deformable triangular 3d meshes -
%       file exchange - MATLAB central,” Apr. 2013.
%       http://www.mathworks.com/matlabcentral/fileexchange/41504-ray-casting-for-deformable-triangular-3d-meshes/content/opcodemesh/matlab/opcodemesh.m
%
%   [3] P. Terdiman, “OPCODE,” Aug. 2002. 
%       http://www.codercorner.com/Opcode.htm
%
%   [4] A. Reichinger, “Kinect pattern uncovered | azt.tm’s blog,” Mar. 2011.
%       https://azttm.wordpress.com/2011/04/03/kinect-pattern-uncovered/

function varargout = KinectSimulator_IR(vertex,face,normalf,varargin)

% DEFAULT PARAMETERS ======================================================
% Depth simulator parameters ----------------------------------------------
% Option to plot IR image from depth image simulator
isPlotIR = false;

% IR simulator parameters -------------------------------------------------
% IR intensity and noise models 
model_Intensity = @(i,r,n,l) i.*5.90e+08.*dot(-n,l)'./r.^2;
model_Speckle   = @(i) i.*gamrnd(4.54,0.196,size(i));
model_Detector  = @(i) i-0.126+10.4.*randn(size(i));
% Size of correlation window used for depth estimation 
corrWind = [9 9];
% Size of sub-ray grid to simulate phyiscal dot cross-sectional area 
nsub = [7 17];
% Option to add a wall at depth
isWall = false;
% Option to plot IR images 
isPlot = true;
% Option to load idealized binary replication of the Kinect dot pattern
isLoadPattern = true;
% Option to quantize the IR image to 10-bit value
isQuant10 = true;
% Option to be able to quantize if not fractional intensities
isQuantOK = true;
% Force horizontal lines to be epipolar rectified 
adjRowShift = .5; % pix
adjColShift = .5; % pix

% Kinect parameters -------------------------------------------------------
% Resolution of real outputted Kinect IR image (rows x cols)
ImgRes = [480 640];   % pix
% Field of view of transmitter/receiver (vertFOV x horzFOV)
ImgFOV = [45.6 58.5]; % deg
% Minimum and maximum operational depths of the Kinect sensor (min x max)
ImgRng = [800 4000];  % mm
% Distance between IR transmitter and receiver
baseRT = 75; % mm

% ERROR MESSAGES ==========================================================
narginchk(3, inf)
nargoutchk(0, 3)
if size(vertex,1) ~= 3 || size(vertex,2) < 4 || length(size(vertex)) > 2
    error('Input ''vertex'' must have the form 3xn, and must have at least 3 values.')
end
if size(face,1) ~= 3 || length(size(face)) > 2
    error('Input ''face'' must have the form 3xm, and must have at least 3 values.')
end
if size(normalf,1) ~= 3 || length(size(normalf)) ~= length(size(face))
    error('Input ''normalf'' must have the form 3xm, and must equal the number of face values.')
end
if min(vertex(3,:)) < 0
    error('CAD model must have only positive depth (i.e. must be in front of the camera).')
end

% SET INPUT PARAMETERS ====================================================
if nargin == 4 || nargin == 5
    error('The IR intensity and noise models need to be specified.')
end
if nargin > 5
    if strcmp(varargin{1},'none') && (~strcmp(varargin{2},'none') || ~strcmp(varargin{3},'none'))
        error('Noise models must be set to ''none'' if intensity model is set to ''none''.');
    else
        % IR intensity model
        if strcmp(varargin{1},'default')
            model_Intensity = @(i,r,n,l) i.*5.90e+08.*dot(-n,l)'./r.^2;
        elseif strcmp(varargin{1},'simple')
            model_Intensity = @(i,r,n,l) i.*5.96e+08./r.^2;
        elseif strcmp(varargin{1},'none')
            model_Intensity = @(i,r,n,l) i.*ones(size(r));
            isQuantOK = false;
        elseif ischar(varargin{1})
            error('The argument ''%s'' is not a valid IR intensity model option.', varargin{1})
        elseif nargin(varargin{1}) ~= 4
            error('IR intensity model must have the form ''@(Iu,r,n,l) fnc''.')
        else % User inputted model
            model_Intensity = varargin{1};
        end
        % IR speckle noise model
        if strcmp(varargin{2},'default')
            model_Speckle = @(i) i.*gamrnd(4.54,0.196,size(i));
        elseif strcmp(varargin{2},'none')
            model_Speckle = @(i) i;
        elseif ischar(varargin{2})
            error('The argument ''%s'' is not a valid IR noise model option.', varargin{2})
        elseif nargin(varargin{2}) ~= 1
            error('IR speckle noise model must have the form ''@(I) fnc''.')
        else % User inputted model
            model_Speckle = varargin{2};
        end
        % IR detector noise model
        if strcmp(varargin{3},'default')
            model_Detector = @(i) i-0.126+10.4.*randn(size(i));
        elseif strcmp(varargin{3},'none')
            model_Detector = @(i) i;
        elseif ischar(varargin{3})
            error('The argument ''%s'' is not a valid IR noise model option.', varargin{3})
        elseif nargin(varargin{3}) ~= 1
            error('IR detector noise model must have the form ''@(I) fnc''.')
        else % User inputted model
            model_Detector = varargin{3};
        end
    end
end
if nargin > 6
    % Option for adding wall
    wallDist = varargin{4};
    if isempty(wallDist)
        isWall = false;
    elseif strcmp(wallDist,'max')
        isWall = true;
    elseif ~isnumeric(wallDist) || wallDist <= 0 || length(wallDist) ~= 1
        error('Wall depth must be a single value greater than zero.')
    else
        isWall = true;
    end
end
if nargin > 7
    % Concatenate cell (meant for use with Kinect depth simulator)
    if iscell(varargin{5})
        varargin = [varargin(1:4) varargin{5}];
    end
    k = 5;
    while k < length(varargin)+1
        switch varargin{k}
            % Depth Simulator Parameters
            case 'refine' 
                k = k+2;
            case 'quant11'
                k = k+2;
            case 'displayIR'
                if strcmp(varargin{k+1},'off')
                    isPlotIR = false;
                    isPlot = false;
                elseif strcmp(varargin{k+1},'on')
                    isPlotIR = true;
                    isPlot = true;
                else
                    error('Display option must be set to ''off'' or ''on''.')
                end
                k = k+2;
            % IR Simulator Parameters
            case 'window'
                corrWind = varargin{k+1};
                if length(corrWind) ~= 2 || sum(mod(corrWind,2)) ~= 2 || sum(corrWind>0) ~= 2 || ~isnumeric(corrWind)
                    error('Correlation window must have two odd integer values greater than zero.')
                elseif corrWind(1) > 15
                    error('Number of rows in correlation window cannot exceed 15 pixels.')
                end
                k = k+2;
            case 'subray'
                nsub = varargin{k+1};
                if length(nsub) ~= 2 || sum(rem(nsub,1)==0) ~=2 || sum(nsub>0) ~= 2 || ~isnumeric(nsub)
                    error('Sub-ray grid must have two integer values greater than zero.')
                end
                k = k+2;
            case 'pattern'
                dotPattern = varargin{k+1};
                isLoadPattern = false;
                if sum([size(find(dotPattern==0),1) size(find(dotPattern==1),1)]) ~= numel(dotPattern)
                    error('Dot pattern image must be binary.')
                end
                k = k+2;
            case 'quant10'
                if strcmp(varargin{k+1},'off')
                    isQuant10 = false;
                elseif strcmp(varargin{k+1},'on')
                    isQuant10 = true;
                else
                    error('Quantize IR image option must be set to ''off'' or ''on''.')
                end
                k = k+2;
            case 'display'
                if strcmp(varargin{k+1},'off')
                    isPlot = false;
                elseif strcmp(varargin{k+1},'on')
                    isPlot = true;
                else
                    error('Display option must be set to ''off'' or ''on''.')
                end
                k = k+2;
            % Kinect Parameters
            case 'imgfov'
                ImgFOV = varargin{k+1}; 
                if length(ImgFOV) ~= 2 || sum(ImgFOV>0) ~= 2 || ~isnumeric(ImgFOV)
                    error('Image field of view must have two values greater than zero.')
                end
                k = k+2;
            case 'imgrng'
                ImgRng = varargin{k+1};
                if length(ImgRng) ~= 2 || sum(ImgRng>0) ~= 2 || ~isnumeric(ImgRng)
                    error('Operational range of depths must have two values greater than zero.')
                elseif ImgRng(1) > ImgRng(2)
                    error('Min depth must be less than max depth.')
                end
                k = k+2;
            otherwise
                if isnumeric(varargin{k})
                    error('The argument ''%d'' is not a valid input parameter.', varargin{k})
                elseif ischar(varargin{k})
                    error('The argument ''%s'' is not a valid input parameter.', varargin{k})
                else
                    error('Not a valid input parameter')
                end
        end
    end
end

% =========================================================================
% PROCESS IR IMAGE ========================================================
% =========================================================================

% PREPROCESS PARAMETERS ===================================================
% Determine horizontal and vertical focal lengths
ImgFOV = ImgFOV*(pi/180); % rad
FocalLength = [ImgRes(2)/(2*tan(ImgFOV(2)/2)); ImgRes(1)/(2*tan(ImgFOV(1)/2))]; % pix

% Number of rows and columns to pad IR image for cross correlation
corrRow = (corrWind(1)-1)/2;
corrCol = (corrWind(2)-1)/2;

% Load idealized binary replication of the Kinect dot pattern -------------
dOff_min = (baseRT*FocalLength(1)/ImgRng(1));
dOff_max = (baseRT*FocalLength(1)/ImgRng(2));
if isLoadPattern 
    load('default_load_files\kinect_pattern_3x3.mat')
    % Check if depth range provides enough coverage for reference images
    if dOff_min-dOff_max > size(dotPattern,2)/3
        warning('Depth range too large for default dot pattern in order to achieve no pattern overlap.')
    end
else
    % Check if depth range provides enough coverage for reference images
    if dOff_min-dOff_max > size(dotPattern,2)
        warning('Depth range too large for size of dot pattern in order to achieve no pattern overlap.')
    end
    minRow = ImgRes(1) + corrWind(1) - 1;
    if size(dotPattern,1) < minRow
        warning(['Dot pattern must have at least ' num2str(minRow) ' rows for a correlation window with ' num2str(corrWind(1)) ' rows'])
    end
    % Force horizontal lines to be epipolar rectified 
    if mod(size(dotPattern,1),2) == 0
        adjRowShift = 0;
    end
    if mod(size(dotPattern,2),2) == 0
        adjColShift = 0;
    end
end

% IR dot pattern and padded image sizes
ImgResPad  = ImgRes+corrWind-1;
ImgSizeDot = size(dotPattern);

% Increase resolution of dot pattern with sub-rays ------------------------
% Minimum and maximum depth values of 3D scene
if isWall
    if strcmp(wallDist,'max') || nargin < 7
        wallDist = ImgRng(2);
    end
    if wallDist < ImgRng(1) || wallDist > ImgRng(2)
        error('The wall depth must be between the min and max operational depths.')
    end
    minDepth = min([min(vertex(3,:)) wallDist]); % mm
    maxDepth = max([max(vertex(3,:)) wallDist]); % mm
else
    minDepth = min(vertex(3,:)); % mm
    maxDepth = max(vertex(3,:)); % mm
end

% Add columns of dot pattern to left and right side based on disparity equation
minDisparity = round((baseRT*FocalLength(1))/min([ImgRng(1) minDepth]));
maxDisparity = round((baseRT*FocalLength(1))/max([ImgRng(2) maxDepth]));

% Number of cols cannot exceed size of dot pattern (for simplicity of coding)
pixShftLeft_T = min([ImgSizeDot(2),max([0, floor((ImgRes(2)-ImgSizeDot(2))/2)+1+minDisparity+corrCol])]);
pixShftRght_T = min([ImgSizeDot(2),max([0, floor((ImgRes(2)-ImgSizeDot(2))/2)+1-maxDisparity+corrCol])]);

dotAddLeft = dotPattern(:,end-pixShftLeft_T+1:end);
dotAddRght = dotPattern(:,1:pixShftRght_T);
dotAdd     = [dotAddLeft dotPattern dotAddRght];
ImgSizeAdd = size(dotAdd);

% Increase resolution of dot pattern to nsub_Nsubpx image size
dotInc  = reshape(repmat(dotAdd,nsub(2),1),ImgSizeAdd(1),[]);
dotInc  = reshape(repmat(dotInc',nsub(1),1),ImgSizeAdd(2)*nsub(2),[])';
dotIndx = find(dotInc==1);
ImgSizeInc = size(dotInc);

% Pre-compute IR speckle noise for each dot
IR_speckle = model_Speckle(ones(ImgSizeAdd));
IR_speckle = reshape(repmat(IR_speckle,nsub(2),1),ImgSizeAdd(1),[]);
IR_speckle = reshape(repmat(IR_speckle',nsub(1),1),ImgSizeAdd(2)*nsub(2),[])';
IR_speckle = IR_speckle(dotIndx);

% Set up scene by adding wall CAD to object CAD ---------------------------
if isWall
    maxPixRow = max([ceil(ImgSizeDot(1)/2) ceil(ImgRes(1)/2)]); 
    maxPixCol = max([ceil(ImgSizeDot(2)/2) ceil(ImgRes(2)/2)]); 

    wallX1 =  ImgRng(2)*tan((maxPixCol+pixShftLeft_T)*ImgFOV(2)/ImgRes(2));
    wallX2 = -ImgRng(2)*tan((maxPixCol+pixShftRght_T)*ImgFOV(2)/ImgRes(2)) - baseRT;
    wallY1 =  ImgRng(2)*tan((maxPixRow+corrRow)*ImgFOV(1)/ImgRes(1));
    wallY2 = -ImgRng(2)*tan((maxPixRow+corrRow)*ImgFOV(1)/ImgRes(1));

    % Set parameters of wall CAD model
    vertex_wall = [wallX1   wallX1   wallX2   wallX2;...
                   wallY1   wallY2   wallY1   wallY2;...
                   wallDist wallDist wallDist wallDist];
    vertex      = [vertex vertex_wall];
    
    face_wallAdd = [size(vertex,2)-3 size(vertex,2)-3;...
                    size(vertex,2)-2 size(vertex,2)-1;...
                    size(vertex,2)   size(vertex,2)];
    face         = [face face_wallAdd];
                
    norm_wall = [0  0;...
                 0  0;...
                -1 -1];
    normalf   = [normalf norm_wall];
end

% SET UP TRANSMITTER RAYS FOR RAY CASTING =================================
% Setup collision structure 
t = opcodemesh(vertex,face);

% Create angles of rays for each sub-pixel from transmitter 
vertPixLeft_T = ImgSizeDot(1)/2-1/(nsub(1)*2);
horzPixLeft_T = ImgSizeDot(2)/2-1/(nsub(2)*2)+pixShftLeft_T;

% Convert index to subscript values (based on ind2sub)
jpix_y = rem(dotIndx-1, ImgSizeInc(1)) + 1;
jpix_x = (dotIndx - jpix_y)/ImgSizeInc(1) + 1;

% Adjust subscript value to sensor coordinate system values
spix_x = horzPixLeft_T-(jpix_x-1)/nsub(2)+adjColShift;
spix_y = vertPixLeft_T-(jpix_y-1)/nsub(1)+adjRowShift;

% Determine 3D location of where each ray intersects max range
X_T = spix_x * maxDepth / FocalLength(1);
Y_T = spix_y * maxDepth / FocalLength(2);
Z_T = maxDepth*ones(size(dotIndx));

% Set up origin and directions for each ray
orig_T = [zeros(size(X_T))-baseRT zeros(size(Y_T)) zeros(size(Z_T))]';
dir_T  = [           X_T-baseRT              Y_T              Z_T]';

% Find intersecting points from line of sight vectors 
[~,d_T,t_T,~] = t.intersect(orig_T,dir_T-orig_T);
indxHit = find(t_T);

% OCCLUSION DETECTION =====================================================
% Set up receiver rays based on where transmitted rays intersected CAD
X_R = d_T(indxHit).*X_T(indxHit)-baseRT;
Y_R = d_T(indxHit).*Y_T(indxHit);
Z_R = d_T(indxHit).*Z_T(indxHit);

% Find intersecting points from line of sight vectors 
orig_R = [zeros(size(X_R)) zeros(size(Y_R)) zeros(size(Z_R))]';
dir_R  = [           X_R              Y_R              Z_R]';

% Find intersecting points from line of sight vectors 
[~,d_R,t_R,b_R] = t.intersect(orig_R,dir_R-orig_R);
indxMiss = find(~t_R);
d_R(indxMiss) = [];
t_R(indxMiss) = [];
b_R(:,indxMiss) = [];
indxHit(indxMiss) = [];

% Build array of rays that are not occluded
t_T = t_T(indxHit); 
tol = 1e-3;
% Find sub-rays that intersect near the edge of facets
indxBary = sum(b_R>1-tol) + sum(b_R<tol) + (sum(b_R)>1-tol) + (sum(b_R)<tol);
% Find sub-rays that intersect near the edge AND close to each other
indxDist = ~logical( indxBary(:) + (d_R(:)>1-tol) - 2 );
% Find sub-rays that intersect the same facet
indxKeep = find(indxDist + (t_T(:)==t_R(:)));
if isempty(indxKeep), indxKeep = zeros(0,1); end

% Reduce precision to avoid rounding error
d_T = round(d_T*1e5)/1e5;
X_R = d_T(indxHit(indxKeep)).*X_T(indxHit(indxKeep));
Y_R = d_T(indxHit(indxKeep)).*Y_T(indxHit(indxKeep));
Z_R = d_T(indxHit(indxKeep)).*Z_T(indxHit(indxKeep));

% Filter speckle noise for all intersecting dots
IR_speckle = IR_speckle(indxHit(indxKeep));

% Find range of all dots wrt transmitter
rng_T = sqrt(X_R.^2 + Y_R.^2 + Z_R.^2);

% Find surface normal for all intersecting dots
sn_T = normalf(:,t_R(indxKeep));

% Find the IR light direction for all sub-rays
ld_T = bsxfun(@rdivide,[X_R Y_R Z_R]',rng_T');

% PROJECT IR DOTS INTO PIXEL COORDINATE SYSTEM ============================
% For transmitted rays that have unoccluded IR dots, find 3D sensor coordinates
center = zeros(2,size(indxKeep,1));
center(1,:) = ImgRes(2)/2 +.5 - FocalLength(1).* (X_R-baseRT)'./Z_R';
center(2,:) = ImgRes(1)/2 +.5 - FocalLength(2).* Y_R'./Z_R';

% GENERATE OUTPUT IR IMAGE ================================================
subIntensity = 1/(nsub(1)*nsub(2));

% Round row and col values to pixel it will be added in to
center = round(center);

% Padded IR image with correlation window (main image used) ---------------
% Filter out values that fall outside operational depths
indxFilt = find(ceil(Z_R)>=ImgRng(1));
indxFilt = indxFilt(floor(Z_R(indxFilt))<=ImgRng(2));

% Filter out values that fall outside image frame
minFiltRow = 1-corrRow;
maxFiltRow = ImgResPad(1)-corrRow;
minFiltCol = 1-corrCol;
maxFiltCol = ImgResPad(2)-corrCol;

indxFilt = indxFilt(center(1,indxFilt)>=minFiltCol);
indxFilt = indxFilt(center(1,indxFilt)<=maxFiltCol);
indxFilt = indxFilt(center(2,indxFilt)>=minFiltRow);
indxFilt = indxFilt(center(2,indxFilt)<=maxFiltRow);

% Preprocess intensity values for each sub-ray
intensity_T = model_Intensity(subIntensity,rng_T(indxFilt),sn_T(:,indxFilt),ld_T(:,indxFilt));

% Introduce IR speckle noise model
intensity_T = intensity_T.*IR_speckle(indxFilt);

% Determine image pixel indices for each sub-ray
indxCent = (center(2,indxFilt) + corrRow) + (center(1,indxFilt) + corrCol - 1).*ImgResPad(1);

% Sum sub-rays within pixels
IR_intensities = accumarray(indxCent',intensity_T');

% Add in sub-ray intensities to pixels
IRimg = single(zeros(ImgResPad));
IRimg(IR_intensities>0) = IR_intensities(IR_intensities>0);

% Determine index values for pixels with IR dots
indxDot = find(IRimg>0);

% Introduce IR detector noise model
IRimg(indxDot) = model_Detector(IRimg(indxDot));

% Quantize image to 10 bit if intensity models are included ---------------
if isQuant10 && isQuantOK
    % Round intensity to closest integer value
    IRimg = round(IRimg);
    % Saturate high intensity values above max allowable integer
    IRimg(IRimg > 2^10) = 2^10;
    % Saturate low intensity values below min allowable integer
    IRimg(IRimg < 0) = 0;
end

% Store image into function output variable
varargout{1} = IRimg;

% Cropped IR image for displaying -----------------------------------------
if nargout > 1 || isPlotIR
    IRimgOut = IRimg;
    
    % Include values that fall outside operational depths
    indxOut = [find(ceil(Z_R)<ImgRng(1)); find(floor(Z_R)>ImgRng(2))];
    
    % Filter out values that fall outside image frame
    indxOut = indxOut(center(1,indxOut)>=minFiltCol);
    indxOut = indxOut(center(1,indxOut)<=maxFiltCol);
    indxOut = indxOut(center(2,indxOut)>=minFiltRow);
    indxOut = indxOut(center(2,indxOut)<=maxFiltRow);
    
    % Preprocess intensity values for each sub-ray
    intensity_T = model_Intensity(subIntensity,rng_T(indxOut),sn_T(:,indxOut),ld_T(:,indxOut));
    
    % Introduce IR speckle noise model
    intensity_T = intensity_T.*IR_speckle(indxOut);
    
    % Determine image pixel indices for each sub-ray
    indxCent = (center(2,indxOut) + corrRow) + (center(1,indxOut) + corrCol - 1).*ImgResPad(1);
    
    % Sum sub-rays within pixels
    IR_intensities = accumarray(indxCent',intensity_T');
    
    % Add in sub-ray intensities to pixels
    IRimgOut(IR_intensities>0) = IR_intensities(IR_intensities>0);

    % Determine index values for pixels with IR dots from filtered sub-rays
    IRimg_temp = IRimgOut;
    IRimg_temp(indxDot) = 0;
    indxDot = find(IRimg_temp>0);

    % Introduce IR detecor noise model
    IRimgOut(indxDot) = model_Detector(IRimgOut(indxDot));
    
    % Crop IRimgOut to fit display IR image 
    rowRange = 1+corrRow:ImgRes(1)+corrRow;
    colRange = 1+corrCol:ImgRes(2)+corrCol;

    IRimg_disp = IRimgOut(rowRange,colRange);
    
    % Quantize image to 10 bit if intensity models are included -----------
    if isQuant10 && isQuantOK
        % Round intensity to closest integer value
        IRimg_disp = round(IRimg_disp);
        % Saturate high intensity values above max allowable integer
        IRimg_disp(IRimg_disp > 2^10) = 2^10;
        % Saturate low intensity values below min allowable integer
        IRimg_disp(IRimg_disp < 0) = 0;
    end
    
    % Store image into function output variable
    varargout{2} = IRimg_disp;
end

% Full IR image -----------------------------------------------------------
if nargout > 2
    indxFilt = [indxFilt(:); indxOut(:)];
    
    % Find values that weren't used in main image
    center(:,indxFilt) = [];
    
    % Filter speckle noise
    IR_speckle(indxFilt) = [];
    
    % Filter range vector
    rng_T(indxFilt) = [];
    
    % Filter surface normal vector
    sn_T(:,indxFilt) = [];
    
    % Filter IR light direction vector
    ld_T(:,indxFilt) = [];
    
    % Fill full image with values from main image
    cropRow = max([0, ((ImgSizeDot(1)-ImgRes(1))/2)+adjRowShift]);
    
    minCol_full = min([min(center(1,:)), 1-corrCol]);
    maxCol_full = max([max(center(1,:))+1-minCol_full, ImgRes(2)+corrCol+1-minCol_full]);
    
    rowShft_full = cropRow;
    colShft_full = 1 - minCol_full;
    
    rowRange = (minFiltRow:maxFiltRow)+rowShft_full;
    colRange = (minFiltCol:maxFiltCol)+colShft_full;
    
    IRimg_full = single(zeros([ImgSizeDot(1), maxCol_full]));
    IRimg_full(rowRange,colRange) = IRimgOut;

    % Preprocess intensity values for each sub-ray
    intensity_T = model_Intensity(subIntensity,rng_T,sn_T,ld_T);
    
    % Introduce IR speckle noise model
    intensity_T = intensity_T.*IR_speckle;
    
    % Determine image pixel indices for each sub-ray
    indxCent = (center(2,:) + rowShft_full) + (center(1,:) + colShft_full - 1).*size(IRimg_full,1);
    
    % Sum sub-rays within pixels
    IR_intensities = accumarray(indxCent',intensity_T');
    
    % Add in sub-ray intensities to pixels
    IRimg_full(IR_intensities>0) = IR_intensities(IR_intensities>0);
    
    % Determine index values for pixels with IR dots from filtered sub-rays
    IRimg_temp = IRimg_full;
    IRimg_temp(rowRange,colRange) = 0;
    indxDot = find(IRimg_temp>0);

    % Introduce IR detecor noise model
    IRimg_full(indxDot) = model_Detector(IRimg_full(indxDot));

    % Quantize image to 10 bit if intensity models are included -----------
    if isQuant10 && isQuantOK
        % Round intensity to closest integer value
        IRimg_full = round(IRimg_full);
        % Saturate high intensity values above max allowable integer
        IRimg_full(IRimg_full > 2^10) = 2^10;
        % Saturate low intensity values below min allowable integer
        IRimg_full(IRimg_full < 0) = 0;
    end

    % Store image into function output variable
    varargout{3} = IRimg_full;
end

% DISPLAY OUTPUT IR IMAGES ================================================
if isPlot
    % Determine max allowable intensity
    if isQuantOK
        imgMax = 2^10;
%         imgMax = model_Intensity(1,minDepth);
%         imgMax = min([2^10, model_Intensity(1,minDepth)]);  
    else
        imgMax = 1;       
    end
    
    % Determine bright center dots locations
    oneBlock = ImgSizeDot/3;
    oneBlockHalf = round(oneBlock/2);
    brightDotX = [oneBlockHalf(1) oneBlockHalf(1)+oneBlock(1) oneBlockHalf(1)+2*oneBlock(1)];
    brightDotY = [oneBlockHalf(2) oneBlockHalf(2)+oneBlock(2) oneBlockHalf(2)+2*oneBlock(2)];
    brightRepX = repmat(brightDotX,1,3)';
    brightRepY = reshape(repmat(brightDotY,3,1),1,[])';
    
    X_bright = ((ImgSizeDot(2)/2-1/2)-(brightRepY-1)+adjColShift) * maxDepth / FocalLength(1);
    Y_bright = ((ImgSizeDot(1)/2-1/2)-(brightRepX-1)+adjRowShift) * maxDepth / FocalLength(2);
    Z_bright = maxDepth*ones(size(X_bright));
    
    orig_bright = [zeros(size(X_bright))-baseRT zeros(size(Y_bright)) zeros(size(Z_bright))]';
    dir_bright  = [           X_bright-baseRT              Y_bright              Z_bright]';
    
    [~,d_bright,~,~] = t.intersect(orig_bright,dir_bright-orig_bright);
    
    xvals_bright = d_bright.*X_bright-baseRT;
    yvals_bright = d_bright.*Y_bright;
    zvals_bright = d_bright.*Z_bright;
    
    brightPlotX = ImgRes(2)/2 +.5 - FocalLength(1) .* xvals_bright./zvals_bright;
    brightPlotY = ImgRes(1)/2 +.5 - FocalLength(2) .* yvals_bright./zvals_bright;
    
    % Determine where dots had to be added to original dot pattern on the left
    if pixShftLeft_T > 0
        jrow = 0;
        for irow = 1:size(dotAddLeft,1)
            indxMax = find(dotAddLeft(irow,:)==1, 1, 'last' );
            if ~isempty(indxMax)
                % If the dot is on the last 3 columns of the added pattern
                if indxMax >= size(dotAddLeft,2) - 2
                    jrow = jrow + 1;
                    addDotLeftX(jrow) = irow;
                    addDotLeftY(jrow) = indxMax;
                end
            end
        end

        X_add = ((ImgSizeDot(2)/2-1/2+pixShftLeft_T)-(addDotLeftY'-1)+adjColShift) * maxDepth / FocalLength(1);
        Y_add = ((ImgSizeDot(1)/2-1/2)-(addDotLeftX'-1)+adjRowShift) * maxDepth / FocalLength(2);
        Z_add = maxDepth*ones(size(addDotLeftX'));

        orig_add = [zeros(size(X_add))-baseRT zeros(size(Y_add)) zeros(size(Z_add))]';
        dir_add  = [           X_add-baseRT              Y_add              Z_add]';

        [~,d_add,~,~] = t.intersect(orig_add,dir_add-orig_add);

        xvals_add = d_add.*X_add-baseRT;
        yvals_add = d_add.*Y_add;
        zvals_add = d_add.*Z_add;

        addPlotXLeft = ImgRes(2)/2 +.5 - FocalLength(1) .* xvals_add./zvals_add;
        addPlotYLeft = ImgRes(1)/2 +.5 - FocalLength(2) .* yvals_add./zvals_add;
    end
    
    % Determine where dots had to be added to original dot pattern on the right
    if pixShftRght_T > 0
        jrow = 0;
        for irow = 1:ImgSizeDot(1)
            indxMin = find(dotAddRght(irow,:)==1, 1, 'first' );
            if ~isempty(indxMin)
                % If the dot is on the first 3 columns of the added pattern
                if indxMin <= 3
                    jrow = jrow + 1;
                    addDotRghtX(jrow) = irow;
                    addDotRghtY(jrow) = indxMin + ImgSizeDot(2) + pixShftLeft_T;
                end
            end
        end
        
        X_add = ((ImgSizeDot(2)/2-1/2+pixShftLeft_T)-(addDotRghtY'-1)+adjColShift) * maxDepth / FocalLength(1);
        Y_add = ((ImgSizeDot(1)/2-1/2)-(addDotRghtX'-1)+adjRowShift) * maxDepth / FocalLength(2);
        Z_add = maxDepth*ones(size(addDotRghtX'));
        
        orig_add = [zeros(size(X_add))-baseRT zeros(size(Y_add)) zeros(size(Z_add))]';
        dir_add  = [           X_add-baseRT              Y_add              Z_add]';
        
        [~,d_add,~,~] = t.intersect(orig_add,dir_add-orig_add);
        
        xvals_add = d_add.*X_add-baseRT;
        yvals_add = d_add.*Y_add;
        zvals_add = d_add.*Z_add;
        
        addPlotXRght = ImgRes(2)/2 + .5 - FocalLength(1) .* xvals_add./zvals_add;
        addPlotYRght = ImgRes(1)/2 + .5 - FocalLength(2) .* yvals_add./zvals_add;
    end
    
    % Plot padded IR image ------------------------------------------------
    if ~isPlotIR
        figure, imshow(IRimg,[0 imgMax])
%         title('IR image of cropped projected dot pattern for correlation window')

        % Plot stars over bright center dots
        hold on
        if isLoadPattern
            plot(brightPlotX+corrCol,brightPlotY+corrRow,'y*')
        else
            plot(median(brightPlotX)+corrCol,median(brightPlotY)+corrRow,'y*')
        end
        hold off

        % Plot dashed line around where IR image will be cropped for displaying
        cropPlotX_corr = [corrCol+1, corrCol+ImgRes(2), corrCol+ImgRes(2), corrCol+1, corrCol+1];
        cropPlotY_corr = [corrRow+1, corrRow+1, corrRow+ImgRes(1), corrRow+ImgRes(1), corrRow+1];

        hold on, plot(cropPlotX_corr,cropPlotY_corr,'y:'), hold off
    end
    
    % Plot cropped IR image (size of IR image Kinect displays) ------------
    if nargout > 1 || isPlotIR
        figure, imshow(IRimg_disp,[0 imgMax])
%         title('IR image size of Kinect output')

        % Plot stars over bright center dots
        hold on
        if isLoadPattern
            plot(brightPlotX,brightPlotY,'y*')
        else
            plot(median(brightPlotX),median(brightPlotY),'y*')
        end
        hold off
    end
    
    % Plot full IR image --------------------------------------------------
    if nargout > 2       
        figure, imshow(IRimg_full,[0 imgMax])
        title('IR image of full projected dot pattern')

        % Plot stars over bright center dots
        hold on
        if isLoadPattern
            plot(brightPlotX+colShft_full,brightPlotY+rowShft_full,'y*')
        else
            plot(median(brightPlotX)+colShft_full,median(brightPlotY)+rowShft_full,'y*')
        end
        hold off

        % Plot dashed line around where IR image will be cropped for padded img
        cropX1_corr = minFiltCol + colShft_full;
        cropX2_corr = maxFiltCol + colShft_full;
        cropY1_corr = minFiltRow + rowShft_full;
        cropY2_corr = maxFiltRow + rowShft_full;
        
        cropPlotX_disp = [cropX1_corr, cropX2_corr, cropX2_corr, cropX1_corr, cropX1_corr];
        cropPlotY_disp = [cropY1_corr, cropY1_corr, cropY2_corr, cropY2_corr, cropY1_corr];

        hold on, plot(cropPlotX_disp,cropPlotY_disp,'y:'), hold off

        % Plot where dots had to be added to original dot pattern on the left
        if pixShftLeft_T > 0
            hold on, plot(addPlotXLeft+colShft_full,addPlotYLeft+rowShft_full,'b.'), hold off
        end

        % Plot where dots had to be added to original dot pattern on the right
        if pixShftRght_T > 0
            hold on, plot(addPlotXRght+colShft_full,addPlotYRght+rowShft_full,'b.'), hold off
        end
    end
end