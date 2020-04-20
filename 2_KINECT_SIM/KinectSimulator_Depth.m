%KINECTSIMULATOR_DEPTH Simulate Kinect depth images. 
%   KINECTSIMULATOR_DEPTH is a program developed to simulate high fidelity
%   Kinect depth images by closely following the Microsoft Kinect's 
%   mechanics. For a detailed description of how this simulator was 
%   developed, please refer to [1]. If this simulator is used for 
%   publication, please cite [1] in your references.
%
%   DEPTHimg = KinectSimulator_Depth(vertex,face,normalf) returns the 
%   simulated depth image 'DEPTHimg'. The parameters 'vertex' and 'face' 
%   define the CAD model of the 3D scene used to generate an image.
%
%       vertex  - 3xn, n vertices of each 3D coordinate that defines the  
%                 CAD model.
%       face    - 3xm, m facets, each represented by 3 vertices that  
%                 defines the CAD model.
%       normalf - 3xm, m facets, representing the normal direction of each
%                 facet.
%
%   The depth image simulator calls the function KINECTSIMULATOR_IR to
%   generate a series of non-noisy reference IR images and the output noisy 
%   IR image of the 3D scene to estimate the noisy depth image. Note, The 
%   IR image simulator program utilizes a Matlab wrapper written by Vipin 
%   Vijayan [2] of the ray casting program OPCODE. The Original OPCODE
%   program was written by Pierre Terdiman [3].
%   
%   Depth image estimation undergoes two steps. The first step estimates an
%   integer disparity value by correlating a pixel window of the binary  
%   measured image to a binary reference image of a flat wall at either the   
%   min or max operational depth. This step finds a best match between a  
%   window centered around a pixel of the projected IR pattern from the 
%   measurement and reference images. The epipolar lines of the transmitter 
%   and receiver are assumed coincident, which means the pattern can only
%   shift by a varying number of column pixels. The program then performs a 
%   sub-pixel refinement step to estimate a fractional disparity value. 
%   This is done by finding the minimum sum of absolute differences between  
%   the noisy measured IR image and a non-noisy reference image of a flat 
%   wall at a distance computed by the initial integer disparity 
%   estimation. This step estimates where the location where IR dots in the  
%   window template split pixels. The depth 'z' is computed from a given  
%   disparity 'd' by
%
%       z = b*f / d,
%
%   where 'f' is the horizontal focal length of the Kinect sensor, and 'b' 
%   is the baseline distance between the transmitter and receiver. The 
%   default baseline distance is fixed to 75 mm, and the focal length is 
%   computed from the inputted FOV, where the default is 571.4 pixels.
%
%   [DEPTHimg, IRimg] = KinectSimulator_Depth(vertex,face,normalf) returns  
%   the simulated noisy IR image IRimg_disp, generated from the 
%   KINECTSIMULATOR_IR function. This image has the same size and 
%   resolution of the real output Kinect IR image (480 rows x 640 columns). 
%   Note, all intersecting IR dots are displayed, regardless of the 
%   operational range of depths.
%   
%   DEPTHimg = KinectSimulator_Depth(vertex,face,normalf,IR_intensity,IR_speckle,IR_detector) 
%   allows the user to specify a different IR intensity and noise model for
%   the IR image simulator.
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
%                       lighting direction to fit the model.
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
%   DEPTHimg = KinectSimulator_Depth(vertex,face,normalf,IR_intensity,IR_speckle,IR_detector,wallDist) 
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
%   DEPTHimg = KinectSimulator_Depth(vertex,face,normalf,IR_intensity,IR_speckle,IR_detector,wallDist,options) 
%   gives the user the option to change default Kinect input parameters and
%   the default IR and depth simulator parameters. The options are listed 
%   as follows:
%
%       Depth Simulator Parameters
%           'refine'  - The interpolation factor to perform sub-pixel
%                       refinement. Since the Kinect performs interpolation
%                       by a factor of 1/8th of a pixel, the default
%                       for this option is set to 8.
%
%           'quant11' - The option to quantize depth values to real
%                       allowable 11-bit depth values outputted by the 
%                       Kinect sensor. The 'default' option loads an array
%                       of depth values collected from real data outputted 
%                       by a Kinect for Windows sensor. The user may also
%                       set this option to 'off', or input a new array of
%                       quantized depth values, all of which must be
%                       greater than zero.
%
%         'displayIR' - The user may set this option to 'on' to display
%                       the noisy IR image of the 3D scene. The default is
%                       set to 'off'.
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
%                       turned to 'off'. 
%
%       Kinect Parameters
%           'imgfov'  - The field of view of Kinect's transmitter/receiver.
%                       The default is set to 45.6 x 58.5 degrees for the 
%                       verticle and horizontal FOVs, respectively, i.e. 
%                       [45.6 58.5].
%
%           'imgrng'  - The minimum and maximum operational depths of the
%                       Kinect sensor. Dots with depths that fall outside 
%                       of this range are filtered out from the IRimg and 
%                       IRimg_disp images. This is important for the depth
%                       image simulator because reference images generated
%                       at the min and max depths with only be able to find
%                       matches in the simulated measured IR image between
%                       the set operational depth range. The default is set
%                       to 800 mm for the minimum depth, and 4000 mm for
%                       the maximum depth, i.e. [800 4000].
%
%   Notes about the options:
%       By limiting disparity estimation to an 1/8th of a pixel, this
%       simulator in essence quantizes depth similar to the way Kinect 
%       quantizes depth images into 11-bit values. However, the estimated
%       horizontal FOV and simulated disparity values most likely differ 
%       from the exact Kinect parameters, and therefore setting 'quant11' 
%       to 'off' will result in output depths different from real Kinect 
%       depth values.
%
%       Keeping the IR image quantization option 'quant10' to 'on' will
%       result in introducing more noise to the output IR values on the
%       order of 10*log10(2^10) = 30.1 dB, which impacts depth estimation
%       in the depth image simulator. Depending on the inputted IR  
%       intensity and noise models, this may introduce erroneous depth 
%       error, so the user can choose to set this option to 'off' to avoid
%       this.        
%
%       If 'imgrng' is set to a larger range, processing will be slower
%       because pixel templates of the measured IR image need to be 
%       compared to more columns of pixel templates preprocessed reference  
%       image array. Also, if the range is smaller, the error in depth 
%       image estimates will be smaller.
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

function varargout = KinectSimulator_Depth(vertex,face,normalf,varargin)

% DEFAULT PARAMETERS ======================================================
% Depth simulator parameters ----------------------------------------------
% Number of levels to perform interpolation for sub-pixel accuracy
nlev = 8;
% Option to quantize depth image into 11-bit value
isQuant11 = true;
% Option to use depth quantization model
isQuantLoad = true;
% Option to plot IR image from depth image simulator
isPlotIR = false;

% IR simulator parameters -------------------------------------------------
% IR intensity and noise models 
model_Intensity = @(i,r,n,l) i.*5.90e+08.*dot(-n,l)'./r.^2;
model_Speckle   = @(i) i.*gamrnd(4.54,0.196,size(i));
model_Detector  = @(i) i-0.126+10.4.*randn(size(i));
% Size of correlation window used for depth estimation 
corrWind = [9 9];
% Option to add a wall at depth
isWall = false;
% Option to load idealized binary replication of the Kinect dot pattern
isLoadPattern = true;
% If IR intensity model is set to 'none', turn off IR image quantizing
isQuantOK = true;

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
nargoutchk(0, 2)
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
if nargin == 3
    varargin{1} = 'default';
end
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
    k = 5;
    while k < length(varargin)+1
        switch varargin{k}
            % Depth Simulator Parameters
            case 'refine'
                nlev = varargin{k+1};
                if length(nlev) ~= 1 || rem(nlev,1)~=0 || nlev<=0 || ~isnumeric(nlev)
                    error('Sub-pixel refinement value must be integer greater than zero.')
                end
                k = k+2;
            case 'quant11'
                if strcmp(varargin{k+1},'off')
                    isQuant11 = false;
                elseif strcmp(varargin{k+1},'default')
                    isQuant11 = true;
                    isQuantLoad = true;
                elseif ischar(varargin{k+1})
                    error('The argument ''%s'' is not a valid depth quantization model option.', varargin{k+1})
                elseif sum(varargin{k+1}<=0) > 0 || ~isnumeric(varargin{k+1})
                    error('Quantized depth list must have values greater than 0 mm.')
                else
                    isQuant11 = true;
                    isQuantLoad = false;
                    distReal = varargin{k+1};
                end
                k = k+2;
            case 'displayIR'
                if strcmp(varargin{k+1},'off')
                    isPlotIR = false;
                elseif strcmp(varargin{k+1},'on')
                    isPlotIR = true;
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
                if size(find(dotPattern==0),1)+size(find(dotPattern==1),1) ~= numel(dotPattern)
                    error('Dot pattern image must be binary.')
                end
                k = k+2;
            case 'quant10'
                if strcmp(varargin{k+1},'off')
                elseif strcmp(varargin{k+1},'on')
                else
                    error('Quantize IR image option must be set to ''off'' or ''on''.')
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
% PROCESS DEPTH IMAGE =====================================================
% =========================================================================

% PREPROCESS PARAMETERS ===================================================
% Determine horizontal and vertical focal lengths
ImgFOV = ImgFOV*(pi/180); % rad
FocalLength = [ImgRes(2)/(2*tan(ImgFOV(2)/2)); ImgRes(1)/(2*tan(ImgFOV(1)/2))]; % pix

% Number of rows and columns to pad IR image for cross correlation
corrRow = (corrWind(1)-1)/2;
corrCol = (corrWind(2)-1)/2;

% Find min and max depths for ref image so dots intersect one pixel -------
if baseRT*FocalLength(1)/ImgRng(2) < 1
    error('Maximum depth is too large to compute good max reference image.')
end

% Set new depth and find offset disparity for minimum reference image
dOff_min   = ceil(baseRT*FocalLength(1)/ImgRng(1));
minRefDpth = baseRT*FocalLength(1)/dOff_min;

% Set new depth and find offset disparity for maximum reference image
dOff_max   = floor(baseRT*FocalLength(1)/ImgRng(2));
maxRefDpth = baseRT*FocalLength(1)/dOff_max;

% Number of disparity levels between min and max depth 
numIntDisp = dOff_min - dOff_max + 1;

% Number of pixels in correlation window
windSize = corrWind(1)*corrWind(2);

% Preprocess depths for all simulated disparities
disp_all  = dOff_min:-1/nlev:dOff_max;
depth_all = baseRT*FocalLength(1)./disp_all;

% Load idealized binary replication of the Kinect dot pattern -------------
if isLoadPattern 
    load('default_load_files\kinect_pattern_3x3.mat')
    % Check if depth range provides enough coverage for reference images
    if dOff_min-dOff_max > size(dotPattern,2)/3
        error(sprintf('Depth range too large for default dot pattern in order to achieve no pattern overlap.\nHint: Try a minimum depth of at least 204 mm.'))
    end
else
    % Check if depth range provides enough coverage for reference images
    if dOff_min-dOff_max > size(dotPattern,2)
        error('Depth range too large for size of dot pattern in order to achieve no pattern overlap.')
    end
    minRow = ImgRes(1) + corrWind(1) - 1;
    if size(dotPattern,1) < minRow
        error(['Dot pattern must have at least ' num2str(minRow) ' rows for a correlation window with ' num2str(corrWind(1)) ' rows'])
    end
end

% IR dot pattern sizes
ImgSizeDot = size(dotPattern);

% Preprocess reference IR images and indices ------------------------------
[IR_ref, IR_ind] = Preprocess_RefImgs(varargin);
    
% Quantize simulated depths -----------------------------------------------
if isQuant11
    % Load default list of quantized depths
    if isQuantLoad
        load('default_load_files/real_distances.mat')
        distReal = [1:distReal(1) distReal(2:end)];
    end
    
    % Find closest quantized depth value
    distQuant = zeros(size(depth_all));
    for idist = 1:length(depth_all)
        [~,indx] = min(abs(depth_all(idist)-distReal));
        distQuant(idist) = distReal(indx);
    end
    depth_all = distQuant;
end

% GENERATE OUTPUT DEPTH IMAGE =============================================
% Set up scene by adding wall CAD to object CAD ---------------------------
if isWall
    if strcmp(wallDist,'max') || nargin < 7
        wallDist = ImgRng(2);
    end
    if wallDist < ImgRng(1) || wallDist > ImgRng(2)
        error('The wall depth must be between the min and max operational depths.')
    end
    
    % Make corners of the wall span FOV at max distance
    minDisparity = ceil((baseRT*FocalLength(1))/minRefDpth);
    maxDisparity = floor((baseRT*FocalLength(1))/maxRefDpth);

    % Number of cols cannot exceed size of dot pattern (for simplicity of coding)
    pixShftLeft_T = min([ImgSizeDot(2),max([0, floor((ImgRes(2)-ImgSizeDot(2))/2)+1+minDisparity+corrCol])]);
    pixShftRght_T = min([ImgSizeDot(2),max([0, floor((ImgRes(2)-ImgSizeDot(2))/2)+1-maxDisparity+corrCol])]);

    maxPixRow = max([ceil(ImgSizeDot(1)/2) ceil(ImgRes(1)/2)]); 
    maxPixCol = max([ceil(ImgSizeDot(2)/2) ceil(ImgRes(2)/2)]); 

    wallX1 =  maxRefDpth*tan((maxPixCol+pixShftLeft_T)*ImgFOV(2)/ImgRes(2));
    wallX2 = -maxRefDpth*tan((maxPixCol+pixShftRght_T)*ImgFOV(2)/ImgRes(2)) - baseRT;
    wallY1 =  maxRefDpth*tan((maxPixRow+corrRow)*ImgFOV(1)/ImgRes(1));
    wallY2 = -maxRefDpth*tan((maxPixRow+corrRow)*ImgFOV(1)/ImgRes(1));

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

% Generate IR image of object ---------------------------------------------
% Change input parameter options for measured IR image 
if nargin < 7
    varargin_now = {};
else
    varargin_now = varargin(5:end);
end

varargin_now{end+1} = 'display';
varargin_now{end+1} = 'off';

% Binary IR image
IR_bin = KinectSimulator_IR(vertex,face,normalf,'none','none','none',[],varargin_now);

% Noisy IR image
if isPlotIR
    varargin_now{end-1} = 'display';
    varargin_now{end}   = 'on';
end
if isQuantOK
    if nargout > 1
        [IR_now, IRimg] = KinectSimulator_IR(vertex,face,normalf,model_Intensity,model_Speckle,model_Detector,[],varargin_now);
        
        % Store image into function output variable
        varargout{2} = IRimg;
    else
        IR_now = KinectSimulator_IR(vertex,face,normalf,model_Intensity,model_Speckle,model_Detector,[],varargin_now);
    end
else
    if nargout > 1
        [IR_now, IRimg] = KinectSimulator_IR(vertex,face,normalf,'none','none','none',[],varargin_now);
        
        % Store image into function output variable
        varargout{2} = IRimg;
    else
        IR_now = KinectSimulator_IR(vertex,face,normalf,'none','none','none',[],varargin_now);
    end
end

% Estimate depth image of object with correlation window ------------------
DEPTHimg = zeros(ImgRes);

for ipix = 1:prod(ImgRes)
    % Binary window
    window_bin = IR_bin(IR_ind(:,ipix));
    
    % Noisy window
    window_now = IR_now(IR_ind(:,ipix));
    
    if sum(window_now) ~= 0
        % Estimate integer disparity with binary IR image -----------------
        snorm_ref = IR_ref(IR_ind(:,ipix),nlev,:);
        snorm_ref = logical(reshape(snorm_ref,windSize,numIntDisp));
        snorm_now = window_bin - sum(window_bin) / windSize;
        snorm_now = repmat(snorm_now,1,numIntDisp);

        % Maximize horizontal covariance
        horzCov_ref = sum(snorm_ref.*snorm_now);
        [~,dispInd] = max(horzCov_ref);
        dispLookup  = (dispInd-1)*nlev+1;

        % Sub-pixel refinement with noisy IR image ------------------------
        window_sub = IR_ref(IR_ind(:,ipix),:,dispInd);
        window_now = repmat(window_now,1,2*nlev-1);

        % Minimize sum of absolute differences
        horzCov_sub = sum(abs(window_sub-window_now));
        [~,dispInd] = min(horzCov_sub);
        dispLookup  = dispLookup + dispInd - nlev;
        
        % Convert disparity to depth from lookup table --------------------
        DEPTHimg(ipix) = depth_all(dispLookup);
    end
end

% Store image into function output variable
varargout{1} = DEPTHimg;