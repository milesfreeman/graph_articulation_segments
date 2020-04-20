%PREPROCESS_REFIMGS Preprocess reference images and window indices. 
%   PREPROCESS_REFIMGS is a function called by KINECTSIMULATOR_DEPTH to
%   preprocess reference image and index arrays for faster depth image
%   processing.
%
%   [IR_REF, IR_IND] = Preprocess_RefImgs(varargin_ref) returns arrays
%   IR_REF and IR_IND corresponding to the input parameters specified in
%   the KINECTSIMULATOR_DEPTH function call. 
%
%       IR_REF - Contains all reference images at depths corresponding to
%                integer disparities between the minimum and maximum
%                operation depths. This array also contains all possible
%                sub-pixel shifts at each integer disparity depth. These
%                reference images are the same size as the padded IR image
%                returned by the KINECTSIMULATOR_IR function.
%                This array is therefore 3 dimensional, with a size of
%
%                   numel(IRimg) x 2*nlev-1 x numIntDisp
%
%                where numel(IRimg) is the total number of pixels in the
%                padded IR image, nlev is the number of levels set for 
%                sub-pixel refinement, and numIntDisp is the total number 
%                of integer disparities between the minimum and maximum 
%                operational depths.
%
%       IR_IND - Contains the indices of all pixels within the correlation
%                window centered on each IR image pixel. The IR image in
%                this case is the same size and resolution of the real
%                outputted Kinect IR image (480 rows x 640 columns). 
%                This array is therefore 2 dimensional, with a size of 
%
%                   windSize x numel(IRimg_disp)
%
%                where windSize is the total number of pixels in the
%                correlation window (i.e. numel(corrWind), and 
%                numel(IRimg_disp) is the total number of pixels in the
%                output Kinect IR image (i.e. 307200 pixels).

function [IR_ref, IR_ind] = Preprocess_RefImgs(varargin_ref)

% DEFAULT PARAMETERS ======================================================
% Depth simulator parameters ----------------------------------------------
% Number of levels to perform interpolation for sub-pixel accuracy
nlev = 8;

% IR simulator parameters -------------------------------------------------
% Size of correlation window used for depth estimation 
corrWind = [9 9];
% Option to load idealized binary replication of the Kinect dot pattern
isLoadPattern = true;
% Option to quantize the IR image to 10-bit value
isQuant10 = true;
% If IR intensity model is set to 'none', turn off IR image quantizing
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
narginchk(1, 1)
nargoutchk(2, 2)

% SET INPUT PARAMETERS ====================================================
% IR intensity model
if strcmp(varargin_ref{1},'default')
    model_Intensity = @(i,r,n,l) i.*5.90e+08.*dot(-n,l)'./r.^2;
elseif strcmp(varargin_ref{1},'simple')
    model_Intensity = @(i,r,n,l) i.*5.96e+08./r.^2;
elseif strcmp(varargin_ref{1},'none')
    model_Intensity = @(i,r,n,l) i.*ones(size(r));
    isQuantOK = false;
else % User inputted model
    model_Intensity = varargin_ref{1};
end
if length(varargin_ref) > 4
    k = 5;
    while k < length(varargin_ref)+1
        switch varargin_ref{k}
            % Depth Simulator Parameters
            case 'refine'
                nlev = varargin_ref{k+1};
                k = k+2;
            case 'quant11'
                k = k+2;
            case 'displayIR'
                k = k+2;
            % IR Simulator Parameters
            case 'window'
                corrWind = varargin_ref{k+1};
                k = k+2;
            case 'subray'
                k = k+2;
            case 'pattern'
                dotPattern = varargin_ref{k+1};
                isLoadPattern = false;
                k = k+2;
            case 'quant10'
                if strcmp(varargin_ref{k+1},'off')
                    isQuant10 = false;
                elseif strcmp(varargin_ref{k+1},'on')
                    isQuant10 = true;
                end
                k = k+2;
            % Kinect Parameters
            case 'imgfov'
                ImgFOV = varargin_ref{k+1};
                k = k+2;
            case 'imgrng'
                ImgRng = varargin_ref{k+1};
                k = k+2;
        end
    end
end

% =========================================================================
% PREPROCESS PARAMETERS ===================================================
% =========================================================================

% Load idealized binary replication of the Kinect dot pattern -------------
if isLoadPattern 
    load('default_load_files\kinect_pattern_3x3.mat')
else
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

% Number of pixels in correlation window
windSize = corrWind(1)*corrWind(2);

% Preprocess indices for reference and noisy IR images ====================
IR_ind   = uint32(nan(windSize,prod(ImgRes)));

ipix = 0;
for ipix_col = 1:ImgRes(2)
    for ipix_row = 1:ImgRes(1)
        ipix = ipix + 1;

        % Determine indices for correlation window
        row_now = repmat((ipix_row:ipix_row+corrWind(1)-1)',1,corrWind(2));
        col_now = repmat(ipix_col:ipix_col+corrWind(2)-1,corrWind(1),1);
        ind_now = row_now(:) + (col_now(:) - 1).*ImgResPad(1);

        % Store values
        IR_ind(:,ipix)   = ind_now;
    end
end

% Preprocess reference IR images ==========================================
% Determine horizontal and vertical focal lengths
ImgFOV = ImgFOV*(pi/180); % rad
FocalLength = [ImgRes(2)/(2*tan(ImgFOV(2)/2)); ImgRes(1)/(2*tan(ImgFOV(1)/2))]; % pix

% Number of rows and columns to pad IR image for cross correlation
corrRow = (corrWind(1)-1)/2;
corrCol = (corrWind(2)-1)/2;

% Set new depth and find offset disparity for minimum reference image
dOff_min   = ceil(baseRT*FocalLength(1)/ImgRng(1));
minRefDpth = baseRT*FocalLength(1)/dOff_min;

% Set new depth and find offset disparity for maximum reference image
dOff_max   = floor(baseRT*FocalLength(1)/ImgRng(2));
maxRefDpth = baseRT*FocalLength(1)/dOff_max;

% Number of disparity levels between min and max depth 
numIntDisp = dOff_min - dOff_max + 1;

% Preprocess depths for all simulated disparities
disp_all  = dOff_min:-1/nlev:dOff_max;
depth_all = baseRT*FocalLength(1)./disp_all;

% Add columns of dot pattern to left and right side based on disparity equation
minDisparity = ceil((baseRT*FocalLength(1))/minRefDpth);
maxDisparity = floor((baseRT*FocalLength(1))/maxRefDpth);

% Number of cols cannot exceed size of dot pattern (for simplicity of coding)
pixShftLeft_T = min([ImgSizeDot(2),max([0, floor((ImgRes(2)-ImgSizeDot(2))/2)+1+minDisparity+corrCol])]);
pixShftRght_T = min([ImgSizeDot(2),max([0, floor((ImgRes(2)-ImgSizeDot(2))/2)+1-maxDisparity+corrCol])]);

% Preprocess parameters for transmitter rays ------------------------------
% Generage reference image of entire IR pattern projection
dotAddLeft = dotPattern(:,end-pixShftLeft_T+1:end);
dotAddRght = dotPattern(:,1:pixShftRght_T);
dotAdd     = [dotAddLeft dotPattern dotAddRght];
dotIndx    = find(dotAdd==1);
ImgSizeAdd = size(dotAdd);

% Convert index to subscript values (based on ind2sub)
jpix_y = rem(dotIndx-1, ImgSizeAdd(1)) + 1;
jpix_x = (dotIndx - jpix_y)/ImgSizeAdd(1) + 1;

% Determine where IR dots split to the left of the main pixel
indxLeft = jpix_x>1;
dotIndxLeft = jpix_y(indxLeft) + (jpix_x(indxLeft) - 2).*ImgSizeAdd(1);
indxRght = jpix_x<ImgSizeAdd(2);
dotIndxRght = jpix_y(indxRght) + jpix_x(indxRght).*ImgSizeAdd(1);

% Crop reference image to fit padded size 
minFiltRow = 1-corrRow;
maxFiltRow = ImgResPad(1)-corrRow;
cropRow    = max([0, ((ImgSizeDot(1)-ImgRes(1))/2)+adjRowShift]);

rowRange = (minFiltRow:maxFiltRow)+cropRow;
colRange = 2:2-1+ImgResPad(2)-1+1;

% Create angles of rays for each sub-pixel from transmitter
vertPixLeft_T = ImgSizeDot(1)/2-1/2;
horzPixLeft_T = ImgSizeDot(2)/2-1/2+pixShftLeft_T;

% Adjust subscript value to sensor coordinate system values
spix_x = horzPixLeft_T-(jpix_x-1)+adjColShift;
spix_y = vertPixLeft_T-(jpix_y-1)+adjRowShift;

% Determine 3D location of where each ray intersects unit range
X_T = spix_x / FocalLength(1);
Y_T = spix_y / FocalLength(2);
Z_T = ones(size(dotIndx));
XYZ_T = sqrt(X_T.^2 + Y_T.^2 + Z_T.^2);

% Find surface normal for all intersecting dots
sn_T = [zeros(1,size(dotIndx,1));...
        zeros(1,size(dotIndx,1));...
        -1*ones(1,size(dotIndx,1))];

% Find the IR light direction for all sub-rays
ld_T = bsxfun(@rdivide,[X_T Y_T Z_T]',XYZ_T');

% Preprocess fractional intensity arrays
leftMain = reshape(1-1/nlev:-1/nlev:1/nlev,1,1,nlev-1);
leftSplt = 1 - leftMain;
rghtMain = reshape(1/nlev:1/nlev:1-1/nlev,1,1,nlev-1);
rghtSplt = 1 - rghtMain;

% Preprocess reference images with intensities for lookup table -----------
IR_ref = single(nan(prod(ImgResPad),2*nlev-1,numIntDisp));

for idisp = 1:numIntDisp
    idepth = depth_all((idisp-1)*nlev+1);

    % Compute range of all dots wrt transmitter
    rng_T = idepth*XYZ_T;

    % Compute intensities for all dots
    intensity_T = model_Intensity(1,rng_T,sn_T,ld_T);
    
    % Compute reference image where IR dots interesect one pixel
    IR_ref_main = zeros(size(dotAdd));
    IR_ref_main(dotIndx) = intensity_T;
    IR_ref_main = IR_ref_main(rowRange,colRange+idisp-1);
    
    % Store reference image
    IR_ref(:,nlev,idisp) = IR_ref_main(:);
    
    if idisp == 1
        % Compute reference images where IR dots split with left pixel
        IR_ref_left = zeros(size(dotAdd));
        IR_ref_left(dotIndxLeft) = intensity_T(indxLeft);
        IR_ref_left = IR_ref_left(rowRange,colRange+idisp-1);
        IR_ref_left = bsxfun(@times,repmat(IR_ref_main,1,1,nlev-1),leftMain)+...
                      bsxfun(@times,repmat(IR_ref_left,1,1,nlev-1),leftSplt);
         
        % Store reference images
        IR_ref(:,nlev+1:2*nlev-1,idisp) = reshape(IR_ref_left,[],nlev-1);
    elseif idisp == numIntDisp
        % Compute reference images where IR dots split with right pixel
        IR_ref_rght = zeros(size(dotAdd));
        IR_ref_rght(dotIndxRght) = intensity_T(indxRght);
        IR_ref_rght = IR_ref_rght(rowRange,colRange+idisp-1);
        IR_ref_rght = bsxfun(@times,repmat(IR_ref_main,1,1,nlev-1),rghtMain)+...
                      bsxfun(@times,repmat(IR_ref_rght,1,1,nlev-1),rghtSplt);
                  
        % Store reference images
        IR_ref(:,1:nlev-1,idisp) = reshape(IR_ref_rght,[],nlev-1);
    else
        % Compute reference images where IR dots split with left pixel
        IR_ref_left = zeros(size(dotAdd));
        IR_ref_left(dotIndxLeft) = intensity_T(indxLeft);
        IR_ref_left = IR_ref_left(rowRange,colRange+idisp-1);
        IR_ref_left = bsxfun(@times,repmat(IR_ref_main,1,1,nlev-1),leftMain)+...
                      bsxfun(@times,repmat(IR_ref_left,1,1,nlev-1),leftSplt);
                  
        % Compute reference images where IR dots split with right pixel
        IR_ref_rght = zeros(size(dotAdd));
        IR_ref_rght(dotIndxRght) = intensity_T(indxRght);
        IR_ref_rght = IR_ref_rght(rowRange,colRange+idisp-1);
        IR_ref_rght = bsxfun(@times,repmat(IR_ref_main,1,1,nlev-1),rghtMain)+...
                      bsxfun(@times,repmat(IR_ref_rght,1,1,nlev-1),rghtSplt);
        
        % Store reference images
        IR_ref(:,1:nlev-1,idisp) = reshape(IR_ref_rght,[],nlev-1);
        IR_ref(:,nlev+1:2*nlev-1,idisp) = reshape(IR_ref_left,[],nlev-1);
    end
end
if isQuant10 && isQuantOK
    IR_ref = round(IR_ref);
end