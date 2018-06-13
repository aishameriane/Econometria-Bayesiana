
%  procedure to do a kalman filter with measurement error for the system:
%       Zt+1 = FZt + vt+1,   Evv' = Q
%       Yt = H'Zt + ut, Euu' = R
%

% 
% Inputs:
% 	y, Txnvars matrix of data
% 	f,h,q,r: see above
%
% Output:
% 	like, log likelihood value
% 
% Guilherme Valle Moura 2008
% 

function [like,res,S]= kalman(y,f,h,q,r)


[T,nvars]=size(y);
rowsff = size(f,1);

% Initialize states
zt = zeros(rowsff,1);
Sig = reshape(inv(eye(rowsff^2) - kron(f,f))*q(:),rowsff,rowsff);
if isinf(Sig)
   Sig = 100;
end



for t=1:T    
    omegt  = h'*Sig*h+r; % cov[y,y_hat(t|t-1)]    
    hphinv = inv(omegt);
    phhhp  = Sig*h*hphinv; %f^(-1)*Kalman gain
    yres   = y(t,:)'-h'*zt;   
    res(:,t) = yres;
    ztt    = zt+phhhp*yres;    % Updated states z_t|t
    S(:,t) = ztt;
    ptt    = Sig-phhhp*h'*Sig; % MSE associated with updated projection
    zt     = f*ztt;            % Forecast of next period's state
    Sig    = f*ptt*f'+q;       %MSE of forecast   
    llfn(t)= -.5*nvars*log(2*pi)+0.5*log(det(hphinv)) - 0.5*yres'*hphinv*yres;% loglikelihood contribution
end

like = sum(llfn); % loglikelihood
