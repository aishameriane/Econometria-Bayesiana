function [lks,z,epsdraw,p]=kalmansmoother(y,f,h,q,r)

%  procedure to do a kalman filter and smoother with measurement error for the system:
%       Zt+1 = FZt + vt+1,   Evv' = Q
%       Yt = H'Zt + ut, Euu' = R
%

% 
% Inputs:
% 	y, Txnvars matrix of data
% 	f,h,q,r: see above
%
% Output:
% 	lks: log likelihood value
%   z: smoothed zs
%   epsdraws: smoothed system inovations
%   p: vcv matrices of zs 
% 
% Guilherme Valle Moura 31/03/2009




[T,nvars]=size(y);
rowsff = size(f,1);

% Initialize states
zttm1 = zeros(rowsff,1);
z     = [zttm1' zttm1'];
pttm1 = reshape(inv(eye(rowsff^2) - kron(f,f))*q(:),rowsff,rowsff);

p = [pttm1 pttm1 pttm1];
iq = eye(rowsff);
sttm1 = eye(rowsff);


for t=1:T    
    omegt  = h'*pttm1*h+r; % cov[y,y_hat(t|t-1)]    
    hphinv = pinv(omegt);
    phhhp  = pttm1*h*hphinv; %f^(-1)*Kalman gain
    yres   = y(t,:)'-h'*zttm1;   
    wyres(t,:) = (f'*h*hphinv*yres)';
    sttm1  = [sttm1;f'*(iq-h*hphinv*h'*pttm1)];
    ptt    = pttm1-phhhp*h'*pttm1; % MSE associated with updated projection
    pttm1  = f*ptt*f'+q;   %MSE of forecast
    p      = [p;ptt pttm1 pttm1];    
    ztt    = zttm1+phhhp*yres; % Updated states z_t|t      
    zttm1  = f*ztt;     % Forecast of next period's state
    z      = [z;ztt' zttm1'];       
    llfn(t)= -.5*nvars*log(2*pi)+0.5*log(det(hphinv)) - 0.5*yres'*hphinv*yres;% loglikelihood contribution
end
% Delete first rows of p and z
p = p(rowsff+1:size(p,1),:);
sttm1 = sttm1(rowsff+1:size(sttm1,1),:);
z = z(2:size(z,1),:);
T = size(z,1);
lks = sum(llfn); % loglikelihood

% Start Smoother
epsdraw = 0;
ztcapt = z(T,1:rowsff)';
ptcapt = p(rowsff*(T-1)+1:rowsff*T,1:rowsff);
p(rowsff*(T-1)+1:rowsff*T,2*rowsff+1:3*rowsff) = ptcapt;
z(T,rowsff+1:2*rowsff) = ztcapt';
epsdraw = zeros(T,rowsff);
i = T;
mttm1 = wyres(T,:)';
while i>1
    i=i-1;
    % reload kalman filter values
    ptt = p(rowsff*(i-1)+1:rowsff*i,1:rowsff);
    ztt = z(i,1:rowsff)';
    ztcapt = ztt + ptt*mttm1;
    epsdraw(i+1,:) = z(i+1,rowsff+1:2*rowsff)-(f*ztcapt)';  
    z(i,rowsff+1:2*rowsff) = ztcapt';
    mttm1 = sttm1(rowsff*i+1:rowsff*(i+1),1:rowsff)*mttm1+wyres(i,:)';
end
%z = z(:,rowsff+1:end);

    

    
    


