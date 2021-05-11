function K = kernel(X,Y,kerneltype)
% 功能：支持多种核运算;
% 语法结构：K = kernel(X,Y,kerneltype)，kerneltype选择核技巧
% 'linear'：线性内积
%           K(v1,v2) = <v1,v2>
% 'gaussian'：高斯核 %
%           K(v1,v2)=exp(-gama||v1-v2||^2)
% 'sigmoid':sigmoid核；双曲正切函数
%           K(v1,v2)=tanh(gama<v1,v2>+c)    
% 'mullinear':多项式核
%           K（v1,v2）=<v1,v2>^d;d为多项式的次数
% 'triangle':三角核
%           K（v1,v2）=-||v1-v2||^d

% 在svm中运用线性，高斯或者sigmoid效果比较好
switch kerneltype
    case 'linear' % 线性内积
        K = X'*Y;
    case 'sigmoid'
        belta = 0.01;
        theta = 0.001;
        K = tanh(belta*X*Y+theta);
    case 'gaussian'% k(v1,v2) = exp(-||v1-v2||^2/(2sigma^2))
        delta = 2*1.414;
        delta = delta*delta;
        XX = sum(X'.*X',2);
        YY = sum(Y'.*Y',2);
        XY = X'*Y;
        K = abs(repmat(XX,[1 size(YY,1)]) + repmat(YY',[size(XX,1) 1]) - 2*XY);
        K = exp(-K./delta);
    case 'mullinear'
        K = (X'*Y).^2;
%     case'triangle'
%         K = -norm(X-Y,1)^2;
        
end
end