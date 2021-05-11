function svm = train_svm(train_data,train_label,kertype,C)
% 功能说明：完成SVM训练
% 语法习惯与参数列表：svm = train_svm(train_data,train_label,kertype,B)
% input:
% train_data:训练数据
% train_label：训练数据的类别
% kertype：核函数的类别
% C 惩罚参数
% B 为变量约束中的上界
% output:
% svm：是一个结构体，包含属性如下：
% svm.a :得到的凸二次规划的解
% svm.data ： 支持向量
% svm.label ：支持向量的类别
% ------------*************************···········
% ------------关键函数quadprog的一些说明···········
% 函数quadprog：用于解二次规划问题
% 问题描述：
% min（x）: 0.5·x'·H·x + f'·x
%      
%           A·x <= b,    
% s.t.:   Aeq·x  = beq;
%         lb <=x <= ub;
% 
% 全参数语法结构：x = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
% 变量说明：
% H,A,Qeq是矩阵，f,b,beq,lb,ub,x是向量
% options:选择优化算法并进行设置
% 优化选项设置，对属性进行设置：
% 使用 optimoptions 或 optimset 创建选项(属性)；
% 指定为 optimoptions 的输出或 optimset 等返回的结构体。

% 变量初始化以及超参设置
n = length(train_label); % 对变量的自由约束，上下界
H = (train_label'*train_label).*kernel(train_data,train_data,kertype);% H为yi*yj*K(xi,xj)
f = -ones(n,1); % 保证f为列向量，原式中包含转置操作
A = [];% 不含不等约束
b = [];% 不含不等约束
Aeq = train_label;  % s.t.: aY = 0;
beq = 0;            % s.t.: aY = 0;
lb = zeros(n,1);    % 解：a 的范围  
ub = C*ones(n,1);   % 0 <= a <= C
a0 = zeros(n,1);    % a0是解的初始近似值
options = optimset;  % 'interior-point-convex'（默认优化算法）
options.Display = 'iter-detailed';% 显示优化步骤

% x = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options) 使用 options 中指定的优化选项求解上述问题。
% 使用 optimset 创建 options。如果不提供初始点，设置 x0 = []。
a = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);

% 寻找支持向量；a>e 则认为train_data为支持向量 函数find查找逻辑为真的索引  
e = 1e-4;      

sv_index = find(abs(a)>e);
svm.a = a(sv_index);
svm.data = train_data(:,sv_index);% 作图显示支持向量位置
svm.label = train_label(sv_index);
end