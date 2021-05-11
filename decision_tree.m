% 使用ID3决策树算法预测销量高低
clear ;

% 数据预处理
disp('正在进行数据预处理...');
[matrix,attributes_label,attributes] =  id3_preprocess();

% 构造ID3决策树，其中id3()为自定义函数
disp('数据预处理完成，正在进行构造树...');
tree = id3(matrix,attributes_label,attributes);

% 打印并画决策树
[nodeids,nodevalues] = print_tree(tree);
tree_plot(nodeids,nodevalues);

disp('ID3算法构建决策树完成！');

function [ matrix,attributes,activeAttributes ] = id3_preprocess(  )
% ID3算法数据预处理，把字符串转换为0,1编码

% 输出参数：
% matrix： 转换后的0,1矩阵；
% attributes: 属性和Label；
% activeAttributes : 属性向量，全1；

% 读取数据
txt = {  '序号'    '天气'    '是否周末'    '是否有促销'    '销量'
        ''        '坏'      '是'          '是'            '高'  
        ''        '坏'      '是'          '是'            '高'  
        ''        '坏'      '是'          '是'            '高'  
        ''        '坏'      '否'          '是'            '高'  
        ''        '坏'      '是'          '是'            '高'  
        ''        '坏'      '否'          '是'            '高'  
        ''        '坏'      '是'          '否'            '高'  
        ''        '好'      '是'          '是'            '高'  
        ''        '好'      '是'          '否'            '高'  
        ''        '好'      '是'          '是'            '高'  
        ''        '好'      '是'          '是'            '高'  
        ''        '好'      '是'          '是'            '高'  
        ''        '好'      '是'          '是'            '高'  
        ''        '坏'      '是'          '是'            '低'  
        ''        '好'      '否'          '是'            '高'  
        ''        '好'      '否'          '是'            '高'  
        ''        '好'      '否'          '是'            '高'  
        ''        '好'      '否'          '是'            '高'  
        ''        '好'      '否'          '否'            '高'  
        ''        '坏'      '否'          '否'            '低'  
        ''        '坏'      '否'          '是'            '低'  
        ''        '坏'      '否'          '是'            '低'  
        ''        '坏'      '否'          '是'            '低'  
        ''        '坏'      '否'          '否'            '低'  
        ''        '坏'      '是'          '否'            '低'  
        ''        '好'      '否'          '是'            '低'  
        ''        '好'      '否'          '是'            '低'  
        ''        '坏'      '否'          '否'            '低'  
        ''        '坏'      '否'          '否'            '低'  
        ''        '好'      '否'          '否'            '低'  
        ''        '坏'      '是'          '否'            '低'  
        ''        '好'      '否'          '是'            '低'  
        ''        '好'      '否'          '否'            '低'  
        ''        '好'      '否'          '否'            '低'  }
attributes=txt(1,2:end);
activeAttributes = ones(1,length(attributes)-1);
data = txt(2:end,2:end);

% 针对每列数据进行转换
[rows,cols] = size(data);
matrix = zeros(rows,cols);
for j=1:cols
    matrix(:,j) = cellfun(@trans2onezero,data(:,j));
end

end

function flag = trans2onezero(data)
    if strcmp(data,'坏') ||strcmp(data,'否')...
        ||strcmp(data,'低')
        flag =0;
        return ;
    end
    flag =1;
end


function [ tree ] = id3( examples, attributes, activeAttributes )
% ID3 算法 ，构建ID3决策树
    ...参考：https://github.com/gwheaton/ID3-Decision-Tree

% 输入参数：
% example: 输入0、1矩阵；
% attributes: 属性值，含有Label；
% activeAttributes: 活跃的属性值；-1,1向量，1表示活跃；

% 输出参数：
% tree：构建的决策树；

% 提供的数据为空，则报异常
if (isempty(examples));
    error('必须提供数据！');
end

% 常量
numberAttributes = length(activeAttributes);
numberExamples = length(examples(:,1));

% 创建树节点
tree = struct('value', 'null', 'left', 'null', 'right', 'null');

% 如果最后一列全部为1，则返回“true”
lastColumnSum = sum(examples(:, numberAttributes + 1));

if (lastColumnSum == numberExamples);
    tree.value = 'true';
    return
end
% 如果最后一列全部为0，则返回“false”
if (lastColumnSum == 0);
    tree.value = 'false';
    return
end

% 如果活跃的属性为空，则返回label最多的属性值
if (sum(activeAttributes) == 0);
    if (lastColumnSum >= numberExamples / 2);
        tree.value = 'true';
    else
        tree.value = 'false';
    end
    return
end

% 计算当前属性的熵
p1 = lastColumnSum / numberExamples;
if (p1 == 0);
    p1_eq = 0;
else
    p1_eq = -1*p1*log2(p1);
end
p0 = (numberExamples - lastColumnSum) / numberExamples;
if (p0 == 0);
    p0_eq = 0;
else
    p0_eq = -1*p0*log2(p0);
end
currentEntropy = p1_eq + p0_eq;

% 寻找最大增益
gains = -1*ones(1,numberAttributes); % 初始化增益

for i=1:numberAttributes;
    if (activeAttributes(i)) % 该属性仍处于活跃状态，对其更新
        s0 = 0; s0_and_true = 0;
        s1 = 0; s1_and_true = 0;
        for j=1:numberExamples;
            if (examples(j,i)); 
                s1 = s1 + 1;
                if (examples(j, numberAttributes + 1)); 
                    s1_and_true = s1_and_true + 1;
                end
            else
                s0 = s0 + 1;
                if (examples(j, numberAttributes + 1)); 
                    s0_and_true = s0_and_true + 1;
                end
            end
        end

        % 熵 S(v=1)
        if (~s1);
            p1 = 0;
        else
            p1 = (s1_and_true / s1); 
        end
        if (p1 == 0);
            p1_eq = 0;
        else
            p1_eq = -1*(p1)*log2(p1);
        end
        if (~s1);
            p0 = 0;
        else
            p0 = ((s1 - s1_and_true) / s1);
        end
        if (p0 == 0);
            p0_eq = 0;
        else
            p0_eq = -1*(p0)*log2(p0);
        end
        entropy_s1 = p1_eq + p0_eq;

        % 熵 S(v=0)
        if (~s0);
            p1 = 0;
        else
            p1 = (s0_and_true / s0); 
        end
        if (p1 == 0);
            p1_eq = 0;
        else
            p1_eq = -1*(p1)*log2(p1);
        end
        if (~s0);
            p0 = 0;
        else
            p0 = ((s0 - s0_and_true) / s0);
        end
        if (p0 == 0);
            p0_eq = 0;
        else
            p0_eq = -1*(p0)*log2(p0);
        end
        entropy_s0 = p1_eq + p0_eq;

        gains(i) = currentEntropy - ((s1/numberExamples)*entropy_s1) - ((s0/numberExamples)*entropy_s0);
    end
end

% 选出最大增益
[~, bestAttribute] = max(gains);
% 设置相应值
tree.value = attributes{bestAttribute};
% 去活跃状态
activeAttributes(bestAttribute) = 0;

% 根据bestAttribute把数据进行分组
examples_0= examples(examples(:,bestAttribute)==0,:);
examples_1= examples(examples(:,bestAttribute)==1,:);

% 当 value = false or 0, 左分支
if (isempty(examples_0));
    leaf = struct('value', 'null', 'left', 'null', 'right', 'null');
    if (lastColumnSum >= numberExamples / 2); % for matrix examples
        leaf.value = 'true';
    else
        leaf.value = 'false';
    end
    tree.left = leaf;
else
    % 递归
    tree.left = id3(examples_0, attributes, activeAttributes);
end
% 当 value = true or 1, 右分支
if (isempty(examples_1));
    leaf = struct('value', 'null', 'left', 'null', 'right', 'null');
    if (lastColumnSum >= numberExamples / 2); 
        leaf.value = 'true';
    else
        leaf.value = 'false';
    end
    tree.right = leaf;
else
    % 递归
    tree.right = id3(examples_1, attributes, activeAttributes);
end

% 返回
return
end

function [nodeids_,nodevalue_] = print_tree(tree)
% 打印树，返回树的关系向量
global nodeid nodeids nodevalue;
nodeids(1)=0; % 根节点的值为0
nodeid=0;
nodevalue={};
if isempty(tree) 
    disp('空树！');
    return ;
end

queue = queue_push([],tree);
while ~isempty(queue) % 队列不为空
     [node,queue] = queue_pop(queue); % 出队列

     visit(node,queue_curr_size(queue));
     if ~strcmp(node.left,'null') % 左子树不为空
        queue = queue_push(queue,node.left); % 进队
     end
     if ~strcmp(node.right,'null') % 左子树不为空
        queue = queue_push(queue,node.right); % 进队
     end
end

% 返回 节点关系，用于treeplot画图
nodeids_=nodeids;
nodevalue_=nodevalue;
end

function visit(node,length_)
% 访问node 节点，并把其设置值为nodeid的节点
    global nodeid nodeids nodevalue;
    if isleaf(node)
        nodeid=nodeid+1;
        fprintf('叶子节点，node: %d\t，属性值: %s\n', ...
        nodeid, node.value);
        nodevalue{1,nodeid}=node.value;
    else % 要么是叶子节点，要么不是
        %if isleaf(node.left) && ~isleaf(node.right) % 左边为叶子节点,右边不是
        nodeid=nodeid+1;
        nodeids(nodeid+length_+1)=nodeid;
        nodeids(nodeid+length_+2)=nodeid;

        fprintf('node: %d\t属性值: %s\t，左子树为节点：node%d，右子树为节点：node%d\n', ...
        nodeid, node.value,nodeid+length_+1,nodeid+length_+2);
        nodevalue{1,nodeid}=node.value;
    end
end

function flag = isleaf(node)
% 是否是叶子节点
    if strcmp(node.left,'null') && strcmp(node.right,'null') % 左右都为空
        flag =1;
    else
        flag=0;
    end
end

function tree_plot( p ,nodevalues)
% 参考treeplot函数

[x,y,h]=treelayout(p);
f = find(p~=0);
pp = p(f);
X = [x(f); x(pp); NaN(size(f))];
Y = [y(f); y(pp); NaN(size(f))];

X = X(:);
Y = Y(:);

    n = length(p);
    if n < 500,
        hold on ; 
        plot (x, y, 'ro', X, Y, 'r-');
        nodesize = length(x);
        for i=1:nodesize
%            text(x(i)+0.01,y(i),['node' num2str(i)]); 
            text(x(i)+0.01,y(i),nodevalues{1,i}); 
        end
        hold off;
    else
        plot (X, Y, 'r-');
    end;

xlabel(['height = ' int2str(h)]);
axis([0 1 0 1]);

end

function [ newqueue ] = queue_push( queue,item )
% 进队

% cols = size(queue);
% newqueue =structs(1,cols+1);
newqueue=[queue,item];

end


function [ item,newqueue ] = queue_pop( queue )
% 访问队列

if isempty(queue)
    disp('队列为空，不能访问！');
    return;
end

item = queue(1); % 第一个元素弹出
newqueue=queue(2:end); % 往后移动一个元素位置

end

function [ length_ ] = queue_curr_size( queue )
% 当前队列长度
length_= length(queue);
end

