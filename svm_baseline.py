
# node_feature = {}
# for each_node in range(0,80) :
#     if not each_node in node_feature:
#         mapped = len(node_feature)+100
#         node_feature[mapped] = each_node
# print(node_feature)
# print(node_feature[109])
import numpy as np
import torch
# node_edges={1:2,3:5,4:6,8:10,11:13,14:100}
# sorting = [0,1]
#
# sorting2 = [1]
# sorting2 = np.stack(sorting2)
# sorting3 = [9]
# sorting3 = np.stack(sorting3)
# concat_tag = torch.LongTensor(sorting).view(-1,1)
#
# node_tag = torch.zeros(2,2)
# node_tag.scatter_(1,concat_tag,1)
# node_feat = []
#
# tmp = torch.from_numpy(sorting2).type('torch.FloatTensor')
# node_feat.append(tmp)
# tmp2 = torch.from_numpy(sorting3).type('torch.FloatTensor')
# node_feat.append(tmp2)
#
# node_feat = torch.cat(node_feat,0)
# node_tag1 = node_tag.type_as(node_feat)
# test1 = torch.zeros(3,2)
# test2= torch.randn(2,2)
# node_feat = torch.cat([test2 ,test1],1)
import numpy as np
loss= 0.1
acc = 0.89
test = np.array([loss,acc])*100

print(test)
# print(list(node_edges.keys())[list(node_edges.values()).index(100)])
