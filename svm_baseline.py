
node_feature = {}
for each_node in range(0,80) :
    if not each_node in node_feature:
        mapped = len(node_feature)+100
        node_feature[mapped] = each_node
print(node_feature)
print(node_feature[109])
