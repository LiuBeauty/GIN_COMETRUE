import matplotlib.pyplot as plt

acc = [0.2,0.3,0.6]
epoch = [1,2,3]
loss = [0.8,0.6,0.4]
plt.figure(1)
plt.plot(epoch,acc,label = 'jfsd')
plt.savefig('./Pic/tt.png')
plt.figure(2)
plt.plot(epoch,loss,label='sdf')
plt.savefig('./Pic/tt2.png')