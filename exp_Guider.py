import os

# exp = [[12,3],[24,6],[96,24],[96,48]]
# exp = [[3,1],[3,2],[3,3]]
exp = [[12,3],[96,48]]

# lr = [2e-5,3e-5,4e-5]
lr = [1.6e-4,1.5e-4,1.4e-4]
# alpha = [0,0.5]
for lr in lr:
     for seq_in,seq_out in exp:
          os.system('python main_Guider.py --seq_in={:d} --seq_out={:d} --lr={:f}'.format(seq_in,seq_out,lr))
# # lrs = [2e-4,1e-4,5e-5,1e-5,1e-6]
# for lr in lrs:k
#     os.system('python main_Guider.py --alpha=0.5 --hier=True --seq_in=12 --epoch=1000 --seq_out=3 --lr={:f} --pre_hier=True'.format(lr))

