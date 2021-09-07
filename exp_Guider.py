import os

exp = [[12,3],[24,6],[96,24],[96,48]]
# alpha = [0,0.5]
for seq_in,seq_out in exp:
     os.system('python main_Guider.py --seq_in={:d} --seq_out={:d}'.format(seq_in,seq_out))
# # lrs = [2e-4,1e-4,5e-5,1e-5,1e-6]
# for lr in lrs:
#     os.system('python main_Guider.py --alpha=0.5 --hier=True --seq_in=12 --epoch=1000 --seq_out=3 --lr={:f} --pre_hier=True'.format(lr))


