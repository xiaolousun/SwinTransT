import torch
import collections

# path_old = '/home/cx/TransT/models/DETR_ep0397.pth.tar'
# path_new = '/home/cx/TransT/models/transt_N2.pth'
# # path_1 = '/home/cx/TransT/transt_para.pth'
# # path_2 = '/home/cx/TransT/transt_1.pth'
# # path_3 = '/home/cx/TransT/atom_default.pth'
#
# old = torch.load(path_old)
# new = torch.load(path_new)
#
# old_net = old['net']
# new_net = new['net']
#
# old2new = {}
# old2new_net = collections.OrderedDict()
#
# for key in old_net.keys():
#     if key == 'transformer.decoder.layers.0.norm2.weight':
#         new_key = 'featurefusion_network.decoder.layers.0.norm1.weight'
#         old2new_net[new_key] = old_net[key]
#     elif key == 'transformer.decoder.layers.0.norm2.bias':
#         new_key = 'featurefusion_network.decoder.layers.0.norm1.bias'
#         old2new_net[new_key] = old_net[key]
#     elif key == 'transformer.decoder.layers.0.norm3.weight':
#         new_key = 'featurefusion_network.decoder.layers.0.norm2.weight'
#         old2new_net[new_key] = old_net[key]
#     elif key == 'transformer.decoder.layers.0.norm3.bias':
#         new_key = 'featurefusion_network.decoder.layers.0.norm2.bias'
#         old2new_net[new_key] = old_net[key]
#     elif key[0:11] == 'transformer':
#         new_key = 'featurefusion_network' + key[11:]
#         old2new_net[new_key] = old_net[key]
#     else:
#         old2new_net[key] = old_net[key]
# old2new['net'] = old2new_net
# old2new['constructor'] = new['constructor']
# torch.save(old2new, '/home/cx/TransT/models/transt_N2_got10k_397.pth')

# path = '/home/cx/cx1/trdimp_net.pth.tar'
# model = torch.load(path)
# net = model['net']
# torch.save(net, '/home/cx/cx1/trdimp.pth')

# path1 = '/home/cx/cx1/work_dir/work_dir_mt_2t_e464_iou/checkpoints/ltr/transt/transt_ddp/TransTiouh_ep0090.pth.tar'
# path1 = '/home/xlsun/xlsun/code/TransT-M/results/SwinTransT-M/SwinTransT_ep0500.pth.tar'
# path2 = '/home/xlsun/xlsun/code/TransT-M/results/SwinTransT_Cvt_iouhead/checkpoints/SwinTransTiouh_ep0066.pth.tar'
# model1 = torch.load(path1)
# model2 = torch.load(path2)
# net1 = model1['net']
# net2 = model2['net']
# model3 = {}
# net3 = collections.OrderedDict()
# for key in net1.keys():
#     if key[0:6] == 'transt':
#         net3[key[7:]] = net1[key]
#     else:
#         net3[key] = net1[key]

# model3['net'] = net3
# model3['constructor'] = model2['constructor']
# torch.save(model3, '/home/cx/cx1/e461e90.pth')


# path1 = '/home/xlsun/xlsun/code/TransT-M/results/SwinTransT-M/SwinTransT_ep0500.pth.tar'
path1 = '/home/xlsun/xlsun/code/TransT/results/SwinTransT_Cvt3/checkpoints/ltr/transt/swin_transt_qkvcnn_encoderfusion/SwinTransT_ep0050.pth.tar'
path2 = '/home/xlsun/xlsun/code/TransT/results/SwinTransT_Cvt3_withoutconv/SwinTransT_ep0050.pth.tar'
model1 = torch.load(path1)
model2 = torch.load(path2)
net1 = model1['net']
net2 = model2['net']

net1_keys = set(net1.keys())
net2_keys = set(net2.keys())

iouh_keys = list(net1_keys.difference(net2_keys))

model3 = {}
net3 = collections.OrderedDict()

for key in net2.keys():
    if 'featurefusion_network' in key:
        net3[key] = net2[key]

for key in net1.keys():
    if 'backbone' in key:
        net3[key] = net1[key]

# for key in net1.keys():
#     if 'backbone' in key and 'dwconv' in key:
#         continue
#     else:
#         net3[key] = net1[key]


# net3 = net1.copy()
# for key in iouh_keys:
#     net3[key] = net2[key]

model3['net'] = net3
model3['constructor'] = model2['constructor']
torch.save(model3, '/home/xlsun/xlsun/code/TransT/results/SwinTransT_Cvt3_transfer/SwinTransT_cvt3_encoderfusioin_transfer2_ep0050.pth')

