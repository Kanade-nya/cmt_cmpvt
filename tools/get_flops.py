from ptflops import get_model_complexity_info

from models.cmt_pyvit import cmt_pyvit, cmt_pyvit_xxs
from models.edgevit import edgevit_s,edgevit_xxs
# from models.pvt_v2 import pvt_v2_b2
from models.paraViT import paraViT_xxs
from models.v3_shunted import v3_shunted
from models.v4_shunted import v4_shunted_xxs
from models.conv_pvt2mlp12 import conv_pvt2mlp_tiny12,conv_pvt2mlp_light
from models.coat_v1 import coat_v1,coat_v1_light
from models.v5_shunted import v5_shunted, v5_shunted_middle,v5_shunted_large
model = v5_shunted_large()
# xxs: 0.55 gflops 4.07M
# red2_light(仿造xxs):    516.15 MMac     3.37 M
# red2_light[2,2,4,2]:     696.69 MMac     3.79 M
# red2[2,2,5,1]:   2.41GMac 13.07M 用的是这个
# red2[2,2,4,2]:   2.41GMac  15.18 M
# red2_small[3,3,4,3] 19.32 M ,[3,3,6,3] 21.46,[3,3,9,3] 24.66

macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# cmt _ xxs 0.64gflops , 4.35 参数
# paraViT 603.63 , 4.07,
# 修改后？ ->  embed_dims=[36,72,144,288], 750.28 7.35
# 修改后？ ->  num_heads=[2, 4, 8, 16], 598.42 ,5.87
