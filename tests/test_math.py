import torch
from matrixsplat.render import blend_reference,blend_matrix,weights,render_dense,Camera
from matrixsplat.scenes import make_scene

def test_matrix_matches_independent_loop():
    torch.manual_seed(1);w=torch.rand(17,23,dtype=torch.float64);f=torch.rand(23,7,dtype=torch.float64);b=torch.rand(7,dtype=torch.float64)
    torch.testing.assert_close(blend_matrix(w,f,.1,b),blend_reference(w,f,.1,b),atol=1e-12,rtol=1e-12)

def test_quotient_gradient_finite_difference():
    torch.manual_seed(2);w=torch.rand(5,4,dtype=torch.float64,requires_grad=True);f=torch.rand(4,3,dtype=torch.float64,requires_grad=True);b=torch.rand(3,dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda x,y:blend_matrix(x,y,.1,b),(w,f))

def test_weight_parameters_gradient():
    torch.manual_seed(3);a=torch.rand(5,4,dtype=torch.float64,requires_grad=True);z=torch.linspace(1,3,4,dtype=torch.float64);s=torch.tensor(5.,dtype=torch.float64,requires_grad=True);v=torch.rand(4,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a,s,v:weights(a,z,s,v),(a,s,v))

def test_end_to_end_reference():
    args=make_scene(0,16,3,dtype=torch.float64);cam=Camera(16,16,20)
    torch.testing.assert_close(render_dense(*args,cam),render_dense(*args,cam,mode='reference'),atol=1e-12,rtol=1e-12)

def test_streaming_and_channels_preserve_outputs():
    from matrixsplat.render import render_streamed
    args=make_scene(4,19,17,dtype=torch.float64);cam=Camera(19,17,25)
    torch.testing.assert_close(render_dense(*args,cam),render_streamed(*args,cam,pixel_batch=41,channel_batch=5),atol=1e-12,rtol=1e-12)

def test_projection_gradient():
    from matrixsplat.render import project
    args=make_scene(5,3,3,dtype=torch.float64);means=args[0].requires_grad_();cov=args[1].requires_grad_()
    assert torch.autograd.gradcheck(lambda m,c:project(m,c,Camera())[0],(means,cov))

def test_tiles_match_dense_nonmultiple_image_and_gradients():
    from matrixsplat.tiled import render_tiled
    args=make_scene(8,37,9,dtype=torch.float64);cam=Camera(29,23,29)
    f=args[-1].requires_grad_();args=(*args[:-1],f)
    a=render_dense(*args,cam);b=render_tiled(*args,cam,channel_batch=4)
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    ga=torch.autograd.grad(a.square().sum(),f,retain_graph=True)[0];gb=torch.autograd.grad(b.square().sum(),f)[0]
    torch.testing.assert_close(ga,gb,atol=1e-10,rtol=1e-10)

def test_empty_tiles_and_behind_camera():
    from matrixsplat.tiled import render_tiled
    args=list(make_scene(3,5,3,dtype=torch.float64));args[0][:,2]=-1
    bg=torch.tensor([.2,.3,.4],dtype=torch.float64)
    out=render_tiled(*args,Camera(16,16),bg=bg)
    torch.testing.assert_close(out,bg.expand_as(out))

def test_tiled_geometry_gradients_match_dense():
    from matrixsplat.tiled import render_tiled
    args=list(make_scene(21,12,3,dtype=torch.float64));args=[x.requires_grad_() for x in args];cam=Camera(17,19,21)
    a=render_dense(*args,cam);b=render_tiled(*args,cam)
    ga=torch.autograd.grad(a.square().sum(),args,retain_graph=True);gb=torch.autograd.grad(b.square().sum(),args)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=1e-8,atol=1e-8)

def test_permutation_invariance_and_background():
    torch.manual_seed(22);w=torch.rand(11,16,dtype=torch.float64);f=torch.rand(16,6,dtype=torch.float64);bg=torch.rand(6,dtype=torch.float64);order=torch.randperm(16)
    torch.testing.assert_close(blend_matrix(w,f,.01,bg),blend_matrix(w[:,order],f[order],.01,bg))
    torch.testing.assert_close(blend_matrix(torch.zeros_like(w),f,.01,bg),bg.expand(11,6))

def test_analytic_weight_gradient_matches_autograd():
    torch.manual_seed(11);w=torch.rand(7,8,dtype=torch.float64,requires_grad=True);f=torch.rand(8,5,dtype=torch.float64);bg=torch.rand(5,dtype=torch.float64);upstream=torch.rand(7,5,dtype=torch.float64)
    image=blend_matrix(w,f,.1,bg);grad=torch.autograd.grad((image*upstream).sum(),w)[0]
    analytical=((f[None,:,:]-image[:,None,:])*upstream[:,None,:]).sum(-1)/(w.sum(-1,keepdim=True)+.1)
    torch.testing.assert_close(grad,analytical,atol=1e-12,rtol=1e-12)

def test_standard_ply_import(tmp_path):
    import numpy as np
    from plyfile import PlyData,PlyElement
    from matrixsplat.io import load_ply
    names=['x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3','opacity','f_dc_0','f_dc_1','f_dc_2']
    data=np.zeros(1,dtype=[(x,'f4') for x in names]);data['z']=3;data['rot_0']=1
    for key,value in zip(['scale_0','scale_1','scale_2'],[.1,.2,.3]):data[key]=np.log(value)
    path=tmp_path/'one.ply';PlyData([PlyElement.describe(data,'vertex')]).write(path)
    means,cov,opacity,rgb=load_ply(path)
    torch.testing.assert_close(means,torch.tensor([[0.,0.,3.]]));torch.testing.assert_close(cov,torch.diag(torch.tensor([.01,.04,.09]))[None],atol=1e-7,rtol=1e-5)
    torch.testing.assert_close(opacity,torch.tensor([.5]));torch.testing.assert_close(rgb,torch.full((1,3),.5))
