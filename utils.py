#@title plot_domain_for_model
import torch.nn.functional as F
import torch
import splinecam as splinecam
import matplotlib.pyplot as plt
import os

def plot_domain_for_model(test_model, folder, fig_name, true_model):
    test_model.type(torch.float64)
    xlow,ylow = -1.0, -1.0
    xhigh,yhigh = 1.0, 1.0

    domain = torch.tensor([
        [xlow,ylow],
        [xlow,yhigh],
        [xhigh,yhigh],
        [xhigh,ylow],
        [xlow,ylow]
    ])
    T = splinecam.utils.get_proj_mat(domain)
    NN = splinecam.wrappers.model_wrapper(
        test_model,
        input_shape=(20,),
        T = T,
        dtype = torch.float64,
        device = 'cuda'
    )

    flag =  NN.verify()
    assert flag

    # Get partitions
    out_cyc,endpoints,Abw = splinecam.compute.get_partitions_with_db(domain,T,NN)

    plt.rcParams['figure.dpi'] = 300

    fig,ax = plt.subplots()

    for each in endpoints:
        if each is not None:
            ax.plot(each[:,0],each[:,1],c='r',zorder=1000000000,linewidth=5)

    minval,_ = torch.vstack(out_cyc).min(0)
    maxval,_ = torch.vstack(out_cyc).max(0)

    splinecam.plot.plot_partition(out_cyc, xlims=[minval[0],maxval[0]],alpha=0.3,
                              edgecolor='#a70000',color_range=[.3,.8],ax=ax,colors=['none'],
                              ylims=[minval[1],maxval[1]], linewidth=.5)

    if not os.path.exists(folder):
        os.mkdir(folder)
    path = os.path.join(folder, f'{fig_name}.png')
    plt.savefig(path,bbox_inches='tight',pad_inches=0)
    plt.close()
    test_model.type(torch.float32)
