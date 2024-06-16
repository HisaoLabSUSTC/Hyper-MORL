from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
import numpy as np
import matplotlib
import argparse
import os
from multiprocessing import Process
import sys
def format_value(v):
    formatted_num = "%.2e"%(v)
    parts = formatted_num.split('e')
    
    exponent = int(parts[1])
    base = 10**exponent
    output = "%.2fE%d"%(v/base,exponent)
    return output

class PreferenceMarker:
    def __init__(self, pref_ax, info_ax, obj_ax, prefs, objs, obj_scale):
        self.pref_marker = None
        self.obj_marker = None
        self.pref = None
        self.isMove = False
        self.pref_ax = pref_ax
        self.info_ax = info_ax
        self.obj_ax = obj_ax
        self.prefs = prefs
        self.objs = objs
        self.obj_scale = obj_scale
       
        self.pref_text = info_ax.text(0.02, 0.7, "", fontsize = 15)
        self.obj_text = info_ax.text(0.02, 0.4, "", fontsize = 15)
        self.info_ax.text(0.2, 0.1, "Click for Visualization", fontsize = 15)
        self.button_pos = [0.18, 0.05, 0.8, 0.3]
        self.info_ax.add_patch(plt.Rectangle((self.button_pos[0], self.button_pos[1]), self.button_pos[2] - self.button_pos[0],
                               self.button_pos[3] - self.button_pos[1], facecolor= [0.9, 0.9, 0.9], edgecolor='black', lw=1))    
        self.update_pref(np.array([0.5, 0.5]))

    def update_pref(self, new_pref):
        if self.pref_marker is not None:
            self.pref_marker.remove()

        if self.obj_marker is not None:
            self.obj_marker.remove()
        
        distances = np.sum(np.power(self.prefs - new_pref, 2), axis = -1)
        index = np.argmin(distances)
        self.pref = prefs[index]
        obj_value = self.objs[index]
        self.pref_marker = self.pref_ax.plot(self.pref[0], self.pref[1], marker='o', linestyle = '-', markersize= 20, 
                linewidth=5, markerfacecolor='green', markeredgecolor = 'black')[0]
        self.obj_marker = self.obj_ax.plot(obj_value[0] / self.obj_scale[0], obj_value[1] / self.obj_scale[1], marker='o', linestyle = '-', markersize= 20, 
            linewidth=5, markerfacecolor='green', markeredgecolor = 'black')[0]
        
        self.pref_text.set_text("Input Preference = (%.4f, %.4f)" % (self.pref[0], self.pref[1]))
        self.obj_text.set_text("Objective Values = (%s, %s)" % (format_value(obj_value[0]), format_value(obj_value[1])))
    
    def start_move(self, ix, iy):
        eps = 0.05
        if abs(ix - self.pref[0]) >= eps or abs(iy - self.pref[1]) >= eps: return
        self.isMove = True
        self.last_pos = np.array([ix, iy])

    def end_move(self):
        #print('pref=', self.pref[0], self.pref[1])
        self.isMove = False
    
    def move(self, ix, iy):
        if not self.isMove: return
        dpos = np.array([ix, iy]) - self.last_pos
        v = np.array([1, -1]) / np.sqrt(2)
        new_pref = np.dot(dpos, v) * v + self.pref
        np.clip(new_pref, 0, 1)
        self.update_pref(new_pref)
        self.last_pos = np.array([ix, iy])

def play_mujoco():
    play_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mujoco_play.py')
    print(play_script)
    model_path = os.path.join(dir_name, "pretrained", args.env_name, "Hyper-MORL/default/%d/"%(args.run_id))
    if args.env_name == 'MO-Hopper-v3':
        play_cmd = 'python {} --env {} --model {} --pref1 {} --pref2 {} --pref3 {}'.format(play_script, args.env_name, model_path, *pref_marker.pref)
    else:
        play_cmd = 'python {} --env {} --model {} --pref1 {} --pref2 {}'.format(play_script, args.env_name, model_path, *pref_marker.pref)
    Process(target=os.system, args=(play_cmd,)).start()

def mouse_press(event):
    if event.inaxes == pref_ax:
        if event.button != MouseButton.LEFT: return
        ix, iy = event.xdata, event.ydata
        global pref_marker
        pref_marker.start_move(ix, iy)
    elif event.inaxes == info_ax:
        if event.button != MouseButton.LEFT : return
        ix, iy = event.xdata, event.ydata
        pos = pref_marker.button_pos
        if ix >= pos[0] and iy >= pos[1] and ix <= pos[2] and iy <= pos[3]:
            play_mujoco()
    else:
        return
    
def mouse_release(event):
    if event.button != MouseButton.LEFT: return
    #print('Mouse release')
    global pref_marker
    pref_marker.end_move()

def mouse_move(event):
    if event.inaxes != pref_ax: return
    ix, iy = event.xdata, event.ydata
    global pref_marker
    pref_marker.move(ix, iy)
    fig.canvas.draw()

def draw_2objectives(ax, objs, colors, env_name):
    # Settings
    if objs.shape[1] == 3:
        draw_3objectives(ax, objs, colors)
        return
    env_name_hyper = ["MO-Swimmer-v2", "MO-HalfCheetah-v2", "MO-Walker2d-v2","MO-Ant-v2","MO-Hopper-v2","MO-Humanoid-v2"]
    xlabel = ["Forward speed", "Forward speed","Forward speed","X-axis speed", "Forward speed","Forward speed"]
    ylabel = ["Energy efficiency","Energy efficiency","Energy efficiency", "Y-axis speed","Jumping height","Energy efficiency"]
    xscale = [2,3,3,3,3,4]
    yscale = [2,3,3,3,3,4]
    xvals = [300,3000,3000,3500,4500,10000]
    yvals = [200,3000,3000,3500,6000,10000]
    #ax.set_position(pos=[0.15,0.15,0.81,0.82])
    i = env_name_hyper.index(env_name)

    ax.scatter(objs[:, 0]/(10**xscale[i]), objs[:, 1]/(10**yscale[i]), s=100, marker='o', linewidths=0.1, facecolors=colors, edgecolors=(0.1,0.1,0.1))

    s1 = " ($\\times$$10^{%d}$)"%(xscale[i])
    s2 = " ($\\times$$10^{%d}$)"%(yscale[i])
    ax.set_xlabel(xlabel[i]+r"%s"%(s1))
    ax.set_ylabel(ylabel[i]+r"%s"%(s2))
    ax.set_xlim([0, xvals[i]/(10**xscale[i])])
    ax.set_ylim([0, yvals[i]/(10**yscale[i])])
    return np.array([10**xscale[i], 10**yscale[i]])

def draw_3objectives(ax, objs, colors):
    ax.set_xlabel(r"Forward speed (%s)"%("$\\times 10^3$"))
    ax.set_ylabel(r"Jumping height (%s)"%("$\\times 10^3$"))
    ax.set_zlabel(r"Energy efficiency (%s)"%("$\\times 10^3$"))
    label_pad = -7
    ax.xaxis.labelpad=label_pad
    ax.yaxis.labelpad=label_pad
    ax.zaxis.labelpad=label_pad

    ax.tick_params(axis='both', which='major', pad=-2)
    ax.xaxis.set_tick_params(pad=0)
    #ax.xaxis.get_tick_padding=0
    ax.view_init(elev=25, azim=28)
    ax.scatter(objs[:,0]/1000, objs[:,1]/1000, objs[:,2]/1000,s=4, marker='o', linewidths=0.1, c=np.clip(colors,0,1), edgecolors='black')
    ax.set_xlim([0,3200/1000])
    ax.set_ylim([0,6000/1000])
    ax.set_zlim([0,3000/1000])

if __name__=='__main__':
    if sys.platform == 'linux':
        font = {'family' : 'Liberation Serif',
        'size'   : 15}
    elif 'win' in sys.platform:
        font = {'family' : 'Times New Roman',
        'size'   : 15}
    matplotlib.rc('font', **font)
    plt.rcParams['mathtext.fontset']='cm'


    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default = 'MO-Swimmer-v2') # 'MO-Walker2d'
    parser.add_argument('--run-id', type=int, default = 0)
    args = parser.parse_args()

    dir_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    # Draw figure
    fig = plt.figure(figsize = (9, 6))
    (pref_fig, obj_fig) = fig.subfigures(1, 2, width_ratios=[0.4, 0.6])
    pref_fig.set_facecolor("#faf8de")
    obj_fig.set_facecolor("#f8fcf4")

    obj_ax = obj_fig.subplots()
    objs = np.load(os.path.join(dir_name,"visualization", "sample", "%s_objs_n2000_%d.npz"%(args.env_name, args.run_id)))["objs"]
    obj_ax.set_position([0.18, 0.18, 0.75, 0.7])
    obj_scale = draw_2objectives(obj_ax, objs, 'r', args.env_name)
    # objs, obj_scale = draw_2objectives(obj_ax, os.path.join(dir_name, "visualization", "sample", "%s_objs_n2000_%d.txt"%(args.env_name, args.run_id)), args.env_name)
    prefs = np.load(os.path.join(dir_name,"visualization", "sample", "%s_prefs_n2000.npz"%(args.env_name)))["prefs"]

    obj_ax.set_title('Objective Space')
    axes = pref_fig.subplots(2, 1, gridspec_kw={'height_ratios': [0.65, 0.35]})
    pref_ax, info_ax = axes
    pref_ax.set_title('Preference Space')
    pref_ax.set_position([0.18, 0.45, 0.7, 0.43])

    info_ax.set_position([0.05, 0.1, 0.9, 0.2])
    info_ax.set_xticks([])
    info_ax.set_yticks([])
    # info_ax.axis('off')
    pref_ax.plot([0, 1], [1, 0], color = 'red')
    pref_ax.set_xlabel("Pref. on Obj. 1")
    pref_ax.set_ylabel("Pref. on Obj. 2")
    eps = 0.1
    pref_ax.set_xticks([0, 0.5, 1])
    pref_ax.set_yticks([0, 0.5, 1])
    pref_ax.set_xlim([0 - eps, 1 + eps])
    pref_ax.set_ylim([0 - eps, 1 + eps])

    pref_marker = PreferenceMarker(pref_ax, info_ax, obj_ax, prefs, objs, obj_scale)

    fig.canvas.mpl_connect('button_press_event', mouse_press)
    fig.canvas.mpl_connect('button_release_event', mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', mouse_move)
    plt.show()