from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.backend_bases import MouseButton
import numpy as np
import matplotlib
import argparse
import os
from multiprocessing import Process
import sys
from re import split
import time


def format_value(v):
    formatted_num = "%.2e"%(v)
    parts = formatted_num.split('e')
    
    exponent = int(parts[1])
    base = 10**exponent
    output = "%.2fE%d"%(v/base,exponent)
    return output

class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for subclasses of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()

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
       
        self.pref_text = info_ax.text(0.00, 0.7, "", fontsize = 14) # animated 
        self.obj_text = info_ax.text(0.00, 0.4, "", fontsize = 14)  # animated
        self.info_ax.text(0.2, 0.1, "Click for Visualization", fontsize = 14)
        self.button_pos = [0.18, 0.05, 0.9, 0.3]
        self.info_ax.add_patch(plt.Rectangle((self.button_pos[0], self.button_pos[1]), self.button_pos[2] - self.button_pos[0],
                               self.button_pos[3] - self.button_pos[1], facecolor= [0.9, 0.9, 0.9], edgecolor='black', lw=1))    
        self.update_pref(np.array([1/3, 1/3, 1/3]))

    def pref2pos(self, pref):
        """
            Convert the preference to the position in the screen coordinate.
        """
        x, y, _ = proj3d.proj_transform(*pref, self.pref_ax.get_proj())
        return x, y

    def pos2pref(self, pos):
        """
            Convert the position in the screen coordinate to the preference (sum(w) = 1).
        """ 
        # obtain the inverse of the projection matrix
        inv_proj_mat = np.linalg.inv(self.pref_ax.get_proj())
        x, y, z = proj3d.proj_transform(*pos, -1, inv_proj_mat)
        p1 = np.array([x, y, z])
        x, y, z = proj3d.proj_transform(*pos, -0.5, inv_proj_mat)
        p2 = np.array([x, y, z])
        # calculate the intersection of the line and the preference space
        t = (1 - np.sum(p2)) / (np.sum(p1) - np.sum(p2))
        res = p1 * t + p2 * (1-t)
        np.clip(res, 0, 1)
        res = res / np.sum(res)
        return res
    
    def update_pref(self, new_pref):
        # if self.pref_marker is not None:
        #     self.pref_marker.remove()

        # if self.obj_marker is not None:
        #     self.obj_marker.remove()
        
        distances = np.sum(np.power(self.prefs - new_pref, 2), axis = -1)
        index = np.argmin(distances)
        self.pref = prefs[index]
        obj_value = self.objs[index]
        obj_x, obj_y, obj_z = obj_value[0] / self.obj_scale[0], obj_value[1] / self.obj_scale[1], obj_value[2] / self.obj_scale[2]
        if self.pref_marker is None:
            self.pref_marker= self.pref_ax.plot(*self.pref, markersize= 10, marker='o',
                                                     c='green', animated=True, zorder = 10)[0]
            #print('obj_value:',obj_x, obj_y, obj_z)
            self.obj_marker = self.obj_ax.quiver(obj_x, obj_y, obj_z + 2, 0, 0, -1, length =2, animated=True)
            self.bm = BlitManager(fig.canvas, [self.pref_marker, self.pref_text, self.obj_text, self.obj_marker])
               
        else:
            
            self.pref_marker._verts3d = tuple(self.pref)
            # modify 2d segmemtns
            #seg = self.obj_marker.get_segments()
            # t1, t2, _ = proj3d.proj_transform(obj_x, obj_y, obj_z, self.obj_ax.get_proj())
            # print('seg=',seg)
            # p = np.array([t1, t2]) - seg[0][0]
            # for i in range(3):
            #     for j in range(2):
            #         seg[i][j] += p
            # print('updated_seg=',seg)
            #self.obj_marker.set_segments(seg)
            # modify 3d segments
            seg = self.obj_marker._segments3d
            #print('seg=', self.obj_marker.get_segments())
            p = np.array([obj_x, obj_y, obj_z]) - seg[0][0]
            for i in range(3):
                for j in range(2):
                    seg[i][j] += p 
            self.obj_marker.set_segments(seg) 
            self.obj_marker.do_3d_projection()
            #print(print('updated_seg=',seg))         
            
            #print(self.obj_marker._transOffset.get_matrix())
        #
        # self.obj_marker = self.obj_ax.plot(obj_value[0] / self.obj_scale[0], obj_value[1] / self.obj_scale[1], obj_value[2] / self.obj_scale[2],marker='o', linestyle = '-', markersize= 10, 
        #     linewidth=5, markerfacecolor='white', markeredgecolor = 'black')[0]
        #self.obj_marker = self.obj_ax.quiver(obj_x, obj_y, obj_z + 2, 0, 0, -1, length =2)
        self.pref_text.set_text("Input Preference = (%.3f, %.3f, %.3f)" % (self.pref[0], self.pref[1],self.pref[2]))
        self.obj_text.set_text("Objective Values = (%s, %s, %s)" % (format_value(obj_value[0]), format_value(obj_value[1]), format_value(obj_value[2])))
    
    def start_move(self, ix, iy):
        eps = 0.05
        px, py = self.pref2pos(self.pref)
        #print('pref pos:', px, py)
        #print('mouse:', ix, iy)
        if abs(ix - px) >= eps or abs(iy - py) >= eps: return
        self.isMove = True

        self.last_pos = np.array([ix, iy])

    def end_move(self):
        #print('pref=', self.pref[0], self.pref[1])
        self.isMove = False
    


    def move(self, ix, iy):
        if not self.isMove: return
        
        
        new_pos = np.array([ix, iy]) - self.last_pos + self.pref2pos(self.pref)
        new_pref = self.pos2pref(new_pos)
        #print('new_pref=',new_pref)
        self.update_pref(new_pref)
        self.last_pos = np.array([ix, iy])



def play_mujoco():
    play_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mujoco_play.py')
    model_path = os.path.join(dir_name, "pretrained", args.env_name, "Hyper-MORL/default/%d/"%(args.run_id))
    if args.env_name == 'MO-Hopper-v3':
        play_cmd = 'python {} --env {} --model {} --pref1 {} --pref2 {} --pref3 {}'.format(play_script, args.env_name, model_path, *pref_marker.pref)
    else:
        play_cmd = 'python {} --env {} --model {} --pref1 {} --pref2 {}'.format(play_script, args.env_name, model_path, *pref_marker.pref)
    Process(target=os.system, args=(play_cmd,)).start()

def mouse_press(event):
    if event.inaxes == pref_ax:
        if event.button != MouseButton.LEFT: return
        #print('Mouse pressed',xyz[0]/sum(xyz), xyz[1]/sum(xyz), xyz[2]/sum(xyz))
        global pref_marker
        pref_marker.start_move(event.xdata, event.ydata)
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
    st = time.time()
    ix, iy = event.xdata, event.ydata
    global pref_marker
    pref_marker.move(ix, iy)
    #fig.canvas.draw()
    pref_marker.bm.update()
    
def draw_3objectives(ax, objs, colors):
    ax.set_xlabel(r"Forward speed (%s)"%("$\\times 10^3$"))
    ax.set_ylabel(r"Jumping height (%s)"%("$\\times 10^3$"))
    ax.set_zlabel(r"Energy efficiency (%s)"%("$\\times 10^3$"))
    label_pad = 0
    ax.xaxis.labelpad=label_pad
    ax.yaxis.labelpad=label_pad
    ax.zaxis.labelpad=label_pad

    ax.tick_params(axis='both', which='major', pad=-1)
    ax.xaxis.set_tick_params(pad=0)
    #ax.xaxis.get_tick_padding=0
    ax.view_init(elev=25, azim=28)
    ax.scatter(objs[:,0]/1000, objs[:,1]/1000, objs[:,2]/1000,s=4, marker='o', linewidths=0.1, c=[1, 0, 0], edgecolors=[0, 0, 0])
    ax.set_xlim([0,3200/1000])
    ax.set_ylim([0,6000/1000])
    ax.set_zlim([0,3000/1000])
    return np.array([1000, 1000, 1000])

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
    parser.add_argument('--env-name', type=str, default = 'MO-Hopper-v3') # 'MO-Walker2d'
    parser.add_argument('--run-id', type=int, default = 0)
    args = parser.parse_args()

    dir_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    # Draw figure
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize = (9, 6))
    gs0 = gridspec.GridSpec(2, 5, figure=fig)
    pref_ax = fig.add_subplot(gs0[0:1, 0:2], projection= '3d', computed_zorder=False)
    info_ax = fig.add_subplot(gs0[1:2, 0:2])
    obj_ax = fig.add_subplot(gs0[0:2, 2:5], projection= '3d')
    obj_ax.set_position(obj_ax.get_position().translated(0.06, 0.05))
    #(pref_fig, obj_fig) = fig.subfigures(1, 2, width_ratios=[0.4, 0.6])
    #pref_fig.set_facecolor("#faf8de")
    #obj_fig.set_facecolor("#f8fcf4")
    # draw background color
    from matplotlib.patches import Rectangle
    fig.add_artist(Rectangle((0, 0), width=.44, height=1, facecolor='#faf8de', zorder=0))
    fig.add_artist(Rectangle((0.44, 0), width=.56, height=1, facecolor='#f8fcf4', zorder=0))



    #obj_ax = obj_fig.subplots(subplot_kw={'projection': '3d'})
    objs = np.load(os.path.join(dir_name,"visualization", "sample", "%s_objs_n20100_%d.npz"%(args.env_name, args.run_id)))["objs"]
    #obj_ax.set_position([0, 0, 1, 1])
    obj_scale = draw_3objectives(obj_ax, objs, 'r')
    # objs, obj_scale = draw_2objectives(obj_ax, os.path.join(dir_name, "visualization", "sample", "%s_objs_n2000_%d.txt"%(args.env_name, args.run_id)), args.env_name)
    prefs = np.load(os.path.join(dir_name,"visualization", "sample", "%s_prefs_n20100.npz"%(args.env_name)))["prefs"]

    obj_ax.set_title('Objective Space')
    #pref_ax = pref_fig.add_subplot(2, 1, 1, projection= '3d', computed_zorder=False)
    #info_ax = pref_fig.add_subplot(2, 1, 2)
    #axes = pref_fig.subplots(2, 1, gridspec_kw={'height_ratios': [0.65, 0.35]},  subplot_kw={'projection': '3d'})
    #pref_ax, info_ax = axes
    pref_ax.set_title('Preference Space')
    pref_ax.set_position(pref_ax.get_position().translated(-0.05, 0))

    pref_ax.xaxis.set_ticks([0, 0.5, 1])
    pref_ax.yaxis.set_ticks([0, 0.5, 1])
    pref_ax.zaxis.set_ticks([0, 0.5, 1])

    info_ax.set_position(info_ax.get_position().translated(-0.1, 0).expanded(1.1,1))
    info_ax.set_xticks([])
    info_ax.set_yticks([])
    info_ax.axis('off')
    pref_ax.scatter(prefs[:, 0], prefs[:, 1], prefs[:, 2],s = 1, color = [1, 0, 0], zorder = 1)
    # print('ax_data:', proj3d.proj_transform(1, 0, 0, pref_ax.get_proj()))
    # print('ax_data:', proj3d.proj_transform(0, 1, 0, pref_ax.get_proj()))
    # x, y, _ = proj3d.proj_transform(0, 0, 1, pref_ax.get_proj())
    # #pref_ax.add_patch(plt.Circle((x, y),  facecolor= [0, 1, 0], edgecolor='black', lw=1))    

    # print('ax_data:',x, y)
    # print('ax_data:',pref_ax.transData.transform((x, y)))
    # print('ax_data:', pref_ax.transData.transform((0,1,0)))
    # print('ax_data:', pref_ax.transData.transform((0,0,1)))

    pref_ax.disable_mouse_rotation()
    pref_ax.set_xlabel("Pref. on Obj. 1")
    pref_ax.set_ylabel("Pref. on Obj. 2")
    pref_ax.set_zlabel("Pref. on Obj. 3")
    pref_ax.view_init(elev=45, azim=45)
    eps = 0.1
    # # pref_ax.set_xticks([0, 0.5, 1])
    # # pref_ax.set_yticks([0, 0.5, 1])
    pref_ax.set_xlim([0 - eps, 1 + eps])
    pref_ax.set_ylim([0 - eps, 1 + eps])
    pref_ax.set_zlim([0 - eps, 1 + eps])
    pref_marker = PreferenceMarker(pref_ax, info_ax, obj_ax, prefs, objs, obj_scale)

    fig.canvas.mpl_connect('button_press_event', mouse_press)
    fig.canvas.mpl_connect('button_release_event', mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', mouse_move)
    plt.show()
