import Projet_Moving_Wall_Class as pc
import numpy as np

import os
from matplotlib import cm
from matplotlib.collections import EllipseCollection
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(999)
Snapshot_output_dir = './SnapshotsMonomers'
if not os.path.exists(Snapshot_output_dir): os.makedirs(Snapshot_output_dir)

Conf_output_dir = './ConfsMonomers'
Path_ToConfiguration = Conf_output_dir+'/FinalMonomerConf.p'
# if os.path.isfile( Path_ToConfiguration ):
#     '''Initialize system from existing file'''
#     mols = pc.Monomers( FilePath = Path_ToConfiguration )
# else:
'''Initialize system with following parameters'''
# create directory if it does not exist
if not os.path.exists(Conf_output_dir): os.makedirs(Conf_output_dir)
#define parameters
NumberOfMonomers = 100
N_left_Chamber = 70
L_xMin, L_xMax = 0, 3*(313**0.5)/4
L_yMin, L_yMax = 0, (313**0.5)/4
NumberMono_per_kind = np.array([ 70, 30])
Radiai_per_kind = np.array([ 0.2, 0.11])
Densities_per_kind = np.array([ 2.2, 2])
k_BT_right = 1
k_BT_left= 5
# call constructor, which should initialize the configuration

mols = pc.Monomers(NumberOfMonomers, N_left_Chamber, L_xMin, L_xMax, L_yMin, L_yMax, NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT_right, k_BT_left )

mols.snapshot( FileName = Snapshot_output_dir+'/InitialConf.png', Title = '$t = 0$')
#we could initialize next_event, but it's not necessary
#next_event = pc.CollisionEvent( Type = 'wall or other, to be determined', dt = 0, mono_1 = 0, mono_2 = 0, w_dir = 0)

'''define parameters for MD simulation'''
t = 0.0
dt = 0.02#evenement régulier
NumberOfFrames = 500
next_event = mols.compute_next_event()
Densite_droite=[]
Densite_gauche=[]
Densite_tot=[]
KTemperature_droite=[]
KTemperature_gauche=[]
KTemperature_tot=[]
Energietotal=[]

def MolecularDynamicsLoop( frame ):
    global t, mols, next_event

    t_next_frame = t + dt

    while t + next_event.dt <= t_next_frame:
        t += next_event.dt
        mols.pos += mols.vel * next_event.dt
        mols.wall_pos += mols.wall_vel * next_event.dt
        mols.compute_new_velocities(next_event)
        next_event = mols.compute_next_event()

    remain_t = t_next_frame - t #equal dt if no event between frames
    mols.pos += mols.vel * remain_t
    mols.wall_pos += mols.wall_vel * remain_t
    t += remain_t
    next_event.dt-=remain_t

    TEMPS.append(t)
    POSITION.append(mols.wall_pos[0])
    Densite_gauche.append(sum(mols.mass[:mols.N_left_chamber])/(L_yMax*(mols.wall_pos[0]-mols.wall_epais/2)))
    Densite_tot.append(sum(mols.mass[:])/(L_yMax*(L_xMax-mols.wall_epais)))
    Densite_droite.append(sum(mols.mass[mols.N_left_chamber:])/(L_yMax*(L_xMax-mols.wall_pos[0]+mols.wall_epais/2)))
    KTemperature_droite.append(sum(1/2*mols.mass[mols.N_left_chamber:]*(mols.vel[mols.N_left_chamber:]**2).sum(1))/(mols.NM-mols.N_left_chamber))
    KTemperature_gauche.append(sum(1/2*mols.mass[:mols.N_left_chamber]*(mols.vel[:mols.N_left_chamber]**2).sum(1))/(mols.N_left_chamber))
    KTemperature_tot.append(sum(1/2*mols.mass[:]*(mols.vel[:]**2).sum(1))/mols.NM)

    Energietotal.append(sum(1/2*mols.mass[:mols.N_left_chamber]*(mols.vel[:mols.N_left_chamber]**2).sum(1))+sum(1/2*mols.mass[mols.N_left_chamber:]*(mols.vel[mols.N_left_chamber:]**2).sum(1))+1/2*mols.wall_mass*mols.wall_vel[0]**2)

    plt.title( '$t = %.4f$, remaining frames = %d' % (t, NumberOfFrames-(frame+1)) )
    collection.set_offsets( mols.pos )
    moving_wall.set_x(mols.wall_pos[0]-mols.wall_epais/2)
    return collection
    return moving_wall

'''We define and initalize the plot for the animation'''
fig, ax = plt.subplots()
L_xMin, L_yMin = mols.BoxLimMin #not defined if initalized by file
L_xMax, L_yMax = mols.BoxLimMax #not defined if initalized by file
BorderGap = 0.1*(L_xMax - L_xMin)
ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)
ax.set_aspect('equal')

# confining hard walls plotted as dashed lines
rect = mpatches.Rectangle((L_xMin,L_yMin), L_xMax-L_xMin, L_yMax-L_yMin, linestyle='dashed', ec='gray', fc='None')
ax.add_patch(rect)

moving_wall = mpatches.Rectangle((mols.wall_pos[0]-mols.wall_epais/2,0), mols.wall_epais, L_yMax-L_yMin, linestyle='-', ec='red', fc='None')
ax.add_patch(moving_wall)


# plotting all monomers as solid circles of individual color
MonomerColors = np.linspace(0.2,0.95,mols.NM)
Width, Hight, Angle = 2*mols.rad, 2*mols.rad, np.zeros(mols.NM)
collection = EllipseCollection(Width, Hight, Angle, units='x', offsets=mols.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
collection.set_array(MonomerColors)
collection.set_clim(0, 1) # <--- we set the limit for the color code
ax.add_collection(collection)

'''Create the animation, i.e. looping NumberOfFrames over the update function'''
Delay_in_ms = 33.3 #dely between images/frames for plt.show()
TEMPS=[]
POSITION=[]
ani = FuncAnimation(fig, MolecularDynamicsLoop, frames=NumberOfFrames, interval=Delay_in_ms, blit=False, repeat=False)
plt.show()

'''Save the final configuration and make a snapshot.'''
#write the function to save the final configuration
mols.save_configuration(Path_ToConfiguration)
mols.snapshot( FileName = Snapshot_output_dir + '/FinalConf.png', Title = '$t = %.4f$' % t)

plt.plot(TEMPS,POSITION)
plt.title("Position du mur en fonction du temps")
plt.xlabel("temps")
plt.ylabel("Position du mur")
plt.show()

plt.plot(TEMPS,Densite_gauche,label="densité gauche")
plt.plot(TEMPS,Densite_droite,label="densité droite")
plt.plot(TEMPS,Densite_tot,label="densité totale")
plt.title("Densité en fonction du temps")
plt.xlabel("temps")
plt.ylabel("Densité")
plt.legend()
plt.show()

plt.plot(TEMPS,KTemperature_droite,label="k_B*Température droite")
plt.plot(TEMPS,KTemperature_gauche,label="k_B*Température gauche")
plt.plot(TEMPS,KTemperature_tot,label="k_B*Température totale")
plt.title("K_B*Température en fonction du temps")
plt.xlabel("temps")
plt.ylabel("K_B*Température")
plt.legend()
plt.show()


plt.plot(TEMPS,Energietotal,label="Energie du syteme global")
plt.grid()
plt.xlabel("temps")
plt.ylabel("Energie totale")
plt.legend()
plt.show()