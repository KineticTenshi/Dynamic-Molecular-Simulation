import numpy as np
from matplotlib import cm
from matplotlib.collections import EllipseCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pickle

class CollisionEvent:
    def __init__(self, Type = 'wall or other', dt = np.inf, mono_1 = 0, mono_2 = 0, w_dir = 1):

        self.Type = Type
        self.dt = dt
        self.mono_1 = mono_1
        self.mono_2 = mono_2  # only important for interparticle collisions
        self.w_dir = w_dir # only important for wall collisions

class Monomers:
    def __init__(self, NumberOfMonomers = 4, N_left_Chamber = 2, L_xMin = 0, L_xMax = 1, L_yMin = 0, L_yMax = 1, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), k_BT_right = 1, k_BT_left=1, FilePath = './Configuration.p'):
        try:
            self.__dict__ = pickle.load( open( FilePath, "rb" ) )
            print("IMPORTANT! System is initialized from file %s, i.e. other input parameters of __init__ are ignored!" % FilePath)
        except:
            assert ( NumberOfMonomers > 0 )
            assert ( (L_xMin < L_xMax) and (L_yMin < L_yMax) )
            self.NM = NumberOfMonomers
            self.N_left_chamber = N_left_Chamber
            self.DIM = 2 #dimension of system
            self.BoxLimMin = np.array([ L_xMin, L_yMin])
            self.BoxLimMax = np.array([ L_xMax, L_yMax])
            self.mass = -1*np.ones( self.NM ) # Masses, negative mass means not initialized
            self.rad = -1*np.ones( self.NM ) # Radiai, negative radiai means not initialized
            self.pos = np.zeros( (self.NM, self.DIM) ) # preferable pour correspondre à votre approche
            self.vel = np.empty( (self.NM, self.DIM) ) # Velocities, not initalized but desired shape
            self.mono_pairs = np.array( [ (k,l) for k in range(self.NM) for l in range( k+1,self.NM ) ] )
            self.list_mono = np.array([k for k in range(self.NM)])
            self.next_wall_coll = CollisionEvent( 'wall', np.inf, 0, 0, 0)
            self.next_mono_coll = CollisionEvent( 'mono', np.inf, 0, 0, 0)
            self.next_moving_wall_coll = CollisionEvent( 'moving wall', np.inf, 0, 0, 0)#new event: collision with the moving wall
            # parameters of the moving wall
            self.wall_pos=np.array([L_xMax/2,0])
            self.wall_epais=np.array([0.1])
            self.wall_vel=np.array([0.0,0.0])
            self.wall_mass=np.array([1.0])
            self.assignRadiaiMassesVelocities(NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT_right, k_BT_left) #differentiation of the left and right temperature
            self.assignRandomMonoPos()

    def save_configuration(self, FilePath = 'MonomerConfiguration.p'):
        '''Saves configuration. Callable at any time during simulation.'''
        #print( self.__dict__ )

    def assignRadiaiMassesVelocities(self, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), k_BT_right = 1, k_BT_left = 1 ):

        assert( sum(NumberMono_per_kind) == self.NM )
        assert( isinstance(Radiai_per_kind,np.ndarray) and (Radiai_per_kind.ndim == 1) )
        assert( (Radiai_per_kind.shape == NumberMono_per_kind.shape) and (Radiai_per_kind.shape == Densities_per_kind.shape))

        NumberMono_per_kind_left = np.array([50, 20])
        NumberMono_per_kind_right = NumberMono_per_kind - NumberMono_per_kind_left

        #Initilization of the left chamber
        compteur=0
        for k in range (self.N_left_chamber):
            if (k+1)>sum(NumberMono_per_kind_left[:compteur+1]):
                compteur+=1
            self.rad[k]=Radiai_per_kind[compteur]
            self.mass[k]=Densities_per_kind[compteur]*np.pi*Radiai_per_kind[compteur]**2#mass=density*Surface

        #Initilization of the right chamber
        compteur=0
        for k in range (self.NM - self.N_left_chamber):
            if (k+1)>sum(NumberMono_per_kind_right[:compteur+1]):
                compteur+=1
            self.rad[k + self.N_left_chamber]=Radiai_per_kind[compteur]
            self.mass[k + self.N_left_chamber]=Densities_per_kind[compteur]*np.pi*Radiai_per_kind[compteur]**2

        '''initialize velocities'''
        assert( k_BT_right > 0 )

        vitesse=np.ones((self.NM, self.DIM))-2*np.random.rand(self.NM, self.DIM)#random intitailsation of velocity
        #Now we have to normalize the velocities in order to have the desired temperature in the left and right chamber
        N_right=self.NM-self.N_left_chamber
        E_right=N_right*self.DIM*k_BT_right/2#Energy of the left chamber
        E_left=self.N_left_chamber*self.DIM*k_BT_left/2#Energy of the left chamber

        Somme_left=sum((self.mass[:self.N_left_chamber]*(vitesse[:self.N_left_chamber]**2).sum(1))/2)#Kinetic energy in the left chamber
        Somme_right=sum((self.mass[self.N_left_chamber:]*(vitesse[self.N_left_chamber:]**2).sum(1))/2)#Kinetic energy in the right chamber

        self.vel[:self.N_left_chamber]=vitesse[:self.N_left_chamber]*(E_left/Somme_left)**(1/2)
        self.vel[self.N_left_chamber:]=vitesse[self.N_left_chamber:]*(E_right/Somme_right)**(1/2)

        #Print of initial conditions
        print("température droite initiale",sum(1/2*self.mass[self.N_left_chamber:]*(self.vel[self.N_left_chamber:]**2).sum(1))/(self.NM-self.N_left_chamber))
        print("température gauche initiale",sum(1/2*self.mass[:self.N_left_chamber]*(self.vel[:self.N_left_chamber]**2).sum(1))/(self.N_left_chamber))
        print("temperature totale",sum(1/2*self.mass[self.N_left_chamber:]*(self.vel[self.N_left_chamber:]**2).sum(1))/(self.NM-self.N_left_chamber)+sum(1/2*self.mass[:self.N_left_chamber]*(self.vel[:self.N_left_chamber]**2).sum(1))/(self.N_left_chamber))
        print("masse",self.mass[0])

    def assignRandomMonoPos(self, start_index = 0 ):
        assert ( min(self.rad) > 0 )#otherwise not initialized
        mono_new, infiniteLoopTest = start_index, 0

        Left_Chamber_Min = self.BoxLimMin
        Left_Chamber_Max = np.array([(self.wall_pos[0]-(self.wall_epais[0]/2)), self.BoxLimMax[1]])

        Right_Chamber_Min = np.array([(self.wall_pos[0]+(self.wall_epais[0]/2)), self.BoxLimMin[1]])
        Right_Chamber_Max = self.BoxLimMax

        #left chamber
        #The first monomer is initialised
        self.pos[mono_new] = np.random.uniform(Left_Chamber_Min+self.rad[mono_new, None], Left_Chamber_Max-self.rad[mono_new,None])

        mono_new+=1
        #The position of the monomers is initialized for one monomer at a time according to the remaining space
        while mono_new < self.N_left_chamber :
            flag = True
            while flag :
                self.pos[mono_new] = np.random.uniform(Left_Chamber_Min+self.rad[mono_new, None], Left_Chamber_Max-self.rad[mono_new,None])

                delta_r_ij = np.where(self.pos != 0, self.pos - self.pos[mono_new], 0)
                delta_r_ij_sq = (delta_r_ij**2).sum(1)

                min_distance = np.where(delta_r_ij_sq == 0, np.Inf, delta_r_ij_sq)
                min_distance = np.argmin(min_distance)
                if (delta_r_ij_sq[min_distance])**(1/2) > self.rad[min_distance]+self.rad[mono_new] :
                    flag = False
            mono_new += 1

        #right chamber
        #the code is similar
        self.pos[mono_new] = np.random.uniform(Right_Chamber_Min+self.rad[mono_new, None], Right_Chamber_Max-self.rad[mono_new,None])

        mono_new+=1

        while mono_new < self.NM :
            flag = True
            while flag :
                self.pos[mono_new] = np.random.uniform(Right_Chamber_Min+self.rad[mono_new, None], Right_Chamber_Max-self.rad[mono_new,None])

                delta_r_ij = np.where(self.pos != 0, self.pos - self.pos[mono_new], 0)
                delta_r_ij_sq = (delta_r_ij**2).sum(1)

                min_distance = np.where(delta_r_ij_sq == 0, np.Inf, delta_r_ij_sq)
                min_distance = np.argmin(min_distance)
                if (delta_r_ij_sq[min_distance])**(1/2) > self.rad[min_distance]+self.rad[mono_new] :
                    flag = False
            mono_new += 1

    def Wall_time(self):
        Correct_Wall=(self.BoxLimMax - self.rad[:,None]) * (self.vel > 0) + (self.BoxLimMin + self.rad[:,None]) * (self.vel < 0)
        CollTime = (Correct_Wall-self.pos)/self.vel
        minColl_index=np.argmin(CollTime)
        collision_disk,wall_direction = divmod(minColl_index,2)
        minCollTime=CollTime[collision_disk,wall_direction]

        self.next_wall_coll.dt = minCollTime
        self.next_wall_coll.mono_1 = collision_disk
        self.next_wall_coll.w_dir = wall_direction

    def moving_wall_time (self):
        CollisionDist_sq = (self.rad + self.wall_epais/2)**2#minimal distance between a monomer and the wall
        del_VecPos = self.pos[:,0] - self.wall_pos[0]
        del_VecVel = self.vel[:,0] - self.wall_vel[0]
        del_VecPos_sq = (del_VecPos**2) #|dr|^2
        a = (del_VecVel**2) #|dv|^2
        c = del_VecPos_sq - CollisionDist_sq # initial distance
        b = 2 * (del_VecPos * del_VecVel) # 2( dr \cdot dv )

        Omega = b**2 - 4.* a * c
        RequiredCondition = ((b < 0) & (Omega > 0))
        DismissCondition = np.logical_not( RequiredCondition )
        b[DismissCondition] = -np.inf
        Omega[DismissCondition] = 0
        del_t = ( b + np.sqrt(Omega) ) / (-2*a)

        minCollTime = del_t[np.argmin(del_t)]
        collision_disk = self.list_mono[np.argmin(del_t)]

        self.next_moving_wall_coll.dt = minCollTime
        self.next_moving_wall_coll.mono_1 = collision_disk

    def Mono_pair_time(self): #calcul collision particule
        mono_i = self.mono_pairs[:,0] # List of collision partner 1
        mono_j = self.mono_pairs[:,1] # List of collision partner 2

        CollisionDist_sq = (self.rad[mono_i] + self.rad[mono_j])**2 # distance squared at which collision partners touch

        del_VecPos = self.pos[mono_i] - self.pos[mono_j] # r_i - r_j
        del_VecVel = self.vel[mono_i] - self.vel[mono_j] # v_i - v_j

        del_VecPos_sq = (del_VecPos**2).sum(1) #|dr|^2
        a = (del_VecVel**2).sum(1) #|dv|^2
        c = del_VecPos_sq - CollisionDist_sq # initial distance
        b = 2 * (del_VecPos * del_VecVel).sum(1) # 2( dr \cdot dv )
        Omega = b**2 - 4.* a * c
        RequiredCondition = ((b < 0) & (Omega > 0))
        DismissCondition = np.logical_not( RequiredCondition )
        b[DismissCondition] = -np.inf
        Omega[DismissCondition] = 0
        del_t = ( b + np.sqrt(Omega) ) / (-2*a)

        minCollTime = del_t[np.argmin(del_t)]
        collision_disk_1 = mono_i[np.argmin(del_t)]
        collision_disk_2 = mono_j[np.argmin(del_t)]

        self.next_mono_coll.dt = minCollTime
        self.next_mono_coll.mono_1 = collision_disk_1
        self.next_mono_coll.mono_2 = collision_disk_2


    def compute_next_event(self):
        self.Wall_time()
        self.Mono_pair_time()
        self.moving_wall_time()
        #We need to compare the date of the three possible events: the collision of monomers, the collision of monomer and the moving wall and the collision of monomer and the wall of the box
        if self.next_mono_coll.dt>self.next_wall_coll.dt:
            if self.next_moving_wall_coll.dt > self.next_wall_coll.dt :
                return self.next_wall_coll
            else:
                return self.next_moving_wall_coll
        else:
            if self.next_moving_wall_coll.dt>self.next_mono_coll.dt:
                return self.next_mono_coll
            else:
                return self.next_moving_wall_coll

    def compute_new_velocities(self, next_event):
        if next_event.Type=='wall':
            self.vel[next_event.mono_1,next_event.w_dir]=-self.vel[next_event.mono_1,next_event.w_dir]#inversion of the velocity

        elif next_event.Type== 'mono':
            deltax=(self.pos[next_event.mono_2]-self.pos[next_event.mono_1])/np.linalg.norm(self.pos[next_event.mono_2]-self.pos[next_event.mono_1])
            deltav=self.vel[next_event.mono_1]-self.vel[next_event.mono_2]
            produit_scalaire=np.dot(deltav,deltax)
            self.vel[next_event.mono_1]=self.vel[next_event.mono_1]-2*self.mass[next_event.mono_2]/(self.mass[next_event.mono_1]+self.mass[next_event.mono_2])*produit_scalaire*deltax
            self.vel[next_event.mono_2]=self.vel[next_event.mono_2]+2*self.mass[next_event.mono_1]/(self.mass[next_event.mono_1]+self.mass[next_event.mono_2])*produit_scalaire*deltax

        else:#See the report for details of the calculation (conservation of momentum and conservattion of energy)
            vel_p=self.vel[next_event.mono_1, 0]
            vel_wall=self.wall_vel[0]
            M = self.mass[next_event.mono_1] + self.wall_mass
            self.vel[next_event.mono_1, 0]=((self.mass[next_event.mono_1]-self.wall_mass)/M)*vel_p + ((2*self.wall_mass)/M)*vel_wall
            self.wall_vel[0]=((2*self.mass[next_event.mono_1])/M)*vel_p+((self.wall_mass-self.mass[next_event.mono_1])/M)*vel_wall

        '''
        Function updates the velocities of the monomer(s)
        involved in collision event.
        Update depends on event type.
        Ellastic wall collisions in x direction reverse vx.
        Ellastic pair collisions follow: https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
        '''

    def snapshot(self, FileName = './snapshot.png', Title = '$t = $?'):
        '''
        Function saves a snapshot of current configuration,
        i.e. particle positions as circles of corresponding radius,
        velocities as arrows on particles,
        blue dashed lines for the hard walls of the simulation box.
        '''
        fig, ax = plt.subplots( dpi=300 )
        L_xMin, L_xMax = self.BoxLimMin[0], self.BoxLimMax[0]
        L_yMin, L_yMax = self.BoxLimMin[1], self.BoxLimMax[1]
        BorderGap = 0.1*(L_xMax - L_xMin)
        ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
        ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)

        #--->plot hard walls (rectangle)
        rect = mpatches.Rectangle((L_xMin,L_yMin), L_xMax-L_xMin, L_yMax-L_yMin, linestyle='dashed', ec='gray', fc='None')
        ax.add_patch(rect)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ position')
        ax.set_ylabel('$y$ position')

        #--->plot monomer positions as circles
        MonomerColors = np.linspace( 0.2, 0.95, self.NM)
        Width, Hight, Angle = 2*self.rad, 2*self.rad, np.zeros( self.NM )
        collection = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
        collection.set_array(MonomerColors)
        collection.set_clim(0, 1) # <--- we set the limit for the color code
        ax.add_collection(collection)

        #--->plot velocities as arrows
        ax.quiver( self.pos[:,0], self.pos[:,1], self.vel[:,0], self.vel[:,1] , units = 'dots', scale_units = 'dots')

        plt.title(Title)
        plt.savefig(FileName)
        plt.close()