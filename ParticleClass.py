import numpy as np
from matplotlib import cm
from matplotlib.collections import EllipseCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pickle
import random
            
class CollisionEvent:
    """
    Object contains all information about a collision event
    which are necessary to update the velocity after the collision.
    For MD of hard spheres (with hard bond-length dimer interactions)
    in a rectangular simulation box with hard walls, there are only
    two distinct collision types:
    1) wall collision of particle i with vertical or horizontal wall
    2) external (or dimer bond-length) collision between particle i and j
    """
    def __init__(self, Type = 'wall or other', dt = np.inf, mono_1 = 0, mono_2 = 0, w_dir = 1):
        """
        Type = 'wall' or other
        dt = remaining time until collision
        mono_1 = index of monomer
        mono_2 = if inter-particle collision, index of second monomer
        w_dir = if wall collision, direction of wall
        (   w_dir = 0 if wall in x direction, i.e. vertical walls
            w_dir = 1 if wall in y direction, i.e. horizontal walls   )
        """
        self.Type = Type
        self.dt = dt
        self.mono_1 = mono_1
        self.mono_2 = mono_2  # only importent for interparticle collisions
        self.w_dir = w_dir # only important for wall collisions
        
    def __str__(self):
        if self.Type == 'wall':
            return "Event type: {:s}, dt: {:.8f}, p1 = {:d}, dim = {:d}".format(self.Type, self.dt, self.mono_1, self.w_dir)
        else:
            return "Event type: {:s}, dt: {:.8f}, p1 = {:d}, p2 = {:d}".format(self.Type, self.dt, self.mono_1, self.mono_2)

class Monomers:
    """
    Class for event-driven molecular dynamics simulation of hard spheres:
    -Object contains all information about a two-dimensional monomer system
    of hard spheres confined in a rectengular box of hard walls.
    -A configuration is fully described by the simulation box and
    the particles positions, velocities, radiai, and masses.
    -Initial configuration of $N$ monomers has random positions (without overlap)
    and velocities of random orientation and norms satisfying
    $E = \sum_i^N m_i / 2 (v_i)^2 = N d/2 k_B T$, with $d$ being the dimension,
    $k_B$ the Boltzmann constant, and $T$ the temperature.
    -Class contains all functions for an event-driven molecular dynamics (MD)
    simulation. Essentail for inter-particle collsions is the mono_pair array,
    which book-keeps all combinations without repetition of particle-index
    pairs for inter-particles collisions, e.g. for $N = 3$ particles
    indices = 0, 1, 2
    mono_pair = [[0,1], [0,2], [1,2]]
    -Monomers can be initialized with individual radiai and density = mass/volume.
    For example:
    NumberOfMonomers = 7
    NumberMono_per_kind = [ 2, 5]
    Radiai_per_kind = [ 0.2, 0.5]
    Densities_per_kind = [ 2.2, 5.5]
    then monomers mono_0, mono_1 have radius 0.2 and mass 2.2*pi*0.2^2
    and monomers mono_2,...,mono_6 have radius 0.5 and mass 5.5*pi*0.5^2
    -The dictionary of this class can be saved in a pickle file at any time of
    the MD simulation. If the filename of this dictionary is passed to
    __init__ the simulation can be continued at any point in the future.
    IMPORTANT! If system is initialized from file, then other parameters
    of __init__ are ignored!
    """
    def __init__(self, NumberOfMonomers = 4, L_xMin = 0, L_xMax = 1, L_yMin = 0, L_yMax = 1, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), k_BT = 5, FilePath = './Configuration.p'):
        try:
            self.__dict__ = pickle.load( open( FilePath, "rb" ) )
            print("IMPORTANT! System is initialized from file %s, i.e. other input parameters of __init__ are ignored!" % FilePath)
        except:
            assert ( NumberOfMonomers > 0 )
            assert ( (L_xMin < L_xMax) and (L_yMin < L_yMax) )
            self.NM = NumberOfMonomers
            self.DIM = 2 #dimension of system
            self.BoxLimMin = np.array([ L_xMin, L_yMin])
            self.BoxLimMax = np.array([ L_xMax, L_yMax])
            self.mass = -1*np.ones( self.NM ) # Masses, negative mass means not initialized
            self.rad = -1*np.ones( self.NM ) # Radiai, negative radiai means not initialized
            self.pos = np.zeros( (self.NM, self.DIM) ) # Positions, np.zeros rather than np.empty as it fits our initialization better
            self.vel = np.empty( (self.NM, self.DIM) ) # Velocities, not initalized but desired shape
            self.mono_pairs = np.array( [ (k,l) for k in range(self.NM) for l in range( k+1,self.NM ) ] )
            self.next_wall_coll = CollisionEvent( 'wall', np.inf, 0, 0, 0)
            self.next_mono_coll = CollisionEvent( 'mono', np.inf, 0, 0, 0)

            self.assignRadiaiMassesVelocities(NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT )
            self.assignRandomMonoPos( )
    
    def save_configuration(self, FilePath = 'MonomerConfiguration.p'):
        '''Saves configuration. Callable at any time during simulation.'''
        #print( self.__dict__ )
    
    def assignRadiaiMassesVelocities(self, NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT):
        '''
        Make this a PRIVATE function -> cannot be called outside class definition.
        '''
        '''initialize radiai and masses'''

        assert( sum(NumberMono_per_kind) == self.NM )
        assert( isinstance(Radiai_per_kind,np.ndarray) and (Radiai_per_kind.ndim == 1) )
        assert( (Radiai_per_kind.shape == NumberMono_per_kind.shape) and (Radiai_per_kind.shape == Densities_per_kind.shape))
        
        NumberMono_per_kind = [ 10, 5]
        Radiai_per_kind = [ 0.3, 0.5]
        Densities_per_kind = [ 2.2, 5.5]
        
        compteur=0                                                              #set counter
        for k in range (self.NM):
            if (k+1)>sum(NumberMono_per_kind[:compteur+1]):                     #assign radiai and masses as long as the counter
                compteur+=1                                                     #has not went through the number of one kind of mono
            self.rad[k]=Radiai_per_kind[compteur]                               #then go to the next kind of mono
            self.mass[k]=Densities_per_kind[compteur]*np.pi*Radiai_per_kind[compteur]**2
        '''initialize velocities'''
        assert( k_BT > 0 )
        # E_kin = sum_i m_i /2 v_i^2 = N * dim/2 k_BT https://en.wikipedia.org/wiki/Ideal_gas_law#Energy_associated_with_a_gas
        
        #-->> your turn
        vitesse=np.ones((self.NM, self.DIM))-2*np.random.rand(self.NM, self.DIM)
        E=self.NM*self.DIM*k_BT/2
        Somme=sum(self.mass*(vitesse[:,0]**2+vitesse[:,1]**2)/2)#somme des mi/2 vi^2 pour i allant de 1 à N
        masse=self.mass
        Masse=np.transpose(np.array([masse.tolist(),masse.tolist()]))

        self.vel=vitesse*(E/Somme)**(1/2)*(2/Masse)**(1/2) # j'ai décommenté cette ligne
    
    def assignRandomMonoPos(self, start_index = 0 ):
        '''
        Make this a PRIVATE function -> cannot be called outside class definition.
        Initialize random positions without overlap between monomers and wall.
        '''
        assert ( min(self.rad) > 0 )#otherwise not initialized
        mono_new, infiniteLoopTest = start_index, 0                             #set of counters
        BoxLength = self.BoxLimMax - self.BoxLimMin                             #compute limits where center of mono can be initialized
        #initialize first monomer before going into loop to avoid infinite loop
        self.pos[mono_new] = np.random.uniform(self.BoxLimMin+self.rad[mono_new, None], self.BoxLimMax-self.rad[mono_new,None])
        mono_new+=1
        #initialization loop
        while mono_new < self.NM and infiniteLoopTest < 10**4:
            infiniteLoopTest += 1
            flag = True                                                         #set the flag to stay in the loop as long as it is
            while flag :                                                        #needed
                self.pos[mono_new] = np.random.uniform(self.BoxLimMin+self.rad[mono_new, None], self.BoxLimMax-self.rad[mono_new,None])
                delta_r_ij = np.where(self.pos != 0, self.pos - self.pos[mono_new], 0) #compute distance between mono and all mono
                delta_r_ij_sq = (delta_r_ij**2).sum(1)                                 #previously initialized and take square of norm
                
                min_distance = np.where(delta_r_ij_sq == 0, np.Inf, delta_r_ij_sq)     #replace 0 (not initialized) in delta_r_i_sq
                min_distance = np.argmin(min_distance)                                 #by inf for the search of smallest distance
                #if the smallest distance between new mono and old monos is greater than sum of radiai, then initialization is correct
                if delta_r_ij_sq[min_distance] > (self.rad[min_distance]+self.rad[mono_new])**2 :
                    flag = False                                                #if initialization is correct, set flag to false to
            mono_new += 1                                                       #exit the loop and go to the next mono

    def __str__(self, index = 'all'):
        if index == 'all':
            return "\nMonomers with:\nposition = " + str(self.pos) + "\nvelocity = " + str(self.vel) + "\nradius = " + str(self.rad) + "\nmass = " + str(self.mass)
        else:
            return "\nMonomer at index = " + str(index) + " with:\nposition = " + str(self.pos[index]) + "\nvelocity = " + str(self.vel[index]) + "\nradius = " + str(self.rad[index]) + "\nmass = " + str(self.mass[index])
        
    def Wall_time(self):
        #############################
        # calcul du temps pour rejoindre un mur en dimension 2
        #on détermine le temps min, on repère la particule en question et le mur
        #puis mise à jour des valeurs
        #UTILISATION MASQUE
        '''
        -Function computes list of remaining time dt until future
        wall collision in x and y direction for every particle.
        Then, it stores collision parameters of the event with
        the smallest dt in the object next_wall_coll.
        -Meaning of future:
        if v > 0: solve BoxLimMax - rad = x + v * dt
        else:     solve BoxLimMin + rad = x + v * dt
        '''
        
        #compute distance from particles to correct wall according to sign of their speed
        Correct_Wall=(self.BoxLimMax - self.rad[:,None]) * (self.vel > 0) + (self.BoxLimMin + self.rad[:,None]) * (self.vel < 0)
        CollTime = (Correct_Wall-self.pos)/self.vel                             #compute time before wall collision
        minColl_index=np.argmin(CollTime)                                       #search for index of smallest collision time
        collision_disk,wall_direction = divmod(minColl_index,2)                 #get index and wall direction using euclidean division
        minCollTime=CollTime[collision_disk,wall_direction]                     #get smallest collision time using the computed index

        self.next_wall_coll.dt = minCollTime                                    #set of simulation parameters, collision time
        self.next_wall_coll.mono_1 = collision_disk                             #index of the monomer colliding
        #self.next_wall_coll.mono_2 = not necessary
        self.next_wall_coll.w_dir = wall_direction                              #direction of the wall colliding with mono
        
        
    def Mono_pair_time(self): #calcul collision particule
        '''
        - Function computes list of remaining time dt until
        future external collition between all combinations of
        monomer pairs without repetition. Then, it stores
        collision parameters of the event with
        the smallest dt in the object next_mono_coll.
        - If particles move away from each other, i.e.
        scal >= 0 or Omega < 0, then remaining dt is infinity.
        '''
        mono_i = self.mono_pairs[:,0] # List of collision partner 1
        mono_j = self.mono_pairs[:,1] # List of collision partner 2

        CollisionDist_sq = (self.rad[mono_i] + self.rad[mono_j])**2 # distance squared at which collision partners touch

        del_VecPos = self.pos[mono_i] - self.pos[mono_j] # r_i - r_j
        del_VecVel = self.vel[mono_i] - self.vel[mono_j] # v_i - v_j
        del_VecPos_sq = (del_VecPos**2).sum(1)   #|dr|^2

        a = (del_VecVel**2).sum(1)               # |dv|^2
        c = del_VecPos_sq - CollisionDist_sq     # initial distance
        b = 2 * (del_VecPos * del_VecVel).sum(1) # 2( dr \cdot dv )

        Omega = b**2 - 4.* a * c                                                #compute discriminant
        RequiredCondition = ((b < 0) & (Omega > 0))                             #we want only one solution of the quadratic equation
        DismissCondition = np.logical_not( RequiredCondition )                  #which is such that Omega is real and b < 0
        b[DismissCondition] = -np.inf                                           #b that does not satisfy conditions is set as -inf
        Omega[DismissCondition] = 0                                             #Omega that does not satisfy conditions is set as 0
        del_t = ( b + np.sqrt(Omega) ) / (-2*a)                                 #so when we compute collision times, those with
                                                                                #undesired conditions will be +inf and won't interfere
        minCollTime = del_t[np.argmin(del_t)]                                   #for the search of smallest collision time
        collision_disk_1 = mono_i[np.argmin(del_t)]      #set index of first colliding mono
        collision_disk_2 = mono_j[np.argmin(del_t)]      #set index of second colliding mono
        
        self.next_mono_coll.dt = minCollTime             #set collision time
        self.next_mono_coll.mono_1 = collision_disk_1
        self.next_mono_coll.mono_2 = collision_disk_2
#       self.next_mono_coll.w_dir = not necessary
        
    def compute_next_event(self):
        '''
        Function gets event information about:
        1) next possible wall event
        2) next possible pair event 
        Function returns event info of event with
        minimal time, i.e. the clostest in future.
        '''
        
        #This is not correct! you have to write the code!
        self.Wall_time()                                                        #call method to compute wall collision time
        self.Mono_pair_time()                                                   #call method to compute particle collision time
        if self.next_mono_coll.dt>self.next_wall_coll.dt:
            return self.next_wall_coll
        else:
            return self.next_mono_coll
            
    def compute_new_velocities(self, next_event):
        ##############
        #inversion de la vitesse lors du choc sur le mur (attention en x ou en y)
        if next_event.Type=='wall':
            self.vel[next_event.mono_1,next_event.w_dir]=-self.vel[next_event.mono_1,next_event.w_dir]#inversion de la vitesse
        else: #compute new speeds according to physics
            deltax=(self.pos[next_event.mono_2]-self.pos[next_event.mono_1])/np.linalg.norm(self.pos[next_event.mono_2]-self.pos[next_event.mono_1])
            deltav=self.vel[next_event.mono_1]-self.vel[next_event.mono_2]
            produit_scalaire=np.dot(deltav,deltax)
            self.vel[next_event.mono_1]=self.vel[next_event.mono_1]-2*self.mass[next_event.mono_2]/(self.mass[next_event.mono_1]+self.mass[next_event.mono_2])*produit_scalaire*deltax
            self.vel[next_event.mono_2]=self.vel[next_event.mono_2]+2*self.mass[next_event.mono_1]/(self.mass[next_event.mono_1]+self.mass[next_event.mono_2])*produit_scalaire*deltax
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

        
class Dimers(Monomers):
    """
    --> Class derived from Monomers.
    --> See also comments in Monomer class.
    --> Class for event-driven molecular dynamics simulation of hard-sphere
    system with DIMERS (and monomers). Two hard-sphere monomers form a dimer,
    and experience additional ellastic collisions at the maximum
    bond length of the dimer. The bond length is defined in units of the
    minimal distance of the monomers, i.e. the sum of their radiai.
    -Next to the monomer information, the maximum dimer bond length is needed
    to fully describe one configuration.
    -Initial configuration of $N$ monomers has random positions without overlap
    and separation of dimer pairs is smaller than the bond length.
    Velocities have random orientations and norms that satisfy
    $E = \sum_i^N m_i / 2 (v_i)^2 = N d/2 k_B T$, with $d$ being the dimension,
    $k_B$ the Boltzmann constant, and $T$ the temperature.
    -Class contains all functions for an event-driven molecular dynamics (MD)
    simulation. Essentail for all inter-particle collsions is the mono_pair array
    (explained in momonmer class). Essentail for the ellastic bond collision
    of the dimers is the dimer_pair array which book-keeps index pairs of
    monomers that form a dimer. For example, for a system of $N = 10$ monomers
    and $M = 2$ dimers:
    monomer indices = 0, 1, 2, 3, ..., 9
    dimer_pair = [[0,2], [1,3]]
    -Monomers can be initialized with individual radiai and density = mass/volume.
    For example:
    NumberOfMonomers = 10
    NumberOfDimers = 2
    bond_length_scale = 1.2
    NumberMono_per_kind = [ 2, 2, 6]
    Radiai_per_kind = [ 0.2, 0.5, 0.1]
    Densities_per_kind = [ 2.2, 5.5, 1.1]
    then monomers mono_0, mono_1 have radius 0.2 and mass 2.2*pi*0.2^2
    and monomers mono_2, mono_3 have radius 0.5 and mass 5.5*pi*0.5^2
    and monomers mono_4,..., mono_9 have radius 0.1 and mass 1.1*pi*0.1^2
    dimer pairs are: (mono_0, mono_2), (mono_1, mono_3) with bond length 1.2*(0.2+0.5)
    see bond_length_scale and radiai
    -The dictionary of this class can be saved in a pickle file at any time of
    the MD simulation. If the filename of this dictionary is passed to
    __init__ the simulation can be continued at any point in the future.
    IMPORTANT! If system is initialized from file, then other parameters
    of __init__ are ignored!
    """
    def __init__(self, NumberOfMonomers = 4, NumberOfDimers = 2, L_xMin = 0, L_xMax = 1, L_yMin = 0, L_yMax = 1, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), bond_length_scale = 1.2, k_BT = 1, FilePath = './Configuration.p'):
        #if __init__() defined in derived class -> child does NOT inherit parent's __init__()
        try:
            self.__dict__ = pickle.load( open( FilePath, "rb" ) )
            print("IMPORTANT! System is initialized from file %s, i.e. other input parameters of __init__ are ignored!" % FilePath)
        except:
            assert ( (NumberOfDimers > 0) and (NumberOfMonomers >= 2*NumberOfDimers) )
            assert ( bond_length_scale > 1. ) # is in units of minimal distance of respective monomer pair
            Monomers.__init__(self, NumberOfMonomers, L_xMin, L_xMax, L_yMin, L_yMax, NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT )
            self.ND = NumberOfDimers
            self.dimer_pairs = np.array([[k,self.ND+k] for k in range(self.ND)])#choice 2 -> more practical than [2*k,2*k+1]
            mono_i, mono_j = self.dimer_pairs[:,0], self.dimer_pairs[:,1]
            self.bond_length = bond_length_scale * ( self.rad[mono_i] + self.rad[mono_j] )
            self.next_dimer_coll = CollisionEvent( 'dimer', 0, 0, 0, 0)
            
            '''
            Positions initialized as pure monomer system by monomer __init__.
            ---> Reinitalize all monomer positions, but place dimer pairs first
            while respecting the maximal distance given by the bond length!
            '''
            self.assignRandomDimerPos()
            self.assignRandomMonoPos( 2*NumberOfDimers )
    
    def assignRandomDimerPos(self):
        '''
        Make this is a PRIVATE function -> cannot be called outside class definition
        initialize random positions without overlap between monomers and wall
        '''
        dimer_new_index, infiniteLoopTest = 0, 0
        BoxLength = self.BoxLimMax - self.BoxLimMin
        while dimer_new_index < self.ND and infiniteLoopTest < 10**4:
            infiniteLoopTest += 1
            mono_i, mono_j = dimer_new = self.dimer_pairs[dimer_new_index]
            
            # Your turn to place the dimers one after another such that
            # there are no overlaps between monomers and hard walls and
            # dimer pairs are not further appart than their max bond length.
        if dimer_new_index != self.ND:
            print('Failed to initialize all dimer positions.\nIncrease simulation box size!')
            exit()
        
        
    def __str__(self, index = 'all'):
        if index == 'all':
            return Monomers.__str__(self) + "\ndimer pairs = " + str(self.dimer_pairs) + "\nwith max bond length = " + str(self.bond_length)
        else:
            return "\nDimer pair " + str(index) + " consists of monomers = " + str(self.dimer_pairs[index]) + "\nwith max bond length = " + str(self.bond_length[index]) + Monomers.__str__(self, self.dimer_pairs[index][0]) + Monomers.__str__(self, self.dimer_pairs[index][1])

    def Dimer_pair_time(self):
        '''
        Function computes list of remaining time dt until
        future dimer bond collition for all dimer pairs.
        Then, it stores collision parameters of the event with
        the smallest dt in the object next_dimer_coll.
        '''
        mono_i = self.dimer_pairs[:,0] # List of collision partner 1
        mono_j = self.dimer_pairs[:,1] # List of collision partner 2

        dvx=self.vel[mono_i,0]-self.vel[mono_j,0]                               #exact same code as mono_pair_time but...
        dvy=self.vel[mono_i,1]-self.vel[mono_j,1]
        dx=self.pos[mono_i,0]-self.pos[mono_j,0]
        dy=self.pos[mono_i,1]-self.pos[mono_j,1]
        Ri=self.rad[mono_i]
        Rj=self.rad[mono_j]

        a=dvx**2+dvy**2
        b=2*(dvx*dx+dvy*dy)
        c=dx**2+dy**2-(self.bond_length*(Ri+Rj))**2
        D=b**2-4*a*c

        dt = np.where(D > 0, (-b + np.sqrt(D))/(2*a), np.array([np.inf]*self.NM)) #except this time we only need positive discriminant
        minCollTime = np.min(dt)

        collision_disk_1 = self.mono_pairs[np.argmin(dt),0]
        collision_disk_2 = self.mono_pairs[np.argmin(dt),1]

        self.next_dimer_coll.dt = minCollTime
        self.next_dimer_coll.mono_1 = collision_disk_1
        self.next_dimer_coll.mono_2 = collision_disk_2

    def compute_next_event(self):
        '''
        Function gets event information about:
        1) next possible wall event
        2) next possible pair event
        Function returns event info of event with
        minimal time, i.e. the clostest in future.
        '''

        self.Wall_time()
        self.Mono_pair_time()
        self.Dimer_pair_time()
        if self.next_wall_coll.dt < self.next_mono_coll.dt:
            if self.next_wall_coll.dt < self.next_dimer_coll.dt:
                return self.next_wall_coll
            else:
                return self.next_dimer_coll
        else:
            if self.next_dimer_coll.dt < self.next_mono_coll.dt:
                return self.next_dimer_coll
            else:
                return self.next_mono_coll
        
    def snapshot(self, FileName = './snapshot.png', Title = ''):
        '''
        ---> Overwriting snapshot(...) of Monomers class!
        Function saves a snapshot of current configuration,
        i.e. monomer positions as circles of corresponding radius,
        dimer bond length as back empty circles (on top of monomers)
        velocities as arrows on monomers,
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
        COLORS = np.linspace(0.2,0.95,self.ND+1)
        MonomerColors = np.ones(self.NM)*COLORS[-1] #unique color for monomers
        # recolor each monomer pair with individual color
        MonomerColors[self.dimer_pairs[:,0]] = COLORS[:len(self.dimer_pairs)]
        MonomerColors[self.dimer_pairs[:,1]] = COLORS[:len(self.dimer_pairs)]

        #plot solid monomers
        Width, Hight, Angle = 2*self.rad, 2*self.rad, np.zeros( self.NM )
        collection = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
        collection.set_array(MonomerColors)
        collection.set_clim(0, 1) # <--- we set the limit for the color code
        ax.add_collection(collection)
        
        #plot bond length of dimers as black cicles
        Width, Hight, Angle = self.bond_length, self.bond_length, np.zeros( self.ND )
        mono_i = self.dimer_pairs[:,0]
        mono_j = self.dimer_pairs[:,1]
        collection_mono_i = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos[mono_i],
                       transOffset=ax.transData, edgecolor = 'k', facecolor = 'None')
        collection_mono_j = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos[mono_j],
                       transOffset=ax.transData, edgecolor = 'k', facecolor = 'None')
        ax.add_collection(collection_mono_i)
        ax.add_collection(collection_mono_j)

        #--->plot velocities as arrows
        ax.quiver( self.pos[:,0], self.pos[:,1], self.vel[:,0], self.vel[:,1] , units = 'dots', scale_units = 'dots')
        
        plt.title(Title)
        plt.savefig( FileName)
        plt.close()
