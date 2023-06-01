"""
Laurent Colpaert - Thesis 2022-2023
"""
from enum import Enum
from math import  exp
import numpy as np
from numpy import linalg as LA

from utility import distToCircle, distToRect

SIZE = 41

class behaviours(Enum):
    """
    The value corresponding is the position in the behaviour vector
    """
    PHI = 60
    DUTY_FACTOR = 61
    AAC = 1
    HOMING = 2

class Behaviour:
    def __init__(self, b1 : behaviours, b2 : behaviours,r1 : int = 1, r2 : int = 2) -> None:
        self.b1 = b1
        self.range1 = self.set_range(self.b1)
        self.b2 = b2
        self.range2 = self.set_range(self.b2)
        self.r1 = r1
        self.r2 = r2

    def setup(self,circle_goal, nRbt, iteration,arenaD : float, patches : list,obstacles : list) -> None:
        self.circle_goal = circle_goal
        self.nRbt = nRbt
        self.iteration = iteration
        self.arenaD = arenaD
        self.patches = patches
        self.obstacles = obstacles


    def retrieve_behaviour_fct(self) -> list:
        """
        Retrieve the two behaviour function selected
        """
        return self.get_behaviours(self.b1),self.get_behaviours(self.b2)

    def compute_features(self,swarm_pos : list) -> list:
        features = []
        phi= self.compute_phi(swarm_pos=swarm_pos)
        features.extend(iter(phi))
        df = self.duty_factor(swarm_pos=swarm_pos)
        features.append(df)
        return features

    def get_range_position(self, position : int) -> list:
        from simulation import Mission
        """
        In function of the position in the behaviour vector return the range corresponding
        """  
        mission = Mission.SHELTER #TODO: change this line -> ugly :3
        if mission == Mission.SHELTER:
            if position +1<= 100:
                return self.range1
            else:
                return self.range2
            
        if position +1<= behaviours.PHI.value:
            return self.range1
        elif position +1== behaviours.DUTY_FACTOR.value:
            return self.range2

    def get_behaviours(self, behaviour : behaviours) -> list:
        """
        Retrieve a behaviour function based on the input behaviour
        """
        if behaviour.value == 1: #DUTY_FACTOR
            return self.duty_factor
        elif behaviour.value == 2: #PHI
            return self.compute_phi
        
    def set_range(self, b : behaviours) -> list: 
        """
        Return the corresponding range depending on the behaviour chosen
        """
        if b.value == behaviours.DUTY_FACTOR.value: #DUTY_FACTOR
            return [0,1]
        elif b.value == behaviours.PHI.value: #PHI
            return [0,1]
        elif b.value == behaviours.AAC.value:
            return [0,1]
        elif b.value == behaviours.HOMING.value: #PHI
            return [0,1]
        
        
    def duty_factor(self,swarm_pos : list)-> float:
        """
        Compute the duty factor.
        It's the amout of time that all the robot have spent in the final landmark

        Args:
            -None
        Returns:
            -float: the value of the behaviour
        """
        return sum(
            distToCircle(self.circle_goal, pos,self.obstacles,self.arenaD) < self.circle_goal[2]
            for pos in swarm_pos[:-20]
        )/(self.nRbt * self.iteration)

    def compute_phi(self, swarm_pos : list)-> float:
        """
        Compute the phi behaviour.
        It's the distance of each robot from the landmarks

        Args:
            -None
        Returns:
            -float: the value of the behaviour
        """
        phi_tot = []
        for p in self.patches:
            phi = []
            patch = p.copy()

            for pos in swarm_pos[-self.nRbt:]:
                if(len(patch) == 3):
                    distance = distToCircle(patch, pos,self.obstacles,self.arenaD)
                else:
                    distance = distToRect(patch, pos,self.obstacles,self.arenaD)
                phi.append(distance)

            h = (2*np.log(10))/(self.arenaD**2)
            phi = [exp(- h * self.arenaD * pos) for pos in phi]
            phi.sort(reverse=True)
            phi_tot.extend(iter(phi))

        #Distance from the closest robot 
        phi = []
        for i in range(self.nRbt):
            neighbors = swarm_pos[-self.nRbt:].copy()
            neighbors.pop(i)
            distance = min(
                LA.norm(np.array(swarm_pos[-self.nRbt + i]) - np.array(n), ord=2)
                for n in neighbors
            )
            phi.append(distance)

        h = (2*np.log(10))/(self.arenaD**2)
        phi = [exp(- h * self.arenaD * pos) for pos in phi]
        phi.sort(reverse=True)

        phi_tot.extend(iter(phi))
        # return sum(phi_tot) / len(phi_tot)
        return phi_tot
    
    def set_behaviour1(self, b1 : behaviours) -> None:
        self.b1 = b1

    def set_behaviour2(self, b2 : behaviours) -> None:
        self.b2 = b2