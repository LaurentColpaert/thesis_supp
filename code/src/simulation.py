"""
Laurent Colpaert - Thesis 2022-2023
"""
import ast
import enum
from math import cos, exp, sin, sqrt
import os
import re
import subprocess
import time
from xml.dom import minidom
from numpy import linalg as LA


import numpy as np
from behaviour import Behaviour

from utility import distToCircle, distToRect, retrieve_patches

class Mission(enum.Enum):
    AAC = 1
    FORBID = 2
    HOMING = 3
    SHELTER = 4
    TRAIN = 5

class Simulation():
    """
    Class Simulation : contain the function necessary to launch an argos simulation based on AutoMoDe
    """
    def __init__(self, behaviour : Behaviour, mission : Mission = Mission.AAC, visualization : bool = False) -> None:
        self.swarm_pos = []
        self.mission = mission
        if mission == Mission.AAC:
            self.argos_file = "aacVisu.argos" if visualization else "aac.argos"
        elif mission == Mission.FORBID:
            self.argos_file = "forbidVisu.argos" if visualization else "forbid.argos"
        elif mission == Mission.HOMING:
            self.argos_file = "homingVisu.argos" if visualization else "homing.argos"
        elif mission == Mission.SHELTER:
            self.argos_file = "shelterVisu.argos" if visualization else "shelter.argos"
        elif mission == Mission.TRAIN:
            self.argos_file = "repertoireTrainingVisu.argos"

        #Besqt PFSM
        self.pfsm = "--fsm-config --nstates 4 --s0 4 --att0 3.25 --n0 2 --n0x0 0 --c0x0 5 --p0x0 0.23 --n0x1 2 --c0x1 0 --p0x1 0.70 --s1 2 --n1 3 --n1x0 0 --c1x0 4 --w1x0 8.91 --p1x0 7 --n1x1 1 --c1x1 0 --p1x1 0.15 --n1x2 2 --c1x2 3 --w1x2 1.68 --p1x2 10 --s2 1 --n2 1 --n2x0 0 --c2x0 3 --w2x0 6.93 --p2x0 4 --s3 4 --att3 3.71 --n3 2 --n3x0 0 --c3x0 1 --p3x0 0.50 --n3x1 2 --c3x1 5 --p3x1 0.62"
        self.arenaD = 3
        self.nRbt = 20
        self.iteration = 1200
        # Patch = [x,y,r]
        self.patches, self.obstacles,self.circle_goal = retrieve_patches(self.argos_file, self.mission)
        self.behaviour = behaviour
        self.behaviour.setup(self.circle_goal,self.nRbt,self.iteration,self.arenaD,self.patches,self.obstacles)
        
    def read_file(self,filename: str = 'position.txt'):
        """
        Read the value of a file and retrieve all the line into a list

        Args:
            -filename (str): the name of the file 

        Returns:
            -None
        """
        with open(f"/home/laurent/Documents/Polytech/MA2/thesis/argos/{filename}") as f:
            self.swarm_pos.extend(ast.literal_eval(line) for line in f)

        self.swarm_pos = [list(t) for t in self.swarm_pos]

        os.remove(f"/home/laurent/Documents/Polytech/MA2/thesis/argos/{filename}")

    def run_simulation(self)-> tuple:
        """
        Run an argos simulation and compute the behaviour and fitness

        Args:
            -None
        Returns:
            -tuple(float): the value of the behaviour and fitness
        """
        command = f"cd /home/laurent/Documents/Polytech/MA2/thesis/argos; /home/laurent/AutoMoDe/bin/automode_main -c {self.argos_file} -n {self.pfsm}"
        process = subprocess.Popen(f"{command}",stdout=subprocess.PIPE, shell = True)
        result = process.stdout.read().decode().strip()
        pattern = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-]?\d+)?,\s*[-+]?\d+(?:\.\d+)?(?:[eE][-]?\d+)?\n")
        test = pattern.findall(result)

        self.swarm_pos = []
        self.swarm_pos.extend(
            (float(elem.split(',')[0]), float(elem.split(',')[1].split('\n')[0]))
            for elem in test
        )

        pattern = r"Fitness\s.*"
        match = re.findall(pattern,result)
        fitness_from_file = match
        print("Fitness from argos ", fitness_from_file)
        time.sleep(3)

        # self.read_file()
        # features = self.behaviour.compute_features(self.swarm_pos)
        # print("Features : ", features)
        features = []
        fitness = self.compute_fitness()
        print("Fitness : ", fitness)
        return features,fitness
    
    def run_simulation_std_out(self) -> tuple:
        """
        Run an argos simulation and compute the behaviour and fitness

        Args:
            -None
        Returns:
            -tuple(float): the value of the behaviour and fitness
        """
        command = f"cd /home/laurent/Documents/Polytech/MA2/thesis/argos; /home/laurent/AutoMoDe/bin/automode_main -c {self.argos_file} -n {self.pfsm}"
        process = subprocess.Popen(f"{command}",stdout=subprocess.PIPE, shell = True)
        result = process.stdout.read().decode().strip()
        pattern = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-]?\d+)?,\s*[-+]?\d+(?:\.\d+)?(?:[eE][-]?\d+)?\n")
        test = pattern.findall(result)

        self.swarm_pos = []
        self.swarm_pos.extend(
            (float(elem.split(',')[0]), float(elem.split(',')[1].split('\n')[0]))
            for elem in test
        )
        time.sleep(3)
        features = self.behaviour.compute_features(self.swarm_pos)
        # print("Features : ", features)
        # print("Features len: ", len(features))
        fitness = self.compute_fitness()
        # print("Fitness : ", fitness)
        return features,fitness
    
    def run_simulation_std_out_feature(self) -> tuple:
        """
        Run an argos simulation and compute the behaviour and fitness

        Args:
            -None
        Returns:
            -tuple(float): the value of the behaviour and fitness
        """
        command = f"cd /home/laurent/Documents/Polytech/MA2/thesis/argos; /home/laurent/AutoMoDe/bin/automode_main -c {self.argos_file} -n {self.pfsm}"
        process = subprocess.Popen(f"{command}",stdout=subprocess.PIPE, shell = True)
        result = process.stdout.read().decode().strip()
        pattern = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-]?\d+)?,\s*[-+]?\d+(?:\.\d+)?(?:[eE][-]?\d+)?\n")
        test = pattern.findall(result)

        self.swarm_pos = []
        self.swarm_pos.extend(
            (float(elem.split(',')[0]), float(elem.split(',')[1].split('\n')[0]))
            for elem in test
        )
        # time.sleep(3)
        # features = self.behaviour.compute_features(self.swarm_pos)
        # print("Features : ", features)
        # print("Features len: ", len(features))
        fitness = self.compute_fitness()
        # print("Fitness : ", fitness)
        return fitness,15


    def compute_fitness(self)-> int:
        """
        Compute the fitness of a run = the number of epuck inside the white circle

        Args:
            -None
        Returns:
            -int: the value of the fitness
        """
        if self.mission == Mission.AAC:
            return sum(
                distToCircle(self.circle_goal, pos,self.obstacles,self.arenaD) < self.circle_goal[2]
                for pos in self.swarm_pos[:-20]
            )/(self.iteration * self.nRbt)
        elif self.mission == Mission.SHELTER:
            return sum(
                distToRect(self.circle_goal, pos,self.obstacles,self.arenaD) < self.circle_goal[2]
                for pos in self.swarm_pos[-20:]
            )
        elif self.mission == Mission.HOMING:
            return sum(
                distToCircle(self.circle_goal, pos,self.obstacles,self.arenaD) < self.circle_goal[2]
                for pos in self.swarm_pos[-20:]
            )/self.nRbt
