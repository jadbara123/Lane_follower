import os
import pygame
import math
import cv2
import numpy as np
import time
import new_sim

class initialNode():
    def __init__(self, name, state, action):
        self.state = state
        self.child = "?"
        self.visits = 1
        self.score = 0
        self.parent = "slf"
        self.action = action
        self.sirali_dugum_sayisi = 10
        self.name = name
        self.file_path = "file_path"
        self.fully_expanded = None
        self.is_terminal = None
        self.terminal_type = 0

    def save(self):
        with open(str(self.file_path)+"/"+str(self.name)+".txt", "w") as f:
            f.write(str(self.state) + "*")
            if self.child != []:
                for i in range(0, len(self.child)):
                    f.write(str(self.child[i]) + ":")
            else:
                f.write(str(self.child) + ":")
            f.write(str("*"))
            f.write(str(self.parent) + "*")
            f.write(str(self.score) + "*")
            f.write(str(self.visits) + "*")
            f.write(str(self.action) + "*")
            f.write(str(self.fully_expanded) + "*")
            f.write(str(self.is_terminal) + "*")
            f.write(str(self.terminal_type))
            f.close()

class TreeNode():
    def __init__(self, state, parent, action, name):
        self.state = state
        self.action = action
        self.parent = parent
        self.name = str(self.parent) + "-" + str(name)
        self.child = []
        self.visits = 1
        self.score = 0
        self.file_path = "file_path"
        self.terminal_type = 0
        self.fully_expanded = None
        self.is_terminal = None

    def save(self):
        with open(str(self.file_path)+"/"+str(self.name)+".txt", "w") as f:
            f.write(str(self.state) + "*")
            if self.child != []:
                for i in self.child:
                    f.write(str(self.child[i]) + ":")
            else:
                f.write(str("?") + ":")
            f.write(str("*"))
            f.write(str(self.parent) + "*")
            f.write(str(self.score) + "*")
            f.write(str(self.visits) + "*")
            f.write(str(self.action) + "*")
            f.write(str(self.fully_expanded) + "*")
            f.write(str(self.is_terminal) + "*")
            f.write(str(self.terminal_type))
            f.close()

class MCTS():

    def __init__(self, state):
        self.delay = 0
        self.node_file_path = "file_path"
        self.exploration_constant = 0.5
        self.state = state
        self.sirali_dugum_sayisi = 10

    def get_data(self, node_file):
        with open(str(self.node_file_path)+"/"+str(node_file)+".txt", "r") as f:
            text = f.read()
            node_list = text.split('*')
            state = node_list[0]
            child = node_list[1].split(':')
            parent = node_list[2]
            visits = node_list[4]
            score = node_list[3]
            action = node_list[5]
            is_terminal = node_list[7]
            fully_exp = node_list[6]
            terminal_type = node_list[8]
        return score, parent, visits, state, action, is_terminal, fully_exp, terminal_type, child

    def update_node(self, node_name, instance, change):
        if str(instance) == "state":
            instance = 0
        elif str(instance) == "child":
            instance = 1
        elif str(instance) == "parent":
            instance = 2
        elif str(instance) == "score":
            instance = 3
        elif str(instance) == "visits":
            instance = 4
        elif str(instance) == "action":
            instance = 5
        elif str(instance) == "fully_expanded":
            instance = 6
        elif str(instance) == "is_terminal":
            instance = 7
        elif str(instance) == "terminal_type":
            instance = 8
        else:
            print("wtf")
        node_name = str(self.node_file_path+"/"+str(node_name)+".txt")
        with open(node_name, "r") as f:
            text = f.read()
            node_list = text.split('*')
            if instance == 1:
                if node_list[instance] == '?:':
                    change = change
                else:
                    change = str(node_list[instance]+":"+str(change))
            node_list[instance] = str(change)
            with open(node_name, "w") as f:
                for i in range(0, len(node_list)):
                    f.write(str(node_list[i]))
                    if len(node_list)-1 != i:
                        f.write("*")
                f.close()

    def expand(self, node, train = True):
        _,_,_,node_state,node_action,_,_,_,child = self.get_data(node)
        if train:
            action = carlo.get_action()
        else:
            action = carlo.get_action(False)
        state = carlo.get_state()
        if node_state != state or node_action != action:
            parent = node
            child_nodes = []
            if str(child[0]) != "?":
                for i in child:
                    i = i.split("-")
                    i = i[-1]
                    child_nodes.append(i)
                child_nodes.sort()
                name = str(int(child_nodes[-1])+1)
            else:
                name = "1"
            action = [action]
            for i in action:
                child_node = TreeNode(state, parent, i, name)
                name = child_node.name
                child_node.save()
                child = str(name)
                fully_expanded = fully_expanded = True
                self.update_node(node, "child", str(child))
                self.update_node(node, "fully_expanded", fully_expanded)
            #print("debug9")
        else:
            pass

    def select(self, state=np.array, child_nodes=None, eğitim = True):
        # simülasyondan gelen state vektörünün sağlığını kontrol et bir hata gerçekleşiyor olabilir
        flag_create_init_node = False
        flag_start_node = False
        full_node_list = os.listdir(self.node_file_path)
        node_list = []
        if child_nodes is None:
            #print("debug7")
            for i in full_node_list:
                if len(str(i).split(".txt")[0].split("-")) == 1:
                    node_list.append(i.split(".txt")[0])
        elif child_nodes is not None:
            #print("debug4")
            node_list = child_nodes
        if node_list != []:
            if len(node_list[0].split("-")) == 1:
                flag_start_node = True
            #print("debug5")
            desired_nodes = []
            for file in node_list:
                node_path = str(file)
                _, _, _, node_state, _, _, _, _, _ = self.get_data(node_path)
                if str(state) == str(node_state):
                    #print("debug6")
                    desired_nodes.append(node_path)
            if desired_nodes != []:
                if flag_start_node:
                    prev_score = 0
                    for i in desired_nodes:
                        score,_,_,_,_,_,_,_,_ = self.get_data(i) 
                        desired_node = i
                else:
                    desired_node = self.choose_node(desired_nodes)
                return desired_node
            else:
                flag_create_init_node = True
        else:
            flag_create_init_node = True
        if flag_create_init_node:
            #print("debug10")
            if child_nodes is None:
                head_nodes = []
                if os.listdir(self.node_file_path) != []:
                    for i in os.listdir(self.node_file_path):
                        i = i.split(".txt")[0].split("-")
                        if len(i) == 1:
                            i = i[0]
                            head_nodes.append(int(i))
                    head_nodes.sort()
                    name = int(head_nodes[-1])+1
                else:
                    name = 1
                if eğitim:
                    action = carlo.get_action()
                else:
                    action = carlo.get_action(False)
                node = initialNode(str(name), state, action)
                name = node.name
                #print("debug1")
                node.save()
            else:
                _, parent, _, _, _, _, _, _, _ = self.get_data(child_nodes[0])
                name = parent
                self.expand(name)
            desired_node = name
            #print("debug3")
            return desired_node

    def rollout(self, node):
        _, _, _, state, action, _, fully_exp, _, _, = mcts.get_data(node)
        carlo.run()
        carlo.act(action)
        if str(False) ==  str(self.is_terminal(node)) and str(True) == str(fully_exp):
            #print(node)
            _, _, _, _, _, _, _, _, children = self.get_data(node)
            stt = carlo.get_state()
            new_node = self.select(stt, children)
            a = self.rollout(new_node)
            if a == "0":
                return "0"
        else:
            return "0"
    
    def is_terminal(self, node):
        #print("debug13")
        flag, terminal_type = carlo.is_terminal()
        if flag:
            self.update_node(node, "terminal_type", -terminal_type)
            self.update_node(node, "is_terminal", flag) 
            return flag
        elif flag == False:
            #print("debug14")
            name = str(node).split("-")
            if len(name) > self.sirali_dugum_sayisi:
                flag = True
                terminal_type = 1
            self.update_node(node, "is_terminal", flag) 
            self.update_node(node, "terminal_type", terminal_type) 
            return flag        

    def backprop(self, node):
        score, parent, visits, action, _, _, is_terminal, terminal_type, _ = self.get_data(node)
        #print("+++++++++"+str(visits))
        if action == 0:
            score = int(score)-0.5
        self.update_node(node, "visits", int(visits) + 1)
        print(int(score) + int(terminal_type))
        self.update_node(node, "score", str(int(score) + int(terminal_type)))
        if str(parent) != "slf": 
            while len(str(parent).split(".txt")[0].split("-")) != 1:
                score, _, visits, _, _, _, _, _, _ = self.get_data(parent)
                if action == 0:
                    score = int(score)-0.5
                self.update_node(parent, "visits", int(visits) + 1)
                self.update_node(parent, "score", str(int(score) + int(terminal_type)))
                _, parent, _, _, _, _, _, _, _ = self.get_data(parent)
            score, _, visits, _, _, _, _, _, _ = self.get_data(parent)
            if action == 0:
                score = int(score)-0.5
            self.update_node(parent, "visits", int(visits) + 1)
            self.update_node(parent, "score", str(int(score) + int(terminal_type)))
            _, parent, _, _, _, _, _, _, _ = self.get_data(parent)
            return 0
        else:
            return 0

    def choose_node(self, node_list):
        # here you should use the UCT function to choose the appropriate action in a given state
        uct_scores = []
        #print("debug8")
        for i in node_list:
            print(i)
            score, parent, visits, _, _, _, _, _, _ = self.get_data(i)
            _, _, p_visits, _, _, _, _, _, _ = self.get_data(parent)
            print(str(math.log(int(p_visits) / (int(visits))))+" "+str(p_visits)+" "+str(visits))
            uct = (int(score) / (int(visits))) + self.exploration_constant * math.sqrt(math.log((int(p_visits)+1) / (int(visits))))
            uct_scores.append(uct)
        #print(uct_scores)
        desired_node = node_list[uct_scores.index(max(uct_scores))]
        return desired_node

carlo = new_sim.Carlo()
prev = None
child = None
action = None
delay = 0
sirali_dugum_sayisi = 10
flag_egitim = False
while 1:
    carlo.run()
    state = carlo.get_state()
    rt_action = carlo.get_action()
    mcts = MCTS(state=state)
    print("---"+str(delay))
    if flag_egitim and rt_action != 0:
        if str(state) != str(prev) or str(action) != str(rt_action):
            if child is not None:
                mcts.expand(desired_node)
                _,_,_,_,_,_,_,_,child = mcts.get_data(desired_node)
            desired_node = mcts.select(state, child)
        _,_,_,node_state,action,is_terminal,_,_,child = mcts.get_data(desired_node)
        prev = node_state
        delay = 0
        if mcts.is_terminal(desired_node):
            mcts.backprop(desired_node)
            child = None
    elif not flag_egitim:
        if str(action) != str(rt_action) or delay == 0 or delay > 2:
            #print(child)
            if child is not None:
                _,_,_,_,_,_,_,_,child = mcts.get_data(desired_node)
                if child[0] == "?":
                    child = None
            desired_node = mcts.select(state, child, False)
            flag = mcts.is_terminal(desired_node)
            _,_,_,node_state,action,is_terminal,_,_,child = mcts.get_data(desired_node)
            prev = node_state
            #print(carlo.get_action())
            delay = 0
            if flag:
                child = None
        else:
            desired_node = desired_node
        _, _, _, state, action, _, fully_exp, _, _, = mcts.get_data(desired_node)
        carlo.act(action)
        #print(carlo.get_action() != 0)
        while carlo.get_action() != 0 and len(desired_node.split("-")) < sirali_dugum_sayisi:
            carlo.run()
            print("True")
            if str(state) != str(prev) or str(action) != str(rt_action):
                if child is not None:
                    mcts.expand(desired_node)
                    _,_,_,_,_,_,_,_,child = mcts.get_data(desired_node)
                desired_node = mcts.select(state, child)
            _,_,_,node_state,action,is_terminal,_,_,child = mcts.get_data(desired_node)
            prev = node_state
            if mcts.is_terminal(desired_node):
                mcts.backprop(desired_node)
                child = None
        delay += 1


pygame.quit()
cv2.destroyAllWindows()