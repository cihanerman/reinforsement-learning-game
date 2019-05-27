#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 17:54:56 2019
@project_name: Make to Game whit Reinforsement Learning
@author: cihanerman
"""
# import library
import pygame
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# window size
WIDTH = 300
HEIGHT = 300
FPS = 30 # how fast game is
# colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255, 0, 0) # RGB
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
# player model
class Gamer(pygame.sprite.Sprite):
    
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20))
        self.image.fill(BLACK)
        self.rect = self.image.get_rect()
        self.radius = 10
        pygame.draw.circle(self.image, RED, self.rect.center, self.radius)
        self.rect.centerx = WIDTH / 2
        self.rect.bottom = HEIGHT - 1
        self.speedx = 0
        
    def update(self, action):
        self.speedx = 0
        
        if action == 0:
            self.speedx = -3
        elif action == 1:
            self.speedx = 3
        else:
            self.speedx = 0
            
        self.rect.x += self.speedx
        
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        
        if self.rect.left < 0:
            self.rect.left = 0
            
    def getCoordinates(self):
        return (self.rect.x, self.rect.y)
            
class Foe(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.radius = 5
        pygame.draw.circle(self.image, GREEN, self.rect.center, self.radius)
        self.rect.x = random.randrange(0, WIDTH - self.rect.width)
        self.rect.y = random.randrange(3,7)
        self.speedx = 0
        self.speedy = 4
        
    def update(self):
        self.rect.x += self.speedx
        self.rect.y += self.speedy
        
        if self.rect.top > HEIGHT + 10:
            self.rect.x = random.randrange(0, WIDTH - self.rect.width)
            self.rect.y = random.randrange(3,7)
            self.speedy = 3
    
    def getCoordinates(self):
        return (self.rect.x, self.rect.y)
    
class DQLAgent:
    def __init__(self):
        self.state_size = 6
        self.action_size = 3 # right, left, dont move
        
        self.gamma = 0.95
        self.learning_rate = 0.01
        
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen = 1000)
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(48, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(self.action_size, activation = 'linear'))
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def action(self, state):
        state = np.array(state)
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state)
            next_state = np.array(next_state)
            
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state, train_target, verbose = 0)
            
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
class Env(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.all_sprite = pygame.sprite.Group()
        self.foes = pygame.sprite.Group()
        self.gamer = Gamer()
        self.all_sprite.add(self.gamer)
        self.foe1 = Foe()
        self.foe2 = Foe()
        self.foe3 = Foe()
        self.all_sprite.add(self.foe1)
        self.all_sprite.add(self.foe2)
        self.all_sprite.add(self.foe3)
        self.foes.add(self.foe1)
        self.foes.add(self.foe2)
        self.foes.add(self.foe3)
        
        self.reward = 0
        self.total_reward = 0
        self.done = False
        self.agent = DQLAgent()
        
    def findDistance(self, a, b):
        d = a - b
        return d
    
    def step(self, action):
        state_list = []
        
        self.gamer.update(action)
        self.foes.update()
        
        next_gamer_state = self.gamer.getCoordinates()
        next_foe1_state = self.foe1.getCoordinates()
        next_foe2_state = self.foe2.getCoordinates()
        next_foe3_state = self.foe3.getCoordinates()
        
        state_list.append(self.findDistance(next_gamer_state[0],next_foe1_state[0]))
        state_list.append(self.findDistance(next_gamer_state[1],next_foe1_state[1]))
        state_list.append(self.findDistance(next_gamer_state[0],next_foe2_state[0]))
        state_list.append(self.findDistance(next_gamer_state[1],next_foe2_state[1]))
        state_list.append(self.findDistance(next_gamer_state[0],next_foe3_state[0]))
        state_list.append(self.findDistance(next_gamer_state[1],next_foe3_state[1]))
        
        return [state_list]
    
    def initialStates(self):
        self.all_sprite = pygame.sprite.Group()
        self.foes = pygame.sprite.Group()
        self.gamer = Gamer()
        self.all_sprite.add(self.gamer)
        self.foe1 = Foe()
        self.foe2 = Foe()
        self.foe3 = Foe()
        self.all_sprite.add(self.foe1)
        self.all_sprite.add(self.foe2)
        self.all_sprite.add(self.foe3)
        self.foes.add(self.foe1)
        self.foes.add(self.foe2)
        self.foes.add(self.foe3)
        
        self.reward = 0
        self.total_reward = 0
        self.done = False
        
        state_list = []
        
        gamer_state = self.gamer.getCoordinates()
        foe1_state = self.foe1.getCoordinates()
        foe2_state = self.foe2.getCoordinates()
        foe3_state = self.foe3.getCoordinates()
        
        state_list.append(self.findDistance(gamer_state[0],foe1_state[0]))
        state_list.append(self.findDistance(gamer_state[1],foe1_state[1]))
        state_list.append(self.findDistance(gamer_state[0],foe2_state[0]))
        state_list.append(self.findDistance(gamer_state[1],foe2_state[1]))
        state_list.append(self.findDistance(gamer_state[0],foe3_state[0]))
        state_list.append(self.findDistance(gamer_state[1],foe3_state[1]))
        
        return [state_list]
    
    def run(self):
        state = self.initialStates()
        running = True
        batch_size = 24
        
        while running:
            self.reward = 2
            clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            action = self.agent.action(state)
            next_state = self.step(action)
            self.total_reward += self.reward
            
            hits = pygame.sprite.spritecollide(self.gamer, self.foes, False, pygame.sprite.collide_circle)
            
            if hits:
                self.reward = -100
                self.total_reward += self.reward
                self.done = True
                running = False
                print('Total reward: ',self.total_reward)
                
            self.agent.remember(state, action, self.reward, next_state, self.done)
            
            state = next_state
            
            self.agent.replay(batch_size)
            
            self.agent.adaptiveEGreedy()
            
            screen.fill(WHITE)
            self.all_sprite.draw(screen)
            
            pygame.display.flip()
            
        pygame.quit()
        
if __name__ == '__main__':
    env = Env()
    l = []
    t = 0
    
    while True:
        t += 1
        print('Epsilon: ', t)
        l.append(env.total_reward)
        
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('My Rl Game')
        clock = pygame.time.Clock()
        
        env.run()                
