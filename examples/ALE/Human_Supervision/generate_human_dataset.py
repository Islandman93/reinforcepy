#!/usr/bin/env python

# ale_python_test_pygame_player.py
# Author: Ben Goodrich
#
# This modified ale_python_test_pygame.py to provide a fully interactive experience allowing the player
# to play. RAM Contents, current action, and reward are also displayed.
# keys are:
# arrow keys -> up/down/left/right
# z -> fire button
import sys
from ale_python_interface import ALEInterface
import numpy as np
import pygame
import time
import os
import pickle
import copy
from scipy.misc import imresize
from pygame import surfarray

key_action_tform_table = (
0, #00000 none
2, #00001 up
5, #00010 down
2, #00011 up/down (invalid)
4, #00100 left
7, #00101 up/left
9, #00110 down/left
7, #00111 up/down/left (invalid)
3, #01000 right
6, #01001 up/right
8, #01010 down/right
6, #01011 up/down/right (invalid)
3, #01100 left/right (invalid)
6, #01101 left/right/up (invalid)
8, #01110 left/right/down (invalid)
6, #01111 up/down/left/right (invalid)
1, #10000 fire
10, #10001 fire up
13, #10010 fire down
10, #10011 fire up/down (invalid)
12, #10100 fire left
15, #10101 fire up/left
17, #10110 fire down/left
15, #10111 fire up/down/left (invalid)
11, #11000 fire right
14, #11001 fire up/right
16, #11010 fire down/right
14, #11011 fire up/down/right (invalid)
11, #11100 fire left/right (invalid)
14, #11101 fire left/right/up (invalid)
16, #11110 fire left/right/down (invalid)
14  #11111 fire up/down/left/right (invalid)
)


ale = ALEInterface()
rom = b'../roms/breakout.bin'

ale.loadROM(rom)
legal_actions = ale.getMinimalActionSet()
print(legal_actions)

(screen_width,screen_height) = ale.getScreenDims()
print("width/height: " +str(screen_width) + "/" + str(screen_height))

(display_width,display_height) = (1024,420)

#init pygame
pygame.init()
screen = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption("Arcade Learning Environment Player Agent Display")

game_surface = pygame.Surface((screen_width,screen_height))

pygame.display.flip()

#init clock
clock = pygame.time.Clock()

skipFrame = 3
episode = 0
total_reward = 0.0

actions = list()
rewards = list()
states = list()
curractions = list()
currrewards = list()
currstates = list()

while(episode < 1):
    #get the keys
    # event = pygame.event.wait()
    # print(event.type)
    keys = 0
    pressed = pygame.key.get_pressed()
    keys |= pressed[pygame.K_UP]
    keys |= pressed[pygame.K_DOWN]  <<1
    keys |= pressed[pygame.K_LEFT]  <<2
    keys |= pressed[pygame.K_RIGHT] <<3
    keys |= pressed[pygame.K_z] <<4
    a = key_action_tform_table[keys]
    curractions.append(a)
    # if keys == 0:
    #     time.sleep(0.2)

    #clear screen
    screen.fill((0,0,0))

    #get atari screen pixels and blit them
    reward = 0
    gamescreen = ale.getScreenRGB()
    currstates.append(gamescreen)

    reward += ale.act(a)
    total_reward += reward
    currrewards.append(reward)

    frames = np.swapaxes(gamescreen, 0, 1)
    frames = surfarray.make_surface(frames)
    screen.blit(pygame.transform.scale(frames, (screen_width*2, screen_height*2)),(0,0))

    #get RAM
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size),dtype=np.uint8)
    ale.getRAM(ram)


    #Display ram bytes
    font = pygame.font.SysFont("Ubuntu Mono",32)
    text = font.render("RAM: " ,1,(255,208,208))
    screen.blit(text,(330,10))

    font = pygame.font.SysFont("Ubuntu Mono",25)
    height = font.get_height()*1.2

    line_pos = 40
    ram_pos = 0
    while(ram_pos < 128):
        ram_string = ''.join(["%02X "%ram[x] for x in range(ram_pos,min(ram_pos+16,128))])
        text = font.render(ram_string,1,(255,255,255))
        screen.blit(text,(340,line_pos))
        line_pos += height
        ram_pos +=16

    #display current action
    font = pygame.font.SysFont("Ubuntu Mono",32)
    text = font.render("Current Action: " + str(a) ,1,(208,208,255))
    height = font.get_height()*1.2
    screen.blit(text,(330,line_pos))
    line_pos += height

    #display reward
    font = pygame.font.SysFont("Ubuntu Mono",30)
    text = font.render("Total Reward: " + str(total_reward) ,1,(208,255,255))
    screen.blit(text,(330,line_pos))

    pygame.display.flip()

    #process pygame event queue
    exit=False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit=True
            break
    if(pressed[pygame.K_q]):
        exit = True
    if(exit):
        break

    #delay to 60fps
    clock.tick(60.)

    if(ale.game_over()):
        # TODO: append state tp1
        episode_frame_number = ale.getEpisodeFrameNumber()
        frame_number = ale.getFrameNumber()
        print("Frame Number: " + str(frame_number) + " Episode Frame Number: " + str(episode_frame_number))
        print("Episode " + str(episode) + " ended with score: " + str(total_reward))
        ale.reset_game()
        total_reward = 0.0
        episode = episode + 1
        actions.append(copy.deepcopy(np.asarray(curractions, dtype=np.int8)))
        rewards.append(copy.deepcopy(np.asarray(currrewards, dtype=np.int8)))
        states.append(copy.deepcopy(np.asarray(currstates, dtype=np.uint8)))
        curractions.clear()
        currrewards.clear()
        currstates.clear()
with open(os.getcwd() + '/human_dataset.pkl', 'wb') as outFile:
    pickle.dump((states, actions, rewards), outFile)

