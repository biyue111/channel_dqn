#!/bin/bash
# Update the environment, then run the agent
source ~/anaconda3/bin/activate Jamming
cp ChannelEnv.py ../gym/gym/envs/classic_control/ChannelEnv.py
cp channelConfig.py ../gym/gym/envs/classic_control/channelConfig.py
python ./channel1.py
