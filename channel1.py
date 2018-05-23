import random, numpy, math, gym
import matplotlib.pyplot as plt

#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import channelConfig as channelConfig
#plt.ion()
global literation           #Number of test literation

OBSERV_BATCH = channelConfig.OBSERV_BATCH
USER_CNT = channelConfig.USER_CNT

class Brain:

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        # self.model.load_weights("cartpole-basic.h5")

    def _createModel(self):
        model = Sequential()

        model.add(Dense(output_dim=64, activation='relu', input_dim=OBSERV_BATCH))#stateCnt
        model.add(Dense(output_dim=64, activation='relu'))          #######################################
        model.add(Dense(output_dim=actionCnt, activation='linear'))

        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        #return self.predict(s.reshape(1, stateCnt), target=target).flatten()

        print ("**************################", np.array(s))
        return self.predict(np.array(s).reshape(1, OBSERV_BATCH), target=target).flatten()############################

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)
        #print ("samples\n\n",self.samples)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 100000
BATCH_SIZE = 256

GAMMA = 0.7

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001      # speed of decay

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        if random.random() < self.epsilon:
            print("*****random step*****")
            return random.randint(1, self.actionCnt - 1)
        else:
            print("*****predict step*****")
            pred = self.brain.predictOne(s)
            if numpy.argmax(pred) == 0:
                index = numpy.argsort(pred)
                return index[len(index)-2]
            else:
                return numpy.argmax(pred)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)
        #print (batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[0] for o in batch ])
        states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)

        #x = numpy.zeros((batchLen, self.stateCnt))
        x = numpy.zeros((batchLen, np.array(OBSERV_BATCH))) #######################################
        y = numpy.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        self.channel_cnt = self.env.env.channel_cnt
        self.pick_times = [0 for x in range(self.channel_cnt)]
        self.overall_step = 0.000
        self.hundred_step = [0.000 for x in range(0, 100)]
        self.overall_connect = 0.000
        self.overall_one_step = 0.000
        self.hundred_one_step = [0.000 for x in range(0, 100)]
        self.avg_step = 0
        self.avg_step_100 = 0

    def run(self, agent):
        ls_s = self.env.reset()
        R = 0
        step = 0                       #sum of steps in run_time period
        run_time = 0
        global literation           #Number of test literation

        single_step = 0                #step for every successful try
        f_agent = open("agent.csv", "w")
        f_agent.write('user_number,state,action,reward,next_state\n')
        f_channel = open("channel_available.csv", "w")
        for i in range(self.channel_cnt):
            f_channel.write(str(i+1))
            if i < self.channel_cnt-1:
                f_channel.write(',')
            else:
                f_channel.write('\n')
        state_batch_ls = [[0] * OBSERV_BATCH] * USER_CNT
        next_state_batch_ls =  [[0] * OBSERV_BATCH] * USER_CNT

        for i in range(USER_CNT):
            state_batch_ls[i][0] = ls_s[i]

        for i in range(0, literation):
            self.env.render()

            #a = agent.act(s)
            ls_a = [0] * USER_CNT
            for j in range(USER_CNT):
                ls_a[j] = agent.act(state_batch_ls[j][0])
            print("acts------------:", *ls_a)

            step += 1
            single_step += 1

            #self.env.env.setStateBatch(state_batch_ls)

            ls_s_, r, done, info = self.env.step(ls_a)
            channel_available = self.env.env.getChannelAvailable()
            #self.env.env.getRewardChart()
            #self.env.env.getTChart()
            print("reward:", r)

           # if done: # terminal state
           #     self.pick_times[s_-1-self.channel_cnt] += 1
           #     run_time += 1
           #     if (single_step == 1):
           #         self.overall_one_step += 1
           #     single_step = 0
           #     #s_ = None           # it's ok to run without this line, almost no influence to performance
           #     #agent.observe( (s, a, r, s_) )
           #     #s_ = self.env.reset()
           #     print ("done\n")
            for k in range(USER_CNT):
                for j in range(OBSERV_BATCH - 1, 0, -1):
                    next_state_batch_ls[k][j] = state_batch_ls[k][j-1]
                next_state_batch_ls[k][0]= ls_s_[k]
            #agent.observe( (s, a, r, s_) ) #include add to memory
                print ( "stuff to be observed#############",(np.array(state_batch_ls[k]), ls_a[k], r, np.array(next_state_batch_ls[k])) )
                agent.observe( (np.array(state_batch_ls[k]), ls_a[k], r, np.array(next_state_batch_ls[k])) )###################################

            agent.replay()
            for k in range(USER_CNT):
                record_str = str(k)+','+str(ls_s[k])+','+str(ls_a[k])+','+str(r)+','+str(ls_s_[k])+'\n'
                f_agent.write(record_str)
            for j in range(self.channel_cnt):
                f_channel.write(str(channel_available[j+1]))
                if j < self.channel_cnt-1:
                    f_channel.write(',')
                else:
                    f_channel.write('\n')

            ls_s = ls_s_
            for k in range(USER_CNT):
                for i in range(0, OBSERV_BATCH):
                    state_batch_ls[k][i] = next_state_batch_ls[k][i]

            R += r

        f_agent.close()
        f_channel.close()

    def draw_plots(self):
        global literation
        f_agent = open("agent.csv", "r")
        f_agent.readline()
        channel_chosen_cnt = [0] * self.channel_cnt
        success_connexion = 0
        block_num = 30
        block_cnt = 0
        local_ind_ls = []
        local_success_ls = []
        local_success = 0

        for i in range(0, literation):
            step_str = f_agent.readline()
            s = int(step_str.split(",")[1]) # Read the current state
            channel_chosen_cnt[self.env.env.getChannelNumber(s) - 1] += 1
            if not self.env.env.isChannelBlocked(s): # success connexion
                success_connexion += 1
                local_success += 1

            block_cnt += 1
            if block_cnt % block_num == 0:
                # count the success number of block_num communications
                block_cnt = 0
                local_ind_ls.append(i)
                local_success_ls.append(local_success / block_num)
                local_success = 0
                block_cnt = 0
        f_agent.close()

        f_channel = open("channel_available.csv", "r")
        f_channel.readline()
        jammed_time_cnt = [0] * self.channel_cnt
        for i in range(0, literation):
           step_str = f_channel.readline()
           s = step_str.split(",")
           for j in range(0, self.channel_cnt):
               if int(s[j]) == 0:
                   jammed_time_cnt[j] += 1
        f_channel.close()

        # Draw charts of users performence
        plt.subplot(3,1,1) # Draw outage probability
        plt.plot(local_ind_ls, local_success_ls)
        plt.ylabel('Success Rate')
        plt.subplot(3,1,2)
        plt.bar(range(1, self.channel_cnt+1), channel_chosen_cnt, color="blue", align="center")


        # Draw chart of jammer performence
        plt.subplot(3,1,3)
        plt.bar(range(1, self.channel_cnt+1), jammed_time_cnt, color="yellow", align="center")

        plt.show()

#            if done:
#                break

    #        if (run_time == 10):   #modifiable, output frequency
    #            break

        #print("Total reward:", R)
        #print("Steps taken:", step)
        #self.hundred_step[int(self.overall_connect % 100)] = step
        #self.overall_step += float(step)

#        #if (step == 1):
#        #    self.overall_one_step += 1
#        #    self.hundred_one_step[int(self.overall_connect % 100)] = 1
#        #else:
#        #    self.hundred_one_step[int(self.overall_connect % 100)] = 0
        #
        #self.overall_connect += 10

        #avg_step = float(self.overall_step/self.overall_connect)
        #print("Average steps:\t\t\t\t", avg_step)
        #if int(self.overall_connect) > 100:
        #    avg_step_100 = float(sum(self.hundred_step)/100)
        #else:
        #    avg_step_100 = float(sum(self.hundred_step)/int(self.overall_connect))
        #print("Average steps of latest 100 tries:\t", avg_step_100)

        #for i in range (self.channel_cnt):
        #    if self.pick_times[i] != 0:
        #        print("Channel %d picked times: %d" %(i+1, self.pick_times[i]))

        #avg_one_step = float(self.overall_one_step/self.overall_connect)
        #print("Overall success rate:\t\t\t%", avg_one_step * 100)
#        #if int(self.overall_connect) > 100:
#        #    avg_one_step_100 = float(sum(self.hundred_one_step)/100)
#        #else:
#        #    avg_one_step_100 = float(sum(self.hundred_one_step)/int(self.overall_connect))
#        #print("Success rate of latest 100 tries:\t%", avg_one_step_100 * 100)

        #print("Overall run time:", int(self.overall_connect))
        #print("\n")
###############################################################################   CHART PART
#        ##if int(self.overall_connect) < 245:
        #x_scale = numpy.log(int(self.overall_connect))
#        ##else:
#        ##    x_scale = numpy.log10(int(self.overall_connect)) + 3.1
        #plt.subplot(2,1,1)
#        #plt.axis()
        #if int(self.overall_connect) < 245:
        #    plt.scatter(x_scale, avg_step, c = 'r')
        #    plt.scatter(x_scale, avg_step_100, c = 'b')
        #    plt.scatter(x_scale, avg_one_step, c = 'y')
        #    plt.pause(0.00001)
        #elif (int(self.overall_connect) < 1000) and (int(self.overall_connect) % 10 == 0):
        #    plt.scatter(x_scale, avg_step, c = 'r')
        #    plt.scatter(x_scale, avg_step_100, c = 'b')
        #    plt.scatter(x_scale, avg_one_step, c = 'y')
        #    plt.pause(0.00001)
        #elif (int(self.overall_connect) < 2000) and (int(self.overall_connect) % 30 == 0):
        #    plt.scatter(x_scale, avg_step, c = 'r')
        #    plt.scatter(x_scale, avg_step_100, c = 'b')
        #    plt.scatter(x_scale, avg_one_step, c = 'y')
        #    plt.pause(0.00001)
        #elif (int(self.overall_connect) < 30000) and (int(self.overall_connect) % 100 == 0):
        #    plt.scatter(x_scale, avg_step, c = 'r')
        #    plt.scatter(x_scale, avg_step_100, c = 'b')
        #    plt.scatter(x_scale, avg_one_step, c = 'y')
        #    plt.pause(0.00001)

        #plt.subplot(2,1,2)
        #if (int(self.overall_connect) % 100 == 0):
        #    plt.bar(range(len(self.pick_times)), self.pick_times, color="blue")



#-------------------- MAIN ----------------------------
global literation
literation = channelConfig.literation
PROBLEM = 'Channel-v0'
env = Environment(PROBLEM)

stateCnt  = np.array(1)#env.env.observation_space.shape[0]
actionCnt = env.channel_cnt + 1

agent = Agent(stateCnt, actionCnt)

#env.run(agent)
#env.run(agent)
#env.run(agent)
try:
    #while True:
    env.run(agent)
    env.draw_plots()
finally:
    agent.brain.model.save("channel.h5")
