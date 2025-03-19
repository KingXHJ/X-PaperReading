import numpy as np 

class Agent:
    def __init__(self,OOXX_index, Epsilon=0.1,LearningRate=0.1):
        self.value = np.zeros((3,3,3,3,3,3,3,3,3))
        self.currentState = np.zeros(9)
        self.previousState = np.zeros(9)
        self.index = OOXX_index
        self.epsilon = Epsilon
        self.alpha = LearningRate


    def reset(self):
        self.currentState = np.zeros(9)
        self.previousState = np.zeros(9)


    def actionTake(self,State):
        state = State.copy()
        available = np.where(state==0)[0]
        length = len(available)
        if length == 0 :
            return state
        else:
            random = np.random.uniform(0,1)
            if random < self.epsilon:
                choose = np.random.randint(length)
                state[available[choose]] = self.index
            else:
                tempValue = np.zeros(length)
                for i in range(length):
                    tempState = state.copy()
                    tempState[available[i]] = self.index
                    tempValue[i] = self.value[tuple(tempState.astype(int))]
                choose = np.where(tempValue == np.max(tempValue))[0]
                chooseIndex = np.random.randint(len(choose))
                state[available[choose[chooseIndex]]] = self.index
            return state
    

    def valueUpdate(self, State):
        self.currentState = State.copy()
        self.value[tuple(self.previousState.astype(int))] += \
            self.alpha * (self.value[tuple(self.currentState.astype(int))] - \
                          self.value[tuple(self.previousState.astype(int))])
        self.previousState = self.currentState.copy()
