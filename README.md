## ※ Project Structure
    2024_DQN-DDQN/
    │
    ├── main.py                # Main entry point of the program
    ├── config.py              # Hyperparameters and argument parsing
    ├── environment.py         
    ├── memory.py              # Experience replay class
    ├── model.py               # Neural network (CNN) class
    ├── runner.py              
    └── README.md

## 1. 2013 DQN (Deep Q-Network)
**CNN과 Experience Replay Memory 사용**

$$target\ value(y_t)=r_t + \gamma * max_{a'}Q(s_{t+1}, a'; \theta)$$

## 2. 2015 DQN
**Target network를 사용하여 움직이는 target에 대한 학습 불안정성 해결 (이중네트워크)**
1. Target value와 Action value를 구분하여 기준값과 결과값이 동시에 움직이지 않는다.
2. C 스텝 동안 target network를 고정시켜 해당 구간동안 원하는 방향으로 업데이트 하고, 이후에 target network와 main network를 동기화시켜 bias를 줄인다.

$$target\ value(y_t)=r_t + \gamma * max_{a'}Q_{target}(s_{t+1}, a'; {\theta}^-)$$

## 3. DDQN (Double DQN)
**DQN에서 action value를 과대평가(overestimation)하는 문제 발생**
1. Main network : 다음 상태($s_{t+1}$)에서 Q-value를 최대화할 수 있는 행동 $\hat{a}$ 선택
2. Target network : 다음 상태($s_{t+1}$)에서 행동 $\hat{a}$에 대한 평가

$$\hat{a}=max_{a'}Q_{main}(s_{t+1}, a'; \theta)$$
$$target\ value(y_t)=r_t + \gamma * Q_{target}(s_{t+1}, \hat{a}; {\theta}^-)$$

### Reference
- https://github.com/jordanlei/deep-reinforcement-learning-cartpole/tree/master
- https://people.engr.tamu.edu/guni/csce642/files/dqn.pdf
- https://www.nature.com/articles/nature14236
- https://ojs.aaai.org/index.php/AAAI/article/view/10295
