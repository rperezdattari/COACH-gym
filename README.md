# COACH-gym
A python implementation of the COACH algorithm for the Cartpole problem in OpenAI gym.

This code is based on the following publications:
1. [Interactive learning of continuous actions from corrective advice communicated by humans](http://robocup.oss-cn-beijing.aliyuncs.com/symposium%2FRoboCup_Symposium_2015_submission_20.pdf) 
2. [An Interactive Framework for Learning Continuous Actions Policies Based on Corrective Feedback](https://link.springer.com/article/10.1007/s10846-018-0839-z)

## Installation

To use the code, it is necessary to first install the gym toolkit: https://github.com/openai/gym

Then, the files in the `gym` folder of this repository should be replaced/added in the installed gym folder on your PC.

### Requirements
* NumPy
* PyGame

## Usage

To run the code just type in the terminal inside the folder `COACH-gym`:

```python 
python main.py
```
Along with the rendered environment, a small black window should appear when running the code. To be able to give feedback to the agent, this window must be selected/clicked with the computed mouse. 

## Comments

The COACH algorithm is designed to work with problems of continuous actions spaces. Given that the Cartpole environment of gym was designed to work with discret action spaces, a modified continuous version of this environment is used.

This code has been tested in `Ubuntu 16.04` and `python >= 3.5`.



