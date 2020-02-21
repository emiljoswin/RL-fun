from typing import List, Tuple, Dict, Union
import numpy as np

#Note: The value converges to the 'exact same' value as long as the terminals are initialized to 0

class State:
    def __init__(self, value: int=0) -> None:
        self._value = value
        self._policy = {
            'U': 0.25,
            'D': 0.25,
            'L': 0.25,
            'R': 0.25
        }

class Grid:
    def __init__(self, size: int=4, terminals: List=[0, 15], discount: int=0.5, state_value: int=0) -> None:
        self._size = size
        self._grid = size * size
        self._dicount = discount
        self._terminals = terminals
        self._states = [State() for _ in range(size*size)]
        self._reward = -1 # TODO - refactor

        self.set_state_value(0)
        for i in terminals:
            self._states[i]._value = state_value

    def set_state_value(self, value: Union[int, None]) -> None:
        if value is None:
            print('random_initilization')
            for state in self._states:
                state._value = np.random.randint(-10, 10)
            return

        for state in self._states:
            state._value = value

    def cord_to_index(self, row: int, col: int) -> int: 
        return row * self._size + col

    def get_next_state_and_reward(self, state_num: int, state: State, action: str) -> Tuple[State, int]:

        r, c = divmod(state_num, self._size)
        
        if action == 'U':
            r -= 1
        if action == 'D':
            r += 1
        if action == 'L':
            c -= 1
        if action == 'R':
            c += 1
        
        if r < 0:
            new_r = 0
        elif r > self._size-1:
            new_r = self._size-1
        else:
            new_r = r
        
        if c < 0:
            new_c = 0
        elif c > self._size-1:
            new_c = self._size-1
        else:
            new_c = c

        idx = self.cord_to_index(new_r, new_c)
        new_state = self._states[idx]
        return new_state, self._reward

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            self.print_values()
            stable = self.improve_policy()
            if stable: 
                print('policy Stabilized as ')
                self.print_policy()
                break
            self.print_policy()
            print('Policy not stabilized. Continuing')

    def policy_evaluation(self, eps=1e-3) -> None:
        for step in range(500):
            print(f'performing step: {step}')
            new_state_values = [0]*len(self._states)
            for i, state in enumerate(self._states):
                new_value = 0
                if i in self._terminals:
                    new_state_values[i] = 0.
                    continue

                for action in state._policy:
                    next_state, reward = self.get_next_state_and_reward(i, state, action)
                    new_value += state._policy[action]*(reward + self._dicount*next_state._value)
                new_state_values[i] = new_value
            
            diff = [new_state_values[i] - self._states[i]._value for i in range(len(self._states))]
            converge = sum([1 for d in diff if abs(d) < eps  ])
            if converge == len(self._states):# Every state has converged
                print(f'Converged at steps: {step}, returning')
                return

            for i in range(len(new_state_values)):
                self._states[i]._value = new_state_values[i]

    def improve_policy(self) -> bool:
        """
            Extract new policy and if the new policy is not same as old
            policy, update the policy. 
            Else, return False indicating that the policy has stabilized.
        """
        policies = []
        for i, state in enumerate(self._states):
            if i in self._terminals: 
                policies.append(state._policy)
                continue

            best_actions = []
            for action in state._policy:
                next_state, reward = self.get_next_state_and_reward(i, state, action)
                q_value = self._reward + self._dicount*next_state._value
                best_actions.append({'action': action, 'value': q_value})
            best_actions = sorted(best_actions, reverse=True, key=lambda k: k['value'])
            best_action_value = best_actions[0]['value']

            actions = []
            for elem in best_actions:
                if elem['value'] == best_action_value:
                    actions.append(elem['action'])
            prob = 1/(len(actions))
            policy = {}
            for a in actions:
                policy[a] = prob
            policies.append(policy)

        if not self.check_policy_stability(policies):
            for i, state in enumerate(self._states):
                if i in self._terminals: continue
                state._policy = policies[i] 
            return False

        return True

    def check_policy_stability(self, new_policy) ->bool:
        for i, state in enumerate(self._states):
            if i in self._terminals: continue
            new_p = new_policy[i]
            old_p = state._policy
            if len(new_p) != len(old_p):
                return False
            for k in old_p:
                if k in new_p and new_p[k] == old_p[k]:
                    continue
                else: return False
        
        return True

    def print_values(self) -> None:
        for r in range(self._size):
            for c in range(self._size):
                idx = self.cord_to_index(r, c)
                print('\t' , round(self._states[idx]._value, 2) , end='')
            print('\n')


    def print_policy(self) -> None:
        for r in range(self._size):
            for c in range(self._size):
                idx = self.cord_to_index(r, c)
                p = ''.join(list(self._states[idx]._policy.keys()))
                print('\t', p, end='')
            print('\n')


if __name__ == '__main__':
    g = Grid(size=4, discount=1)
    g.policy_iteration()
    # g.print_policy()


