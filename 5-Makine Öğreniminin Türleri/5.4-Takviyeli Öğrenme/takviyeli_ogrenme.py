import gym
import numpy as np

# Ortamın oluşturulması
env = gym.make('CartPole-v1')

# Hiperparametreler
alpha = 0.1  # Öğrenme oranı
gamma = 0.99  # Gelecekteki ödüllerin indirim oranı
epsilon = 0.1  # Epsilon-greedy strateji için başlangıç epsilon değeri
num_episodes = 1000  # Eğitim bölümlerinin sayısı
max_steps_per_episode = 100  # Her bölümdeki maksimum adım sayısı

# Q-tablosunun başlatılması
n_actions = env.action_space.n
state_space_bins = [20, 20, 20, 20]  # Durum uzayı için bin sayıları
q_table = np.random.uniform(low=-1, high=1, size=(state_space_bins + [n_actions]))

# Sürekli durumu ayrık duruma dönüştürme fonksiyonu
def discretize_state(state):
    bins = [
        np.linspace(-4.8, 4.8, state_space_bins[0]),
        np.linspace(-4, 4, state_space_bins[1]),
        np.linspace(-0.418, 0.418, state_space_bins[2]),
        np.linspace(-4, 4, state_space_bins[3])
    ]
    return tuple(
        int(np.digitize(state[i], bins[i]) - 1) for i in range(len(state))
    )

# Q-learning algoritması
for episode in range(num_episodes):
    state = discretize_state(env.reset())
    done = False
    for step in range(max_steps_per_episode):
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state)
        
        if done:
            reward = -100  # Direğin devrilmesini büyük bir ceza ile ödüllendiriyoruz

        q_table[state][action] = q_table[state][action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
        )
        
        state = next_state
        
        if done:
            break
    
    epsilon = max(0.01, epsilon * 0.995)  # Epsilon'u azaltarak keşfi azaltmak

    if episode % 100 == 0:
        print(f"Episode: {episode}")

# Eğitilen modelin test edilmesi
for episode in range(5):
    state = discretize_state(env.reset())
    done = False
    for step in range(max_steps_per_episode):
        env.render()
        action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state)
        state = next_state
        if done:
            break

env.close()
