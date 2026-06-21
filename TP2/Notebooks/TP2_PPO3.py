import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from IPython.display import clear_output
import matplotlib.pyplot as plt



# Redes neurais do ator e do crítico
class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super().__init__()

        layers = []
        in_features = state_dim

        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.Tanh())
            in_features = h

        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(in_features, action_dim)

    def forward(self, x):
        h = self.net(x)
        logits = self.out(h)
        return logits


class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_layers):
        super().__init__()

        layers = []
        in_features = state_dim

        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.Tanh())
            in_features = h

        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(in_features, 1)

    def forward(self, x):
        h = self.net(x)
        value = self.out(h).squeeze(-1)
        return value



# PPO no Blackjack
class PPOBlackjack:
    def __init__(self, parameters):
        self.parameters = parameters

        self.env = gym.make("Blackjack-v1", natural=parameters.get("natural", False))
        self.state_dim = 32 + 11 + 2  # mesma estrutura lógica do seu código tabular
        self.action_dim = self.env.action_space.n

        self.gamma = parameters.get("gamma", 0.99)
        self.clip_eps = parameters.get("clip_eps", 0.2)
        self.lr = parameters.get("lr", 3e-4)
        self.update_epochs = parameters.get("update_epochs", 4)
        self.batch_episodes = parameters.get("batch_episodes", 32)
        self.entropy_coef = parameters.get("entropy_coef", 0.01)
        self.value_coef = parameters.get("value_coef", 0.5)

        self.episodes = parameters.get("episodes", 100000)

        
        self.hidden_layers = parameters.get(
        "hidden_layers",
        [64, 64]
        )
        
        self.hidden_layers = parameters.get("hidden_layers", [64, 64])

        self.actor = ActorNet(
        self.state_dim,
        self.action_dim,
        parameters["actor_hidden_layers"]
        ) 

        self.critic = CriticNet(
        self.state_dim,
        parameters["critic_hidden_layers"]
        )
        
        

        self.opt = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr
        )

        self.save_model = parameters.get("save_model", False)
        self.model_file = parameters.get("model_file", "ppo_blackjack.pt")

        self.reset_buffers()

    def reset_buffers(self):
        self.batch_states = []
        self.batch_actions = []
        self.batch_old_logps = []
        self.batch_returns = []
        self.batch_advantages = []

    def obs_to_state(self, obs):
        # obs = (player_sum, dealer_card, usable_ace)
        p, d, a = obs
        x = np.zeros(self.state_dim, dtype=np.float32)

        x[int(p)] = 1.0
        x[32 + int(d)] = 1.0
        x[32 + 11 + int(bool(a))] = 1.0

        return x

    @torch.no_grad()
    def select_action(self, state_vec):
        
        s = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        

        logits = self.actor(s)
        value = self.critic(s)
        
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)

        return int(action.item()), float(logp.item()), float(value.item())

    def collect_episode(self):
        obs, _ = self.env.reset()
        done = False

        states = []
        actions = []
        old_logps = []
        values = []
        rewards = []

        while not done:
            s = self.obs_to_state(obs)
            a, logp, v = self.select_action(s)

            obs, r, terminated, truncated, _ = self.env.step(a)
            done = terminated or truncated

            states.append(s)
            actions.append(a)
            old_logps.append(logp)
            values.append(v)
            rewards.append(r)

        # retorno descontado por episódio
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.append(R)
        returns.reverse()

        # calcula vantagem : A = R - V
        advantages = np.array(returns, dtype=np.float32) - np.array(values, dtype=np.float32)

        self.batch_states.extend(states)
        self.batch_actions.extend(actions)
        self.batch_old_logps.extend(old_logps)
        self.batch_returns.extend(returns)
        self.batch_advantages.extend(advantages.tolist())

        return float(np.sum(rewards))

    def update(self):
        
        states = torch.tensor(np.array(self.batch_states), dtype=torch.float32)
      
        actions = torch.tensor(self.batch_actions, dtype=torch.int64)
        
        old_logps = torch.tensor(self.batch_old_logps, dtype=torch.float32)
        
        returns = torch.tensor(self.batch_returns, dtype=torch.float32)
        
        advantages = torch.tensor(self.batch_advantages, dtype=torch.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_epochs):
            
            logits = self.actor(states)
            values = self.critic(states)

            dist = Categorical(logits=logits)

            new_logps = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logps - old_logps)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = (returns - values).pow(2).mean()

            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        self.reset_buffers()

    def save(self):
        #torch.save(self.net.state_dict(), self.model_file)
        checkpoint = {
        "actor": self.actor.state_dict(),
        "critic": self.critic.state_dict()
        }

        torch.save(checkpoint, self.model_file)



# Treinamento
if __name__ == "__main__":

    parameters = {
        "episodes": 100000,
        "gamma": 0.99,
        "clip_eps": 0.2,
        "lr": 3e-4,
        "update_epochs": 4,
        "batch_episodes": 32,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "natural": False,
        "save_model": False,
        "model_file": "ppo_blackjack.pt",
        "actor_hidden_layers": [32, 16, 16],
        "critic_hidden_layers": [32, 16, 16]
    }

    agent = PPOBlackjack(parameters)

    print("=== Ator ===")
    print(agent.actor)

    print("\n=== Crítico ===")
    print(agent.critic)

    rewards = []
    avg_rewards = []

    print("Treinando PPO...")

    for i in range(parameters["episodes"]):

        r = agent.collect_episode()

        rewards.append(r)
        avg_rewards.append(np.mean(rewards[-500:]))

        if (i + 1) % parameters["batch_episodes"] == 0:
            agent.update()

        # Apenas imprime progresso
        if (i + 1) % 5000 == 0:
            media = np.mean(rewards[-5000:])
            print(
                f"Episódio {i+1:6d}/{parameters['episodes']} "
                f"| Média últimos 5000 = {media:.4f}"
            )

    # Atualiza
    if len(agent.batch_states) > 0:
        agent.update()

    # Salva
    if parameters["save_model"]:
        agent.save()

    # Estatísticas
    n = min(5000, len(rewards))

    print("\n=== Resultados finais ===")
    print(f"Recompensa média últimos {n} episódios: "
          f"{np.mean(rewards[-n:]):.4f}")

    # =========================
    # Gráfico final
    # =========================

    plt.figure(figsize=(10, 6))

    plt.plot(
        avg_rewards,
        linewidth=2,
        label="Média móvel (500)"
    )

    plt.xlabel("Episódio")
    plt.ylabel("Recompensa")
    plt.title("PPO - Blackjack")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.show()