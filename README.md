# 📘 Tópicos Especiais em Sinais e Sistemas  
## Introdução ao Aprendizado por Reforço  

Repositório destinado aos Trabalhos Práticos (TPs) da disciplina **Tópicos Especiais em Sinais e Sistemas: Introdução ao Aprendizado por Reforço**, do Programa de Pós-Graduação em Engenharia Elétrica da Universidade Federal de Minas Gerais (UFMG).

---

## 👨‍🎓 Alunos
- Henrique Marques Cruz  
- Matheus Tadeu Alves de Carvalho  

---

## 🎯 TP1 – Blackjack com Reinforcement Learning

### 📌 Descrição
O **Trabalho Prático 1 (TP1)** consiste na implementação de algoritmos de Aprendizado por Reforço para o problema do Blackjack, utilizando o ambiente da OpenAI Gym.

### ⚙️ Objetivos
- Modelar o problema como um Processo de Decisão de Markov (MDP)  
- Implementar algoritmos de RL (ex: Monte Carlo ou Q-learning)  
- Estimar uma política ótima  
- Avaliar o desempenho do agente  

### 🧠 Ambiente
- `Blackjack-v1` (OpenAI Gym)
- Estados:
  - Soma das cartas do jogador  
  - Carta visível do dealer  
  - Presença de ás utilizável (usable ace)  
- Ações:
  - `0`: parar (stick)  
  - `1`: pedir carta (hit)  

---

## 🎯 TP2 – PPO para Blackjack com Redes Neurais

### 📌 Descrição
O **Trabalho Prático 2 (TP2)** explora a aplicação do algoritmo **Proximal Policy Optimization (PPO)** para o problema do Blackjack. Diferentemente do TP1, o PPO utiliza **redes neurais artificiais** como aproximadores de função em uma arquitetura ator-crítico.

### ⚙️ Objetivos
- Implementar o algoritmo PPO
- Utilizar redes neurais para representar a política e a função valor
- Avaliar diferentes arquiteturas de redes neurais
- Comparar os resultados com os métodos tabulares do TP1
- Analisar o impacto da complexidade da rede no desempenho do agente

### 🤖 Método
- Algoritmo: PPO (Proximal Policy Optimization)
- Arquitetura: Actor-Critic
- Codificação dos estados: One-Hot Encoding
- Redes avaliadas:
  - 32
  - 32-16
  - 32-16-16

### 📈 Principais Resultados
Os experimentos mostraram que as arquiteturas com duas e três camadas escondidas obtiveram desempenho superior aos métodos tabulares estudados no TP1, atingindo recompensas médias próximas de **−0,06**, enquanto SARSA e Q-Learning convergiram para aproximadamente **−0,10**.

---

## 📌 Observações

Este repositório tem fins exclusivamente acadêmicos, sendo parte das atividades da disciplina no âmbito da pós-graduação da UFMG.
