
# Planning Agents

This repository demonstrates the **Planning Agent** pattern in Agentic AI. Unlike a standard LLM that immediately generates a response, a Planning Agent first decomposes a complex task into smaller, manageable steps and then executes those steps systematically.

Planning improves reasoning quality, reduces hallucinations, and enables agents to solve multi-step problems more reliably. Modern agent frameworks commonly separate planning from execution for complex workflows. :contentReference[oaicite:0]{index=0}

---

## Overview

A Planning Agent follows this workflow:

```text
          User Request
                │
                ▼
        📝 Planning Agent
                │
      Break Goal into Tasks
                │
                ▼
      Prioritized Action Plan
                │
                ▼
      Execute Step by Step
                │
                ▼
         Final Response
```

Instead of asking the LLM to solve everything in one prompt, the agent first decides:

- What needs to be done?
- In what order?
- Which tools or APIs are required?
- What information is missing?
- How should intermediate results be used?

---

## 🚀 Features

- Goal decomposition
- Multi-step reasoning
- Task prioritization
- Sequential execution
- Modular architecture
- Easy integration with external tools
- Better accuracy on complex problems

---

## 📂 Project Structure

```text
8_Planning Agents/
│
├── main.py              # Main application
├── planner.py           # Planning logic
├── executor.py          # Executes generated plan
├── prompts.py           # Prompt templates
├── requirements.txt
└── README.md
```

> *(Update this structure if your project contains different files.)*

---

## ⚙️ How It Works

### Step 1: User Query

```text
Plan a 5-day trip to Japan with a budget of $2,000.
```

---

### Step 2: Planning Phase

The Planning Agent creates something like:

```text
1. Determine cities to visit
2. Estimate transportation costs
3. Select hotels
4. Suggest attractions
5. Calculate food expenses
6. Verify budget
7. Generate itinerary
```

---

### Step 3: Execution

Each task is solved individually.

Example:

```text
Task 1 → Find destinations

↓

Task 2 → Estimate transportation

↓

Task 3 → Choose hotels

↓

Task 4 → Produce final itinerary
```

---

## 🛠️ Installation

Clone the repository

```bash
git clone https://github.com/REEYABORIKAR/Agentic-AI.git
```

Navigate to the project

```bash
cd Agentic-AI/8_Planning\ Agents
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python main.py
```

---

## 🧩 Planning Architecture

```text
                    User Goal
                        │
                        ▼
                Planning Agent
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
    Task 1          Task 2          Task 3
        │               │               │
        └───────────────┼───────────────┘
                        │
                        ▼
                 Execution Agent
                        │
                        ▼
                 Final Response
```

---

## 💡 Example

### Input

```text
Create a study plan to learn Python in 30 days.
```

### Generated Plan

```text
Day 1–5:
Python Basics

Day 6–10:
Functions

Day 11–15:
Object-Oriented Programming

Day 16–20:
File Handling

Day 21–25:
Projects

Day 26–30:
Interview Preparation
```

---

## 🎯 Advantages of Planning Agents

- Better reasoning
- Reduced hallucinations
- Handles long tasks efficiently
- Easier debugging
- Modular execution
- Improved scalability
- Supports tool usage and workflows

---

## 📚 Applications

- Travel planning
- Research assistants
- Software development
- Report generation
- AI coding assistants
- Workflow automation
- Project management
- Data analysis
- Educational tutors

---

## 🧰 Tech Stack

- Python
- Large Language Models (LLMs)
- Agentic AI
- Prompt Engineering
- Planning & Reasoning
- Tool Calling (optional)

---

## 📖 Learning Objectives

After exploring this project, you will understand:

- What a Planning Agent is
- Why planning improves AI performance
- How tasks are decomposed
- How execution follows a generated plan
- How to build your own planning-based AI agent

---

## 🔮 Future Improvements

- Dynamic replanning
- Parallel task execution
- Memory integration
- Tool selection
- Web search integration
- Multi-agent collaboration
- Human-in-the-loop approval
- Visual execution graph

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

## 📜 License

This project is intended for educational and learning purposes.

---

## ⭐ Support

If you found this project useful, consider giving the repository a **⭐ Star** to support future Agentic AI tutorials.
