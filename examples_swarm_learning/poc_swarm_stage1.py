import json
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

"""
core indea: a set of steps and the tools to execute them
"""

client = OpenAI()

# Customer Service Routine
# --------------------------------------------stage 1----------------------------------------------------
# The main power of routines is their simplicity and robustness. 
# Notice that these instructions contain conditionals much like a state machine or branching in code. 
# LLMs can actually handle these cases quite robustly for small and medium sized routine, with the added benefit of having "soft" adherance – the LLM can naturally steer the conversation without getting stuck in dead-ends.
system_message = (
    "You are a customer support agent for ACME Inc."
    "Always answer in a sentence or less."
    "Follow the following routine with the user:"
    "1. First, ask probing questions and understand the user's problem deeper.\n"
    " - unless the user has already provided a reason.\n"
    "2. Propose a fix (make one up).\n"
    "3. ONLY if not satesfied, offer a refund.\n"
    "4. If accepted, search for the ID and then execute refund."
    ""
)

# To execute a routine, let's implement a simple loop that goes through the steps.
# 1. Gets user input. 获取用户输入。
# 2. Appends user message to messages.
# 3. Calls the model. 调用模型。
# 4. Appends model response to messages.

def run_full_turn(system_message, messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message}
        ] + messages,
    )
    message = response.choices[0].message
    messages.append(message)

    if message.content: print("Assistant:", message.content)
    return message


messages = []
while True:
    user = input("User: ")
    messages.append({"role": "user", "content": user})
    run_full_turn(system_message, messages)
