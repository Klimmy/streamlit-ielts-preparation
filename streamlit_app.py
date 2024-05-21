import streamlit as st
import replicate
import os
from transformers import AutoTokenizer
import json


# App title
st.set_page_config(page_title="IELTS General Writing Assistant")
st.title("Your IELTS Writing Assistant")
st.markdown("1. To begin, paste your IELTS task and your answer.\n\n2. If you don't have a task, press the button on the left to generate one.\n\n3. Once you have both task and answer, click 'Evaluate' button to get a feedback from LLM Arctic Snowflake.\n\n4. Good luck with your IELTS exam :raised_hands:  ")
st.markdown("**Please make sure to save your answer and/or task somewhere outside of this app to avoid loosing your work!**")


def main():
    """Execution starts here."""
    display_sidebar_ui()
    upload_prompt_templates()
    display_input_fields()
    display_evaluation()
    display_footer()


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to make sure we're not sending too much text
    text to the Model. Eventually we will replace this with ArcticTokenizer
    """
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")


def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)


@st.cache_resource(show_spinner=False)
def get_llamaguard_deployment():
    return replicate.deployments.get("snowflake/llamaguard")


def check_safety(prompt, disable=False) -> bool:
    if disable:
        return True

    # with st.spinner("Checking prompt safety ..."):
    deployment = get_llamaguard_deployment()
    prediction = deployment.predictions.create(
      input = { "prompt": f"[INST] Task: Check if there is unsafe content in 'Agent' messages in conversations according our safety policy with the below categories.\n\n<BEGIN UNSAFE CONTENT CATEGORIES>\nO1: Violence and Hate.\nShould not\n- Help people plan or engage in violence.\n- Encourage people to engage in violence.\n- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.\n- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.\nCan\n- Provide information on violence and discrimination.\n- Discuss the topics of hate and violence, and can discuss historical events involving violence.\nO2: Sexual Content.\nShould not\n- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.\nCan\n- Discuss topics related to sexuality and sexual education.\n- Provide general information regarding sexual activities.\n- Provide guidance on topics related to sexual health.\nO3: Criminal Planning.\nShould not\n- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.\n- Encourage people to engage in criminal activities or condone criminal behavior.\nCan\n- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.\n- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.\nO4: Guns and Illegal Weapons.\nShould not\n- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.\n- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.\nCan\n- Discuss firearms and the arguments for and against firearm ownership.\nO5: Regulated or Controlled Substances.\nShould not\n- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.\n- Assist or encourage people to create such substances when it is illegal to do so.\nCan\n- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).\n- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.\nO6: Self-Harm.\nShould not\n- Encourage people to harm themselves.\n- Romanticize or condone self-harm or suicide.\n- Provide information regarding the methods of suicide or self-harm.\n- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.\nCan\n- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).\nShould\n- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.\n<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\n\n{prompt}\n\n<END CONVERSATION>\n\nProvide your safety assessment for Agent in the above conversation:\n- First line must read 'safe' or 'unsafe'.\n- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]" }
    )
    prediction.wait()
    output = prediction.output

    if output is not None and "unsafe" in output:
        return False
    else:
        return True


def get_text_between(text, start_word, end_word=None):
    try:
        start_index = text.find(start_word)

        if start_index == -1:
            return ""  # start_word not found

        start_index += len(start_word)

        if end_word:
            end_index = text.find(end_word, start_index)

            if end_index == -1:
                return ""
            return text[start_index:end_index].strip()
        else:
            return text[start_index:].strip()

    except TypeError:
        return ""


# Function for generating Snowflake Arctic response
def generate_arctic_response(prompt):
    if get_num_tokens(prompt) >= 2000:
        st.error("Essay is too long. Please keep it under 2000 tokens.")
        st.stop()

    for event in replicate.stream("snowflake/snowflake-arctic-instruct",
                           input={"prompt": prompt,
                                  "temperature": 1.00,
                                  "top_p": 1.00,
                                  "max_new_tokens": 2000
                                  }):
        yield str(event)


def generate_task_text():
    response = generate_arctic_response(st.session_state.generation_prompt)
    full_response = "".join(response)
    if not check_safety(full_response, disable=True):
        st.write("Something went wrong with safety response of the LLM")
        st.session_state.task_description = ""
    else:
        st.session_state.task_description = full_response


def generate_evaluation_response():
    st.session_state.evaluation_prompt = st.session_state.evaluation_prompt.replace("{prompt_task}", st.session_state.task_description)
    st.session_state.evaluation_prompt = st.session_state.evaluation_prompt.replace("{prompt_answer}", st.session_state.task_answer)
    response = generate_arctic_response(st.session_state.evaluation_prompt)
    full_response = "".join(response)
    try:
        response_parsed = json.loads(full_response, strict=False)
    except Exception as e:
        response_parsed = {
            "feedback": get_text_between(full_response, '"feedback"', '"band"'),
            "band": get_text_between(full_response, '"band"', '"improved_answer"'),
            "improved_answer": get_text_between(full_response, '"improved_answer"')
            }

    if not check_safety(response_parsed, disable=True):
        st.write("Something went wrong with safety response of the LLM")
    else:
        st.session_state.evaluated_answer = response_parsed


def upload_prompt_templates():
    if st.session_state.task_option == 'Task 1 (General)':
        with open('prompt_templates/task_1_generation.txt') as f:
            st.session_state.generation_prompt = f.read()
        with open('prompt_templates/task_1_evaluation.txt') as f:
            st.session_state.evaluation_prompt = f.read()
    elif st.session_state.task_option == 'Task 2 (General/Academic)':
        with open('prompt_templates/task_2_generation.txt') as f:
            st.session_state.generation_prompt = f.read()
        with open('prompt_templates/task_2_evaluation.txt') as f:
            st.session_state.evaluation_prompt = f.read()


# Replicate Credentials
def display_sidebar_ui():
    with st.sidebar:
        st.title('Settings')
        if 'REPLICATE_API_TOKEN' in st.secrets:
            replicate_api = st.secrets['REPLICATE_API_TOKEN']
        else:
            replicate_api = st.text_input('Enter Replicate API token:', type='password')
            if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
                st.warning('Please enter your Replicate API token.', icon='⚠️')
                st.markdown("**Don't have an API token?** Head over to [Replicate](https://replicate.com) to sign up for one.")

        os.environ['REPLICATE_API_TOKEN'] = replicate_api

        st.session_state.task_option = st.selectbox(
            'Select the Task',
             ['Task 1 (General)', 'Task 2 (General/Academic)']
             )

        st.button('Generate random task text', key="user_requested_task_generation", on_click=generate_task_text)


def display_input_fields():
    st.text_area("Task description:", key="task_description", height=200, max_chars=2000, placeholder=f'Paste IELTS Writing {st.session_state.task_option} here or click the button on the left to generate a random one')
    st.text_area("Your Answer:", key="task_answer", height=200, max_chars=2000, placeholder='Type or paste your answer here and click "Evalualte" button below')
    st.button('Evaluate', key="evaluation_requested", on_click=generate_evaluation_response)


def display_evaluation():
    if "evaluated_answer" in st.session_state:
        left_column, middle_column, right_column = st.columns(3)
        with left_column:
            st.metric("**Band:**", st.session_state.evaluated_answer['band'])
        with middle_column:
            st.markdown("**Feedback:**\n\n" + st.session_state.evaluated_answer["feedback"])
        with right_column:
            st.markdown("**Improved answer:**\n\n" + st.session_state.evaluated_answer["improved_answer"])


def display_footer():
    st.markdown("---")
    st.markdown("Made by [Kliment Merzlyakov](https://www.linkedin.com/in/kmerzlyakov/)")
    st.markdown("Check out my [Analytics Engineering blog](https://klimmy.hashnode.dev/)")


if __name__ == "__main__":
    main()
