import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

## Function To get response from LLaMA 2 model
def getLLamaResponse(input_text, no_words, story_genre):

    ### LLaMA2 model
    llm = CTransformers(model='llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})
    
    ## Prompt Template
    template = """
        Write a story in the {story_genre} genre with the topic "{input_text}"
        within {no_words} words.
    """
    
    prompt = PromptTemplate(input_variables=["story_genre", "input_text", 'no_words'],
                            template=template)
    
    ## Generate the response from the LLaMA 2 model
    response = llm(prompt.format(story_genre=story_genre, input_text=input_text, no_words=no_words))
    print(response)
    return response


st.set_page_config(page_title="Generate Stories",
                   page_icon='📖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Stories 📖")

input_text = st.text_input("Enter the Story Topic")

## Creating two more columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('Number of Words')
with col2:
    story_genre = st.selectbox('Story Genre',
                               ('Fantasy', 'Sci-Fi', 'Mystery', 'Romance', 'Horror', 'Adventure'), index=0)

submit = st.button("Generate")

## Final response
if submit:
    st.write(getLLamaResponse(input_text, no_words, story_genre))
