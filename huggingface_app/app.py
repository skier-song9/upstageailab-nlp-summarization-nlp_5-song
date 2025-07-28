import gradio as gr
import pandas as pd
import os
import re
# from dotenv import load_dotenv
import sys
import gc

sys.path.append(os.path.dirname(__file__))
from solar_api import retranslate_single

# --- Configuration ---
# Assuming the script is run from the root of the project
# load_dotenv()
DATA_DIR = './data'
TRAIN1_CSV = os.path.join(DATA_DIR, "train_1.csv")
TRAIN2_CSV = os.path.join(DATA_DIR, "train_2.csv")
DEV_CSV = os.path.join(DATA_DIR, "dev.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_SOLAR_CSV = os.path.join(DATA_DIR, "train_solar_results_filtered")
DEV_SOLAR_CSV = os.path.join(DATA_DIR, "val_solar_results_filtered.csv")

# UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

# --- Load Data ---
# try:
t1 = pd.read_csv(TRAIN1_CSV)
t2 = pd.read_csv(TRAIN2_CSV)
train_df = pd.concat([t1,t2])
del t1, t2
gc.collect()
val_df = pd.read_csv(DEV_CSV)
test_df = pd.read_csv(TEST_CSV)
llm_train_df = None
llm_val_df = None

dfs = []
for i in range(8):
    df = pd.read_csv(f"{TRAIN_SOLAR_CSV}{i+1}.csv")
    dfs.append(df)
llm_train_df = pd.concat(dfs)
del dfs, df
gc.collect()
llm_val_df = pd.read_csv(DEV_SOLAR_CSV)

# --- Block 1: Explore by fname ---

def get_NER(dialogue):
    # not implemented
    return None

def get_sample_by_index(dataset_name, index, use_NER=True):
    df = train_df if dataset_name == "train" else val_df
    llm_df = llm_train_df if dataset_name == "train" else llm_val_df
    index = int(index)

    if 0 <= index < len(df):
        sample = df.iloc[index]

        # get NER by NER-model if exists.
        ner = get_NER(sample['dialogue']) if use_NER else None
        if ner is None:
            ner = "NER model is not ready."
# summary_solar_filter,topic_solar_filter,dialogue_ko2en_filter,dialogue_en2ko_filter,re_summary_solar_filter,ner_solar_filter
        try:
            llm_sample = llm_df.iloc[index]
        except:
            llm_sample = {'summary_solar_filter': 'N/A','ner_solar_filter': 'N/A','re_summary_solar_filter': 'N/A','topic_solar_filter': 'N/A', 'dialogue_en2ko_filter': 'N/A'}

        return (sample['fname'], sample['dialogue'], sample['summary'], ner, sample['topic'],
                llm_sample['summary_solar_filter'],llm_sample['ner_solar_filter'],llm_sample['re_summary_solar_filter'],llm_sample['topic_solar_filter'],llm_sample['dialogue_en2ko_filter'], index)
    return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", index

def update_outputs(dataset_name, index):
    fname, dialogue, summary, ner, topic, llm_summary, llm_ner, re_summary, re_topic, re_dialogue, _ = get_sample_by_index(dataset_name, index)
    return fname, f"```\n{dialogue}\n```", f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{topic}\n```", f"```\n{llm_summary}\n```", f"```\n{llm_ner}\n```", f"```\n{re_summary}\n```", f"```\n{re_topic}\n```", f"```\n{re_dialogue}\n```"

def next_sample(dataset_name, index):
    df = train_df if dataset_name == "train" else val_df
    new_index = min(int(index) + 1, len(df) - 1)
    fname, dialogue, summary, ner, topic, llm_summary, llm_ner, re_summary, re_topic, re_dialogue, _ = get_sample_by_index(dataset_name, new_index)
    return new_index, fname, f"```\n{dialogue}\n```", f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{topic}\n```", f"```\n{llm_summary}\n```", f"```\n{llm_ner}\n```", f"```\n{re_summary}\n```", f"```\n{re_topic}\n```", f"```\n{re_dialogue}\n```"

def prev_sample(dataset_name, index):
    new_index = max(int(index) - 1, 0)
    fname, dialogue, summary, ner, topic, llm_summary, llm_ner, re_summary, re_topic, re_dialogue, _ = get_sample_by_index(dataset_name, new_index)
    return new_index, fname, f"```\n{dialogue}\n```", f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{topic}\n```", f"```\n{llm_summary}\n```", f"```\n{llm_ner}\n```", f"```\n{re_summary}\n```", f"```\n{re_topic}\n```", f"```\n{re_dialogue}\n```"

def reset_index_on_split_change(dataset_name):
    # When the dataset changes, reset the index to 0 and update the display
    fname, dialogue, summary, ner, topic, llm_summary, llm_ner, re_summary, re_topic, re_dialogue, _ = get_sample_by_index(dataset_name, 0)
    return 0, fname, f"```\n{dialogue}\n```", f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{topic}\n```", f"```\n{llm_summary}\n```", f"```\n{llm_ner}\n```", f"```\n{re_summary}\n```", f"```\n{re_topic}\n```", f"```\n{re_dialogue}\n```"

def get_sample_by_fname(dataset_name, fname):
    df = train_df if dataset_name == "train" else val_df
    
    # Find the index for the given fname
    if fname in df['fname'].values:
        index = df[df['fname'] == fname].index[0]
        # Use the existing function with the found index
        fname, dialogue, summary, ner, topic, llm_summary, llm_ner, re_summary, re_topic, re_dialogue, _ = get_sample_by_index(dataset_name, index)
        return index, fname, f"```\n{dialogue}\n```", f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{topic}\n```", f"```\n{llm_summary}\n```", f"```\n{llm_ner}\n```", f"```\n{re_summary}\n```", f"```\n{re_topic}\n```", f"```\n{re_dialogue}\n```"
    else:
        # fname not found
        return -1, f"fname '{fname}' not found in {dataset_name} dataset.", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"

def request_solar_api(api_key, model_name, dataset_name, index):
    if not api_key or not api_key.strip():
        error_msg = "```\n[오류] UPSTAGE_API_KEY가 입력되지 않았습니다. API 키를 입력하고 다시 시도해주세요.\n```"
        return error_msg, error_msg, error_msg, error_msg, error_msg

    df = train_df if dataset_name == "train" else val_df
    index = int(index)
    row_data = df.iloc[index]
    # process_row expects a tuple of (index, series)
    row_for_api = (index, row_data)

    try:
        # retranslate_single returns (fname, results_list)
        fname, results = retranslate_single(row_for_api, api_key, model=model_name)
        
        # results is a list: [summary, topic, en2ko, re_summary, ner]
        summary, topic, re_dialogue, re_summary, ner = results
        
        # Match the output order for the UI: 
        # llm_summary_out, llm_ner_out, re_summary_out, re_topic_out, re_dialogue_out
        return f"```\n{summary}\n```", f"```\n{ner}\n```", f"```\n{re_summary}\n```", f"```\n{topic}\n```", f"```\n{re_dialogue}\n```"
    except Exception as e:
        # It's possible the API key is invalid, causing an error inside retranslate_single
        error_msg = f"```\n[오류] API 호출에 실패했습니다. API 키가 잘못되었거나 네트워크 문제일 수 있습니다.\n{e}\n```"
        return error_msg, error_msg, error_msg, error_msg, error_msg

# --- Block 2: Explore by Topic ---

# Merge dataframes
train_df['source'] = 'train'
val_df['source'] = 'val'
merged_df = pd.concat([train_df, val_df], ignore_index=True)
merged_df['topic'] = merged_df['topic'].astype(str)
topics = sorted(merged_df['topic'].unique().tolist())
topic_df_grouped = merged_df.groupby('topic')

def get_topic_data(topic_name):
    if topic_name not in topic_df_grouped.groups:
        return "Topic not found.", "Topic not found."

    topic_df = topic_df_grouped.get_group(topic_name)
    
    # Clean and split topic name into keywords
    keywords = re.sub(r"[,\\.]", " ", topic_name, flags=re.IGNORECASE)
    keywords = re.sub(r"\\s+", " ", topic_name, flags=re.IGNORECASE)
    keywords = keywords.strip().split(" ")

    def highlight_keywords(text, keywords):
        for keyword in keywords:
            text = re.sub(f"({re.escape(keyword)})", rf'<span style="background-color: yellow;">{keyword}</span>', text, flags=re.IGNORECASE)
        return text

    train_output = ""
    val_output = ""

    for _, row in topic_df.iterrows():
        dialogue = highlight_keywords(row['dialogue'], keywords)
        summary = highlight_keywords(row['summary'], keywords)
        
        formatted_output = f"fname: {row['fname']}\n\n"
        formatted_output += f"Dialogue:\n{dialogue}\n\n"
        formatted_output += f"Summary:\n{summary}\n\n"
        formatted_output += ("-" * 20) + "\n"

        if row['source'] == 'train':
            train_output += formatted_output
        elif row['source'] == 'val':
            val_output += formatted_output
            
    return train_output, val_output

def update_topic_display_by_name(topic_name):
    return get_topic_data(topic_name)

def change_topic(change, current_topic_name):
    current_index = topics.index(current_topic_name)
    new_index = (current_index + change + len(topics)) % len(topics)
    new_topic = topics[new_index]
    train_output, val_output = get_topic_data(new_topic)
    return new_topic, train_output, val_output


# --- Gradio Interface ---

with gr.Blocks() as demo:
    gr.Markdown("# Data Explorer for Dialogue Summarization")

    with gr.Tab("Explore by Sample (fname)"):
        # State to hold the current index
        current_index = gr.State(0)
        
        with gr.Row():
            split_select = gr.Radio(["train", "val"], label="Dataset", value="train")
            prev_btn = gr.Button("Previous")
            next_btn = gr.Button("Next")
        
        with gr.Row():
            fname_out = gr.Textbox(label="fname", interactive=True)
            search_btn = gr.Button("Search")

        with gr.Row():
            with gr.Column():
                origin_out = gr.Markdown(label="original dataset")
                dialogue_out = gr.Markdown(label="Dialogue")
                summary_out = gr.Markdown(label="Summary")
                ner_out = gr.Markdown(label='NER')
                topic_out = gr.Markdown(label="Topic")
            with gr.Column():
                with gr.Row():
                    UPSTAGE_API_KEY = gr.Textbox(label="Your Upstage API Key", interactive=True)
                    solar_model_select = gr.Dropdown(["solar-pro2", "solar-pro", "solar-mini"], label="Solar Model", value="solar-pro2")
                    solar_btn = gr.Button("Request")
                llm_summary_desc = gr.Markdown(label="desc1")
                llm_summary_out = gr.Markdown(label="LLM Summary")
                llm_ner_desc = gr.Markdown(label="desc2")
                llm_ner_out = gr.Markdown(label='LLM_NER')
                re_summary_desc = gr.Markdown(label="desc3")
                re_summary_out = gr.Markdown(label="LLM Retranslate Summary")
                re_topic_desc = gr.Markdown(label="desc4")
                re_topic_out = gr.Markdown(label="LLM Retranslate Topic")
                re_dialogue_desc = gr.Markdown(label="desc5")
                re_dialogue_out = gr.Markdown(label="LLM Retranslate dialogue")

        # Initial load
        initial_fname, initial_dialogue, initial_summary, initial_ner, initial_topic, initial_llm_summary_out,initial_llm_ner_out,initial_re_summary_out,initial_re_topic_out,initial_re_dialogue_out, _ = get_sample_by_index("train", 0)
        
        # Set initial values
        demo.load(
            lambda:
                (
                "### Original Dataset", 
                'Solar Summary result','Solar NER result','Retranslated(ko>en>ko) Solar Summary','Retranslated(ko>en>ko) Solar topic','Retranslated(ko>en>ko) Dialogue',
                initial_fname, f"```\n{initial_dialogue}\n```", f"```\n{initial_summary}\n```", f"```\n{initial_ner}\n```", f"```\n{initial_topic}\n```",
                f"```\n{initial_llm_summary_out}\n```", f"```\n{initial_llm_ner_out}\n```", f"```\n{initial_re_summary_out}\n```", f"```\n{initial_re_topic_out}\n```",f"```\n{initial_re_dialogue_out}\n```"
                ),
            inputs=None,
            outputs=[
                origin_out, 
                llm_summary_desc,llm_ner_desc,re_summary_desc,re_topic_desc,re_dialogue_desc,
                fname_out, dialogue_out, summary_out, ner_out, topic_out,
                llm_summary_out,llm_ner_out,re_summary_out,re_topic_out,re_dialogue_out
            ]
        )

        # Event handlers for Block 1
        search_btn.click(
            get_sample_by_fname,
            inputs=[split_select, fname_out],
            outputs=[current_index, fname_out, dialogue_out, summary_out, ner_out, topic_out, llm_summary_out,llm_ner_out,re_summary_out,re_topic_out,re_dialogue_out]
        )
        next_btn.click(
            next_sample,
            inputs=[split_select, current_index],
            outputs=[current_index, fname_out, dialogue_out, summary_out, ner_out, topic_out, llm_summary_out,llm_ner_out,re_summary_out,re_topic_out,re_dialogue_out]
        )
        prev_btn.click(
            prev_sample,
            inputs=[split_select, current_index],
            outputs=[current_index, fname_out, dialogue_out, summary_out, ner_out, topic_out, llm_summary_out,llm_ner_out,re_summary_out,re_topic_out,re_dialogue_out]
        )
        split_select.change(
            reset_index_on_split_change,
            inputs=[split_select],
            outputs=[current_index, fname_out, dialogue_out, summary_out, ner_out, topic_out, llm_summary_out,llm_ner_out,re_summary_out,re_topic_out,re_dialogue_out]
        )
        solar_btn.click(
            request_solar_api,
            inputs=[UPSTAGE_API_KEY, solar_model_select, split_select, current_index],
            outputs=[llm_summary_out, llm_ner_out, re_summary_out, re_topic_out, re_dialogue_out]
        )


    with gr.Tab("Explore by Topic"):
        with gr.Row():
            topic_select = gr.Dropdown(topics, label="Select Topic", value=topics[0])
            prev_topic_btn = gr.Button("Previous Topic")
            next_topic_btn = gr.Button("Next Topic")

        with gr.Row():
            train_topic_display = gr.Markdown(label="Train Data")
            val_topic_display = gr.Markdown(label="Validation Data")

        # Initial load for Block 2
        demo.load(
            lambda: get_topic_data(topics[0]),
            inputs=None,
            outputs=[train_topic_display, val_topic_display]
        )

        # Event handlers for Block 2
        topic_select.change(
            update_topic_display_by_name,
            inputs=[topic_select],
            outputs=[train_topic_display, val_topic_display]
        )
        next_topic_btn.click(
            lambda current_topic: change_topic(1, current_topic),
            inputs=[topic_select],
            outputs=[topic_select, train_topic_display, val_topic_display]
        )
        prev_topic_btn.click(
            lambda current_topic: change_topic(-1, current_topic),
            inputs=[topic_select],
            outputs=[topic_select, train_topic_display, val_topic_display]
        )
    
    # with gr.Tab("Explore Validation file"):

    #     pass


if __name__ == "__main__":
    # To run this script, use the command:
    # python src/app/app_gradio.py
    demo.launch()