from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM

# 학습을 위한 tokenizer와 사전 학습된 모델을 불러옵니다.
def load_tokenizer_and_model_for_train(config, device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)
    print('-'*10, f'Model Name : {config["general"]["model_name"]}', '-'*10,)
    model_name = config['general']['model_name']
    model_config = AutoConfig.from_pretrained(model_name)

    # generatino config를 미리 설정해야 한다.
    if hasattr(model_config, 'num_beams'):
        model_config.num_beams = config['inference']['num_beams'] # 생성된 문장에서 특정 크기의 N-gram이 반복되지 않도록 설정합니다.
    if hasattr(model_config, 'max_length'):
        model_config.max_length = config['inference']['generate_max_length'] # 디코더가 생성할 최대 출력 시퀀스 길이(=토큰 개수)
    if hasattr(model_config, 'no_repeat_ngram_size'):
        model_config.no_repeat_ngram_size = config['inference']['no_repeat_ngram_size'] # 더 나은 문장을 탐색하기 위해 빔 서치(Beam Search)에서 유지할 빔의 개수
    if hasattr(model_config, 'length_penalty'):
        model_config.length_penalty = config['inference']['length_penalty'] # > 1로 설정하면 짧은 문장을 선호하도록 페널티를 강화, 1보다 작은 값을 주면 긴 생성문을 선호하게 된다.

    tokenizer_args_dict = {
        "pretrained_model_name_or_path": model_name
    }
    if 't5' in model_name.lower():
        tokenizer_args_dict['legacy'] = 'false'
        tokenizer_args_dict['use_fast'] = 'false'
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args_dict)
    try:
        generate_model = AutoModelForSeq2SeqLM.from_pretrained(config['general']['model_name'],config=model_config)
    except:
        generate_model = AutoModelForCausalLM.from_pretrained(config['general']['model_name'],config=model_config)
    special_tokens_dict={'additional_special_tokens':config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    # pad token을 명시적으로 model의 config에 추가한다.
    # if hasattr(generate_model.config, "pad_token_id"):
    #     print("="*15,"model에 pad_token_id 명시","="*15)
    #     generate_model.config.pad_token_id = tokenizer.pad_token_id

    generate_model.resize_token_embeddings(len(tokenizer)) # 사전에 special token을 추가했으므로 재구성 해줍니다.
    generate_model.to(device)
    print(generate_model.config)

    print('-'*10, 'Load tokenizer & model complete', '-'*10,)
    return generate_model , tokenizer

def load_tokenizer_and_model_for_inference(config, device):
    tokenizer = AutoTokenizer.from_pretrained(
        config['inference']['ckt_dir']
    )
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(config['inference']['ckt_dir'])
    except:
        model = AutoModelForCausalLM.from_pretrained(config['inference']['ckt_dir'])
    return model.to(device), tokenizer