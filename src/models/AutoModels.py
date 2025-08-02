from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast, BatchEncoding

class CustomPreTrainedTokenizerFast(PreTrainedTokenizerFast):
    """
    BOS 토큰을 자동으로 추가하지 않도록 tokenize 함수의 기본 동작을 변경한다.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __call__(
            self, 
            text = None, 
            text_pair = None, 
            text_target = None, 
            text_pair_target = None, 
            add_special_tokens = True, 
            padding = False, 
            truncation = None, 
            max_length = None, 
            stride = 0, 
            is_split_into_words = False, 
            pad_to_multiple_of = None, 
            return_tensors = None, 
            return_token_type_ids = None, 
            return_attention_mask = None, 
            return_overflowing_tokens = False, 
            return_special_tokens_mask = False, 
            return_offsets_mapping = False, 
            return_length = False, 
            verbose = True, 
            **kwargs
        ) -> BatchEncoding:
        # 상위 클래스의 __call__ 메서드를 호출하여 토큰화 결과를 받습니다.
        # 이 시점에는 bos_token과 eos_token이 모두 포함됩니다.
        outputs = super().__call__(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )
        # BOS 토큰이 존재하고, 이를 제거해야 하는 경우에만 로직을 실행합니다.
        if self.bos_token_id is not None:
            # outputs가 BatchEncoding 객체이므로 내부 딕셔너리 값들을 수정합니다.
            if outputs['input_ids'][0] == self.bos_token_id:
                # bos_token으로 시작할 경우 bos_token을 제거한다.
                outputs['input_ids'] = outputs['input_ids'][1:] 
                # attention_mask가 있다면 함께 수정합니다.
                if 'attention_mask' in outputs:
                    outputs['attention_mask'] = outputs['attention_mask'][1:]
        # EOS 토큰이 존재하고, 이를 제거해야 하는 경우에만 로직을 실행합니다.
        if self.eos_token_id is not None:
            # outputs가 BatchEncoding 객체이므로 내부 딕셔너리 값들을 수정합니다.
            if outputs['input_ids'][-1] == self.eos_token_id:
                # bos_token으로 시작할 경우 bos_token을 제거한다.
                outputs['input_ids'] = outputs['input_ids'][:-1] 
                # attention_mask가 있다면 함께 수정합니다.
                if 'attention_mask' in outputs:
                    outputs['attention_mask'] = outputs['attention_mask'][:-1]        

        if return_tensors is not None:
            import torch
            from transformers.file_utils import PaddingStrategy

            outputs_as_tensors = {}
            for key, value in outputs.items():
                if isinstance(value, list) and isinstance(value[0], list):
                    # outputs_as_tensors[key] = torch.tensor(value)
                    # 패딩이 필요할 수 있으므로, from_list를 사용합니다.
                    if key == 'input_ids':
                        pad_token = self.pad_token_id
                    else:
                        pad_token = 0 # special_tokens_mask나 attention_mask의 패딩 값
                    
                    max_len = max(len(sublist) for sublist in value)
                    padded_list = [sublist + [pad_token] * (max_len - len(sublist)) for sublist in value]
                    outputs_as_tensors[key] = torch.tensor(padded_list)
                else:
                    outputs_as_tensors[key] = torch.tensor(value)
            
            return BatchEncoding(outputs_as_tensors)
            
        return outputs
        

# 학습을 위한 tokenizer와 사전 학습된 모델을 불러옵니다.
def load_tokenizer_and_model_for_train(config, device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)
    print('-'*10, f'Model Name : {config["general"]["model_name"]}', '-'*10,)
    model_name = config['general']['model_name']
    model_config = AutoConfig.from_pretrained(model_name)

    # generation config를 미리 설정해야 한다.
    if hasattr(model_config, 'num_beams'):
        model_config.num_beams = config['inference']['num_beams'] # 생성된 문장에서 특정 크기의 N-gram이 반복되지 않도록 설정합니다.
    if hasattr(model_config, 'max_length'):
        model_config.max_length = config['inference']['generate_max_length'] # 디코더가 생성할 최대 출력 시퀀스 길이(=토큰 개수)
    if hasattr(model_config, 'no_repeat_ngram_size'):
        model_config.no_repeat_ngram_size = config['inference']['no_repeat_ngram_size'] # 더 나은 문장을 탐색하기 위해 빔 서치(Beam Search)에서 유지할 빔의 개수
    if hasattr(model_config, 'length_penalty'):
        model_config.length_penalty = config['inference']['length_penalty'] # > 1로 설정하면 짧은 문장을 선호하도록 페널티를 강화, 1보다 작은 값을 주면 긴 생성문을 선호하게 된다.

    tokenizer_args_dict = {
        "pretrained_model_name_or_path": model_name,
        "legacy": False
    }
    if 't5' in model_name.lower():
        tokenizer_args_dict['use_fast'] = False # 만약 T5 toeknizer 로드 중 오류가 발생한다면 'false' 문자열로 변경
        # fast_tokenizer를 사용하지 못하는 경우, slow tokenizer를 사용하여 unkown 문자에 대한 토큰화를 진행한다.
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args_dict)

    # tokenizer가 PreTrainedTokenizerFast라면 CustomPreTrainedTokenizerFast으로 바꿔야 한다.
    # 경고는 무시해도 괜찮다.
    if "PreTrainedTokenizerFast" in str(tokenizer.__class__):
        print("=== Change PreTrainedTokenizerFast to CustomPreTrainedTokenizerFast ===")
        tokenizer = CustomPreTrainedTokenizerFast.from_pretrained(**tokenizer_args_dict)

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