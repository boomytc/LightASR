from funasr import AutoModel

model = AutoModel(model="model/paraformer_models/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
                  vad_model="model/vad_models/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                  punc_model="model/punc_models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                  # spk_model="cam++", spk_model_revision="v2.0.2",
                  disable_update=True,
                  )
                  
res = model.generate(input=f"{model.model_path}/example/asr_example.wav", 
            batch_size_s=300, 
            hotword='摩哒')
print(res)
